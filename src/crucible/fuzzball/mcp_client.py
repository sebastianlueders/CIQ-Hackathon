"""Native JSON-RPC 2.0 stdio client for CIQ Fuzzball 4.2 MCP (`fuzzball mcp serve`).

Dynamically inspects `tools/list` `inputSchema.properties` at connection time so every tool call
(`workflow_validate`, `workflow_start`, `workflow_wait`, `workflow_describe`, `workflow_log`,
`object_get`, `cluster_list`, `node_list`, `volume_list`, `workflow_list`) maps onto the exact
argument names and JSON types advertised by the running `fuzzball` binary — eliminating any
`unexpected additional properties` errors across Fuzzball 4.x builds.

All payloads returned by MCP are treated strictly as untrusted data.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)

FUZZFILE_ARG_ALIASES: tuple[str, ...] = (
    "fuzzfile",
    "content",
    "yaml",
    "spec",
    "workflow_yaml",
    "workflow_spec",
    "definition",
    "template",
    "body",
    "text",
    "source",
    "path",
    "file",
    "filename",
    "fuzzfile_path",
    "workflow_path",
    "workflow_file",
    "workflow",
)
PATH_ONLY_PROP_NAMES = frozenset(
    {"path", "file", "filename", "fuzzfile_path", "workflow_path", "workflow_file"}
)
WORKFLOW_META_EXCLUSIONS = frozenset(
    {
        "cluster_id",
        "clusterId",
        "name",
        "workflow_name",
        "workflowName",
        "display_name",
        "displayName",
        "values",
        "priority",
        "preemptible",
        "confirm_endpoint_takeover",
        "confirmEndpointTakeover",
        "dry_run",
        "dryRun",
    }
)


def _norm_key(key: str) -> str:
    return key.replace("-", "").replace("_", "").lower()


class FuzzballMCPError(RuntimeError):
    """Raised on MCP transport, protocol, or tool-execution errors."""


class FuzzballMCPStdioClient:
    """Schema-aware JSON-RPC 2.0 stdio client speaking to `fuzzball mcp serve`."""

    def __init__(
        self,
        command: list[str] | None = None,
        allow_write: bool = False,
        allow_exec: bool = False,
        env: dict[str, str] | None = None,
    ) -> None:
        if command is not None:
            self.command = list(command)
        else:
            env_cmd = os.getenv("CRUCIBLE_FUZZBALL_MCP_CMD", "").strip()
            if env_cmd:
                self.command = shlex.split(env_cmd)
            else:
                fb_exe = "fuzzball"
                if shutil.which(fb_exe) is None and Path("./fuzzball").is_file():
                    fb_exe = str(Path("./fuzzball").resolve())
                self.command = [fb_exe, "mcp", "serve"]
                if allow_write:
                    self.command.append("--allow-write")
                if allow_exec:
                    self.command.append("--allow-exec")

        self.env = env
        self._proc: subprocess.Popen[str] | None = None
        self._next_id = 1
        self.advertised_tools: set[str] = set()
        self.tool_schemas: dict[str, dict[str, Any]] = {}
        self.tool_descriptions: dict[str, str] = {}

    @staticmethod
    def is_binary_available(fuzzball_bin: str = "fuzzball") -> bool:
        env_cmd = os.getenv("CRUCIBLE_FUZZBALL_MCP_CMD", "").strip()
        if env_cmd:
            argv = shlex.split(env_cmd)
            return bool(argv) and (shutil.which(argv[0]) is not None or Path(argv[0]).exists())
        return shutil.which(fuzzball_bin) is not None or Path("./fuzzball").is_file()

    def __enter__(self) -> "FuzzballMCPStdioClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:
        if self._proc is not None:
            return
        exe = self.command[0]
        if shutil.which(exe) is None and not Path(exe).exists():
            raise FuzzballMCPError(f"Fuzzball MCP executable not found: {exe!r}")

        merged_env = os.environ.copy()
        if self.env:
            merged_env.update(self.env)

        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=merged_env,
        )
        self._rpc_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "crucible-autoresearch", "version": "0.2.0"},
            },
        )
        self._send_notification("notifications/initialized", {})
        tools_res = self._rpc_request("tools/list", {})
        for tool in tools_res.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            tname = tool.get("name")
            if not isinstance(tname, str):
                continue
            self.advertised_tools.add(tname)
            schema = tool.get("inputSchema") or tool.get("input_schema") or {}
            if isinstance(schema, dict):
                self.tool_schemas[tname] = schema
            desc = tool.get("description")
            if isinstance(desc, str):
                self.tool_descriptions[tname] = desc

    def dump_tool_catalog(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "description": self.tool_descriptions.get(name, ""),
                "inputSchema": self.tool_schemas.get(name, {}),
            }
            for name in sorted(self.advertised_tools)
        }

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=2.0)
        except Exception:
            self._proc.kill()
        finally:
            self._proc = None

    def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise FuzzballMCPError("MCP session is not connected.")
        msg = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
        self._proc.stdin.write(msg + "\n")
        self._proc.stdin.flush()

    def _rpc_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise FuzzballMCPError("MCP session is not connected.")

        req_id = self._next_id
        self._next_id += 1
        req_line = json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        self._proc.stdin.write(req_line + "\n")
        self._proc.stdin.flush()

        while True:
            line = self._proc.stdout.readline()
            if not line:
                stderr_tail = ""
                if self._proc.stderr:
                    stderr_tail = self._proc.stderr.read()[-500:]
                raise FuzzballMCPError(
                    f"Fuzzball MCP server closed stdout while awaiting '{method}'. stderr: {stderr_tail}"
                )
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            if msg.get("id") != req_id:
                continue
            if "error" in msg:
                raise FuzzballMCPError(f"MCP RPC error on '{method}': {msg['error']}")
            result = msg.get("result")
            return result if isinstance(result, dict) else {"value": result}

    # --- Dynamic `inputSchema` Introspection & Argument Mapping ---

    def _schema_info(self, tool_name: str) -> tuple[dict[str, Any] | None, list[str]]:
        schema = self.tool_schemas.get(tool_name)
        if not isinstance(schema, dict) or "properties" not in schema:
            return None, []
        props = schema.get("properties")
        if not isinstance(props, dict):
            return None, []
        req = schema.get("required")
        req_list = [r for r in req if isinstance(r, str)] if isinstance(req, list) else []
        return props, req_list

    def resolve_argument_key(
        self,
        tool_name: str,
        aliases: tuple[str, ...],
        *,
        fallback_single_string_excluding: frozenset[str] | set[str] | None = None,
    ) -> str | None:
        """Resolve a logical parameter against the live `inputSchema.properties` for `tool_name`.

        - If `tools/list` did not advertise a `properties` schema for this tool, returns `aliases[0]`.
        - If `tools/list` advertised `properties`, ONLY ever returns a key present in `properties`
          (returning `None` when the parameter is unsupported by this server version).
        """
        props, required = self._schema_info(tool_name)
        if props is None:
            return aliases[0] if aliases else None
        if not props:
            return None

        # 1. Exact match in priority order
        for cand in aliases:
            if cand in props:
                return cand

        # 2. Normalized snake_case / camelCase match
        norm_to_actual = {_norm_key(k): k for k in props}
        for cand in aliases:
            actual = norm_to_actual.get(_norm_key(cand))
            if actual is not None:
                return actual

        # 3. Single remaining required/string property fallback
        if fallback_single_string_excluding is not None:
            excluded_norm = {_norm_key(x) for x in fallback_single_string_excluding}
            req_candidates = [
                k
                for k in required
                if k in props
                and _norm_key(k) not in excluded_norm
                and self._prop_accepts_string(props.get(k))
            ]
            if len(req_candidates) == 1:
                return req_candidates[0]
            str_candidates = [
                k
                for k, p_spec in props.items()
                if _norm_key(k) not in excluded_norm and self._prop_accepts_string(p_spec)
            ]
            if len(str_candidates) == 1:
                return str_candidates[0]

        return None

    @staticmethod
    def _prop_accepts_string(prop_spec: Any) -> bool:
        if not isinstance(prop_spec, dict):
            return True
        ptype = prop_spec.get("type")
        if ptype is None:
            return True
        if isinstance(ptype, str):
            return ptype == "string"
        if isinstance(ptype, list):
            return "string" in ptype
        return True

    def _coerce_for_prop(
        self,
        tool_name: str,
        prop_name: str,
        value: Any,
        *,
        is_duration_seconds: bool = False,
    ) -> Any:
        props, _ = self._schema_info(tool_name)
        if not props or prop_name not in props or not isinstance(props[prop_name], dict):
            return value
        pspec = props[prop_name]
        ptype = pspec.get("type")
        if is_duration_seconds and ptype == "string" and isinstance(value, (int, float)):
            return f"{int(value)}s"
        if ptype == "integer" and isinstance(value, (int, float, str)):
            return int(float(value))
        if ptype == "number" and isinstance(value, (int, float, str)):
            return float(value)
        if ptype == "boolean" and not isinstance(value, bool):
            return bool(value)
        return value

    def _build_fuzzfile_args(
        self,
        tool_name: str,
        fuzzfile_yaml: str,
        fuzzfile_path: Path | str | None = None,
    ) -> tuple[dict[str, Any], Path | None]:
        """Construct the Fuzzfile argument dict matching `tool_name`'s exact `inputSchema`."""
        key = self.resolve_argument_key(
            tool_name,
            FUZZFILE_ARG_ALIASES,
            fallback_single_string_excluding=WORKFLOW_META_EXCLUSIONS,
        )
        if key is None:
            props, _ = self._schema_info(tool_name)
            raise FuzzballMCPError(
                f"Could not find a Fuzzfile parameter on MCP tool '{tool_name}'. "
                f"Advertised properties: {sorted((props or {}).keys())}"
            )

        props, _ = self._schema_info(tool_name)
        pspec = (props or {}).get(key) if isinstance(props, dict) else None
        desc_lower = str((pspec or {}).get("description", "")).lower() if isinstance(pspec, dict) else ""
        wants_file_path = key in PATH_ONLY_PROP_NAMES or (
            ("path" in desc_lower or "file" in desc_lower)
            and "yaml" not in desc_lower
            and "string" not in desc_lower
            and "content" not in desc_lower
        )

        temp_created: Path | None = None
        if wants_file_path:
            if fuzzfile_path is not None and Path(fuzzfile_path).exists():
                return {key: str(Path(fuzzfile_path).resolve())}, None
            fd, tmp_name = tempfile.mkstemp(prefix="crucible-", suffix=".yaml")
            os.close(fd)
            temp_created = Path(tmp_name)
            temp_created.write_text(fuzzfile_yaml, encoding="utf-8")
            return {key: str(temp_created)}, temp_created

        return {key: fuzzfile_yaml}, None

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        # Final safety filter: strip any keys not in `inputSchema.properties` when schema is present
        props, _ = self._schema_info(name)
        raw_args = dict(arguments or {})
        if props is not None:
            raw_args = {k: v for k, v in raw_args.items() if k in props}

        res = self._rpc_request(
            "tools/call",
            {"name": name, "arguments": raw_args},
        )
        if res.get("isError"):
            raise FuzzballMCPError(f"Fuzzball MCP tool '{name}' returned error: {res}")
        if "structuredContent" in res:
            return res["structuredContent"]
        content = res.get("content") or []
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and isinstance(first.get("text"), str):
                txt = first["text"].strip()
                try:
                    return json.loads(txt)
                except json.JSONDecodeError:
                    return txt
        return res

    # --- CIQ Fuzzball 4.2 Official Read-Tier Tools ---

    def context_show(self) -> Any:
        return self.call_tool("context_show", {})

    def cluster_list(self) -> Any:
        return self.call_tool("cluster_list", {})

    def node_list(self, fields: list[str] | None = None) -> Any:
        args: dict[str, Any] = {}
        fkey = self.resolve_argument_key("node_list", ("fields",))
        if fkey is not None:
            args[fkey] = fields or ["id", "resource"]
        return self.call_tool("node_list", args)

    def volume_list(self) -> Any:
        return self.call_tool("volume_list", {})

    def workflow_list(self, status: str | None = None) -> Any:
        args: dict[str, Any] = {}
        if status:
            skey = self.resolve_argument_key("workflow_list", ("status",))
            if skey is not None:
                args[skey] = status
            else:
                flt_key = self.resolve_argument_key("workflow_list", ("filter",))
                if flt_key is not None:
                    args[flt_key] = f'status = "{status}"'
        return self.call_tool("workflow_list", args)

    def workflow_validate(
        self,
        fuzzfile_yaml: str,
        fuzzfile_path: Path | str | None = None,
    ) -> dict[str, Any]:
        args, tmp_path = self._build_fuzzfile_args("workflow_validate", fuzzfile_yaml, fuzzfile_path)
        try:
            try:
                out = self.call_tool("workflow_validate", args)
            except FuzzballMCPError as exc:
                # If a single `fuzzfile` prop happened to expect a file path rather than inline YAML
                if (
                    tmp_path is None
                    and fuzzfile_path is not None
                    and Path(fuzzfile_path).exists()
                    and ("no such file" in str(exc).lower() or "file name too long" in str(exc).lower())
                ):
                    only_key = next(iter(args.keys()))
                    out = self.call_tool(
                        "workflow_validate", {only_key: str(Path(fuzzfile_path).resolve())}
                    )
                else:
                    raise
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
        return out if isinstance(out, dict) else {"valid": bool(out), "errors": [], "warnings": []}

    def workflow_describe(self, workflow_id: str, include_spec: bool = False) -> dict[str, Any]:
        args: dict[str, Any] = {}
        id_key = self.resolve_argument_key(
            "workflow_describe",
            ("workflow_id", "id", "workflowId", "uuid", "workflow", "name"),
            fallback_single_string_excluding={"cluster_id", "include_spec", "includeSpec"},
        )
        if id_key is not None:
            args[id_key] = workflow_id
        if include_spec:
            spec_key = self.resolve_argument_key("workflow_describe", ("include_spec", "includeSpec"))
            if spec_key is not None:
                args[spec_key] = True
        out = self.call_tool("workflow_describe", args)
        return out if isinstance(out, dict) else {"raw": str(out)}

    def workflow_wait(self, workflow_id: str, timeout_seconds: int = 7200) -> dict[str, Any]:
        args: dict[str, Any] = {}
        id_key = self.resolve_argument_key(
            "workflow_wait",
            ("workflow_id", "id", "workflowId", "uuid", "workflow", "name"),
            fallback_single_string_excluding={"timeout", "timeout_seconds", "cluster_id"},
        )
        if id_key is not None:
            args[id_key] = workflow_id
        t_key = self.resolve_argument_key(
            "workflow_wait",
            ("timeout_seconds", "timeout", "wait_timeout", "max_wait"),
        )
        if t_key is not None:
            args[t_key] = self._coerce_for_prop(
                "workflow_wait", t_key, timeout_seconds, is_duration_seconds=True
            )
        out = self.call_tool("workflow_wait", args)
        return out if isinstance(out, dict) else {"status": str(out)}

    def workflow_log(self, workflow_id: str, job_name: str) -> str:
        args: dict[str, Any] = {}
        id_key = self.resolve_argument_key(
            "workflow_log",
            ("workflow_id", "id", "workflowId", "uuid", "workflow"),
        )
        if id_key is not None:
            args[id_key] = workflow_id
        j_key = self.resolve_argument_key(
            "workflow_log",
            ("job_name", "job", "job_id", "jobId", "stage", "task", "name"),
        )
        if j_key is not None and j_key != id_key:
            args[j_key] = job_name
        out = self.call_tool("workflow_log", args)
        return out if isinstance(out, str) else json.dumps(out)

    def object_get(self, uri: str, max_inline_bytes: int = 65536) -> Any:
        ref_key = self.resolve_argument_key(
            "object_get",
            ("uri", "ref", "reference", "key", "name", "object", "path", "source"),
            fallback_single_string_excluding={"max_inline_bytes", "maxInlineBytes", "cluster_id"},
        )
        limit_key = self.resolve_argument_key(
            "object_get",
            ("max_inline_bytes", "maxInlineBytes", "max_bytes", "limit"),
        )

        # Try full `fuzzball://scope/...` URI first, then stripped variants if server expects bare ref
        candidates = [uri]
        if uri.startswith("fuzzball://"):
            stripped_scheme = uri.removeprefix("fuzzball://")
            candidates.append(stripped_scheme)
            if "/" in stripped_scheme:
                candidates.append(stripped_scheme.split("/", 1)[1])

        last_err: Exception | None = None
        for cand_ref in candidates:
            args: dict[str, Any] = {}
            if ref_key is not None:
                args[ref_key] = cand_ref
            if limit_key is not None:
                args[limit_key] = self._coerce_for_prop("object_get", limit_key, max_inline_bytes)
            try:
                return self.call_tool("object_get", args)
            except FuzzballMCPError as exc:
                last_err = exc
        assert last_err is not None
        raise last_err

    # --- CIQ Fuzzball 4.2 Official Exec-Tier Tools (`--allow-exec`) ---

    def workflow_start(
        self,
        fuzzfile_yaml: str,
        name: str,
        fuzzfile_path: Path | str | None = None,
        confirm_endpoint_takeover: bool = False,
    ) -> str:
        args, tmp_path = self._build_fuzzfile_args("workflow_start", fuzzfile_yaml, fuzzfile_path)
        name_key = self.resolve_argument_key(
            "workflow_start",
            ("name", "workflow_name", "workflowName", "display_name"),
        )
        if name_key is not None and name_key not in args:
            args[name_key] = name
        if confirm_endpoint_takeover:
            takeover_key = self.resolve_argument_key(
                "workflow_start",
                ("confirm_endpoint_takeover", "confirmEndpointTakeover"),
            )
            if takeover_key is not None:
                args[takeover_key] = True

        try:
            try:
                out = self.call_tool("workflow_start", args)
            except FuzzballMCPError as exc:
                if (
                    tmp_path is None
                    and fuzzfile_path is not None
                    and Path(fuzzfile_path).exists()
                    and ("no such file" in str(exc).lower() or "file name too long" in str(exc).lower())
                ):
                    fz_key = next(k for k in args if k != name_key)
                    retry_args = dict(args)
                    retry_args[fz_key] = str(Path(fuzzfile_path).resolve())
                    out = self.call_tool("workflow_start", retry_args)
                else:
                    raise
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

        if isinstance(out, dict):
            wf_id = out.get("workflowId") or out.get("workflow_id") or out.get("id")
            if isinstance(wf_id, str) and wf_id:
                return wf_id
        if isinstance(out, str):
            m = UUID_RE.search(out)
            if m:
                return m.group(0)
        raise FuzzballMCPError(f"Unexpected workflow_start response: {out!r}")
