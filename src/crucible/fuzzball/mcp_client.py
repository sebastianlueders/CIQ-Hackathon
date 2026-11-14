"""Native JSON-RPC 2.0 stdio client for CIQ Fuzzball 4.2 MCP (`fuzzball mcp serve`).

Implements persistent stdio session management over `fuzzball mcp serve [--allow-write] [--allow-exec]`
and typed wrappers for CIQ Fuzzball 4.2 MCP tools:
- Read tier  : `cluster_list`, `node_list`, `volume_list`, `workflow_list`, `workflow_validate`,
               `workflow_wait`, `workflow_log`, `object_get`
- Exec tier  : `workflow_start`

All payloads returned by MCP are treated strictly as untrusted data.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any


class FuzzballMCPError(RuntimeError):
    """Raised on MCP transport, protocol, or tool-execution errors."""


class FuzzballMCPStdioClient:
    """JSON-RPC 2.0 stdio client speaking to `fuzzball mcp serve`."""

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
                self.command = ["fuzzball", "mcp", "serve"]
                if allow_write:
                    self.command.append("--allow-write")
                if allow_exec:
                    self.command.append("--allow-exec")

        self.env = env
        self._proc: subprocess.Popen[str] | None = None
        self._next_id = 1
        self.advertised_tools: set[str] = set()

    @staticmethod
    def is_binary_available(fuzzball_bin: str = "fuzzball") -> bool:
        env_cmd = os.getenv("CRUCIBLE_FUZZBALL_MCP_CMD", "").strip()
        if env_cmd:
            argv = shlex.split(env_cmd)
            return bool(argv) and (shutil.which(argv[0]) is not None or Path(argv[0]).exists())
        return shutil.which(fuzzball_bin) is not None

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
                "clientInfo": {"name": "crucible-autoresearch", "version": "0.1.0"},
            },
        )
        self._send_notification("notifications/initialized", {})
        tools_res = self._rpc_request("tools/list", {})
        for tool in tools_res.get("tools") or []:
            if isinstance(tool, dict) and isinstance(tool.get("name"), str):
                self.advertised_tools.add(tool["name"])

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

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        res = self._rpc_request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
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

    def cluster_list(self) -> Any:
        return self.call_tool("cluster_list", {})

    def node_list(self, fields: list[str] | None = None) -> Any:
        return self.call_tool("node_list", {"fields": fields or ["id", "resource"]})

    def volume_list(self) -> Any:
        return self.call_tool("volume_list", {})

    def workflow_list(self, status: str | None = None) -> Any:
        args: dict[str, Any] = {}
        if status:
            args["status"] = status
        return self.call_tool("workflow_list", args)

    def workflow_validate(self, fuzzfile_yaml: str) -> dict[str, Any]:
        out = self.call_tool("workflow_validate", {"workflow": fuzzfile_yaml})
        return out if isinstance(out, dict) else {"valid": bool(out), "errors": [], "warnings": []}

    def workflow_describe(self, workflow_id: str, include_spec: bool = False) -> dict[str, Any]:
        out = self.call_tool(
            "workflow_describe",
            {"workflow_id": workflow_id, "include_spec": include_spec},
        )
        return out if isinstance(out, dict) else {"raw": str(out)}

    def workflow_wait(self, workflow_id: str, timeout_seconds: int = 7200) -> dict[str, Any]:
        out = self.call_tool(
            "workflow_wait",
            {"workflow_id": workflow_id, "timeout_seconds": timeout_seconds},
        )
        return out if isinstance(out, dict) else {"status": str(out)}

    def workflow_log(self, workflow_id: str, job_name: str) -> str:
        out = self.call_tool("workflow_log", {"workflow_id": workflow_id, "job_name": job_name})
        return out if isinstance(out, str) else json.dumps(out)

    def object_get(self, uri: str, max_inline_bytes: int = 65536) -> Any:
        return self.call_tool(
            "object_get",
            {"uri": uri, "max_inline_bytes": max_inline_bytes},
        )

    # --- CIQ Fuzzball 4.2 Official Exec-Tier Tools (`--allow-exec`) ---

    def workflow_start(
        self,
        fuzzfile_yaml: str,
        name: str,
        confirm_endpoint_takeover: bool = False,
    ) -> str:
        args: dict[str, Any] = {"workflow": fuzzfile_yaml, "name": name}
        if confirm_endpoint_takeover:
            args["confirm_endpoint_takeover"] = True
        out = self.call_tool("workflow_start", args)
        if isinstance(out, dict):
            wf_id = out.get("workflowId") or out.get("workflow_id") or out.get("id")
            if isinstance(wf_id, str) and wf_id:
                return wf_id
        raise FuzzballMCPError(f"Unexpected workflow_start response: {out!r}")
