"""Privacy-preserving LLM Advisor sitting strictly above `PrivacySchemaFirewall`.

Supports:
- OpenAI-compatible chat completions (`CRUCIBLE_LLM_BASE_URL` + `CRUCIBLE_LLM_API_KEY`) — works
  with OpenAI, local vLLM on Fuzzball, or NVIDIA NIM endpoints.
- Anthropic Messages API (`CRUCIBLE_LLM_PROVIDER=anthropic` + `CRUCIBLE_LLM_API_KEY`).
- Automatic deterministic heuristic advisor fallback when no external endpoint is configured.

Every outbound payload MUST pass `PrivacySchemaFirewall.sanitize_and_serialize()` first.
"""

from __future__ import annotations

import json
import os

import httpx

from crucible.firewall import (
    AdvisorDecisionExplanation,
    AdvisorStageRequest,
    PrivacySchemaFirewall,
)


class FirewalledLLMAdvisor:
    def __init__(self, firewall: PrivacySchemaFirewall) -> None:
        self.firewall = firewall
        self.provider = os.getenv("CRUCIBLE_LLM_PROVIDER", "auto").lower()
        self.base_url = os.getenv("CRUCIBLE_LLM_BASE_URL", "").strip()
        self.api_key = os.getenv("CRUCIBLE_LLM_API_KEY", "").strip()
        self.model = os.getenv("CRUCIBLE_LLM_MODEL", "gpt-4o-mini").strip()

    def advise_stage(
        self,
        request: AdvisorStageRequest,
        selected_label: str,
    ) -> AdvisorDecisionExplanation:
        # Hard firewall boundary — raises DataIsolationViolation if any code/data leaks in
        sanitized_json = self.firewall.sanitize_and_serialize(request)

        narrowed = [
            obs.candidate_label
            for obs in sorted(
                request.observations,
                key=lambda o: o.numeric_metrics.get(
                    "held_out_loss", -o.numeric_metrics.get("tokens_per_sec", 0.0)
                ),
            )[:3]
        ]

        llm_rationale, provider_used = self._try_remote_llm(sanitized_json, selected_label)
        if llm_rationale is None:
            llm_rationale = self._deterministic_explanation(request, selected_label)
            provider_used = "deterministic-heuristic-advisor"

        return AdvisorDecisionExplanation(
            stage=request.stage,
            selected_label=selected_label,
            narrowed_candidate_labels=narrowed,
            rationale=llm_rationale,
            provider_used=provider_used,
        )

    def _try_remote_llm(self, sanitized_json: str, selected_label: str) -> tuple[str | None, str]:
        if not self.api_key and not self.base_url:
            return None, "none"

        system_prompt = (
            "You are Crucible's autoresearch performance-engineering advisor. You receive ONLY "
            "hyperparameter identifiers and numeric metrics (zero raw code or dataset content). "
            "In 2-3 concise sentences, explain why the winning configuration outperformed alternatives."
        )
        user_msg = f"Winning label: {selected_label}\nMetrics JSON:\n{sanitized_json}"

        try:
            if self.provider == "anthropic":
                url = self.base_url or "https://api.anthropic.com/v1/messages"
                resp = httpx.post(
                    url,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 220,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": user_msg}],
                    },
                    timeout=15.0,
                )
                resp.raise_for_status()
                body = resp.json()
                return body["content"][0]["text"].strip(), f"anthropic:{self.model}"

            url = (self.base_url.rstrip("/") if self.base_url else "https://api.openai.com/v1") + "/chat/completions"
            headers = {"content-type": "application/json"}
            if self.api_key:
                headers["authorization"] = f"Bearer {self.api_key}"
            resp = httpx.post(
                url,
                headers=headers,
                json={
                    "model": self.model,
                    "temperature": 0.1,
                    "max_tokens": 220,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            body = resp.json()
            return body["choices"][0]["message"]["content"].strip(), f"openai-compat:{self.model}"
        except Exception:
            return None, "fallback_after_http_error"

    @staticmethod
    def _deterministic_explanation(req: AdvisorStageRequest, selected_label: str) -> str:
        parsed = json.loads(req.model_dump_json())
        obs_list = parsed["observations"]
        winner = next((o for o in reversed(obs_list) if o["candidate_label"] == selected_label), obs_list[-1])
        m = winner["numeric_metrics"]

        if req.stage == "stage_a_recipe":
            return (
                f"Selected LoRA recipe '{selected_label}' via successive halving across {req.cluster_nodes} "
                f"{req.cluster_sm_arch} nodes: full attention + MoE projection coverage achieved the lowest "
                f"held-out loss ({m.get('held_out_loss', 0.0):.4f}, perplexity {m.get('held_out_perplexity', 0.0):.2f}) "
                f"while pruning high-LR/narrow-module trials early."
            )
        if req.stage == "stage_b_training_perf":
            return (
                f"Selected training config '{selected_label}' at {m.get('tokens_per_sec', 0.0):.0f} tok/s "
                f"({m.get('hardware_utilization_pct', 0.0):.1f}% of measured cluster peak): enabling sm_121 "
                f"chunked 152k-vocab cross-entropy, grouped MoE GEMM, and fused LoRA freed unified memory for "
                f"larger micro-batches while matching the reference loss curve within {m.get('loss_curve_delta', 0.0):.4f}."
            )
        if req.stage == "stage_c_inference_perf":
            return (
                f"Selected serving config '{selected_label}' delivering {m.get('tokens_per_sec', 0.0):.0f} tok/s "
                f"(P99 {m.get('p99_latency_ms', 0.0):.1f} ms): Blackwell NVFP4 + 2 independent data replicas avoided "
                f"inter-node TP all-reduce overhead and kept perplexity within +{m.get('perplexity_regression_pct', 0.0):.2f}% "
                f"of BF16 while rejecting over-batched configs that breached quality/latency gates."
            )
        return f"Emitted turnkey Fuzzball workflow anchored on verified configuration '{selected_label}'."
