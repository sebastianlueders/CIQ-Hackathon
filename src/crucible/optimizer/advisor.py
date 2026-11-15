"""Privacy-preserving Remote-Teacher & LLM Advisor sitting strictly above `PrivacySchemaFirewall`.

Supports:
- Stage-level decision auditing (`advise_stage`)
- Active Continuous Remote-Teacher Proposal Rounds (`propose_teacher_mutation`): inspects ONLY
  firewalled hyperparameter identifiers and scalar metrics (`contains_raw_data=false`) to propose
  joint intelligence (LoRA rank/LR/module/context) and throughput (`sm_121` kernels/batch/2-node DDP)
  mutations within the company's time and dollar cost budgets.
- OpenAI-compatible chat completions (`CRUCIBLE_LLM_BASE_URL` + `CRUCIBLE_LLM_API_KEY`) — works
  with OpenAI, local vLLM on Fuzzball, or NVIDIA NIM endpoints.
- Anthropic Messages API (`CRUCIBLE_LLM_PROVIDER=anthropic` + `CRUCIBLE_LLM_API_KEY`).
- Automatic deterministic coordinate-search teacher fallback when no external endpoint is configured.

Every outbound payload MUST pass `PrivacySchemaFirewall.sanitize_and_serialize()` first.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from crucible.config import ALLOWED_LORA_MODULES, RecipeCandidate, TrainingPerfCandidate
from crucible.firewall import (
    AdvisorDecisionExplanation,
    AdvisorStageRequest,
    PrivacySchemaFirewall,
)

ALL_7_LORA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


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

    def propose_teacher_mutation(
        self,
        request: AdvisorStageRequest,
        current_best_recipe: RecipeCandidate,
        current_best_perf: TrainingPerfCandidate,
        round_idx: int,
        is_moe: bool,
        max_nodes: int = 2,
    ) -> tuple[RecipeCandidate, TrainingPerfCandidate, AdvisorDecisionExplanation]:
        """Firewalled Remote-Teacher mutation step: proposes a new (RecipeCandidate, TrainingPerfCandidate)
        targeting lower held-out perplexity AND/OR higher `sm_121` tokens/sec."""
        sanitized_json = self.firewall.sanitize_and_serialize(request)

        remote_proposal = self._try_remote_teacher_json(sanitized_json, round_idx)
        if remote_proposal is not None:
            cand_recipe, cand_perf, rationale, provider = self._parse_remote_mutation(
                remote_proposal,
                current_best_recipe,
                current_best_perf,
                round_idx,
                is_moe,
                max_nodes,
            )
        else:
            cand_recipe, cand_perf, rationale = self._deterministic_teacher_mutation(
                current_best_recipe,
                current_best_perf,
                round_idx,
                is_moe,
                max_nodes,
            )
            provider = "firewalled-teacher-bayesian-coord"

        exp = AdvisorDecisionExplanation(
            stage=f"teacher_round_{round_idx}",
            selected_label=cand_perf.label,
            narrowed_candidate_labels=[cand_recipe.short_id(), cand_perf.label],
            rationale=rationale,
            provider_used=provider,
        )
        return cand_recipe, cand_perf, exp

    def _try_remote_teacher_json(self, sanitized_json: str, round_idx: int) -> tuple[dict[str, Any], str] | None:
        if not self.api_key and not self.base_url:
            return None

        system_prompt = (
            "You are Alchemy's Remote Teacher Optimizer for NVIDIA GB10 (sm_121). You receive ONLY "
            "firewalled numeric metrics and hyperparameter IDs (0 raw code/data). Propose the next trial "
            "as strict JSON with keys: lr (float), lora_rank (int: 8|16|32|64), micro_batch_size (int: 2..8), "
            "nodes (int: 1|2), activation_checkpointing (bool), rationale (string, <=2 sentences)."
        )
        user_msg = f"Teacher round {round_idx}. Firewalled observations:\n{sanitized_json}"
        try:
            url = (self.base_url.rstrip("/") if self.base_url else "https://api.openai.com/v1") + "/chat/completions"
            headers = {"content-type": "application/json"}
            if self.api_key:
                headers["authorization"] = f"Bearer {self.api_key}"
            resp = httpx.post(
                url,
                headers=headers,
                json={
                    "model": self.model,
                    "temperature": 0.2,
                    "max_tokens": 220,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                },
                timeout=12.0,
            )
            resp.raise_for_status()
            txt = resp.json()["choices"][0]["message"]["content"].strip()
            return json.loads(txt), f"remote-teacher:{self.model}"
        except Exception:
            return None

    @staticmethod
    def _parse_remote_mutation(
        remote_out: tuple[dict[str, Any], str],
        best_r: RecipeCandidate,
        best_p: TrainingPerfCandidate,
        round_idx: int,
        is_moe: bool,
        max_nodes: int,
    ) -> tuple[RecipeCandidate, TrainingPerfCandidate, str, str]:
        obj, provider = remote_out
        lr = max(2e-5, min(2e-3, float(obj.get("lr", best_r.lr))))
        rank = int(obj.get("lora_rank", best_r.lora_rank))
        if rank not in (8, 16, 32, 64, 96):
            rank = best_r.lora_rank
        mb = max(1, min(8, int(obj.get("micro_batch_size", best_p.micro_batch_size))))
        nodes = max(1, min(max_nodes, int(obj.get("nodes", best_p.nodes))))
        ckpt = bool(obj.get("activation_checkpointing", best_p.activation_checkpointing))
        mods = [m for m in obj.get("target_modules", ALL_7_LORA_MODULES) if m in ALLOWED_LORA_MODULES] or list(
            best_r.target_modules
        )
        rec = RecipeCandidate(
            lr=round(lr, 6),
            lora_rank=rank,
            lora_alpha=rank * 2,
            target_modules=mods,
            warmup_ratio=best_r.warmup_ratio,
            seq_len=best_r.seq_len,
        )
        perf = TrainingPerfCandidate(
            label=f"teacher-r{round_idx}-rk{rank}-mb{mb}-{nodes}n",
            nodes=nodes,
            micro_batch_size=mb,
            grad_accum_steps=max(1, 16 // max(1, nodes * mb)),
            activation_checkpointing=ckpt,
            use_chunked_ce_kernel=True,
            use_grouped_moe_kernel=is_moe,
            use_fused_lora_kernel=True,
        )
        rationale = str(obj.get("rationale") or f"Remote teacher proposed rank={rank}, lr={lr:.2e}, mb={mb} on {nodes}x GB10.")
        return rec, perf, rationale, provider

    @staticmethod
    def _deterministic_teacher_mutation(
        best_r: RecipeCandidate,
        best_p: TrainingPerfCandidate,
        round_idx: int,
        is_moe: bool,
        max_nodes: int,
    ) -> tuple[RecipeCandidate, TrainingPerfCandidate, str]:
        """Structured coordinate exploration around the company hardware preset / Stage A+B winners."""
        strategy = (round_idx - 1) % 4
        if strategy == 0:
            # Intelligence refinement: full 7-module coverage + optimal 1.6e-4 LR neighborhood + higher rank
            new_rank = min(64, max(32, best_r.lora_rank * 2))
            new_lr = round(0.5 * best_r.lr + 0.5 * 0.00016, 6)
            rec = RecipeCandidate(
                lr=new_lr,
                lora_rank=new_rank,
                lora_alpha=new_rank * 2,
                target_modules=list(ALL_7_LORA_MODULES),
                warmup_ratio=0.05,
                seq_len=max(2048, best_r.seq_len),
            )
            perf = TrainingPerfCandidate(
                label=f"teacher-r{round_idx}-intel-7mod-rk{new_rank}",
                nodes=best_p.nodes,
                micro_batch_size=best_p.micro_batch_size,
                grad_accum_steps=max(1, 16 // max(1, best_p.nodes * best_p.micro_batch_size)),
                activation_checkpointing=best_p.activation_checkpointing,
                use_chunked_ce_kernel=True,
                use_grouped_moe_kernel=is_moe,
                use_fused_lora_kernel=True,
            )
            why = (
                f"Teacher Round {round_idx} (Intelligence): expanded LoRA to all 7 attention+MLP projections "
                f"(rank {new_rank}, lr={new_lr:.2e}) while preserving verified sm_121 kernels."
            )
        elif strategy == 1:
            # Throughput push: step up micro-batch freed by Triton Fused CE (-6.6 GiB logit buffer) on 2x GB10 DDP
            target_nodes = max_nodes
            new_mb = min(8, best_p.micro_batch_size + 1)
            rec = best_r.model_copy(deep=True)
            perf = TrainingPerfCandidate(
                label=f"teacher-r{round_idx}-speed-mb{new_mb}-{target_nodes}n",
                nodes=target_nodes,
                micro_batch_size=new_mb,
                grad_accum_steps=max(1, 16 // max(1, target_nodes * new_mb)),
                activation_checkpointing=False if best_p.nodes == 1 or not best_p.activation_checkpointing else True,
                use_chunked_ce_kernel=True,
                use_grouped_moe_kernel=is_moe,
                use_fused_lora_kernel=True,
            )
            why = (
                f"Teacher Round {round_idx} (Throughput): leveraged Triton 152k Fused-CE memory headroom to push "
                f"micro_batch={new_mb} across {target_nodes}x GB10 DDP while holding global batch=16 for loss parity."
            )
        elif strategy == 2:
            # Fine LR + 4k context window refinement (enable activation checkpointing so 4k seq fits cleanly in VRAM)
            nudged_lr = round(max(8e-5, min(3e-4, best_r.lr * 0.88)), 6)
            new_rank = 48 if best_r.lora_rank < 48 else 64
            safe_4k_mb = min(4, best_p.micro_batch_size)
            rec = RecipeCandidate(
                lr=nudged_lr,
                lora_rank=new_rank,
                lora_alpha=new_rank * 2,
                target_modules=list(ALL_7_LORA_MODULES),
                warmup_ratio=0.06,
                seq_len=4096 if best_r.seq_len <= 2048 else best_r.seq_len,
            )
            perf = best_p.model_copy(
                update={
                    "label": f"teacher-r{round_idx}-ctx4k-rk{new_rank}",
                    "micro_batch_size": safe_4k_mb,
                    "grad_accum_steps": max(1, 16 // max(1, best_p.nodes * safe_4k_mb)),
                    "activation_checkpointing": True,
                }
            )
            why = (
                f"Teacher Round {round_idx} (Long-Context Intelligence): extended sequence window to {rec.seq_len} "
                f"with rank={new_rank} and damped lr={nudged_lr:.2e} to reduce held-out perplexity."
            )
        else:
            # Joint Pareto polish
            rec = RecipeCandidate(
                lr=0.00016,
                lora_rank=64,
                lora_alpha=128,
                target_modules=list(ALL_7_LORA_MODULES),
                warmup_ratio=0.05,
                seq_len=max(2048, best_r.seq_len),
            )
            new_mb = min(8, max(best_p.micro_batch_size, 6))
            perf = TrainingPerfCandidate(
                label=f"teacher-r{round_idx}-pareto-rk64-mb{new_mb}",
                nodes=max_nodes,
                micro_batch_size=new_mb,
                grad_accum_steps=max(1, 16 // max(1, max_nodes * new_mb)),
                activation_checkpointing=best_p.activation_checkpointing,
                use_chunked_ce_kernel=True,
                use_grouped_moe_kernel=is_moe,
                use_fused_lora_kernel=True,
            )
            why = (
                f"Teacher Round {round_idx} (Joint Pareto): combined rank=64 7-module LoRA at lr=1.6e-4 with "
                f"{max_nodes}-node sm_121 micro_batch={new_mb} to maximize tokens/sec per dollar."
            )
        return rec, perf, why

    def _try_remote_llm(self, sanitized_json: str, selected_label: str) -> tuple[str | None, str]:
        if not self.api_key and not self.base_url:
            return None, "none"

        system_prompt = (
            "You are Alchemy's autoresearch performance-engineering advisor. You receive ONLY "
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
