# Alchemy: Turning Private Data into Fast, Fine-Tuned Local Models (Team 3)

**Team 3:** Tushar Sonthalia, Kunlin Han, Sebastian Lueders, Akshat Singhania · **Track:** Multi-Node Training (NVIDIA GB10 `sm_121` · CIQ Fuzzball · VAST Data)

> Bring your private data, a base model, and your hardware. **Alchemy** runs on a thin client workstation (`http://127.0.0.1:8000`), uses a private-enclave **Remote Teacher** to partition uploaded data into leak-free **Train (80%) / Validation (10%) / Test (10%)** SFT splits, dispatches parallel LoRA recipe search, custom Blackwell `sm_121` Triton kernels, 2-node NCCL DDP training, and FP8/NVFP4 vLLM serving sweeps as **CIQ Fuzzball v4 workflows** over **VAST Data** shared volumes — and returns both a single end-to-end Fuzzball workflow (`workflows/pipeline.yaml`) and an unzip-and-run branded company AI copilot (`.zip`). **Zero raw bytes ever leave the local/VAST boundary.**

### Submission Artifacts at a Glance

| Artifact | Location | Purpose |
| --- | --- | --- |
| **Judges One-Pager** | [`docs/one-pager.md`](docs/one-pager.md) ([PDF](docs/one-pager.pdf)) | 1-page technical summary & measured GB10 results |
| **Presentation Slides** | [`docs/slides.md`](docs/slides.md) ([PDF](docs/slides.pdf)) | Architecture, custom `sm_121` kernels, and 2-node scaling |
| **Judged End-to-End Fuzzfile** | [`workflows/pipeline.yaml`](workflows/pipeline.yaml) | Single Fuzzball v4 DAG: `ingest` $\rightarrow$ `fuse-experts` $\rightarrow$ 2-node `train` $\rightarrow$ `serve-and-verify` |
| **Interactive Client Studio & CLI** | [`src/crucible/`](src/crucible/) (`alchemy`) | Full web UI (`http://127.0.0.1:8000`), Remote Teacher SFT synthesizer, & `.zip` app packager |
| **Measured Cluster Benchmarks** | [`results/`](results/summary.md) | Raw logs & JSON metrics from `CIQ-CHI-SPARK` GB10 hardware runs |

---

## Judges Quickstart: Launch & Test Everything in 1 Command

Works out of the box on **macOS or Linux** (Python 3.10+). No GPU is required on your laptop — if live CIQ Fuzzball GPUs are reachable and uncordoned it dispatches via **Fuzzball 4.2 MCP** (`use: team3`), and if you are off-network or cluster nodes are cordoned for judging, it automatically streams the **calibrated `SPARK05`/`SPARK06` GB10 Empirical Twin** so every feature can be tested interactively.

```bash
./scripts/run_team3.sh
```

*(Automatically creates `.venv`, installs the `alchemy` package + FastAPI UI, checks Fuzzball cluster state, and opens **http://127.0.0.1:8000** in your browser.)*

<details>
<summary><strong>Alternative manual setup (without helper script)</strong></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[ui,test]"
alchemy serve --host 127.0.0.1 --port 8000
```
Then open **http://127.0.0.1:8000**.
</details>

---

## 3-Minute Guided Judges Walkthrough (`http://127.0.0.1:8000`)

Notice the 3-step lifecycle tracker at the top of the UI (**`STEP 1: PRE-RUN`** $\rightarrow$ **`STEP 2: RUNNING`** $\rightarrow$ **`STEP 3: FINISHED · OUTPUT READY`**). Final output stays locked until a run finishes so you always know exact readiness.

### Step 1 — Select Base Model & Hardware Preset Warm-Start
1. In panel **1. Select Base Model**, explore the **52 curated open-weight models** across tabs (`Featured`, `Pre-Loaded on VAST`, `Preset Opt Ready`, `Code & Systems`, `Reasoning & Math`, `Frontier Multi-Node`), or paste any HuggingFace `org/model` ID.
2. Click **`Qwen2.5-Coder-1.5B`** (under `Featured` — shows `Min: 1 GPU`, `Pre-Loaded`, and `Preset Opt` saving **`$1.45`** by warm-starting verified `sm_121` kernel configs) or keep **`Qwen3-Coder-30B-A3B`** (`Saves $2.40`).

### Step 2 — Set Company Brand, Drop Private Data & Test Zero-Leak Guard
1. Set **Enterprise Organization / Copilot Brand** (default: **`Apex Dynamics`** — try changing it to any company name).
2. **Test Custom Data Ingestion + Remote Teacher Splits**:
   - Either leave the built-in **LAMMPS C++ & GPU Simulation Corpus** active (`23 Train · 4 Val · 4 Test` Teacher SFT pairs), **or** drag-and-drop any `.py`, `.cpp`, `.md`, `.txt`, `.jsonl`, `.csv`, or `.zip` files onto the dropzone.
   - The **Remote Teacher** (`src/crucible/data/teacher_curriculum.py`) immediately partitions whole documents by stem into **Train (80%) / Val (10%) / Test (10%)** and synthesizes structured SFT instruction-reasoning-response pairs (`teacher_sft_train.jsonl`, `teacher_sft_val.jsonl`, `teacher_sft_test.jsonl`) with **0 cross-split leakage** and **0 cloud bytes egressed**.
3. **Test the Privacy Schema Firewall**: Click **`Verify Zero-Leak Guard`** (top-right header). Alchemy injects a raw C++ snippet into an outbound advisor payload and proves the schema firewall intercepts and blocks it (`0 Bytes` leaked, `SHA-256` audit logged).

### Step 3 — Launch Optimization & Watch Real-Time Dual-GPU Telemetry
1. In panel **3. Continuous Teacher & Cost / Time Budget**:
   - **Execution Mode**: Select `CIQ Fuzzball 4.2 MCP Server · Live Hardware` (auto-falls back to the GB10 Empirical Twin if cluster nodes are cordoned) or `Fuzzball GB10 Empirical Twin`.
   - **Time Budget (min)**: `4` · **Cost Cap ($ USD)**: `3.00` (with **Continuous Remote-Teacher Loop** and **Warm-Start from Company `sm_121` Presets** checked).
2. Click **`Tune on Fuzzball & Generate Local LLM`**:
   - Lifecycle transitions to **`STATE: RUNNING`** with stage progress (`12%` $\rightarrow$ `96%`).
   - Watch **GPU 0 (`CIQ-CHI-SPARK05`) $\leftrightarrow$ GPU 1 (`CIQ-CHI-SPARK06`)** SM utilization, Tensor Core load, LPDDR5X memory, and bidirectional **200GbE RoCE NCCL AllReduce** traffic animate live.
   - Scroll to the 4 lower diagnostic panels to watch **stage-gated loading spinners transition into live streaming** across:
     1. **Real-Time Train & Validation Loss-Curve Equivalence** ($\Delta \le 0.025$ gate vs. baseline)
     2. **Stage B Throughput & Dual-GPU Scaling Waterfall** (including live **Continuous Teacher** rounds)
     3. **Stage C vLLM Serving Sweep** (`BF16`, `FP8`, Blackwell `NVFP4` evaluated on the Sealed Test Split)
     4. **Custom `sm_121` Kernel Parity & Cryptographic SHA-256 Firewall Audit**
   - You can let the ~25s stream complete on its own, **or** click **`Stop & Package Best`** at any moment to wrap up in `< 1s`.

### Step 4 — Test the Finished Company Copilot & Standalone `.zip` App
As soon as the banner turns emerald green (**`✓ STATE: FINISHED · OUTPUT READY`**):
- **Option A — Test Directly in Studio**:
  - Click **`Jump to Tuned Output ↓`** (or scroll to the bottom).
  - Click **`Run Live on Fuzzball GPU`** (or **`Instant Preview`**) to compare the **Untuned Base Model** side-by-side against your **Tuned Private Copilot**.
  - Inspect generated artifacts across tabs: `run_company_app.py`, `teacher_sft_train.jsonl`, `fuzzfile.yaml`, `Modelfile.ollama`, `client_query_llm.py`, `train_recipe.yaml`, and `BUILD_REPORT.md`.
- **Option B — Unzip & Launch the Standalone Employee Web App (`0` pip dependencies)**:
  1. Click **`✓ Output Ready · Download .zip`**.
  2. Run in any terminal:
     ```bash
     cd ~/Downloads && unzip -o "$(ls -t apex-dynamics-private-llm-*.zip | head -1)" -d apex-ai && python3 apex-ai/scripts/run_company_app.py
     ```
  3. Opens a standalone **`Apex Dynamics — Private Engineering Copilot`** chat app at **`http://127.0.0.1:8765`** bundled with your `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`, `data/teacher_sft_*.jsonl`, `fuzzfile.yaml`, `Modelfile.ollama`, and `privacy_audit.jsonl`.

---

## CLI, Judged Fuzzball Workflow & Automated Test Suite

Judges can also verify every subsystem directly from the terminal (activate `.venv` via `source .venv/bin/activate` or prefix with `.venv/bin/`):

```bash
# 1. Run the full automated test suite (ingestion, firewall, sm_121 kernels, MCP, UI, .zip bundle, zero-torch laptop guard)
pytest

# 2. Benchmark & verify numerical parity of all 4 custom sm_121 kernels vs. FP32 PyTorch reference
alchemy bench-kernels

# 3. Validate the judged end-to-end Fuzzball v4 workflow against Team 3 GB10 hardware bounds
alchemy validate-yaml workflows/pipeline.yaml

# 4. Inspect Fuzzball cluster discovery (volumes, nodes, cordon state, 00-fetch.yaml)
alchemy discover

# 5. Run a complete headless 4-stage + Continuous Teacher pipeline from the CLI
alchemy run --backend simulator --output-dir runs/judge-cli-test
alchemy inspect runs/judge-cli-test
```

### Running the Judged End-to-End Fuzzfile Directly on CIQ Fuzzball
When logged into the CIQ Fuzzball cluster (`./fuzzball login`) with uncordoned GB10 nodes:
```bash
./fuzzball workflow validate workflows/pipeline.yaml
./fuzzball workflow start workflows/pipeline.yaml --name alchemy-pipeline
```

---

## Summary of Measured Hardware Results on 2x NVIDIA GB10 (`sm_121`) + VAST Data

All hardware benchmarks were executed as Fuzzball v4 workflows on `team3` (`Qwen3-Coder-30B-A3B` MoE in BF16 on held-out LAMMPS C++ files) and logged under [`results/`](results/):

| Metric | Measured Result on GB10 (`sm_121`) | Artifact / Evidence |
| --- | --- | --- |
| **Held-Out LAMMPS Loss (Perplexity)** | **`0.730 (2.08)` $\rightarrow$ `0.600 (1.82)` (`-17.8%`)** across all 374 held-out files after 2M tokens | `results/pipeline_final_2node.json` |
| **Training Throughput (Stock $\rightarrow$ Tuned)** | **`356` $\rightarrow$ `2,858 tok/s` (`8.0x`)** across 2x GB10 nodes (`grouped_mm` `1.9x` + `torch.compile` `1.5x` + no-ckpt `1.35x` + 2-node DDP `1.98x`, `99%` scaling efficiency) | `results/trials/s4-*`, `results/perf/` |
| **Custom Triton Fused Linear + 152k-Vocab CE** | **`2.1–2.35x` faster**, logit memory **`7.0 GiB` $\rightarrow$ `0.34 GiB`** (`-6.2 GiB` peak training VRAM, exact loss) | `src/kernels/fused_ce.py`, `src/crucible/kernels/chunked_cross_entropy.py` |
| **Custom Triton Grouped MoE GEMM (`sm_121`)** | Registered in `ExpertsInterface` (fwd + backward `dX`) for ragged top-8-of-128 routing | `src/kernels/grouped_gemm.py`, `src/crucible/kernels/grouped_moe_gemm.py` |
| **vLLM Serving (`BF16` $\rightarrow$ `FP8` + LoRA)** | Base `113` $\rightarrow$ `203 tok/s` at `C=16` (`1.79x`, `+0.2%` loss); fine-tuned **FP8 + LoRA reaches `441 tok/s` at `C=64`** (held-out loss `1.293` $\rightarrow$ `1.168`) | `results/serve/` |
| **Shared VAST Data NVMe Fabric** | **`1.64 GB/s`** single-stream safetensors read; 30B checkpoint load cut from `~10 min` $\rightarrow$ **`62 s`** with zero per-trial copying | `results/vast_read.json`, `results/fuse_experts.json` |

---

## Repository Layout

```text
scripts/run_team3.sh      1-command Judges Quickstart launcher (sets up .venv & opens Studio at :8000)
workflows/
  00-fetch.yaml           Stages base models & LAMMPS corpus onto VAST-backed volumes (use: team3)
  pipeline.yaml           Judged end-to-end Fuzzball v4 workflow (ingest -> fuse -> 2-node train -> FP8+LoRA serve)
src/
  crucible/               Alchemy Client Studio (FastAPI + live UI), CLI (`alchemy`), firewall, Teacher SFT, & .zip bundler
  kernels/                GB10 sm_121 Triton kernels (fused 152k-vocab linear+CE & grouped MoE expert GEMM)
  train/, serve/, search/ Cluster trial runners, multinode SSH/NCCL launcher, recipe & serving sweeps
results/                  Measured JSON logs, trial Fuzzfiles, and benchmark summaries from GB10 runs
docs/                     Judges one-pager (`one-pager.pdf`) and presentation deck (`slides.pdf`)
tests/                    End-to-end pytest suite covering splits, Teacher SFT, kernels, firewall, MCP, UI, & bundle export
```
