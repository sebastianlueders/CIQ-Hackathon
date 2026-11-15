# CIQ Supercomputing Local AI Hackathon: Team 3

An autoresearch and performance-engineering agent for local fine-tuning. You bring the data, a base model, and the hardware. The agent:

- searches for the best LoRA recipe
- speeds up training (including our custom sm_121 kernels)
- finds the fastest serving setup that keeps quality

Every experiment runs as a Fuzzball workflow on Fuzzball volumes backed by VAST. The output is a single end-to-end Fuzzball workflow.

## Start here

| Document | For |
|---|---|
| [docs/VISION.md](docs/VISION.md) ([PDF](docs/VISION.pdf)) | the team: the product vision and how the work is split |
| [docs/one-pager.md](docs/one-pager.md) ([PDF](docs/one-pager.pdf)) | the judges: the submission draft (`[X]` = not yet measured) |
| [docs/HANDOFF.md](docs/HANDOFF.md) | anyone writing code: cluster facts, Fuzzfile syntax, sm_121 pitfalls, kernels, plan |

## Layout

```
docs/        documents above; build/ has the PDF build script and styles
workflows/   Fuzzball workflows (judged artifact)
  00-fetch.yaml   pulls the base model and LAMMPS onto VAST-backed volumes (done)
```

Code goes in `src/`, and every measurement goes in `results/`, as laid out in the handoff (Section 7).

## Setup

- **Fuzzball CLI:** put the `fuzzball` binary (v4.3.1) in the repo root. It is git-ignored. Log in with `./fuzzball login`, then check with `./fuzzball node list`.
- **Credentials:** they live in `.env`, which is git-ignored. Never commit it.
- **Run a workflow:** `./fuzzball workflow validate workflows/<file>.yaml`, then `./fuzzball workflow start workflows/<file>.yaml`.
- **Rebuild the PDFs:** `bash docs/build/build-pdfs.sh`. This needs pandoc and xelatex.
