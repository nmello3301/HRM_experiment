# Hierarchical Reasoning Models (HRM) — Thesis Plan & Experiments

**Focus:** Compare **flat** vs **hierarchical** controllers on multi‑step tasks, including tool use, with small/medium OSS models you can run locally (7B–16B; 32B distilled/quantized optional).

## Problem Statement (1‑page)

Large Language Models solve many tasks “in one shot” via instructions, but **multi‑step problems** often benefit from **explicit structure**: planning, decomposition, tool mediation, and verification. This thesis investigates **Hierarchical Reasoning Models (HRM)** vs **flat prompting** across three domains where hierarchy is plausibly beneficial:

1) **Code generation**: planner → coder → tester (tooling: Python runner/unit tests).  
2) **Math‑like reasoning with tools**: Program‑of‑Thought (PoT), emitting short Python codelets executed by a tool node; supervisor enforces plan → subgoals → verify.  
3) **Skill acquisition via curriculum**: training/tuning data staged by step complexity (2→3→4 steps) and tested for generalization to longer chains (5–6 steps).

**Research Questions**
- RQ1: Does an **explicit hierarchy** (planner→subtasks→solver→verifier) **improve solution rate** vs a flat LLM on the **same model size** and **same tool access**?  
- RQ2: When tools are available (Python execution), does **hierarchical tool mediation** outperform flat tool usage in **accuracy, latency, and failure modes**?  
- RQ3: Does a **curriculum** over multi‑step tasks improve **sample efficiency** and **generalization length** for HRM?

**Contributions**
- A reproducible, single‑GPU evaluation suite and codebase comparing **flat vs hierarchical** controllers under controlled settings.
- Clean **ablations**: hierarchy on/off, tool on/off, curriculum vs mixed.
- **Failure taxonomy** for HRM (spec errors, tool misuse, plan drift, verification gaps).

**Compute Scope**
- Inference‑centric with small/light tuning. 7B–16B base models (e.g., Qwen2.5‑7B‑Instruct, Llama‑3.1‑8B‑Instruct, DeepSeek‑Coder‑V2‑Lite ≈16B MoE). Quantization (e.g., GGUF/Q4) keeps VRAM ≤ 24 GB. Optional small LoRA for planner/supervisor.

---

## Experiment Matrix

| ID | Domain | Arms (A/B) | Datasets (small, local) | Models (local‑friendly) | Metrics | Notes |
|----|--------|------------|-------------------------|-------------------------|---------|------|
| **E1** | Code tasks | **A:** HRM planner→coder→tester vs **B:** flat coder | 10–30 curated tasks (toy HumanEval/MBPP‑style). Provided `coding_suite.yaml` starter. | Primary: **DeepSeek‑Coder‑V2‑Lite** (quant) or **Qwen2.5‑Coder‑7B**. Baselines: **Llama‑3.1‑8B‑Instruct**. | pass@k, time‑to‑solve, #iterations, #tool calls | Clean ablation of **hierarchy** on a practical domain. |
| **E2** | Math/PoT | **A:** HRM supervisor w/ PoT tool; **B:** flat with tool | GSM8K slice + synthetic 2–3‑step arithmetic/programming (`math_suite.yaml`). | Qwen2.5‑7B‑Instruct or Llama‑3.1‑8B; optional small LoRA for supervisor | accuracy, wall‑clock, failure categories | Isolates **tool mediation** inside hierarchy. |
| **E3** | Curriculum | **A:** curriculum (2→3→4 steps); **B:** random mix | Same generator; hold‑out 5–6‑step set for extrapolation | Same as E2; re‑use supervisor | accuracy vs #examples (sample efficiency), gen to longer chains | Tests **HRM + curriculum** benefits. |

**Success Criteria (thesis‑worthy if met)**  
- **E1:** HRM ≥ +5–10 pp pass@1 over flat on the same model/tool budget.  
- **E2:** HRM reduces tool‑misuse and spec errors by ≥20% and improves accuracy under time budget.  
- **E3:** Curriculum yields better sample‑efficiency and non‑trivial gains on 5–6‑step generalization.

---

## Quick Start

```bash
# Create and activate env
python -m venv .venv && source .venv/bin/activate    # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Smoke test on 3 toy coding tasks (both arms)
python experiments/exp1_decomp_vs_flat/run.py --suite eval/suites/coding_suite.yaml --model_backend ollama --model_name llama3.1:8b-instruct-q4_K_M --max_iters 3

# View results
ls runs/exp1_decomp_vs_flat
```

**Model Backends**  
- `--model_backend ollama` (local) requires `OLLAMA_BASE_URL` (default http://localhost:11434).  
- `--model_backend openai` (OpenAI‑compatible) requires `OPENAI_API_BASE`, `OPENAI_API_KEY`, and `OPENAI_MODEL`.

---

## Roadmap
- Add unit tests for evaluator and metrics.
- Extend failure taxonomy tagging and produce confusion tables.
- Optional: small LoRA for supervisor using PEFT (fits 24 GB).
