#!/usr/bin/env bash
set -euo pipefail
python experiments/exp1_decomp_vs_flat/run.py --suite eval/suites/coding_suite.yaml --model_backend ollama --model_name llama3.1:8b-instruct-q4_K_M --max_iters 3
