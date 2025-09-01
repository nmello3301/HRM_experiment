import argparse, os, json, time, csv, yaml
from hrm.llm import LLMClient
from hrm.planner import PLANNER_SYSTEM, make_planner_prompt
from hrm.coder import CODER_SYSTEM, make_coder_prompt
from hrm.tester import simple_unit_test
from eval.metrics import pass_at_k

def load_suite(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["tasks"]

def run_arm_hierarchical(llm, task, oracle, max_iters=3):
    # 1) plan
    plan = llm.generate(make_planner_prompt(task), system=PLANNER_SYSTEM, max_tokens=400)
    # 2) code (iterate with naive self-repair by appending last error)
    last_err = ""
    for i in range(max_iters):
        prompt = make_coder_prompt(task, plan) + (f"\n\nIf prior attempt failed, fix this error:\n{last_err}" if last_err else "")
        code = llm.generate(prompt, system=CODER_SYSTEM, max_tokens=800)
        result = simple_unit_test(code, oracle)
        if result["passed"]:
            return True, i+1, plan, result
        last_err = result["stderr"] or result["stdout"]
    return False, max_iters, plan, result

def run_arm_flat(llm, task, oracle, max_iters=3):
    last_err = ""
    BASE = "Solve the task by writing ONLY Python code. If you made a mistake, correct it."
    for i in range(max_iters):
        prompt = f"{BASE}\n\nTask: {task}\n" + (f"Previous run error/logs:\n{last_err}" if last_err else "")
        code = llm.generate(prompt, system="You are a helpful coding assistant. Output ONLY code.", max_tokens=800)
        result = simple_unit_test(code, oracle)
        if result["passed"]:
            return True, i+1, None, result
        last_err = result["stderr"] or result["stdout"]
    return False, max_iters, None, result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=str, required=True)
    ap.add_argument("--model_backend", type=str, default="ollama", choices=["ollama","openai"])
    ap.add_argument("--model_name", type=str, default=None)
    ap.add_argument("--max_iters", type=int, default=3)
    ap.add_argument("--outdir", type=str, default="runs/exp1_decomp_vs_flat")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tasks = load_suite(args.suite)
    llm = LLMClient(backend=args.model_backend, model=args.model_name)

    # Run both arms
    rows = []
    for t in tasks:
        task_id, prompt, oracle = t["id"], t["prompt"], t["oracle"]
        # HRM
        t0 = time.time()
        ok_h, iters_h, plan, res_h = run_arm_hierarchical(llm, prompt, oracle, max_iters=args.max_iters)
        t1 = time.time()
        # Flat
        t2 = time.time()
        ok_f, iters_f, _, res_f = run_arm_flat(llm, prompt, oracle, max_iters=args.max_iters)
        t3 = time.time()

        rows.append({
            "task_id": task_id,
            "ok_h": int(ok_h),
            "iters_h": iters_h,
            "time_h": t1 - t0,
            "ok_f": int(ok_f),
            "iters_f": iters_f,
            "time_f": t3 - t2,
        })

    # Save CSV
    out_csv = os.path.join(args.outdir, "results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Simple aggregate
    successes_h = sum(r["ok_h"] for r in rows)
    successes_f = sum(r["ok_f"] for r in rows)
    attempts = len(rows)
    print(f"HRM pass@1: {pass_at_k(successes_h, attempts, 1):.3f}  ({successes_h}/{attempts})")
    print(f"Flat pass@1: {pass_at_k(successes_f, attempts, 1):.3f}  ({successes_f}/{attempts})")
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
