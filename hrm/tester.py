import subprocess, tempfile, os, textwrap, json, shlex, sys, uuid
from typing import Tuple, Dict, Any

def run_python(code: str, test_input: str = "", timeout: int = 3) -> Tuple[int, str, str]:
    """Run Python code in a temp file, optionally feed stdin. Returns (returncode, stdout, stderr)."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "main.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            proc = subprocess.run(
                [sys.executable, path],
                input=test_input.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
            return proc.returncode, proc.stdout.decode("utf-8", "ignore"), proc.stderr.decode("utf-8", "ignore")
        except subprocess.TimeoutExpired:
            return 124, "", "TIMEOUT"

def simple_unit_test(code: str, oracle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Very small test harness:
    oracle = {
      "stdin": "3\n",    # optional
      "expect_substr": "6"  # any substring we expect in stdout
    }
    """
    rc, out, err = run_python(code, oracle.get("stdin",""))
    passed = (rc == 0) and (oracle.get("expect_substr","") in out)
    return {"passed": passed, "rc": rc, "stdout": out, "stderr": err}
