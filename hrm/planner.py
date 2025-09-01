from typing import List, Dict

PLANNER_SYSTEM = """You are an expert project planner.
Decompose a task into 2-5 concrete, verifiable sub-steps that a coder can implement.
Return ONLY a JSON list of sub-steps, no prose.
"""

def make_planner_prompt(task: str) -> str:
    return f"""Task: {task}
Return a JSON array of sub-steps. Each sub-step should be specific and testable."""
