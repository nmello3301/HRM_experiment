CODER_SYSTEM = """You are a senior software engineer.
Given a task and a list of sub-steps, produce Python code that solves the task.
Write clean, single-file Python. Do not include explanations, only code.
"""

def make_coder_prompt(task: str, substeps_json: str) -> str:
    return f"""Task: {task}
Sub-steps (JSON): {substeps_json}

Write the full Python solution below:"""
