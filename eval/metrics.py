import math
from typing import List

def pass_at_k(successes: int, attempts: int, k: int = 1) -> float:
    """Approximate pass@k for small k. Here it's simply successes/attempts when k=1."""
    if attempts == 0: return 0.0
    if k == 1: return successes / attempts
    # Placeholder for general k; for now assume 1
    return successes / attempts
