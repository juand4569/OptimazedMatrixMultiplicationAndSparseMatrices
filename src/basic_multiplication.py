"""
Basic matrix multiplication implementation O(n³)
"""

import numpy as np


def basic_multiply(A, B):
    """
    Basic matrix multiplication using triple nested loops.
    Complexity: O(n³)
    
    Args:
        A: First matrix (n x m)
        B: Second matrix (m x p)
    
    Returns:
        Result matrix C (n x p)
    """
    if isinstance(A, np.ndarray):
        A = A.tolist()
    if isinstance(B, np.ndarray):
        B = B.tolist()
    
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    
    # Verify dimensions
    if len(B) != m:
        raise ValueError(f"Incompatible dimensions: A is {n}x{m}, B is {len(B)}x{p}")
    
    # Initialize result matrix with zeros
    C = [[0.0 for _ in range(p)] for _ in range(n)]
    
    # Triple nested loop
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i][j] += A[i][k] * B[k][j]
    
    return np.array(C)