"""
Optimized matrix multiplication implementations
"""

import numpy as np


def numpy_multiply(A, B):
    """
    Matrix multiplication using NumPy's optimized implementation.
    Uses BLAS/LAPACK libraries underneath.
    """
    return np.dot(A, B)


def strassen_multiply(A, B, threshold=64):
    """
    Strassen's algorithm for matrix multiplication.
    Complexity: O(n^2.807)
    Falls back to standard multiplication for small matrices.
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    
    n = A.shape[0]
    
    if n <= threshold:
        return np.dot(A, B)
    
    next_pow2 = 2 ** int(np.ceil(np.log2(n)))
    if n != next_pow2:
        A_padded = np.zeros((next_pow2, next_pow2))
        B_padded = np.zeros((next_pow2, next_pow2))
        A_padded[:n, :n] = A
        B_padded[:n, :n] = B
        result = _strassen_recursive(A_padded, B_padded, threshold)
        return result[:n, :n]
    
    return _strassen_recursive(A, B, threshold)


def _strassen_recursive(A, B, threshold):
    """Helper function for Strassen's algorithm recursion"""
    n = A.shape[0]
    
    if n <= threshold:
        return np.dot(A, B)
    
    mid = n // 2
    A11, A12 = A[:mid, :mid], A[:mid, mid:]
    A21, A22 = A[mid:, :mid], A[mid:, mid:]
    B11, B12 = B[:mid, :mid], B[:mid, mid:]
    B21, B22 = B[mid:, :mid], B[mid:, mid:]
    
    M1 = _strassen_recursive(A11 + A22, B11 + B22, threshold)
    M2 = _strassen_recursive(A21 + A22, B11, threshold)
    M3 = _strassen_recursive(A11, B12 - B22, threshold)
    M4 = _strassen_recursive(A22, B21 - B11, threshold)
    M5 = _strassen_recursive(A11 + A12, B22, threshold)
    M6 = _strassen_recursive(A21 - A11, B11 + B12, threshold)
    M7 = _strassen_recursive(A12 - A22, B21 + B22, threshold)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    C = np.zeros((n, n))
    C[:mid, :mid] = C11
    C[:mid, mid:] = C12
    C[mid:, :mid] = C21
    C[mid:, mid:] = C22
    
    return C


def cache_optimized_multiply(A, B, block_size=256):
    """
    Cache-optimized matrix multiplication using blocking/tiling.
    Uses NumPy's highly optimized dot() inside each block.
    This is the correct high-performance approach.
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    
    n, m = A.shape
    m2, p = B.shape
    
    if m != m2:
        raise ValueError(f"Incompatible dimensions: A is {n}x{m}, B is {m2}x{p}")
    
    C = np.zeros((n, p), dtype=float)
    
    for i in range(0, n, block_size):
        i_end = min(i + block_size, n)
        
        for j in range(0, p, block_size):
            j_end = min(j + block_size, p)
            
            for k in range(0, m, block_size):
                k_end = min(k + block_size, m)
                
                # This is the efficient part: NumPy handles each block in optimized C code.
                C[i:i_end, j:j_end] += np.dot(
                    A[i:i_end, k:k_end],
                    B[k:k_end, j:j_end]
                )
    
    return C


def transpose_multiply(A, B):
    """
    Matrix multiplication with B transposed for better cache locality.
    Not used as an optimized method in the final report.
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    
    n, m = A.shape
    m2, p = B.shape
    
    if m != m2:
        raise ValueError(f"Incompatible dimensions")
    
    B_T = B.T.copy()
    C = np.zeros((n, p), dtype=float)
    
    for i in range(n):
        for j in range(p):
            temp_sum = 0.0
            for k in range(m):
                temp_sum += A[i, k] * B_T[j, k]
            C[i, j] = temp_sum
    
    return C
