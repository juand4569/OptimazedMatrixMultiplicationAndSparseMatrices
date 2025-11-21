"""
Utility functions for matrix generation and performance measurement
"""

import numpy as np
import time
import psutil
import os
from functools import wraps


def generate_matrix(n, m=None, seed=42, low=0, high=100):
    """
    Generate a random dense matrix.
    
    Args:
        n: Number of rows
        m: Number of columns (if None, defaults to n for square matrix)
        seed: Random seed for reproducibility
        low: Minimum value
        high: Maximum value
    
    Returns:
        numpy array of shape (n, m)
    """
    if m is None:
        m = n
    
    np.random.seed(seed)
    return np.random.uniform(low, high, (n, m))


def generate_sparse_matrix(n, m=None, sparsity=0.9, seed=42, low=0, high=100):
    """
    Generate a random sparse matrix with specified sparsity level.
    
    Args:
        n: Number of rows
        m: Number of columns (if None, defaults to n for square matrix)
        sparsity: Percentage of zeros (0.0 to 1.0, e.g., 0.9 = 90% zeros)
        seed: Random seed for reproducibility
        low: Minimum value for non-zero elements
        high: Maximum value for non-zero elements
    
    Returns:
        numpy array of shape (n, m) with specified sparsity
    """
    if m is None:
        m = n
    
    np.random.seed(seed)
    
    # Generate random matrix
    matrix = np.random.uniform(low, high, (n, m))
    
    # Create mask for zeros based on sparsity
    zero_mask = np.random.random((n, m)) < sparsity
    matrix[zero_mask] = 0
    
    return matrix


def measure_performance(func):
    """
    Decorator to measure execution time and memory usage of a function.
    
    Args:
        func: Function to measure
    
    Returns:
        Wrapped function that returns (result, metrics_dict)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get process
        process = psutil.Process(os.getpid())
        
        # Record initial memory
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Measure execution time
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Record final memory
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_used = mem_after - mem_before
        
        metrics = {
            'execution_time': execution_time,
            'memory_mb': max(memory_used, 0),  # Avoid negative values
            'function_name': func.__name__
        }
        
        return result, metrics
    
    return wrapper


def benchmark_function(func, A, B, runs=3):
    """
    Benchmark a matrix multiplication function with multiple runs.
    
    Args:
        func: Matrix multiplication function to benchmark
        A: First matrix
        B: Second matrix
        runs: Number of runs to average
    
    Returns:
        Dictionary with averaged metrics
    """
    process = psutil.Process(os.getpid())
    times = []
    memories = []
    
    for _ in range(runs):
        mem_before = process.memory_info().rss / (1024 * 1024)
        
        start_time = time.perf_counter()
        result = func(A, B)
        end_time = time.perf_counter()
        
        mem_after = process.memory_info().rss / (1024 * 1024)
        
        times.append(end_time - start_time)
        memories.append(max(mem_after - mem_before, 0))
    
    return {
        'function_name': func.__name__,
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'avg_memory_mb': np.mean(memories),
        'max_memory_mb': np.max(memories)
    }


def verify_result(C1, C2, tolerance=1e-6):
    """
    Verify that two matrix multiplication results are approximately equal.
    
    Args:
        C1: First result matrix
        C2: Second result matrix
        tolerance: Maximum allowed difference
    
    Returns:
        Boolean indicating if results match
    """
    # Convert sparse to dense if necessary
    if hasattr(C1, 'toarray'):
        C1 = C1.toarray()
    if hasattr(C2, 'toarray'):
        C2 = C2.toarray()
    
    C1 = np.array(C1)
    C2 = np.array(C2)
    
    return np.allclose(C1, C2, rtol=tolerance, atol=tolerance)


def get_matrix_info(matrix):
    """
    Get information about a matrix (size, sparsity, etc.).
    
    Args:
        matrix: Input matrix (dense or sparse)
    
    Returns:
        Dictionary with matrix information
    """
    from scipy import sparse as sp
    
    info = {}
    
    if sp.issparse(matrix):
        info['shape'] = matrix.shape
        info['size'] = matrix.shape[0] * matrix.shape[1]
        info['nnz'] = matrix.nnz
        info['sparsity'] = 100 * (1 - matrix.nnz / info['size'])
        info['format'] = type(matrix).__name__
        info['dtype'] = matrix.dtype
    else:
        matrix = np.array(matrix)
        info['shape'] = matrix.shape
        info['size'] = matrix.size
        info['nnz'] = np.count_nonzero(matrix)
        info['sparsity'] = 100 * (1 - info['nnz'] / info['size'])
        info['format'] = 'dense'
        info['dtype'] = matrix.dtype
    
    return info