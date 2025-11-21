"""
Benchmark script for dense matrix multiplication
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.basic_multiplication import basic_multiply
from src.optimized_multiplication import (numpy_multiply, strassen_multiply, 
                                          cache_optimized_multiply, transpose_multiply)
from src.utils import generate_matrix, benchmark_function, verify_result
import signal
import psutil


def benchmark_dense_matrices(sizes=None, runs=3, save_results=True):
    """Benchmark all dense matrix multiplication methods across different sizes."""
    results = []
    
    # CHANGE: Added transpose_multiply as new method
    # WHY: Another optimization technique to compare
    methods = [
        ('Basic O(n³)', basic_multiply, [10, 50, 100, 250, 500]),
        ('NumPy (BLAS)', numpy_multiply, None),
        ('Strassen', strassen_multiply, None),
        ('Cache Optimized', cache_optimized_multiply, None),
        ('Transpose Method', transpose_multiply, [10, 50, 100, 250, 500])  # NEW
    ]
    
    if sizes is None:
        sizes = [10, 50, 100, 250, 500, 1000, 2000]
    
    for size in sizes:
        print(f"\n{'='*60}")
        print(f"Testing matrix size: {size}x{size}")
        print(f"{'='*60}")
        
        A = generate_matrix(size, seed=42)
        B = generate_matrix(size, seed=43)
        
        for method_name, method_func, allowed_sizes in methods:
            if allowed_sizes is not None and size not in allowed_sizes:
                continue
                
            print(f"\nBenchmarking: {method_name}...")
            
            try:
                metrics = benchmark_function(method_func, A, B, runs=runs)
                
                result = {
                    'size': size,
                    'method': method_name,
                    'avg_time': metrics['avg_time'],
                    'std_time': metrics['std_time'],
                    'min_time': metrics['min_time'],
                    'max_time': metrics['max_time'],
                    'avg_memory_mb': metrics['avg_memory_mb'],
                    'max_memory_mb': metrics['max_memory_mb']
                }
                
                results.append(result)
                
                print(f"  Avg Time: {metrics['avg_time']:.4f}s")
                print(f"  Std Time: {metrics['std_time']:.4f}s")
                print(f"  Avg Memory: {metrics['avg_memory_mb']:.2f} MB")
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                result = {
                    'size': size,
                    'method': method_name,
                    'avg_time': None,
                    'std_time': None,
                    'min_time': None,
                    'max_time': None,
                    'avg_memory_mb': None,
                    'max_memory_mb': None,
                    'error': str(e)
                }
                results.append(result)
    
    if save_results:
        save_results_to_csv(results, 'results/dense_results.csv')
    
    return results


def find_max_matrix_size(method_func, start_size=100, timeout_seconds=30, 
                         max_memory_gb=8):
    """
    NEW FUNCTION: Find maximum matrix size that can be handled.
    
    Tests progressively larger matrices until:
    - Execution takes > timeout_seconds
    - Memory usage exceeds max_memory_gb
    - Error occurs
    
    CHANGE: Added to meet assignment requirement of "maximum matrix size handled"
    WHY: Task explicitly requires reporting this metric
    """
    print(f"\n{'='*60}")
    print(f"Finding maximum matrix size for {method_func.__name__}")
    print(f"{'='*60}")
    
    size = start_size
    max_working_size = size
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Execution timeout")
    
    while True:
        print(f"\nTesting size: {size}x{size}")
        
        try:
            # Check available memory
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            if available_memory_gb < 2:
                print(f"  STOPPED: Low memory ({available_memory_gb:.2f} GB available)")
                break
            
            A = generate_matrix(size, seed=42)
            B = generate_matrix(size, seed=43)
            
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                metrics = benchmark_function(method_func, A, B, runs=1)
                signal.alarm(0)  # Cancel alarm
                
                print(f"  Time: {metrics['avg_time']:.4f}s")
                print(f"  Memory: {metrics['avg_memory_mb']:.2f} MB")
                
                # Check if we should stop
                if metrics['avg_time'] > timeout_seconds:
                    print(f"  STOPPED: Execution time exceeded {timeout_seconds}s")
                    break
                
                if metrics['max_memory_mb'] > max_memory_gb * 1024:
                    print(f"  STOPPED: Memory usage exceeded {max_memory_gb} GB")
                    break
                
                max_working_size = size
                
                # Increase size (more aggressive for small sizes)
                if size < 1000:
                    size = int(size * 1.5)
                else:
                    size = int(size * 1.2)
                    
            except TimeoutError:
                signal.alarm(0)
                print(f"  STOPPED: Timeout at {size}")
                break
                
        except MemoryError:
            print(f"  STOPPED: Memory error at {size}")
            break
        except Exception as e:
            print(f"  STOPPED: Error - {str(e)}")
            break
    
    print(f"\n{'='*60}")
    print(f"Maximum working size: {max_working_size}x{max_working_size}")
    print(f"{'='*60}")
    
    return max_working_size


def verify_implementations(size=100):
    """Verify that all implementations produce the same results."""
    print(f"\nVerifying implementations with {size}x{size} matrices...")
    
    A = generate_matrix(size, seed=42)
    B = generate_matrix(size, seed=43)
    
    reference = numpy_multiply(A, B)
    
    methods = [
        ('Basic O(n³)', basic_multiply, [10, 50, 100, 250, 500]),
        ('Strassen', strassen_multiply, None),
        ('Cache Optimized', cache_optimized_multiply, None),
        ('Transpose Method', transpose_multiply, [10, 50, 100, 250, 500])  # NEW
    ]
    
    verification = {}
    
    for method_name, method_func, allowed_sizes in methods:
        if allowed_sizes is not None and size not in allowed_sizes:
            print(f"  {method_name}: SKIPPED (size {size} not in test range)")
            continue
            
        try:
            result = method_func(A, B)
            matches = verify_result(reference, result)
            verification[method_name] = matches
            status = "✓ PASS" if matches else "✗ FAIL"
            print(f"  {method_name}: {status}")
        except Exception as e:
            verification[method_name] = False
            print(f"  {method_name}: ✗ ERROR - {str(e)}")
    
    return verification


def save_results_to_csv(results, filename):
    """Save benchmark results to CSV file."""
    import csv
    import os
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    if not results:
        return
    
    fieldnames = list(results[0].keys())
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    test_sizes = [10, 50, 100, 250, 500, 1000, 2000]
    
    verify_implementations(size=100)
    results = benchmark_dense_matrices(test_sizes, runs=3)
    
    # NEW: Test maximum size for each optimized method
    print("\n" + "="*60)
    print("FINDING MAXIMUM MATRIX SIZES")
    print("="*60)
    
    for method_name, method_func in [
        ('NumPy (BLAS)', numpy_multiply),
        ('Cache Optimized', cache_optimized_multiply),
        ('Strassen', strassen_multiply)
    ]:
        max_size = find_max_matrix_size(method_func, start_size=2000, 
                                        timeout_seconds=60)
        print(f"{method_name}: {max_size}x{max_size}\n")
    
    print("\n" + "="*60)
    print("Benchmark completed!")
    print("="*60)