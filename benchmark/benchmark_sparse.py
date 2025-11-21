"""
Benchmark script for sparse matrix multiplication
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sparse_multiplication import sparse_multiply_csr, sparse_multiply_csc, create_sparse_matrix, get_sparsity
from src.optimized_multiplication import numpy_multiply
from src.utils import generate_sparse_matrix, benchmark_function


def benchmark_sparse_matrices(sizes, sparsity_levels, runs=3, save_results=True):
    """Benchmark sparse matrix multiplication across different sizes and sparsity levels."""
    results = []
    
    methods = [
        ('CSR Format', sparse_multiply_csr),
        ('CSC Format', sparse_multiply_csc),
        ('Dense (NumPy)', lambda A, B: numpy_multiply(
            A.toarray() if hasattr(A, 'toarray') else A,
            B.toarray() if hasattr(B, 'toarray') else B
        ))
    ]
    
    for size in sizes:
        for sparsity in sparsity_levels:
            print(f"\n{'='*60}")
            print(f"Matrix size: {size}x{size}, Sparsity: {sparsity*100:.0f}%")
            print(f"{'='*60}")
            
            A_dense = generate_sparse_matrix(size, sparsity=sparsity, seed=42)
            B_dense = generate_sparse_matrix(size, sparsity=sparsity, seed=43)
            
            actual_sparsity = get_sparsity(A_dense)
            print(f"Actual sparsity: {actual_sparsity:.2f}%")
            
            for method_name, method_func in methods:
                print(f"\nBenchmarking: {method_name}...")
                
                try:
                    if 'Dense' in method_name:
                        A_test = A_dense
                        B_test = B_dense
                    else:
                        A_test = create_sparse_matrix(A_dense, 
                            format='csr' if 'CSR' in method_name else 'csc')
                        B_test = create_sparse_matrix(B_dense,
                            format='csr' if 'CSR' in method_name else 'csc')
                    
                    metrics = benchmark_function(method_func, A_test, B_test, runs=runs)
                    
                    result = {
                        'size': size,
                        'sparsity': sparsity * 100,
                        'actual_sparsity': actual_sparsity,
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
                    print(f"  Avg Memory: {metrics['avg_memory_mb']:.2f} MB")
                    
                except Exception as e:
                    print(f"  ERROR: {str(e)}")
                    result = {
                        'size': size,
                        'sparsity': sparsity * 100,
                        'actual_sparsity': actual_sparsity,
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
        save_results_to_csv(results, 'results/sparse_results.csv')
    
    return results


def compare_sparse_vs_dense(size, sparsity, runs=3):
    """Direct comparison between sparse and dense multiplication."""
    print(f"\nComparing Sparse vs Dense for {size}x{size}, {sparsity*100:.0f}% sparsity")
    print("="*60)
    
    A_dense = generate_sparse_matrix(size, sparsity=sparsity, seed=42)
    B_dense = generate_sparse_matrix(size, sparsity=sparsity, seed=43)
    
    A_sparse = create_sparse_matrix(A_dense, format='csr')
    B_sparse = create_sparse_matrix(B_dense, format='csr')
    
    print("\nSparse (CSR) multiplication...")
    sparse_metrics = benchmark_function(sparse_multiply_csr, A_sparse, B_sparse, runs=runs)
    
    print("Dense (NumPy) multiplication...")
    dense_metrics = benchmark_function(numpy_multiply, A_dense, B_dense, runs=runs)
    
    speedup = dense_metrics['avg_time'] / sparse_metrics['avg_time']
    
    print(f"\n{'='*60}")
    print("Results:")
    print(f"  Sparse time: {sparse_metrics['avg_time']:.4f}s")
    print(f"  Dense time:  {dense_metrics['avg_time']:.4f}s")
    print(f"  Speedup:     {speedup:.2f}x")
    print(f"  Sparse memory: {sparse_metrics['avg_memory_mb']:.2f} MB")
    print(f"  Dense memory:  {dense_metrics['avg_memory_mb']:.2f} MB")
    print(f"{'='*60}")
    
    return {
        'sparse': sparse_metrics,
        'dense': dense_metrics,
        'speedup': speedup
    }


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
    test_sizes = [100, 500, 1000, 2000, 5000]
    sparsity_levels = [0.5, 0.7, 0.9, 0.95, 0.99]
    
    results = benchmark_sparse_matrices(test_sizes, sparsity_levels, runs=3)
    
    print("\n" + "="*60)
    print("Example: Direct Comparison")
    compare_sparse_vs_dense(size=1000, sparsity=0.9, runs=3)
    
    print("\n" + "="*60)
    print("Sparse benchmark completed!")
    print("="*60)