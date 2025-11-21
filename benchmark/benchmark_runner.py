"""
Main benchmark runner - executes all benchmarks
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_dense import benchmark_dense_matrices, verify_implementations
from benchmark_sparse import benchmark_sparse_matrices, compare_sparse_vs_dense


def run_all_benchmarks(dense_sizes=None, sparse_sizes=None, sparsity_levels=None, runs=3):
    """
    Run all benchmarks (dense and sparse).
    
    Args:
        dense_sizes: List of sizes for dense matrices
        sparse_sizes: List of sizes for sparse matrices
        sparsity_levels: List of sparsity levels to test
        runs: Number of runs per configuration
    """
    # Default configurations
    if dense_sizes is None:
        dense_sizes = [10, 50, 100, 250, 500, 1000]
    
    if sparse_sizes is None:
        sparse_sizes = [100, 500, 1000, 2000]
    
    if sparsity_levels is None:
        sparsity_levels = [0.5, 0.7, 0.9, 0.95, 0.99]
    
    print("\n" + "="*60)
    print("MATRIX MULTIPLICATION BENCHMARK SUITE")
    print("="*60)
    
    # Verify implementations
    print("\n[1/3] Verifying implementations...")
    verify_implementations(size=100)
    
    # Run dense benchmarks
    print("\n[2/3] Running dense matrix benchmarks...")
    dense_results = benchmark_dense_matrices(dense_sizes, runs=runs)
    
    # Run sparse benchmarks
    print("\n[3/3] Running sparse matrix benchmarks...")
    sparse_results = benchmark_sparse_matrices(sparse_sizes, sparsity_levels, runs=runs)
    
    print("\n" + "="*60)
    print("ALL BENCHMARKS COMPLETED!")
    print("="*60)
    print(f"Dense results: results/dense_results.csv")
    print(f"Sparse results: results/sparse_results.csv")
    
    return dense_results, sparse_results


def run_dense_only(sizes=None, runs=3):
    """Run only dense matrix benchmarks."""
    if sizes is None:
        sizes = [10, 50, 100, 250, 500, 1000, 2000]
    
    print("\n" + "="*60)
    print("DENSE MATRIX BENCHMARKS")
    print("="*60)
    
    verify_implementations(size=100)
    results = benchmark_dense_matrices(sizes, runs=runs)
    
    print("\nDense benchmarks completed!")
    return results


def run_sparse_only(sizes=None, sparsity_levels=None, runs=3):
    """Run only sparse matrix benchmarks."""
    if sizes is None:
        sizes = [100, 500, 1000, 2000, 5000]
    
    if sparsity_levels is None:
        sparsity_levels = [0.5, 0.7, 0.9, 0.95, 0.99]
    
    print("\n" + "="*60)
    print("SPARSE MATRIX BENCHMARKS")
    print("="*60)
    
    results = benchmark_sparse_matrices(sizes, sparsity_levels, runs=runs)
    
    print("\nSparse benchmarks completed!")
    return results


def run_quick_test():
    """Run a quick test with small matrices."""
    print("\n" + "="*60)
    print("QUICK TEST MODE")
    print("="*60)
    
    verify_implementations(size=50)
    
    print("\nDense benchmark (small)...")
    benchmark_dense_matrices([10, 50, 100], runs=1)
    
    print("\nSparse benchmark (small)...")
    benchmark_sparse_matrices([100, 500], [0.9, 0.95], runs=1)
    
    print("\nQuick test completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Matrix Multiplication Benchmark Runner')
    parser.add_argument('--mode', choices=['all', 'dense', 'sparse', 'quick'], 
                        default='all', help='Benchmark mode to run')
    parser.add_argument('--runs', type=int, default=3, 
                        help='Number of runs per configuration')
    parser.add_argument('--max-size', type=int, default=2000,
                        help='Maximum matrix size to test')
    
    args = parser.parse_args()
    
    # Generate size ranges based on max_size
    if args.max_size <= 100:
        dense_sizes = [10, 50, 100]
        sparse_sizes = [50, 100]
    elif args.max_size <= 500:
        dense_sizes = [10, 50, 100, 250, 500]
        sparse_sizes = [100, 250, 500]
    elif args.max_size <= 1000:
        dense_sizes = [10, 50, 100, 250, 500, 1000]
        sparse_sizes = [100, 500, 1000]
    else:
        dense_sizes = [10, 50, 100, 250, 500, 1000, 2000]
        sparse_sizes = [100, 500, 1000, 2000, 5000]
    
    # Filter sizes by max_size
    dense_sizes = [s for s in dense_sizes if s <= args.max_size]
    sparse_sizes = [s for s in sparse_sizes if s <= args.max_size]
    
    # Run selected mode
    if args.mode == 'all':
        run_all_benchmarks(dense_sizes, sparse_sizes, runs=args.runs)
    elif args.mode == 'dense':
        run_dense_only(dense_sizes, runs=args.runs)
    elif args.mode == 'sparse':
        run_sparse_only(sparse_sizes, runs=args.runs)
    elif args.mode == 'quick':
        run_quick_test()