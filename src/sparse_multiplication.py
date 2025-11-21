"""
Sparse matrix multiplication implementations
"""

import numpy as np
from scipy import sparse


def create_sparse_matrix(matrix, format='csr'):
    """
    Convert a dense matrix to sparse format.
    
    Args:
        matrix: Dense matrix (numpy array or list)
        format: Sparse format ('csr', 'csc', 'coo', etc.)
    
    Returns:
        Sparse matrix in specified format
    """
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    if format == 'csr':
        return sparse.csr_matrix(matrix)
    elif format == 'csc':
        return sparse.csc_matrix(matrix)
    elif format == 'coo':
        return sparse.coo_matrix(matrix)
    else:
        raise ValueError(f"Unsupported format: {format}")


def sparse_multiply_csr(A, B):
    """
    Sparse matrix multiplication using CSR (Compressed Sparse Row) format.
    Efficient for row slicing and matrix-vector products.
    
    Args:
        A: First matrix (dense or sparse)
        B: Second matrix (dense or sparse)
    
    Returns:
        Result matrix in CSR format
    """
    # Convert to CSR if not already sparse
    if not sparse.issparse(A):
        A = sparse.csr_matrix(A)
    elif not isinstance(A, sparse.csr_matrix):
        A = A.tocsr()
    
    if not sparse.issparse(B):
        B = sparse.csr_matrix(B)
    elif not isinstance(B, sparse.csr_matrix):
        B = B.tocsr()
    
    # Multiply using scipy's optimized implementation
    C = A.dot(B)
    
    return C


def sparse_multiply_csc(A, B):
    """
    Sparse matrix multiplication using CSC (Compressed Sparse Column) format.
    Efficient for column slicing and matrix operations.
    
    Args:
        A: First matrix (dense or sparse)
        B: Second matrix (dense or sparse)
    
    Returns:
        Result matrix in CSC format
    """
    # Convert to CSC if not already sparse
    if not sparse.issparse(A):
        A = sparse.csc_matrix(A)
    elif not isinstance(A, sparse.csc_matrix):
        A = A.tocsc()
    
    if not sparse.issparse(B):
        B = sparse.csc_matrix(B)
    elif not isinstance(B, sparse.csc_matrix):
        B = B.tocsc()
    
    # Multiply using scipy's optimized implementation
    C = A.dot(B)
    
    return C


def sparse_multiply_dense_conversion(A, B):
    """
    Multiply by converting to dense, performing multiplication, 
    then converting back to sparse.
    Useful for comparison to show when sparse methods are beneficial.
    
    Args:
        A: First matrix (sparse)
        B: Second matrix (sparse)
    
    Returns:
        Result matrix in sparse format
    """
    if sparse.issparse(A):
        A_dense = A.toarray()
    else:
        A_dense = np.array(A)
    
    if sparse.issparse(B):
        B_dense = B.toarray()
    else:
        B_dense = np.array(B)
    
    # Dense multiplication
    C_dense = np.dot(A_dense, B_dense)
    
    # Convert back to sparse
    C = sparse.csr_matrix(C_dense)
    
    return C


def get_sparsity(matrix):
    """
    Calculate the sparsity level of a matrix (percentage of zeros).
    
    Args:
        matrix: Dense or sparse matrix
    
    Returns:
        Sparsity as a percentage (0-100)
    """
    if sparse.issparse(matrix):
        total_elements = matrix.shape[0] * matrix.shape[1]
        non_zero = matrix.nnz
        sparsity = 100 * (1 - non_zero / total_elements)
    else:
        matrix = np.array(matrix)
        total_elements = matrix.size
        non_zero = np.count_nonzero(matrix)
        sparsity = 100 * (1 - non_zero / total_elements)
    
    return sparsity