import streamlit as st

def matrix_transpose(A):
    """Return the transpose of matrix A."""
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def matrix_multiply(A, B):
    """Multiply matrix A (m x p) by matrix B (p x n) and return the m x n result."""
    m = len(A)
    p = len(A[0])
    n = len(B[0])
    result = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for k in range(p):
                result[i][j] += A[i][k] * B[k][j]
    return result

def matrix_inverse(A):
    """Compute the inverse of a square matrix A using Gauss-Jordan elimination.
       Raises ValueError if A is singular."""
    n = len(A)
    # Create an augmented matrix [A | I]
    aug = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(A)]
    for i in range(n):
        pivot = aug[i][i]
        if pivot == 0:
            # Swap with a row below that has a non-zero pivot.
            for j in range(i+1, n):
                if aug[j][i] != 0:
                    aug[i], aug[j] = aug[j], aug[i]
                    pivot = aug[i][i]
                    break
            else:
                raise ValueError("Matrix is singular and cannot be inverted")
        # Normalize the pivot row.
        for j in range(2 * n):
            aug[i][j] /= pivot
        # Eliminate the pivot column in all other rows.
        for k in range(n):
            if k != i:
                factor = aug[k][i]
                for j in range(2 * n):
                    aug[k][j] -= factor * aug[i][j]
    # Extract the inverse matrix from the augmented matrix.
    inv = [row[n:] for row in aug]
    return inv

def pseudo_inverse(A):
    """Compute the pseudo inverse of matrix A.
       If m >= n, A⁺ = (AᵀA)⁻¹ Aᵀ; if m < n, A⁺ = Aᵀ (AAᵀ)⁻¹."""
    m = len(A)
    n = len(A[0])
    A_T = matrix_transpose(A)
    if m >= n:
        ATA = matrix_multiply(A_T, A)
        ATA_inv = matrix_inverse(ATA)
        return matrix_multiply(ATA_inv, A_T)
    else:
        AAT = matrix_multiply(A, A_T)
        AAT_inv = matrix_inverse(AAT)
        return matrix_multiply(A_T, AAT_inv)

# ------------------------ Streamlit App ------------------------

st.title("Pseudo Inverse Calculator")
st.write("""
This app computes the pseudo inverse (Moore-Penrose inverse) of a matrix using basic Python math operations.
Please note that the algorithm assumes the matrix has full rank so that the required inverses exist.
""")

# Input: matrix dimensions
rows = st.number_input("Number of rows (m):", min_value=1, value=3, step=1)
cols = st.number_input("Number of columns (n):", min_value=1, value=3, step=1)

st.write("Enter the matrix rows below. Each row should be on a new line and numbers separated by spaces.")
default_matrix = "1 2 3\n4 5 6\n7 8 9" if rows == 3 and cols == 3 else ""
matrix_input = st.text_area("Matrix Input", default_matrix, height=150)

if st.button("Compute Pseudo Inverse"):
    try:
        # Parse the input into a matrix (list of lists)
        A = []
        lines = matrix_input.strip().splitlines()
        if len(lines) != rows:
            st.error(f"Expected {rows} rows but got {len(lines)}.")
        else:
            for line in lines:
                # Split the line into numbers and convert to float
                row_vals = list(map(float, line.strip().split()))
                if len(row_vals) != cols:
                    st.error(f"Each row must have exactly {cols} numbers. Check: '{line}'")
                    A = None
                    break
                A.append(row_vals)
            if A is not None:
                # Compute the pseudo inverse
                A_pinv = pseudo_inverse(A)
                st.subheader("Pseudo Inverse:")
                for r in A_pinv:
                    st.write(r)
    except Exception as e:
        st.error("An error occurred: " + str(e))
