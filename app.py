import streamlit as st

# ---------------- Matrix Operation Functions ----------------

def matrix_transpose(A):
    """Returns the transpose of matrix A."""
    m = len(A)
    n = len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]

def matrix_multiply(A, B):
    """Multiplies matrix A by matrix B."""
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    result = [[0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result

def matrix_add(A, B):
    """Adds two matrices A and B."""
    m = len(A)
    n = len(A[0])
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(m)]

def matrix_identity(n):
    """Returns an n x n identity matrix."""
    I = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        I[i][i] = 1
    return I

def matrix_inverse(A):
    """Computes the inverse of a square matrix A using Gauss-Jordan elimination."""
    n = len(A)
    # Create augmented matrix [A | I]
    aug = [row[:] + identity_row for row, identity_row in zip(A, matrix_identity(n))]
    
    for i in range(n):
        pivot = aug[i][i]
        if abs(pivot) < 1e-12:
            # Swap with a row below that has a nonzero pivot element
            swap_row = None
            for j in range(i+1, n):
                if abs(aug[j][i]) > 1e-12:
                    swap_row = j
                    break
            if swap_row is None:
                raise ValueError("Matrix is singular and cannot be inverted.")
            aug[i], aug[swap_row] = aug[swap_row], aug[i]
            pivot = aug[i][i]
        # Normalize the pivot row
        for j in range(2 * n):
            aug[i][j] /= pivot
        # Eliminate pivot column in other rows
        for k in range(n):
            if k != i:
                factor = aug[k][i]
                for j in range(2 * n):
                    aug[k][j] -= factor * aug[i][j]
                    
    # Extract inverse from augmented matrix
    return [row[n:] for row in aug]

def add_regularization(M, lambda_val):
    """Adds lambda*I to square matrix M (for regularization)."""
    n = len(M)
    I = matrix_identity(n)
    for i in range(n):
        I[i][i] *= lambda_val
    return matrix_add(M, I)

def pseudo_inverse(A):
    """Computes the pseudo inverse of matrix A.
       Uses A⁺ = (AᵀA)⁻¹Aᵀ if rows >= cols,
       and A⁺ = Aᵀ(AAᵀ)⁻¹ if rows < cols.
       Regularizes if needed."""
    m = len(A)
    n = len(A[0])
    At = matrix_transpose(A)
    
    if m >= n:
        AtA = matrix_multiply(At, A)
        try:
            inv_AtA = matrix_inverse(AtA)
        except ValueError:
            AtA_reg = add_regularization(AtA, 1e-10)
            inv_AtA = matrix_inverse(AtA_reg)
        return matrix_multiply(inv_AtA, At)
    else:
        AAt = matrix_multiply(A, At)
        try:
            inv_AAt = matrix_inverse(AAt)
        except ValueError:
            AAt_reg = add_regularization(AAt, 1e-10)
            inv_AAt = matrix_inverse(AAt_reg)
        return matrix_multiply(At, inv_AAt)

# ---------------- Streamlit Interface Functions ----------------

def get_matrix_streamlit():
    m = st.number_input("Rows (m):", min_value=1, value=2, step=1, key="rows")
    n = st.number_input("Columns (n):", min_value=1, value=2, step=1, key="cols")
    st.write("Enter the matrix rows (each row on a new line, space-separated numbers):")
    default_text = "\n".join([" ".join(["0"] * n) for _ in range(m)])
    matrix_text = st.text_area("Matrix", value=default_text, height=150, key="matrix_text")
    matrix = []
    lines = matrix_text.strip().splitlines()
    if len(lines) != m:
        st.error(f"Expected {m} rows but got {len(lines)}.")
        return None
    for i, line in enumerate(lines):
        try:
            row = list(map(float, line.strip().split()))
        except Exception as e:
            st.error(f"Error in row {i+1}: {e}")
            return None
        if len(row) != n:
            st.error(f"Row {i+1} must have exactly {n} numbers.")
            return None
        matrix.append(row)
    return matrix

def print_matrix_streamlit(matrix, title="Matrix"):
    st.subheader(title)
    for row in matrix:
        st.write("  ".join(f"{num:.4f}" for num in row))

# ---------------- Main App ----------------

def main():
    st.title("Matrix Pseudo-Inverse Calculator")
    st.write("Enter the dimensions and values of your matrix. The pseudo-inverse is computed using only Python’s basic math operations.")
    
    with st.form(key="matrix_form"):
        A = get_matrix_streamlit()
        submit_button = st.form_submit_button(label="Calculate Pseudo-Inverse")
    
    if submit_button:
        if A is None:
            return
        st.subheader("Input Matrix:")
        print_matrix_streamlit(A, "Input Matrix:")
        try:
            A_pinv = pseudo_inverse(A)
            print_matrix_streamlit(A_pinv, "Pseudo-Inverse:")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
