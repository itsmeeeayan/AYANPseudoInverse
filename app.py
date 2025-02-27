import streamlit as st

# ---------------- Matrix Operations (unchanged) ----------------

def matrix_transpose(A):
    """Returns the transpose of matrix A."""
    m = len(A)
    n = len(A[0])
    T = [[A[i][j] for i in range(m)] for j in range(n)]
    return T

def matrix_multiply(A, B):
    """Multiplies two matrices A and B."""
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
    C = [[A[i][j] + B[i][j] for j in range(n)] for i in range(m)]
    return C

def matrix_identity(n):
    """Returns an n x n identity matrix."""
    I = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        I[i][i] = 1
    return I

def matrix_inverse(A):
    """Computes the inverse of a square matrix A using Gauss-Jordan elimination."""
    n = len(A)
    # Create an augmented matrix with the identity matrix
    aug = [row[:] + identity_row for row, identity_row in zip(A, matrix_identity(n))]
    for i in range(n):
        pivot = aug[i][i]
        if abs(pivot) < 1e-12:
            # Try to swap with a row below having a non-zero element in this column
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
        for j in range(2*n):
            aug[i][j] /= pivot
        # Eliminate the i-th column elements in other rows
        for k in range(n):
            if k != i:
                factor = aug[k][i]
                for j in range(2*n):
                    aug[k][j] -= factor * aug[i][j]
    # Extract inverse matrix from augmented matrix
    inv = [row[n:] for row in aug]
    return inv

def add_regularization(M, lambda_val):
    """Adds a small regularization term lambda*I to square matrix M."""
    n = len(M)
    I = matrix_identity(n)
    for i in range(n):
        I[i][i] *= lambda_val
    return matrix_add(M, I)

def pseudo_inverse(A):
    """Computes the pseudo inverse of matrix A.
       Uses A⁺ = (AᵀA)⁻¹Aᵀ if rows >= cols,
       and A⁺ = Aᵀ(AAᵀ)⁻¹ if rows < cols.
       If the required inverse is singular, a small regularization is added."""
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
        pseudo = matrix_multiply(inv_AtA, At)
    else:
        AAt = matrix_multiply(A, At)
        try:
            inv_AAt = matrix_inverse(AAt)
        except ValueError:
            AAt_reg = add_regularization(AAt, 1e-10)
            inv_AAt = matrix_inverse(AAt_reg)
        pseudo = matrix_multiply(At, inv_AAt)
        
    return pseudo

# ---------------- Streamlit Input/Output Functions ----------------

def get_matrix_streamlit():
    m = st.number_input("Enter number of rows:", min_value=1, value=3, step=1)
    n = st.number_input("Enter number of columns:", min_value=1, value=3, step=1)
    st.write("Enter the matrix rows (space-separated numbers):")
    default_text = "\n".join([" ".join(["0"] * n) for _ in range(m)])
    matrix_text = st.text_area("Matrix", value=default_text, height=150)
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
            st.error(f"Row {i+1} length does not match the number of columns ({n}).")
            return None
        matrix.append(row)
    return matrix

def print_matrix_streamlit(matrix, title="Matrix"):
    st.subheader(title)
    for row in matrix:
        st.write(" ".join(f"{x:.4f}" for x in row))

# ---------------- Main Streamlit App ----------------

def main():
    st.title("Pseudo Inverse Calculator without using library inverse/pinv functions")
    st.write("We only use Python’s basic math operations.")

    A = get_matrix_streamlit()
    if A is None:
        return

    st.subheader("Input Matrix:")
    print_matrix_streamlit(A, "Input Matrix:")

    try:
        pinv = pseudo_inverse(A)
        print_matrix_streamlit(pinv, "Pseudo Inverse:")
    except Exception as e:
        st.error("Error computing pseudo inverse: " + str(e))

if __name__ == "__main__":
    main()
