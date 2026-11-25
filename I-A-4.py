import numpy as np
import matplotlib.pyplot as plt

# Pauli matrices
X = np.array([[0, 1],
              [1, 0]])

Y = np.array([[0, -1j],
              [1j, 0]])
Z = np.array([[1, 0],
              [0, -1]])

I = np.eye(2)

def tensor_product(*matrices):
    result = np.asarray(matrices[0], dtype=complex)

    for B in matrices[1:]:
        B = np.asarray(B, dtype=complex)
        m, n = result.shape
        p, q = B.shape

        # Allocate the resulting matrix
        new_matrix = np.zeros((m * p, n * q), dtype=complex)

        # Each element of A becomes a block proportional to B
        for i in range(m):
            for j in range(n):
                new_matrix[i*p:(i+1)*p, j*q:(j+1)*q] = (
                    result[i, j] * B
                )
        result = new_matrix
    return result

term1 = tensor_product(Z, X, Y, Z, Y)
term2 = tensor_product(X, I, Y, Z, Y)
M = term1 + term2

# Verify manual implementation against NumPy's Kronecker product
term1_check = np.kron(np.kron(np.kron(np.kron(Z, X), Y), Z), Y)
term2_check = np.kron(np.kron(np.kron(np.kron(X, I), Y), Z), Y)
print("Result matches:", np.allclose(M, term1_check + term2_check))

# Visualization
if np.allclose(M.imag, 0):
    M_print = np.rint(M.real).astype(int)
else:
    M_print = M

np.set_printoptions(linewidth=200, threshold=np.inf)
print(M_print)
