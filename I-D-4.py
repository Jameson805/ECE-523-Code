
import numpy as np
from itertools import product

# Pauli matrices
X = np.array([[0, 1],
              [1, 0]])

Y = np.array([[0, -1j],
              [1j, 0]])

Z = np.array([[1, 0],
              [0, -1]])

I = np.eye(2)
paulis = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

def pauli_decomposition(H, n, tol=1e-10):
    coefficients = {}
    H_reconstructed = np.zeros_like(H, dtype=complex)

    # Iterate through all 4^n Pauli strings
    for labels in product('IXYZ', repeat=n):
        label = ''.join(labels)
        P = np.array([[1]], dtype=complex)
        for p in labels:
            P = np.kron(P, paulis[p])

        # Calculate coefficient using trace formula
        coeff = np.trace(P @ H) / (2**n)
        H_reconstructed += coeff * P

        if abs(coeff) > tol:
            coefficients[label] = np.real_if_close(coeff).item()

    return coefficients, H_reconstructed



phi_JLP = np.diag([-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]) / 15

phi_HO = np.zeros((16, 16), dtype=complex)
for n in range(15):
    phi_HO[n, n+1] = np.sqrt(n+1) / 2
    phi_HO[n+1, n] = np.sqrt(n+1) / 2

jlp_coeffs, jlp_reconstructed = pauli_decomposition(phi_JLP, 4)
ho_coeffs, ho_reconstructed = pauli_decomposition(phi_HO, 4)


def print_decomposition(name, coefficients, original, reconstructed):
    print(f"\n{name} Pauli Decomposition:")
    entries = [
        f"{label}: {coeff:+.4f}"
        for label, coeff in coefficients.items()
    ]

    for i in range(0, len(entries), 4):
        print("    ".join(entries[i:i+4]))

    print(f"\nNonzero coefficients: {len(coefficients)}")
    print("Reconstruction matches:",
          np.allclose(original, reconstructed))

print_decomposition("JLP", jlp_coeffs, phi_JLP, jlp_reconstructed)
print_decomposition("HO", ho_coeffs, phi_HO, ho_reconstructed)
