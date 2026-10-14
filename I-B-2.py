import numpy as np

psi1 = np.array([1, 0, 0, -1j], dtype=complex) / np.sqrt(2)
psi2 = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
psi3 = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
psi4 = np.array([1, 1, 1, 1], dtype=complex) / 2

def gram_schmidt(vectors):
    basis = []
    for u in vectors:
        w = u.copy()
        for v in basis:
            w -= np.vdot(v, w) * v
        norm = np.linalg.norm(w)
        basis.append(w / norm)
    return basis

print("====== Basis 1: psi1, psi2, psi3, psi4 ======")
basis_1 = gram_schmidt([psi1,psi2, psi3, psi4])
np.set_printoptions(precision=3, suppress=True)
for j, q in enumerate(basis_1, start=1):
        print(f"q{j} =\n{q.reshape(-1, 1)}")

print("====== Basis 2: psi4, psi1, psi2, psi3 ======")
basis_2 = gram_schmidt([psi4,psi1, psi2, psi3])
np.set_printoptions(precision=3, suppress=True)
for j, q in enumerate(basis_2, start=1):
        print(f"q{j} =\n{q.reshape(-1, 1)}")

print("====== Basis 3: psi3, psi4, psi1, psi2 ======")
basis_3 = gram_schmidt([psi3,psi4, psi1, psi2])
np.set_printoptions(precision=3, suppress=True)
for j, q in enumerate(basis_3, start=1):
        print(f"q{j} =\n{q.reshape(-1, 1)}")

print("====== Basis 4: psi2, psi3, psi4, psi1 ======")
basis_4 = gram_schmidt([psi2,psi3, psi4, psi1])
np.set_printoptions(precision=3, suppress=True)
for j, q in enumerate(basis_4, start=1):
        print(f"q{j} =\n{q.reshape(-1, 1)}")
    