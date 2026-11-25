import numpy as np
import matplotlib.pyplot as plt

# Pauli matrices
X = np.array([[0, 1],
              [1, 0]])

Y = np.array([[0, -1j],
              [1j, 0]])

I = np.eye(2)

# Hamiltonian, initial state, and bell basis states
H = np.kron(X, Y)
psi0 = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
bell_basis = np.column_stack([
    np.array([1, 0, 0, 1]) / np.sqrt(2),   # Phi+
    np.array([1, 0, 0, -1]) / np.sqrt(2),  # Phi-
    np.array([0, 1, 1, 0]) / np.sqrt(2),   # Psi+
    np.array([0, 1, -1, 0]) / np.sqrt(2)   # Psi-
])
labels = [r'$\Phi^+$', r'$\Phi^-$', r'$\Psi^+$', r'$\Psi^-$']

# Time evolution
times = np.linspace(0, 2, 500)
probabilities = []
for t in times:
    U = np.cos(t) * np.eye(4) - 1j * np.sin(t) * H
    psi = U @ psi0
    amplitudes = bell_basis.conj().T @ psi
    probabilities.append(np.abs(amplitudes)**2)
probabilities = np.array(probabilities)

# Plot
plt.figure(figsize=(9, 5))
for i in range(4):
    plt.plot(times, probabilities[:, i], label=labels[i])
plt.xlabel('Time')
plt.ylabel('Probability')
plt.title('Bell Basis Probability Evolution')
plt.xlim(0, 2)
plt.ylim(0, 1.05)
plt.legend()
plt.grid()
plt.show()
