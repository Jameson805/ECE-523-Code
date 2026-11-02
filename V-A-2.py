import numpy as np
import matplotlib.pyplot as plt

p = 0.15
n_steps = 20

states = {
    "|0>": [0, 0, 1],
    "|1>": [0, 0, -1],
    "|+>": [1, 0, 0],
    "|->": [-1, 0, 0],
    "|+i>": [0, 1, 0],
    "|-i>": [0, -1, 0]
}

trajectories = {}
for name, initial in states.items():
    r = np.array(initial, dtype=float)
    trajectory = [r.copy()]
    for _ in range(n_steps):
        x, y, z = r
        r = np.array([
            np.sqrt(1-p)*x,
            np.sqrt(1-p)*y,
            z
        ])
        trajectory.append(r.copy())
    trajectories[name] = np.array(trajectory)


fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection='3d')

u = np.linspace(0, 2*np.pi, 60)
v = np.linspace(0, np.pi, 30)
X = np.outer(np.cos(u), np.sin(v))
Y = np.outer(np.sin(u), np.sin(v))
Z = np.outer(np.ones_like(u), np.cos(v))
ax.plot_wireframe(X, Y, Z, alpha=0.1)

for name, trajectory in trajectories.items():
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        marker='o',
        markersize=3,
        label=name
    )

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Phase Damping on the Bloch Sphere")
ax.legend()
ax.set_box_aspect([1, 1, 1])
plt.show()
