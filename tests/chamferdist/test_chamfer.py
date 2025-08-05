import torch
from chamferdist import ChamferDistance
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define known point clouds (same as C++ version)
p1 = torch.tensor(
    [
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ],
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [0.5, 0.5, -1.5],
        ],
    ],
    dtype=torch.float32,
).to(device)

p2 = torch.tensor(
    [
        [
            [0.1, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, -1.1, 0.0],
            [1.0, -1.1, 0.0],
            [0.5, 0.5, 0.6],
        ],
        [
            [0.0, 0.0, 0.9],
            [1.0, 0.0, 1.1],
            [0.0, 1.0, 0.9],
            [1.0, 1.0, 1.1],
            [0.5, 0.5, 2.0],
        ],
    ],
    dtype=torch.float32,
).to(device)

print("Source Point Cloud (p1):\n", p1)
print("\nTarget Point Cloud (p2):\n", p2)

# Create ChamferDistance instance
chamfer = ChamferDistance()
start = time.time()
dist = chamfer(p1, p2, bidirectional=True)
end = time.time()
duration = end - start
print("\nTime taken for Chamfer Distance computation:", duration, "seconds")

print("\nChamfer Distance:", dist.item())
