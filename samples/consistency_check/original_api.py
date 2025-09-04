import torch
import numpy as np
from smplx import SMPL
import sys

if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} <model_path> <output_file>")
    sys.exit(1)

model_path = sys.argv[1]
out_file = sys.argv[2]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SMPL(model_path=model_path, batch_size=1, create_global_orient=True, create_body_pose=True, create_betas=True).to(device)
model.eval()

betas = 0.1 * torch.ones((1, model.num_betas), dtype=torch.float32, device=device)
global_orient = 0.7 * torch.ones((1, 3), dtype=torch.float32, device=device)
body_pose = - 0.2 * torch.ones((1, 69), dtype=torch.float32, device=device)
transl = 3.5 * torch.ones((1, 3), dtype=torch.float32, device=device)

output = model.forward(betas=betas, global_orient=global_orient, body_pose=body_pose, transl=transl)
vertices = output.vertices  # (1, V, 3)

with open(out_file, 'w') as f:
    for i in range(vertices.shape[1]):
        f.write(f"{vertices[0, i, 0].item():.8f} {vertices[0, i, 1].item():.8f} {vertices[0, i, 2].item():.8f}\n")

print(f"Vertices written to {out_file}")
