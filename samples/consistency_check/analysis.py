import sys
import numpy as np

if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} <cpp_file> <python_file>")
    sys.exit(1)

cpp_file = sys.argv[1]
py_file = sys.argv[2]

cpp_data = np.loadtxt(cpp_file)
py_data = np.loadtxt(py_file)

if cpp_data.shape != py_data.shape:
    print("Shape mismatch between files!")
    sys.exit(1)

# Compare up to 4 decimals
diff = np.abs(cpp_data - py_data)
mask = diff > 1e-5

if np.any(mask):
    indices = np.argwhere(mask)
    print(f"Mismatches found at {len(indices)} vertices (more than 4 decimals):")
    for idx in indices[:10]:  # show first 10 mismatches
        i = idx[0]
        print(f"Vertex {i}: C++ = {cpp_data[i]}, Python = {py_data[i]}")
else:
    print("All vertices match up to 5 decimal places!")