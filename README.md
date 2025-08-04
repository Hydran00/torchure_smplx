# torchure_smplx

A high-performance C++ implementation of the [SMPL](https://smpl.is.tue.mpg.de/), SMPL-H, and SMPL-X body models using [LibTorch](https://pytorch.org/cppdocs/). Optimized for GPU execution, capable of running up to **1000 inferences per second**.

---

## 🚀 Features

- C++ implementation using LibTorch (PyTorch C++ API)
- Supports SMPL, SMPL-H (TODO), and SMPL-X (TODO) models
- GPU acceleration via CUDA (optional)
- Suitable for real-time performance-critical applications
- Fast model loading using NumPy `.npz` format (no Python runtime required)

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/hydran00/torchure_smplx.git
cd torchure_smplx
```

---

### 2. Install dependencies

- [LibTorch](https://pytorch.org/get-started/locally/)
- [Zlib](https://zlib.net/) (often pre-installed) or with `sudo apt install zlib1g-dev`
- CMake
- CUDA (optional, for GPU builds)

Download the correct LibTorch version for your platform and CUDA version from [https://pytorch.org](https://pytorch.org).

---

### 3. Configure and build

First, determine your LibTorch path (if installed via Python):

```bash
python -c "import torch; print(torch.utils.cmake_prefix_path)"
```

Then configure and build:

```bash
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH="<output from python command>"
make
```

*Optional: To link Torch you can also set the same `Torch_DIR` variable in CMake:*

```bash
cmake .. -DTorch_DIR="/path/to/libtorch/share/cmake/Torch"
make
```
---

### 4. (Optional) Enable CUDA

If you want GPU acceleration, set the `CUDACXX` variable to point to `nvcc`, in Linux the path is usually `/usr/local/cuda/bin/nvcc`.

```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
```

Make sure your LibTorch download matches your CUDA version.

---

### 5. Convert `.pkl` models to `.npz`

This project uses `.npz` format for loading SMPL/SMPL-X models in C++. Convert the official `.pkl` models using the included Python script:

```bash
python3 pkl2npz.py \
  smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl \
  smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl
```

This will produce `.npz` files compatible with the C++ implementation.

---

### 6. Run the benchmark

```bash
./benchmark <path-to-converted-smpl-model.npz>
```

This runs the model and outputs a mesh for a single forward pass. Performance on GPU can reach ~1000 FPS depending on hardware.

### 7. Test the fitting with C++ Chamfer Distance implementation
```
./fitting <path-to-converted-smpl-model.npz>
```
---

## 🧠 Notes

- No TorchScript or ONNX required — this is a native C++ implementation.
- The `.npz` model format is designed to be lightweight and easily loadable using `cnpy`.
- You can easily integrate this library into other robotics, animation, or simulation systems.

---

## 📜 License

Apache 2.0 License (see `LICENSE` file).

---

## 🙋‍♂️ Support

Please open an issue or PR on [GitHub](https://github.com/hydran00/torchure_smplx) if you need help or want to contribute.
