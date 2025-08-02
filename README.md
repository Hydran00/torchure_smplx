# torchure_smplx
Libtorch based C++ implementation of SMPL, SMPL-H, SMPL-X body models for high performance applications. Runs close to 1000 inferences per second on GPU.

# Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/hydran00/torchure_smplx.git
   ```
2. Set your torch and CUDA paths in the `CMakeLists.txt`:  
   you can find it easily if you have pyTorch installed:
   ```bash
    python -c "import torch; print(torch.utils.cmake_prefix_path)"
    ```
    then set the `Torch_DIR` variable in `CMakeLists.txt` to the output path, e.g.:
   ```bash
    set(Torch_DIR /home/user/.local/lib/python3.10/site-packages/torch/share/cmake/Torch)
    ```
    To use CUDA, you just need to set the `CUDACXX` variable to the path of `nvcc`:
   ```bash
   export CUDACXX=/usr/local/cuda/bin/nvcc 
   ```
3. Build the project:
   ```bash
   cd torchure_smplx
   mkdir build
   cd build
   cmake ..
   make
   ```
4. Download the SMPL and/or variants pkl model and convert them with `pkl2npz.py` script in a `.npz` format:
    ```bash
    # example for SMPL model
    python3 pkl2npz.py smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl
    ```
5. Test the benchmark and save the SMPL mesh result:
   ```bash
   ./benchmark <path to your npz SMPL(x/h) model>
   ```
# References
The main code is a revised and improved version of the following repositories:  
[smpl-cpp](https://github.com/Arktische/smpl-cpp)  
[smplxpp](https://github.com/sxyu/smplxpp)

