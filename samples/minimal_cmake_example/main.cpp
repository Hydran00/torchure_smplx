#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "chamfer.h"
#include "smplx.hpp"
using namespace torch::indexing;

#ifdef USE_OPEN3D
#include <open3d/Open3D.h>
#endif

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }

    std::string path = argv[1];
    if (!std::filesystem::exists(path)) {
        std::cerr << "Model path does not exist: " << path << std::endl;
        return 1;
    }

    torch::Device device =
        torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << device << "\n";

    smplx::SMPL smpl(path.c_str(), device);
    smpl.eval();
    ChamferDistance chamfer;

    // Predicted SMPL (same as target but different betas)
#ifdef USE_OPEN3D
    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("SMPL Fitting", 1920, 1080);
    vis.DestroyVisualizerWindow();
#endif

    return 0;
}
