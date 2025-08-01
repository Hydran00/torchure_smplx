#include <fstream>
#include "body_models.hpp"
#include "common.hpp"
#include "smplx.hpp"

void save_obj(const std::string &filename, const torch::Tensor &vertices,
              const torch::Tensor &faces) {
    std::ofstream obj_file(filename);
    if (!obj_file.is_open()) {
        throw std::runtime_error("Could not open OBJ file for writing.");
    }

    // Write vertices
    for (int64_t i = 0; i < vertices.size(0); ++i) {
        auto v = vertices[i];
        obj_file << "v " << v[0].item<float>() << " " << v[1].item<float>()
                 << " " << v[2].item<float>() << "\n";
    }

    // Write faces (1-based indexing)
    for (int64_t i = 0; i < faces.size(0); ++i) {
        auto f = faces[i];
        obj_file << "f " << f[0].item<int64_t>() + 1 << " "
                 << f[1].item<int64_t>() + 1 << " " << f[2].item<int64_t>() + 1
                 << "\n";
    }
}
// Compute FPS for N iterations of forward pass
double compute_fps(smplx::SMPL &model, const torch::Tensor &betas,
                   const torch::Tensor &global_orient,
                   const torch::Tensor &body_pose, const torch::Tensor &transl,
                   int iterations = 1000) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        auto output = model.forward(
            smplx::betas(betas), smplx::global_orient(global_orient),
            smplx::body_pose(body_pose), smplx::transl(transl));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    double seconds = duration / 1000.0;
    return iterations / seconds;
}
int main(int argc, char *argv[]) {
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Cuda available: " << std::boolalpha
              << torch::cuda::is_available() << std::endl;
    std::string path = argv[1];
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }
    if (!std::filesystem::exists(path)) {
        std::cerr << "Model path does not exist: " << path << std::endl;
        return 1;
    }
    std::cout << "Using model path: " << path << std::endl;
    smplx::SMPL smpl(path.c_str(), device);
    smpl.eval();
    int batch_size = 1;
    auto betas = -2 * torch::ones({batch_size, 10},
                                  torch::TensorOptions().device(device));
    auto global_orient =
        torch::ones({batch_size, 3}, torch::TensorOptions().device(device));
    auto body_pose =
        torch::zeros({batch_size, 69}, torch::TensorOptions().device(device));
    auto transl =
        torch::zeros({batch_size, 3}, torch::TensorOptions().device(device));

    int iterations = 1000;
    double fps =
        compute_fps(smpl, betas, global_orient, body_pose, transl, iterations);

    std::cout << "FPS (forward pass, " << iterations << " iterations): " << fps
              << std::endl;

    // auto faces = smpl.faces();
    // save_obj("output_path.obj", output.vertices.value().squeeze(0), faces);
    return 0;
}
