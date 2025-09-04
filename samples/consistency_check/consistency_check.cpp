#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "smplx.hpp"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <output_file>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string out_file = argv[2];
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    smplx::SMPL model(model_path.c_str(), device);
    model.eval();

    int num_samples = 1;

    auto betas = 0.1 * torch::ones({num_samples, model.num_betas()}, torch::kFloat64).to(device);
    auto global_orient = 0.7 * torch::ones({num_samples, 3}, torch::kFloat64).to(device);
    auto body_pose = -0.2 * torch::ones({num_samples, 69}, torch::kFloat64).to(device);
    auto transl = 3.5 * torch::ones({num_samples, 3}, torch::kFloat64).to(device);

    auto output = model.forward(
        smplx::betas(betas), smplx::global_orient(global_orient),
        smplx::body_pose(body_pose), smplx::transl(transl),
        smplx::return_verts(true));

    auto vertices = output.vertices.value(); // (num_samples, V, 3)
    std::ofstream ofs(out_file);
    ofs << std::setprecision(8);

    for (int i = 0; i < num_samples; ++i) {
        auto verts_i = vertices[i];
        for (int j = 0; j < verts_i.size(0); ++j) {
            auto v = verts_i[j];
            ofs << v[0].item<double>() << " " 
                << v[1].item<double>() << " " 
                << v[2].item<double>() << "\n";
        }
    }
    ofs.close();
    std::cout << "Vertices written to " << out_file << std::endl;
    return 0;
}
