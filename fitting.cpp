#include <fstream>
#include "chamfer.h" // ChamferDistance module
#include "smplx.hpp"

void save_obj(const std::string &filename, const torch::Tensor &vertices,
              const torch::Tensor &faces) {
    std::ofstream obj_file(filename);
    if (!obj_file.is_open()) {
        throw std::runtime_error("Could not open OBJ file for writing.");
    }
    obj_file << std::fixed << std::setprecision(8);

    for (int64_t i = 0; i < vertices.size(0); ++i) {
        auto v = vertices[i];
        obj_file << "v " << v[0].item<float>() << " " << v[1].item<float>()
                 << " " << v[2].item<float>() << "\n";
    }

    for (int64_t i = 0; i < faces.size(0); ++i) {
        auto f = faces[i];
        obj_file << "f " << f[0].item<int64_t>() + 1 << " "
                 << f[1].item<int64_t>() + 1 << " " << f[2].item<int64_t>() + 1
                 << "\n";
    }
}

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

    const int batch_size = 1;
    const int num_betas = smpl.num_betas();

    auto faces = smpl.faces();

    // === Target ===
    auto betas_target =
        (-2) * torch::ones({batch_size, num_betas}, torch::kFloat64).to(device);
    auto global_orient =
        torch::zeros({batch_size, 3}, torch::kFloat64).to(device);
    auto body_pose = torch::zeros({batch_size, 69}, torch::kFloat64).to(device);
    auto transl = torch::zeros({batch_size, 3}, torch::kFloat64).to(device);

    auto target_output = smpl.forward(
        smplx::betas(betas_target), smplx::global_orient(global_orient),
        smplx::body_pose(body_pose), smplx::transl(transl));

    auto vertices_target = target_output.vertices.value(); // shape: (1, V, 3)

    save_obj("target.obj", vertices_target.squeeze(0).cpu(), faces.cpu());

    // === Optimization Setup ===
    torch::Tensor betas = torch::zeros({batch_size, num_betas}, torch::kFloat64)
                              .to(device)
                              .set_requires_grad(true);

    torch::optim::Adam optimizer({betas}, torch::optim::AdamOptions(0.01));

    ChamferDistance chamfer;

    const int steps = 300;
    for (int i = 0; i < steps; ++i) {
        optimizer.zero_grad();

        auto output = smpl.forward(
            smplx::betas(betas), smplx::global_orient(global_orient),
            smplx::body_pose(body_pose), smplx::transl(transl));

        auto vertices_pred = output.vertices.value(); // shape: (1, V, 3)

        // Chamfer loss
        auto loss = chamfer.forward(vertices_pred.to(torch::kFloat32),
                                    vertices_target.to(torch::kFloat32), true);

        loss.backward();
        optimizer.step();

        if (i % 10 == 0 || i == steps - 1) {
            std::cout << "Step " << i
                      << ", Chamfer Loss: " << loss.item<float>() << std::endl;
        }
    }

    auto final_output = smpl.forward(
        smplx::betas(betas.detach()), smplx::global_orient(global_orient),
        smplx::body_pose(body_pose), smplx::transl(transl));

    auto final_vertices = final_output.vertices.value();
    save_obj("predicted.obj", final_vertices.squeeze(0).cpu(), faces.cpu());

    return 0;
}
