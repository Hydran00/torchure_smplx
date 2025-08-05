#include <torch/torch.h>
#include <iostream>
#include "chamfer.h" // Assuming this defines ChamferDistance module

int main() {
    // Set device (CPU or CUDA if available)
    torch::Device device(torch::kCUDA);
    if (!torch::cuda::is_available()) {
        std::cout << "CUDA not available, switching to CPU.\n";
        device = torch::Device(torch::kCPU);
    }

    // Manually defined point clouds
    torch::Tensor p1 =
        torch::tensor({{{0.0, 0.0, 0.0},
                        {1.0, 0.0, 0.0},
                        {0.0, 1.0, 0.0},
                        {1.0, 1.0, 0.0},
                        {0.5, 0.5, 0.5}},
                       {{0.0, 0.0, 1.0},
                        {1.0, 0.0, 1.0},
                        {0.0, 1.0, -1.0},
                        {1.0, 1.0, -1.0},
                        {0.5, 0.5, -1.5}}},
                      torch::dtype(torch::kFloat32).device(device));

    torch::Tensor p2 =
        torch::tensor({{{0.1, 0.0, 0.0},
                        {-2.0, 0.0, 0.0},
                        {0.0, -1.1, 0.0},
                        {1.0, -1.1, 0.0},
                        {0.5, 0.5, 0.6}},
                       {{0.0, 0.0, 0.9},
                        {1.0, 0.0, 1.1},
                        {0.0, 1.0, 0.9},
                        {1.0, 1.0, 1.1},
                        {0.5, 0.5, 2.0}}},
                      torch::dtype(torch::kFloat32).device(device));

    std::cout << "Source Point Cloud (p1):\n" << p1 << "\n\n";
    std::cout << "Target Point Cloud (p2):\n" << p2 << "\n\n";

    ChamferDistance chamfer;
    auto start_time = std::chrono::high_resolution_clock::now();
    torch::Tensor dist = chamfer.forward(p1, p2, /*bidirectional=*/true);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    std::cout << "Time taken for Chamfer Distance computation: "
              << duration.count() << " milliseconds\n";
    std::cout << "Chamfer Distance: " << dist.item<float>() << std::endl;

    return 0;
}
