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

#ifdef USE_OPEN3D
std::shared_ptr<open3d::geometry::TriangleMesh>
tensor_to_open3d_mesh(const torch::Tensor &vertices,
                      const torch::Tensor &faces) {
    auto mesh = std::make_shared<open3d::geometry::TriangleMesh>();
    auto verts = vertices.squeeze(0).to(torch::kCPU).contiguous();
    auto tris = faces.to(torch::kCPU).contiguous();

    for (int64_t i = 0; i < verts.size(0); ++i) {
        auto v = verts[i];
        mesh->vertices_.emplace_back(v[0].item<double>(), v[1].item<double>(),
                                     v[2].item<double>());
    }

    for (int64_t i = 0; i < tris.size(0); ++i) {
        auto f = tris[i];
        mesh->triangles_.emplace_back(Eigen::Vector3i(
            f[0].item<int>(), f[1].item<int>(), f[2].item<int>()));
    }

    mesh->ComputeVertexNormals();
    return mesh;
}
std::shared_ptr<open3d::geometry::PointCloud>
tensor_to_open3d_pointcloud(const torch::Tensor &vertices) {
    auto cloud = std::make_shared<open3d::geometry::PointCloud>();
    auto verts = vertices.squeeze(0).to(torch::kCPU).contiguous();

    cloud->points_.resize(verts.size(0));
    for (int64_t i = 0; i < verts.size(0); ++i) {
        auto v = verts[i];
        cloud->points_[i] = Eigen::Vector3d(
            v[0].item<double>(), v[1].item<double>(), v[2].item<double>());
    }
    cloud->PaintUniformColor(Eigen::Vector3d(0.1, 0.1, 0.8));
    return cloud;
}
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

    const int batch_size = 1;
    const int num_betas = smpl.num_betas();
    auto faces = smpl.faces();

    // We will generate a target scan to fit just by moving the betas and body
    // pose of the template SMPL model. If you want to use a real scan, load a
    // point cloud from file (you can use Open3D) and use the points as the
    // vertices_target.

    // Target SMPL
    auto betas_target =
        -2.2 * torch::ones({batch_size, num_betas}, torch::kFloat64).to(device);
    auto global_orient =
        torch::zeros({batch_size, 3}, torch::kFloat64).to(device);
    auto body_pose_target =
        torch::zeros({batch_size, 69}, torch::kFloat64).to(device);
    // move leg
    body_pose_target.index_put_({0, 3 * 1}, 0.5);
    body_pose_target.index_put_({0, 3 * 1 + 2}, -0.5);
    // move elbows
    body_pose_target.index_put_({0, Slice(3 * 18, 3 * 18 + 3)}, 0.5);
    body_pose_target.index_put_({0, Slice(3 * 19, 3 * 19 + 3)}, -0.5);
    // move head
    body_pose_target.index_put_({0, Slice(3 * 15, 3 * 15 + 3)}, 0.3);
    auto transl = torch::zeros({batch_size, 3}, torch::kFloat64).to(device);
    auto target_output = smpl.forward(
        smplx::betas(betas_target), smplx::global_orient(global_orient),
        smplx::body_pose(body_pose_target), smplx::transl(transl));
    auto vertices_target = target_output.vertices.value(); // (1, V, 3)
    save_obj("target.obj", vertices_target.squeeze(0).cpu(), faces.cpu());
    std::cout << "Generated target output" << std::endl;

    // Predicted SMPL (same as target but different betas)
    torch::Tensor betas = torch::zeros({batch_size, num_betas}, torch::kFloat64)
                              .to(device)
                              .set_requires_grad(true);
    torch::Tensor body_pose = torch::zeros({batch_size, 69}, torch::kFloat64)
                                  .to(device)
                                  .set_requires_grad(true);
    auto output_pred =
        smpl.forward(smplx::betas(betas), smplx::global_orient(global_orient),
                     smplx::body_pose(body_pose), smplx::transl(transl));
    auto vertices_pred = output_pred.vertices.value(); // (1, V, 3)

    torch::optim::Adam optimizer({betas, body_pose},
                                 torch::optim::AdamOptions(0.1));
    ChamferDistance chamfer;

#ifdef USE_OPEN3D
    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("SMPL Fitting", 1920, 1080);
    auto cloud_ptr = tensor_to_open3d_pointcloud(vertices_target);
    vis.AddGeometry(cloud_ptr);
    auto mesh_ptr = tensor_to_open3d_mesh(vertices_pred, faces);
    vis.AddGeometry(mesh_ptr);
    vis.UpdateGeometry();
    vis.PollEvents();
    vis.UpdateRender();
#endif
    std::cout << "Press Enter to start optimization..." << std::endl;
    std::cin.get();
    auto start = std::chrono::steady_clock::now();
    while (true) {
        auto duration = std::chrono::steady_clock::now() - start;
        if (duration.count() > 0.5) {
            break;
        }
    }

    std::cout << "Starting optimization..." << std::endl;

    const int steps = 200;
    for (int i = 0; i < steps; ++i) {
        optimizer.zero_grad();

        auto output = smpl.forward(
            smplx::betas(betas), smplx::global_orient(global_orient),
            smplx::body_pose(body_pose), smplx::transl(transl));

        auto vertices_pred = output.vertices.value(); // (1, V, 3)

        auto loss = chamfer.forward(vertices_pred.to(torch::kFloat64),
                                    vertices_target.to(torch::kFloat64), true);
        loss.backward();
        optimizer.step();

        if (i % 10 == 0 || i == steps - 1) {
            std::cout << "Step " << i
                      << ", Chamfer Loss: " << loss.item<float>() << std::endl;
        }

#ifdef USE_OPEN3D
        // Update Open3D mesh every N frames
        if (i % 5 == 0 || i == steps - 1) {
            auto verts = vertices_pred.squeeze(0).to(torch::kCPU).contiguous();
            mesh_ptr->vertices_.clear();
            for (int64_t j = 0; j < verts.size(0); ++j) {
                auto v = verts[j];
                mesh_ptr->vertices_.emplace_back(v[0].item<double>(),
                                                 v[1].item<double>(),
                                                 v[2].item<double>());
            }
            mesh_ptr->ComputeVertexNormals();
            vis.UpdateGeometry(mesh_ptr);
            vis.PollEvents();
            vis.UpdateRender();
        }
#endif
    }

    auto final_output = smpl.forward(
        smplx::betas(betas.detach()), smplx::global_orient(global_orient),
        smplx::body_pose(body_pose), smplx::transl(transl));

    auto final_vertices = final_output.vertices.value();
    save_obj("predicted.obj", final_vertices.squeeze(0).cpu(), faces.cpu());

#ifdef USE_OPEN3D
    std::cout << "Press Q in the viewer window to close.\n";
    vis.Run();
    vis.DestroyVisualizerWindow();
#endif

    return 0;
}
