#include "body_models.hpp"
#include <fstream>
#include <memory>
#include "common.hpp"
#include "lbs.hpp"
#include "utils.hpp"
#include "vertex_joint_selector.hpp"

#define DEBUG true

namespace smplx {

constexpr int SMPL::SHAPE_SPACE_DIM;

auto SMPL::construct(const char *model_path) -> void {
    // Load and check model path
    vertex_joint_selector_ =
        std::make_unique<VertexJointSelector>(vars_.vertex_ids, device_);
    ASSERT_MSG(std::filesystem::exists(model_path), "%s not exist", model_path);
    ASSERT_MSG(check_file_ext(model_path, "json"), "invalid extension %s",
               std::filesystem::path(model_path).extension().string().c_str());

    // Parse JSON file
    std::ifstream file(model_path);
    if (!file.is_open())
        throw std::runtime_error("Could not open model file");
    nlohmann::json data = nlohmann::json::parse(file);
    std::cout << "Model loaded: " << model_path << std::endl;

    // Load model components from JSON
    shapedirs_ = utils::load_tensor(
        data["shapedirs"].get<std::vector<std::vector<std::vector<float>>>>(),
        vars_.dtype, device_);
    auto num_betas = shapedirs_.size(2);

    faces_ = utils::load_tensor(data["f"].get<std::vector<std::vector<int>>>(),
                                torch::kInt32, device_);

    // Handle SMIL kid model if needed (from JSON)
    if (vars_.age == "kid" && vars_.kid_template_path.has_value()) {
        ASSERT_MSG(std::filesystem::exists(vars_.kid_template_path.value()),
                   "Kid template path does not exist");

        std::ifstream kid_file(vars_.kid_template_path.value());
        if (!kid_file.is_open())
            throw std::runtime_error("Could not open kid template JSON file");
        nlohmann::json kid_data = nlohmann::json::parse(kid_file);

        torch::Tensor v_template_adult = utils::load_tensor(
            data["v_template"].get<std::vector<std::vector<float>>>(),
            vars_.dtype, device_);
        torch::Tensor v_template_kid = utils::load_tensor(
            kid_data["v_template"].get<std::vector<std::vector<float>>>(),
            vars_.dtype, device_);

        v_template_kid -= torch::mean(v_template_kid, 0);
        auto v_template_diff =
            torch::unsqueeze(v_template_kid - v_template_adult, 2);

        shapedirs_ = torch::cat({shapedirs_, v_template_diff}, 2);
        ++num_betas;
    }

    vars_.num_betas = num_betas;
    register_buffer("shapedirs", shapedirs_);
    register_buffer("faces_tensor", faces_);

    // Initialize parameters if missing
    if (!vars_.betas.has_value()) {
        vars_.betas.emplace(
            torch::zeros({vars_.batch_size, vars_.num_betas},
                         torch::dtype(vars_.dtype).device(device_))
                .requires_grad_(true));
    }
    if (!vars_.global_orient.has_value()) {
        vars_.global_orient.emplace(torch::zeros(
            {vars_.batch_size, 3}, torch::dtype(vars_.dtype).device(device_)));
    }
    if (!vars_.body_pose.has_value()) {
        vars_.body_pose.emplace(
            torch::zeros({vars_.batch_size, NUM_BODY_JOINTS * 3},
                         torch::dtype(vars_.dtype).device(device_)));
    }
    if (!vars_.transl.has_value()) {
        vars_.transl.emplace(torch::zeros(
            {vars_.batch_size, 3}, torch::dtype(vars_.dtype).device(device_)));
    }

    // Load remaining model buffers
    v_template_ = utils::load_tensor(
        data["v_template"].get<std::vector<std::vector<float>>>(), vars_.dtype,
        device_);
    if (vars_.v_template.has_value())
        v_template_ = vars_.v_template.value().clone().to(device_);

    J_regressor_ = utils::load_tensor(
        data["J_regressor"].get<std::vector<std::vector<float>>>(), vars_.dtype,
        device_);
    posedirs_ = utils::load_tensor(
        data["posedirs"].get<std::vector<std::vector<std::vector<float>>>>(),
        vars_.dtype, device_);
    auto num_pose_basis = posedirs_.size(2);
    posedirs_ = posedirs_.reshape({-1, num_pose_basis}).t().contiguous();

    lbs_weights_ = utils::load_tensor(
        data["weights"].get<std::vector<std::vector<float>>>(), vars_.dtype,
        device_);
    parents_ =
        torch::from_blob((void *)parents,
                         {sizeof(parents) / sizeof(parents[0])}, torch::kLong)
            .clone()
            .to(device_);

    // Register buffers
    register_buffer("parents", parents_);
    register_buffer("lbs_weights", lbs_weights_);
    register_buffer("v_template", v_template_);
    register_buffer("J_regressor", J_regressor_);
    register_buffer("posedirs", posedirs_);

    // Register model parameters
    register_parameter("betas", vars_.betas.value().requires_grad_(true));
    register_parameter("global_orient",
                       vars_.global_orient.value().requires_grad_(true));
    register_parameter("body_pose",
                       vars_.body_pose.value().requires_grad_(true));
    register_parameter("transl", vars_.transl.value().requires_grad_(true));
}

auto SMPL::forward_impl() -> SMPLOutput {
    auto full_pose =
        torch::cat({vars_.global_orient.value(), vars_.body_pose.value()}, 1);

    auto batch_size =
        mmax(vars_.betas.value().size(0), vars_.global_orient.value().size(0),
             vars_.body_pose.value().size(0));

    // Ensure all tensors are of the same batch size
    if (vars_.betas.value().size(0) != batch_size) {
        vars_.betas = vars_.betas.value().expand(
            {int(batch_size / vars_.betas.value().size(0)), -1});
    }
    auto [vertices, joints] = lbs::lbs(
        vars_.betas.value(), full_pose, v_template_, shapedirs_, posedirs_,
        J_regressor_, parents_, lbs_weights_, vars_.pose2rot);

    vertices = vertices.to(device_);
    joints = joints.to(device_);
    joints = vertex_joint_selector_->forward(vertices, joints);

    std::cout << "Vertices shape: " << vertices.sizes() << std::endl;
    std::cout << "Joints shape: " << joints.sizes() << std::endl;

    // WTF ?
    // if (vars_.joint_mapper.has_value() && !vars_.joint_mapper.has_value()) {
    //     joints = vars_.joint_mapper.value()(joints);
    // }

    // if (!vars_.joint_mapper.has_value() && vars_.joint_mapper.has_value()) {
    //     joints = vars_.joint_mapper.value()(joints);
    // }

    joints += vars_.transl.value().unsqueeze(1);
    vertices += vars_.transl.value().unsqueeze(1);

    return {vars_.return_verts ? std::make_optional(vertices) : std::nullopt,
            joints,
            vars_.return_full_pose ? std::make_optional(full_pose)
                                   : std::nullopt,
            vars_.global_orient,
            vars_.betas,
            vars_.body_pose};
}

} // namespace smplx
