#include "smplx.hpp"
#include <memory>

#define DEBUG false

namespace smplx {

constexpr int SMPL::SHAPE_SPACE_DIM;

auto SMPL::construct(const char *model_path) -> void {
    vertex_joint_selector_ =
        std::make_unique<VertexJointSelector>(vars_.vertex_ids, device_);

    ASSERT_MSG(std::filesystem::exists(model_path), "%s not exist", model_path);
    ASSERT_MSG(check_file_ext(model_path, "npz"), "invalid extension %s",
               std::filesystem::path(model_path).extension().string().c_str());

    cnpy::npz_t data;
    try {
        data = cnpy::npz_load(model_path);
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Failed to load npz file: ") +
                                 e.what());
    }

    std::cout << "SMPL model loaded: " << model_path << std::endl;

    auto load_required_tensor = [&](const std::string &name,
                                    torch::Dtype dtype) -> torch::Tensor {
        if (!data.count(name)) {
            throw std::runtime_error("Missing tensor in npz: '" + name + "'");
        }
        try {
            return cnpyToTensor(data.at(name), dtype).to(device_);
        } catch (const std::exception &e) {
            throw std::runtime_error("Failed to load tensor '" + name +
                                     "': " + e.what());
        }
    };

    try {
        shapedirs_ = load_required_tensor("shapedirs", torch::kFloat64);
        auto num_betas = shapedirs_.size(2);

        faces_ = load_required_tensor("f", torch::kUInt32);

        if (vars_.age == "kid" && vars_.kid_template_path.has_value()) {
            ASSERT_MSG(std::filesystem::exists(vars_.kid_template_path.value()),
                       "Kid template path does not exist");

            cnpy::npz_t kid_data =
                cnpy::npz_load(vars_.kid_template_path.value());

            torch::Tensor v_template_adult =
                load_required_tensor("v_template", torch::kFloat64);
            torch::Tensor v_template_kid =
                cnpyToTensor(kid_data.at("v_template"), torch::kFloat64)
                    .to(device_);

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
            vars_.global_orient.emplace(
                torch::zeros({vars_.batch_size, 3},
                             torch::dtype(vars_.dtype).device(device_)));
        }
        if (!vars_.body_pose.has_value()) {
            vars_.body_pose.emplace(
                torch::zeros({vars_.batch_size, NUM_BODY_JOINTS * 3},
                             torch::dtype(vars_.dtype).device(device_)));
        }
        if (!vars_.transl.has_value()) {
            vars_.transl.emplace(
                torch::zeros({vars_.batch_size, 3},
                             torch::dtype(vars_.dtype).device(device_)));
        }

        v_template_ = load_required_tensor("v_template", torch::kFloat64);
        if (vars_.v_template.has_value())
            v_template_ = vars_.v_template.value().clone().to(device_);

        J_regressor_ = load_required_tensor("J_regressor", torch::kFloat64);

        posedirs_ = load_required_tensor("posedirs", torch::kFloat64);
        posedirs_ = posedirs_.reshape({-1, posedirs_.size(2)})
                        .transpose(0, 1)
                        .contiguous();

        lbs_weights_ = load_required_tensor("weights", torch::kFloat64);

        parents_ = torch::from_blob((void *)parents,
                                    {sizeof(parents) / sizeof(parents[0])},
                                    torch::kLong)
                       .clone()
                       .to(device_);

        // Register all buffers and parameters
        register_buffer("parents", parents_);
        register_buffer("lbs_weights", lbs_weights_);
        register_buffer("v_template", v_template_);
        register_buffer("J_regressor", J_regressor_);
        register_buffer("posedirs", posedirs_);

        register_parameter("betas", vars_.betas.value().requires_grad_(true));
        register_parameter("global_orient",
                           vars_.global_orient.value().requires_grad_(true));
        register_parameter("body_pose",
                           vars_.body_pose.value().requires_grad_(true));
        register_parameter("transl", vars_.transl.value().requires_grad_(true));

        // print first cell of each tensor
        if (DEBUG) {
            std::cout << "shapedirs: " << shapedirs_[0] << "\n";
            std::cout << "faces: " << faces_[0] << "\n";
            std::cout << "v_template: " << v_template_[0] << "\n";
            std::cout << "J_regressor: " << J_regressor_[0][0] << "\n";
            std::cout << "posedirs: " << posedirs_[0][0] << "\n";
            std::cout << "lbs_weights: " << lbs_weights_[0][0] << "\n";
        }

    } catch (const std::exception &e) {
        std::cerr << "[ERROR] Model construction failed: " << e.what()
                  << std::endl;
        throw; // rethrow for upstream handling
    }
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

    if (vars_.joint_mapper.has_value()) {
        joints = vars_.joint_mapper.value()(joints);
    }

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
