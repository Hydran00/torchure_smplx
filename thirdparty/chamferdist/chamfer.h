#pragma once
#include <torch/torch.h>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include "knn.h"

// Struct to hold KNN results (like namedtuple _KNN)
struct KNNResult {
    at::Tensor dists; // (N, P1, K)
    at::Tensor idx;   // (N, P1, K)
    at::Tensor knn;   // (N, P1, K, D), optional
};

class KNNPointsFunction : public torch::autograd::Function<KNNPointsFunction> {
  public:
    static torch::Tensor mask_invalid_dists(torch::Tensor dists,
                                            torch::Tensor lengths2, int64_t K,
                                            int64_t P1) {
        auto device = dists.device();
        auto N = dists.size(0);
        torch::Tensor arangeK = torch::arange(K, device);
        torch::Tensor mask = lengths2.view({N, 1}) <= arangeK.view({1, K});
        mask = mask.view({N, 1, K}).expand({N, P1, K});
        dists.masked_fill_(mask, INFINITY);
        auto sort_result = dists.sort(2);
        dists = std::get<0>(sort_result);
        torch::Tensor sort_idx = std::get<1>(sort_result);
        dists.masked_fill_(mask, 0);
        return sort_idx;
    }

    static torch::autograd::tensor_list
    forward(torch::autograd::AutogradContext *ctx, torch::Tensor p1,
            torch::Tensor p2, torch::Tensor lengths1, torch::Tensor lengths2,
            int64_t K, int64_t version, bool return_sorted = true) {
        // Compute KNN indices and distances using custom CUDA/C++ backend
        // NOTE: You should implement this function in your backend (e.g.,
        // knn_points_idx)
        auto knn_result =
            KNearestNeighborIdx(p1, p2, lengths1, lengths2, K, version);
        torch::Tensor idx = std::get<0>(knn_result);
        torch::Tensor dists = std::get<1>(knn_result);

        // Sort by distances if required
        if (K > 1 && return_sorted) {
            if (lengths2.min().item<int64_t>() < K) {
                int64_t P1 = p1.size(1);
                auto sort_idx = mask_invalid_dists(dists, lengths2, K, P1);
                idx = idx.gather(2, sort_idx);
            } else {
                auto sort_result = dists.sort(2);
                dists = std::get<0>(sort_result);
                auto sort_idx = std::get<1>(sort_result);
                idx = idx.gather(2, sort_idx);
            }
        }

        ctx->save_for_backward({p1, p2, lengths1, lengths2, idx});
        ctx->mark_non_differentiable({idx});

        return {dists, idx};
    }

    static torch::autograd::tensor_list
    backward(torch::autograd::AutogradContext *ctx,
             torch::autograd::tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        torch::Tensor p1 = saved[0];
        torch::Tensor p2 = saved[1];
        torch::Tensor lengths1 = saved[2];
        torch::Tensor lengths2 = saved[3];
        torch::Tensor idx = saved[4];

        torch::Tensor grad_dists = grad_outputs[0];
        if (grad_dists.dtype() != torch::kFloat32)
            grad_dists = grad_dists.to(torch::kFloat32);
        if (p1.dtype() != torch::kFloat32)
            p1 = p1.to(torch::kFloat32);
        if (p2.dtype() != torch::kFloat32)
            p2 = p2.to(torch::kFloat32);

        // NOTE: Implement this function in your backend
        auto grads = KNearestNeighborBackward(p1, p2, lengths1, lengths2, idx,
                                              grad_dists);
        torch::Tensor grad_p1 = std::get<0>(grads);
        torch::Tensor grad_p2 = std::get<1>(grads);

        return {
            grad_p1,         // ∂L/∂p1
            grad_p2,         // ∂L/∂p2
            torch::Tensor(), // None for lengths1
            torch::Tensor(), // None for lengths2
            torch::Tensor(), // None for K
            torch::Tensor(), // None for version
            torch::Tensor()  // None for return_sorted
        };
    }
};
// Forward declaration of KNN core function (implemented elsewhere, e.g.
// knn_cpu.cpp / knn.cu)
// Helper for gather operation for KNN neighbors
// knn_gather implementation
torch::Tensor knn_gather(const torch::Tensor &x, const torch::Tensor &idx,
                         const torch::Tensor &lengths) {

    auto x_sizes = x.sizes();
    auto idx_sizes = idx.sizes();

    int64_t N = x_sizes[0];
    int64_t M = x_sizes[1];
    int64_t U = x_sizes[2];

    int64_t _N = idx_sizes[0];
    int64_t L = idx_sizes[1];
    int64_t K = idx_sizes[2];

    if (N != _N) {
        throw std::runtime_error(
            "x and idx must have the same batch dimension.");
    }

    torch::Tensor lengths_ = lengths;
    if (!lengths_.defined()) {
        lengths_ = torch::full(
            {N}, M,
            torch::TensorOptions().dtype(torch::kInt64).device(x.device()));
    }

    // Expand idx for gathering along last dimension U
    auto idx_expanded = idx.unsqueeze(3).expand({N, L, K, U}); // [N, L, K, U]

    // Expand x for broadcasting
    auto x_expanded = x.unsqueeze(2).expand({N, M, K, U}); // [N, M, K, U]

    // Gather along dimension 1 (M)
    auto x_out = x_expanded.gather(1, idx_expanded); // [N, L, K, U]

    // Masking: if lengths.min() < K, zero out invalid indices
    if (lengths_.min().item<int64_t>() < K) {
        auto device = x.device();
        auto arange_k = torch::arange(K, device = device);
        auto mask = lengths_.unsqueeze(1) <= arange_k.unsqueeze(0); // [N, K]

        mask = mask.unsqueeze(1).expand({N, L, K});    // [N, L, K]
        mask = mask.unsqueeze(3).expand({N, L, K, U}); // [N, L, K, U]

        x_out.masked_fill_(mask, 0.0);
    }

    return x_out;
}
KNNResult knn_points(const torch::Tensor &p1, const torch::Tensor &p2,
                     torch::optional<torch::Tensor> lengths1 = torch::nullopt,
                     torch::optional<torch::Tensor> lengths2 = torch::nullopt,
                     int64_t K = 1, int64_t version = -1,
                     bool return_nn = false, bool return_sorted = true) {
    // Check batch and point dimension consistency
    if (p1.size(0) != p2.size(0)) {
        TORCH_CHECK(false, "p1 and p2 must have the same batch size");
    }
    if (p1.size(2) != p2.size(2)) {
        TORCH_CHECK(false, "p1 and p2 must have the same point dimensionality");
    }

    torch::Tensor p1_contig = p1.contiguous();
    torch::Tensor p2_contig = p2.contiguous();

    int64_t P1 = p1_contig.size(1);
    int64_t P2 = p2_contig.size(1);

    torch::Device device = p1.device();

    // Set lengths to full size if not provided
    if (!lengths1.has_value()) {
        lengths1 = torch::full(
            {p1.size(0)}, P1,
            torch::TensorOptions().dtype(torch::kInt64).device(device));
    }
    if (!lengths2.has_value()) {
        lengths2 = torch::full(
            {p1.size(0)}, P2,
            torch::TensorOptions().dtype(torch::kInt64).device(device));
    }

    // Call the custom autograd function
    auto outputs =
        KNNPointsFunction::apply(p1_contig, p2_contig, lengths1.value(),
                                 lengths2.value(), K, version, return_sorted);
    torch::Tensor p1_dists = outputs[0];
    torch::Tensor p1_idx = outputs[1];

    torch::optional<torch::Tensor> p2_nn = torch::nullopt;
    if (return_nn) {
        p2_nn = knn_gather(p2_contig, p1_idx, lengths2.value());
    }

    return KNNResult{p1_dists, p1_idx, p2_nn.value_or(torch::Tensor())};
}

// ChamferDistance class like PyTorch nn.Module
class ChamferDistance : public torch::nn::Module {
  public:
    ChamferDistance() {}

    at::Tensor forward(const at::Tensor &source_cloud,
                       const at::Tensor &target_cloud,
                       bool bidirectional = false, bool reverse = false,
                       const std::string &batch_reduction = "mean",
                       const std::string &point_reduction = "sum") {
        if (!source_cloud.is_cuda() == target_cloud.is_cuda()) {
            // you can adjust for cpu/cuda if needed
        }
        if (source_cloud.device() != target_cloud.device()) {
            throw std::runtime_error(
                "Source and target clouds must be on the same device.");
        }
        if (source_cloud.dim() != 3 || target_cloud.dim() != 3) {
            throw std::runtime_error(
                "Input point clouds must be 3D tensors (N, P, D).");
        }
        auto batchsize_source = source_cloud.size(0);
        auto lengths_source = source_cloud.size(1);
        auto dim_source = source_cloud.size(2);

        auto batchsize_target = target_cloud.size(0);
        auto lengths_target = target_cloud.size(1);
        auto dim_target = target_cloud.size(2);

        if (batchsize_source != batchsize_target) {
            throw std::runtime_error(
                "Source and target pointclouds must have the same batchsize.");
        }
        if (dim_source != dim_target) {
            throw std::runtime_error("Source and target pointclouds must have "
                                     "the same dimensionality.");
        }
        if (bidirectional && reverse) {
            std::cerr << "Warning: Both bidirectional and reverse set to true; "
                         "bidirectional takes precedence.\n";
        }
        if (point_reduction != "sum" && point_reduction != "mean" &&
            point_reduction != "none") {
            throw std::runtime_error(
                "point_reduction must be 'sum', 'mean' or 'none'");
        }
        if (batch_reduction != "sum" && batch_reduction != "mean" &&
            batch_reduction != "none") {
            throw std::runtime_error(
                "batch_reduction must be 'sum', 'mean' or 'none'");
        }

        auto device = source_cloud.device();
        // Length tensors (full lengths)
        at::Tensor lengths_src =
            torch::full({batchsize_source}, lengths_source,
                        torch::dtype(torch::kLong).device(device));
        at::Tensor lengths_tgt =
            torch::full({batchsize_target}, lengths_target,
                        torch::dtype(torch::kLong).device(device));

        // Forward KNN (source -> target)
        KNNResult source_nn =
            knn_points(source_cloud, target_cloud, lengths_src, lengths_tgt, 1);

        // Reverse KNN (target -> source) if needed
        c10::optional<KNNResult> target_nn = c10::nullopt;
        if (reverse || bidirectional) {
            target_nn = knn_points(target_cloud, source_cloud, lengths_tgt,
                                   lengths_src, 1);
        }

        // chamfer distances (N, P)
        at::Tensor chamfer_forward = source_nn.dists.select(-1, 0);
        at::Tensor chamfer_backward;
        if (reverse || bidirectional) {
            chamfer_backward = target_nn->dists.select(-1, 0);
        }

        // Point reduction
        if (point_reduction == "sum") {
            chamfer_forward = chamfer_forward.sum(1);
            if (reverse || bidirectional) {
                chamfer_backward = chamfer_backward.sum(1);
            }
        } else if (point_reduction == "mean") {
            chamfer_forward = chamfer_forward.mean(1);
            if (reverse || bidirectional) {
                chamfer_backward = chamfer_backward.mean(1);
            }
        } // else no reduction at point level (keep per-point)

        // Batch reduction
        if (batch_reduction == "sum") {
            chamfer_forward = chamfer_forward.sum();
            if (reverse || bidirectional) {
                chamfer_backward = chamfer_backward.sum();
            }
        } else if (batch_reduction == "mean") {
            chamfer_forward = chamfer_forward.mean();
            if (reverse || bidirectional) {
                chamfer_backward = chamfer_backward.mean();
            }
        } // else no reduction at batch level

        if (bidirectional) {
            return chamfer_forward + chamfer_backward;
        } else if (reverse) {
            return chamfer_backward;
        } else {
            return chamfer_forward;
        }
    }
};
