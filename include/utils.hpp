#ifndef SMPLX_UTILS_HPP
#define SMPLX_UTILS_HPP
#include <filesystem>
#include <optional>
#include "common.hpp"

namespace smplx {

inline auto check_file_ext(const char *path, const char *extension) -> bool {
    std::string fileExt = std::filesystem::path(path).extension().string();
    std::string ext(extension);

    // trim front .
    if (!fileExt.empty() && fileExt.front() == '.') {
        fileExt.erase(fileExt.begin());
    }

    return std::equal(
        fileExt.begin(), fileExt.end(), ext.begin(), ext.end(),
        [](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

struct ModelOutput {};

struct SMPLOutput {
    std::optional<Tensor> vertices;
    std::optional<Tensor> joints;
    std::optional<Tensor> full_pose;
    std::optional<Tensor> global_orient;
    std::optional<Tensor> betas;
    std::optional<Tensor> body_pose;
    std::optional<Tensor> transl;
    std::optional<Tensor> v_shaped;
};
} // namespace smplx
// Utility: load torch::Tensor from nested std::vectors
namespace utils {
template <typename T>
torch::Tensor load_tensor(const std::vector<std::vector<T>> &data,
                          torch::ScalarType dtype,
                          const torch::Device &device) {
    size_t rows = data.size();
    size_t cols = data[0].size();
    std::vector<T> flat;
    for (const auto &row : data)
        flat.insert(flat.end(), row.begin(), row.end());
    return torch::from_blob(
               flat.data(),
               {static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
               torch::dtype(dtype))
        .clone()
        .to(device);
}

template <typename T>
torch::Tensor load_tensor(const std::vector<std::vector<std::vector<T>>> &data,
                          torch::ScalarType dtype,
                          const torch::Device &device) {
    auto d0 = data.size();
    auto d1 = data[0].size();
    auto d2 = data[0][0].size();
    std::vector<T> flat;
    for (const auto &mat : data)
        for (const auto &row : mat)
            flat.insert(flat.end(), row.begin(), row.end());
    return torch::from_blob(flat.data(),
                            {static_cast<int64_t>(d0), static_cast<int64_t>(d1),
                             static_cast<int64_t>(d2)},
                            torch::dtype(dtype))
        .clone()
        .to(device);
}
} // namespace utils
#endif