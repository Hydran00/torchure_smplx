#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "smplx.hpp"

struct Expected {
    std::vector<int64_t> shape;
    torch::Dtype dtype;
};

bool shapeMatches(const torch::Tensor &tensor,
                  const std::vector<int64_t> &expectedShape) {
    auto actual = tensor.sizes();
    if (actual.size() != expectedShape.size())
        return false;
    for (size_t i = 0; i < expectedShape.size(); ++i) {
        if (actual[i] != expectedShape[i])
            return false;
    }
    return true;
}

int main() {
    std::string filename = "smpl_male.npz";
    cnpy::npz_t npz_file = cnpy::npz_load(filename);
    std::unordered_map<std::string, torch::Tensor> tensors;

    std::unordered_map<std::string, Expected> expected = {
        {"J_regressor_prior", {{24, 6890}, torch::kFloat64}},
        {"J", {{24, 3}, torch::kFloat64}},
        {"J_regressor", {{24, 6890}, torch::kFloat64}},
        {"weights_prior", {{6890, 24}, torch::kFloat64}},
        {"weights", {{6890, 24}, torch::kFloat64}},
        {"vert_sym_idxs", {{6890}, torch::kInt64}},
        {"posedirs", {{6890, 3, 207}, torch::kFloat64}},
        {"v_template", {{6890, 3}, torch::kFloat64}},
        {"shapedirs", {{6890, 3, 10}, torch::kFloat64}},
        {"f", {{13776, 3}, torch::kUInt32}},
        {"kintree_table", {{2, 24}, torch::kUInt32}},
    };

    std::cout << "Testing tensors from '" << filename << "':\n\n";

    for (const auto &pair : npz_file) {
        const std::string &name = pair.first;
        const cnpy::NpyArray &arr = pair.second;

        if (!expected.count(name)) {
            std::cerr << "[WARNING] Unknown variable '" << name
                      << "' — skipping.\n";
            continue;
        }

        torch::Dtype dtype = expected[name].dtype;
        torch::Tensor tensor;

        try {
            tensor = cnpyToTensor(arr, dtype);
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] Could not convert '" << name
                      << "': " << e.what() << "\n";
            continue;
        }

        tensors[name] = tensor;

        std::cout << "- " << name << "\n";
        std::cout << "  dtype: " << tensor.dtype() << "\n";
        std::cout << "  shape: ";
        for (auto s : tensor.sizes())
            std::cout << s << " ";
        std::cout << "\n";

        if (!shapeMatches(tensor, expected[name].shape)) {
            std::cerr << "  ❌ Shape mismatch for '" << name << "' — expected: ";
            for (auto s : expected[name].shape)
                std::cout << s << " ";
            std::cout << "\n";
        } else {
            std::cout << "  ✅ Shape OK\n";
        }

        std::cout << std::endl;
    }

    return 0;
}
