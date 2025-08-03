
#ifndef CNPY_CONVERTER_H
#define CNPY_CONVERTER_H
#include <cnpy.h>
#include <torch/torch.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T>
torch::Tensor convertTensor(const cnpy::NpyArray &arr, torch::Dtype dtype) {
    const T *data_ptr = arr.data<T>();
    const auto &shape = arr.shape;

    if (shape.size() < 2 || shape.size() > 3) {
        throw std::runtime_error("Only 2D or 3D arrays are supported.");
    }

    std::vector<T> flat_data;
    flat_data.reserve(arr.num_vals);

    if (shape.size() == 2) {
        size_t nrows = shape[0];
        size_t ncols = shape[1];

        if (arr.fortran_order) {
            for (size_t i = 0; i < nrows; ++i) {
                for (size_t j = 0; j < ncols; ++j) {
                    flat_data.push_back(data_ptr[j * nrows + i]);
                }
            }
        } else {
            for (size_t i = 0; i < nrows; ++i) {
                for (size_t j = 0; j < ncols; ++j) {
                    flat_data.push_back(data_ptr[i * ncols + j]);
                }
            }
        }
        return torch::from_blob(
                   flat_data.data(),
                   {static_cast<int64_t>(nrows), static_cast<int64_t>(ncols)},
                   torch::TensorOptions().dtype(dtype))
            .clone();
    } else {
        size_t dim0 = shape[0];
        size_t dim1 = shape[1];
        size_t dim2 = shape[2];

        if (arr.fortran_order) {
            for (size_t i = 0; i < dim0; ++i) {
                for (size_t j = 0; j < dim1; ++j) {
                    for (size_t k = 0; k < dim2; ++k) {
                        flat_data.push_back(
                            data_ptr[k * dim1 * dim0 + j * dim0 + i]);
                    }
                }
            }
        } else {
            for (size_t i = 0; i < dim0; ++i) {
                for (size_t j = 0; j < dim1; ++j) {
                    for (size_t k = 0; k < dim2; ++k) {
                        flat_data.push_back(
                            data_ptr[i * dim1 * dim2 + j * dim2 + k]);
                    }
                }
            }
        }
        return torch::from_blob(flat_data.data(),
                                {static_cast<int64_t>(dim0),
                                 static_cast<int64_t>(dim1),
                                 static_cast<int64_t>(dim2)},
                                torch::TensorOptions().dtype(dtype))
            .clone();
    }
}

inline torch::Tensor cnpyToTensor(const cnpy::NpyArray &arr,
                                  torch::Dtype dtype) {
    switch (dtype) {
    case torch::kFloat32:
        return convertTensor<float>(arr, dtype);
    case torch::kFloat64:
        return convertTensor<double>(arr, dtype);
    case torch::kInt32:
        return convertTensor<int32_t>(arr, dtype);
    case torch::kInt64:
        return convertTensor<int64_t>(arr, dtype);
    case torch::kUInt32:
        return convertTensor<uint32_t>(arr, dtype);
    default:
        throw std::runtime_error("Unsupported dtype in cnpyToTensor.");
    }
}

inline void dump_tensor(const torch::Tensor &tensor, const std::string &name) {
    std::ofstream ofs(name + ".txt");
    for (int64_t i = 0; i < tensor.size(0); ++i) {
        for (int64_t j = 0; j < tensor.size(1); ++j) {
            ofs << name << "[" << i << ", " << j
                << "]: " << tensor.index({i, j}).item<double>() << "\n";
        }
    }
    ofs.close();
}
#endif // CNPY_CONVERTER_H