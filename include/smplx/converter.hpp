
#ifndef CNPY_CONVERTER_H
#define CNPY_CONVERTER_H
#include <cnpy.h>
#include <torch/torch.h>
#include <stdexcept>
#include <string>
#include <vector>

// Convert cnpy array to torch tensor with specified dtype
inline torch::Tensor cnpyToTensor(const cnpy::NpyArray &arr,
                                  torch::Dtype dtype) {
    std::vector<int64_t> shape(arr.shape.begin(), arr.shape.end());

    if (dtype == torch::kFloat32) {
        auto data_ptr = arr.data<float>();
        return torch::from_blob(const_cast<float *>(data_ptr), shape,
                                torch::TensorOptions().dtype(torch::kFloat32))
            .clone();
    } else if (dtype == torch::kFloat64) {
        auto data_ptr = arr.data<double>();
        return torch::from_blob(const_cast<double *>(data_ptr), shape,
                                torch::TensorOptions().dtype(torch::kFloat64))
            .clone();
    } else if (dtype == torch::kInt32) {
        auto data_ptr = arr.data<int32_t>();
        return torch::from_blob(const_cast<int32_t *>(data_ptr), shape,
                                torch::TensorOptions().dtype(torch::kInt32))
            .clone();
    } else if (dtype == torch::kInt64) {
        auto data_ptr = arr.data<int64_t>();
        return torch::from_blob(const_cast<int64_t *>(data_ptr), shape,
                                torch::TensorOptions().dtype(torch::kInt64))
            .clone();
    } else if (dtype == torch::kUInt32) {
        auto data_ptr = arr.data<uint32_t>();
        return torch::from_blob(const_cast<uint32_t *>(data_ptr), shape,
                                torch::TensorOptions().dtype(torch::kUInt32))
            .clone();
    } else {
        throw std::runtime_error("Unsupported dtype in cnpyToTensor.");
    }
}

#endif // CNPY_CONVERTER_H