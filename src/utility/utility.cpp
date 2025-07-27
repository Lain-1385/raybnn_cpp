#include "utility.hpp"

torch::Tensor make_tensor_from_buffer(size_t num_elements) {
    auto buffer = std::make_shared<GpuBuffer>(num_elements * sizeof(float));
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto tensor = torch::from_blob(
        buffer->data(),
        {static_cast<long>(num_elements)},
        [buffer](void *p) mutable {
            buffer.reset(); // Tensor释放 → GpuBuffer析构 → 显存释放
        },
        options);

    return tensor;
}