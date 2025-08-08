#include <ATen/core/TensorBody.h>
#include <torch/torch.h>

class sparse {
public:
    static torch::Tensor COO_find(const torch::Tensor &WRowIdxCOO, const torch::Tensor &target_rows);
    static torch::Tensor COO_find_batch(const torch::Tensor &WRowIdxCOO, const torch::Tensor &target_rows, int64_t batch_size);
    static torch::Tensor find_unique(const torch::Tensor &arr, int64_t neuron_size);
};
