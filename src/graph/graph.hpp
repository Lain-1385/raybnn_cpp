#pragma once

#include <cstdint>
#include <torch/torch.h>

class RayBNNGraph {
private:
    torch::Tensor WRowIdx_;
    torch::Tensor WColIdx_;

public:
    torch::Tensor traverse_forward(torch::Tensor &idx_in, int64_t depth, int64_t neuron_size);
    torch::Tensor traverse_backward(torch::Tensor &idx_in, int64_t depth, int64_t neuron_size);
    torch::Tensor delete_loops();
    bool check_connected(const torch::Tensor &in_idx, const torch::Tensor &out_idx, int64_t neuron_size, int64_t depth);

    void delete_loops(const torch::Tensor &last_idx,
                      const torch::Tensor &first_idx,
                      int64_t neuron_size,
                      int64_t depth,
                      torch::Tensor &WValues,
                      torch::Tensor &WRowIdxCOO,
                      torch::Tensor &WColIdx);

    torch::Tensor get_global_weight_idx(int64_t neuron_size, const torch::Tensor &WRowIdxCOO, const torch::Tensor &WColIdx) {
        return WRowIdxCOO * neuron_size + WColIdx;
    }
};