#pragma once

#include <cstdint>
#include <torch/torch.h>

class RayBNNGraph {
private:
    torch::Tensor WRowIdx_;
    torch::Tensor WColIdx_;

public:
    torch::Tensor traverse_forward(torch::Tensor idx_in, int64_t depth);
    torch::Tensor traverse_backward(torch::Tensor &idx_in, int64_t depth);
    torch::Tensor delete_loops();
};