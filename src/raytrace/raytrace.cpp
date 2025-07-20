#include "raytrace.hpp"

using namespace torch;
void raytrace::raytrace3() {
    // Implementation of the ray tracing algorithm
    // This function will utilize the modeldata structure to perform computations
    // related to ray tracing in a neural network context.
}

// Function to filter rays around the target position within connection radius
// give up using norm to save time
// target_pos [1,3]
// input_pos [N,3]
// input_idx [N,1]
std::pair<torch::Tensor, torch::Tensor>
raytrace::filter_rays(const float con_rad, torch::Tensor &target_pos, torch::Tensor &input_pos, torch::Tensor &input_idx) {
    Tensor diff = input_pos - target_pos;
    torch::Tensor dist_squared = diff.pow(2).sum(1);
    torch::Tensor mask = dist_squared < con_rad * con_rad;
    Tensor res_pos = input_pos.index({mask});
    Tensor res_idx = input_idx.index({mask});
    return {res_pos, res_idx};
}