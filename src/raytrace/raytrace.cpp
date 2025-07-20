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
// Output res_pos [M,3] and res_idx [M,1] where M is the number of rays that pass the filter
std::pair<torch::Tensor, torch::Tensor>
raytrace::filter_rays(const float con_rad, torch::Tensor &target_pos, torch::Tensor &input_pos, torch::Tensor &input_idx) {
    Tensor diff = input_pos - target_pos;
    torch::Tensor dist_squared = diff.pow(2).sum(1);
    torch::Tensor mask = dist_squared < con_rad * con_rad;
    Tensor res_pos = input_pos.index({mask});
    Tensor res_idx = input_idx.index({mask});
    return {res_pos, res_idx};
}

// pos_A [N,3]
// pos_B [M,3]
torch::Tensor raytrace::rays_from_neuronsA_to_neuronsB(const float con_rad, const torch::Tensor &pos_A, const torch::Tensor &pos_B) {
    Tensor diff = pos_B.unsqueeze(1) - pos_A.unsqueeze(0); //[M,N,3]
    Tensor dist_squared = diff.pow(2).sum(2);              // [M,N]
    Tensor mask = dist_squared < con_rad * con_rad;        // [M,N]
    return mask.nonzero();
}
