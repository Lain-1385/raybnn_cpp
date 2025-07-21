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

// Function to check if a line segment intersects with spheres defined by block_cells
// line_start [N,3]
// line_end [N,3]
// block_cells [M,3]
// Output: Tensor mask [M,N] where True indicates intersection
// line_start and line_end should diff, otherwise it will cause divided by zero exception
Tensor raytrace::line_sphere_intersect(const torch::Tensor &line_start,
                                       const torch::Tensor &line_end,
                                       const torch::Tensor &block_cells,
                                       const float block_radius) {
    Tensor line_dir = line_end - line_start;                                         // [N,3]
    Tensor line_start_to_block = block_cells.unsqueeze(1) - line_start.unsqueeze(0); // [M,1,3] - [1,N,3] -> [M,N,3]
    Tensor dot_product = (line_start_to_block * line_dir.unsqueeze(0)).sum(2);       // [M,N]
    Tensor line_dir_sq = (line_dir * line_dir).sum(1);                               // [N]
    Tensor projection_ratio = dot_product / line_dir_sq.unsqueeze(0);                // [M,N]
    projection_ratio.clamp_(0, 1);
    Tensor closest_point = line_start.unsqueeze(0) + projection_ratio.unsqueeze(2) * line_dir.unsqueeze(0); // [M,N,3]
    Tensor block_to_closest = closest_point - block_cells.unsqueeze(1);                                     // [M,N,3]
    Tensor dist_squared = (block_to_closest * block_to_closest).sum(2);
    Tensor mask = dist_squared <= block_radius * block_radius; // [M,N]
    return mask;
}