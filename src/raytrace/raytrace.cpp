#include "raytrace.hpp"
#include <cstddef>
#include <cstdint>

constexpr int64_t PRUNE_COUNT_LIMIT = 10000000;
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

// Function to create rays from neurons A to neurons B

// pos_A [N,3]
// pos_B [M,3]
torch::Tensor raytrace::rays_from_neuronsA_to_neuronsB(const float con_rad, const torch::Tensor &pos_A, const torch::Tensor &pos_B) {
    Tensor diff = pos_B.unsqueeze(1) - pos_A.unsqueeze(0); //[M,N,3]
    Tensor dist_squared = diff.pow(2).sum(2);              // [M,N]
    Tensor mask = dist_squared < con_rad * con_rad;        // [M,N]
    return mask.nonzero();
}

// Function to check if a line segment intersects with spheres defined by block_cells
// use projection method to find the closest point on the line segment to each sphere
// line_start [N,3]
// line_end [N,3]
// block_cells [M,3]
// block_radius [M]
// Output: Tensor mask [M,N] where True indicates intersection
// WARNING: line_start and line_end should diff, otherwise it will cause divided by zero exception
Tensor raytrace::line_sphere_intersect(const torch::Tensor &line_start,
                                       const torch::Tensor &line_end,
                                       const torch::Tensor &block_cells,
                                       const torch::Tensor &block_radius) {
    Tensor line_dir = line_end - line_start;                                         // [N,3]
    Tensor line_start_to_block = block_cells.unsqueeze(1) - line_start.unsqueeze(0); // [M,1,3] - [1,N,3] -> [M,N,3]
    Tensor dot_product = (line_start_to_block * line_dir.unsqueeze(0)).sum(2);       // [M,N]
    Tensor line_dir_sq = (line_dir * line_dir).sum(1);                               // [N]
    Tensor projection_ratio = dot_product / line_dir_sq.unsqueeze(0);                // [M,N]
    projection_ratio.clamp_(0, 1);
    Tensor closest_point = line_start.unsqueeze(0) + projection_ratio.unsqueeze(2) * line_dir.unsqueeze(0); // [M,N,3]
    Tensor block_to_closest = closest_point - block_cells.unsqueeze(1);                                     // [M,N,3]
    Tensor dist_squared = (block_to_closest * block_to_closest).sum(2);
    Tensor mask = dist_squared <= block_radius.pow(2).unsqueeze(1); // [M,N]
    return mask;
}

// Function to perform batch processing of line-sphere intersection
// HIGHLIGHT:   Batch processing avoids memory overflow;
//              Adaptive pruning improves computational efficiency;
//              Threshold-based filtering enables fine-grained control over ray selection;
// block_cells [M,3] it should be the hidden neurons or glia cells;
// block_radius [M]
// line_start [N,3] it should be the start point of the rays, normaly we have Na*Nb rays(from neurons set A to neurons set B)
// line_end [N,3];index_start [N];index_end [N]
void raytrace::line_sphere_intersect_batch(const int64_t batch_size,
                                           const int64_t max_allowed_hits, // it can be zero
                                           const torch::Tensor &block_cells,
                                           const torch::Tensor &block_radius,
                                           torch::Tensor &line_start,
                                           torch::Tensor &line_end,
                                           torch::Tensor &index_start,
                                           torch::Tensor &index_end) {
    int64_t num_block_cells = block_cells.size(0);
    size_t prune_period = -1;
    size_t prune_count = 0;
    Tensor hits = torch::zeros_like(index_start, torch::dtype(torch::kLong));
    for (int64_t i = 0; i < num_block_cells; i += batch_size) {
        int64_t end = std::min(i + batch_size, num_block_cells);
        Tensor batch_cells = block_cells.slice(0, i, end);
        Tensor batch_radius = block_radius.slice(0, i, end);
        Tensor mask_intersect = line_sphere_intersect(line_start, line_end, batch_cells, batch_radius); // [M',N]

        if (prune_period == -1) {
            prune_period = mask_intersect.numel() > 0 ? PRUNE_COUNT_LIMIT / mask_intersect.numel() : PRUNE_COUNT_LIMIT;
        }

        hits = hits + mask_intersect.sum(0); // [N]
        std::cout << "hits.shape:" << hits.sizes() << std::endl;

        prune_count += 1;
        if (prune_count > prune_period && end < num_block_cells) {
            Tensor valid_hits = hits <= max_allowed_hits; // [N] kBool
            // early prune the rays based on accumulated hit count
            line_start = line_start.index({valid_hits});
            line_end = line_end.index({valid_hits});
            index_start = index_start.index({valid_hits});
            index_end = index_end.index({valid_hits});
            hits = hits.index({valid_hits});
            prune_count = 0;
            prune_period = -1; // reset prune period
        }
    }
    Tensor valid_hits = hits <= max_allowed_hits; // [N] kBool
    line_start = line_start.index({valid_hits});
    line_end = line_end.index({valid_hits});
    index_start = index_start.index({valid_hits});
    index_end = index_end.index({valid_hits});
}

// WRowIdx [N]
// WColIdx [N]
inline Tensor dedup_and_sort(torch::Tensor WRowIdx, torch::Tensor WColIdx) {
    torch::Tensor indices = torch::stack({WRowIdx, WColIdx}, 1);
    // WARNING:my libtorch has no unique function, so we use torch::_unique
    auto [unique_indices, _] = torch::_unique(indices, true, false);
    return unique_indices;
}
