#include "raytrace.hpp"
#include <ATen/core/TensorBody.h>
#include <cstddef>
#include <cstdint>
#include <random>

constexpr int64_t PRUNE_COUNT_LIMIT = 10000000;
constexpr int64_t RAYTRACE_LIMIT = 10000000;
constexpr int64_t MAX_ALLOWED_HITS_NEURON = 2;
constexpr int64_t MAX_ALLOWED_HITS_GLIA = 0;
constexpr int64_t MAX_SAME_COUNTER = 5;

using namespace torch;

// Function to filter rays around the target position within connection radius
// give up using norm to save time
// target_pos [1,3]
// input_pos [N,3]
// input_idx [N,1]
// Output res_pos [M,3] and res_idx [M,1] where M is the number of rays that pass the filter
std::pair<torch::Tensor, torch::Tensor>
raytrace::filter_rays(const float con_rad, const torch::Tensor &target_pos, const torch::Tensor &input_pos, const torch::Tensor &input_idx) {
    Tensor diff = input_pos - target_pos;
    Tensor dist_squared = diff.pow(2).sum(1);
    Tensor mask = dist_squared < con_rad * con_rad;
    Tensor res_pos = input_pos.index({mask});
    Tensor res_idx = input_idx.index({mask});
    return {res_pos, res_idx};
}

// Function to create rays from neurons A set to neurons B set
// pos_A [N,3]
// pos_B [M,3]
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> raytrace::rays_from_neuronsA_to_neuronsB(const float con_rad,
                                                                                                                const torch::Tensor &pos_A,
                                                                                                                const torch::Tensor &pos_B,
                                                                                                                const torch::Tensor &idx_A,
                                                                                                                const torch::Tensor &idx_B) {
    int64_t N = pos_A.size(0);
    int64_t M = pos_B.size(0);

    Tensor diff = pos_A.unsqueeze(1) - pos_B.unsqueeze(0); // [N,M,3]
    Tensor dist_squared = diff.pow(2).sum(2);              // [N,M]

    Tensor mask = dist_squared < con_rad * con_rad; // [N,M]

    Tensor idx_pairs = torch::nonzero(mask); // [K, 2]

    Tensor tiled_pos_A = pos_A.index_select(0, idx_pairs.select(1, 0));
    Tensor tiled_pos_B = pos_B.index_select(0, idx_pairs.select(1, 1));
    Tensor tiled_idx_A = idx_A.index_select(0, idx_pairs.select(1, 0));
    Tensor tiled_idx_B = idx_B.index_select(0, idx_pairs.select(1, 1));

    return {tiled_pos_A, tiled_pos_B, tiled_idx_A, tiled_idx_B};
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

// Deduplicate and sort (WRowIdx, WColIdx) pairs using hashing.
// Highlights:
// - Faster than lexicographic row-wise unique (torch::unique_dim) on [WRowIdx, WColIdx].
// - Dynamically computes minimal max_col to reduce hash range and sorting cost.
// WRowIdx [N]
// WColIdx [N]
inline void dedup_and_sort(torch::Tensor &WRowIdx, torch::Tensor &WColIdx) {
    // here we use a valid max_col which is as small as possible to reduce overhead in torch::_unique radix/bucket sort
    // unnecessary to use power of 2 as compiler is smart enough
    int64_t max_col = WColIdx.max().item<int64_t>() + 1;
    Tensor hash = WRowIdx * max_col + WColIdx;
    auto [unique_hash, _] = torch::_unique(hash, true, false);
    std::cout << "unique_hash dtype: " << unique_hash.dtype() << std::endl;
    WRowIdx = torch::div(unique_hash, max_col, "floor");
    std::cout << "WRowIdx dtype: " << WRowIdx.dtype() << std::endl;
    WColIdx = unique_hash % max_col;
}

// Function to perform ray tracing with distance limitation in batch
// This function traces rays from sender neurons to hidden neurons, applying distance limits
std::tuple<torch::Tensor, torch::Tensor>
raytrace::raytrace_distance_limited(const modeldata &model_info,
                                    const torch::Tensor &glia_pos,
                                    const torch::Tensor &sender_pos, // the neuron sending rays, it can be hidden neuron itself[Na,3]
                                    const torch::Tensor &hidden_pos,
                                    const std::optional<torch::Tensor> &prev_WRowIdx,
                                    const std::optional<torch::Tensor> &prev_WColIdx) {

    float con_rad = model_info.con_rad;
    size_t max_rounds = model_info.ray_max_rounds;
    bool ray_neuron_intersect = model_info.ray_neuron_intersect;

    Tensor sender_idx = torch::arange(sender_pos.size(0), torch::TensorOptions().dtype(torch::kLong).device(sender_pos.device())); // 1D
    Tensor hidden_idx = torch::arange(hidden_pos.size(0), torch::TensorOptions().dtype(torch::kLong).device(hidden_pos.device())); // 1D

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, sender_pos.size(0) - 1);
    Tensor WRowIdx = torch::empty({0}, torch::TensorOptions().dtype(torch::kLong).device(sender_pos.device()));
    Tensor WColIdx = torch::empty({0}, torch::TensorOptions().dtype(torch::kLong).device(sender_pos.device()));

    int64_t round_prev_con_num = 0;
    size_t same_counter = 0;
    for (size_t round = 0; round < max_rounds; ++round) {
        int64_t random_index = dis(gen);
        Tensor cur_batch_center = sender_pos.slice(0, random_index, random_index + 1); // [1,3]
        auto [cur_sender_pos, cur_sender_idx] = filter_rays(2.0f * con_rad, cur_batch_center, sender_pos, sender_idx);
        if (cur_sender_pos.size(0) == 0)
            continue;

        auto [cur_hidden_pos, cur_hidden_idx] = filter_rays(con_rad, cur_batch_center, hidden_pos, hidden_idx);
        if (cur_hidden_pos.size(0) == 0)
            continue;
        // now we have cur_sender_pos and cur_hidden_pos, cur_sender_idx and cur_hidden_idx, we can compute the rays
        // here tiled_sender_pos, tiled_hidden_pos are the start and end of rays, one to one correspondence
        auto [tiled_sender_pos, tiled_hidden_pos, tiled_sender_idx, tiled_hidden_idx] =
            rays_from_neuronsA_to_neuronsB(con_rad, cur_sender_pos, cur_hidden_pos, cur_sender_idx, cur_hidden_idx);

        // release memory
        cur_sender_pos.reset();
        cur_hidden_pos.reset();
        cur_sender_idx.reset();
        cur_hidden_idx.reset();
        if (tiled_sender_idx.size(0) == 0 || tiled_hidden_idx.size(0) == 0) {
            continue; // no rays found
        }

        if (ray_neuron_intersect) {
            int64_t raytrace_batch_size = 1 + RAYTRACE_LIMIT / tiled_sender_pos.size(0);
            Tensor hidden_radius =
                torch::full({hidden_pos.size(0)}, model_info.neuron_rad, torch::TensorOptions().dtype(torch::kFloat).device(hidden_pos.device()));

            line_sphere_intersect_batch(raytrace_batch_size,
                                        MAX_ALLOWED_HITS_NEURON,
                                        hidden_pos,
                                        hidden_radius,
                                        tiled_sender_pos,
                                        tiled_hidden_pos,
                                        tiled_sender_idx,
                                        tiled_hidden_idx);
        }
        if (tiled_hidden_idx.size(0) == 0 || tiled_sender_idx.size(0) == 0) {
            continue; // no rays found after intersection
        }

        // glial cells intersection
        Tensor glia_radius =
            torch::full({glia_pos.size(0)},
                        model_info.neuron_rad,
                        torch::TensorOptions().dtype(torch::kFloat).device(glia_pos.device())); // glia radius should be the same as neuron radius
        int64_t raytrace_batch_size = 1 + RAYTRACE_LIMIT / tiled_sender_pos.size(0);
        line_sphere_intersect_batch(raytrace_batch_size,
                                    MAX_ALLOWED_HITS_GLIA,
                                    glia_pos,
                                    glia_radius,
                                    tiled_sender_pos,
                                    tiled_hidden_pos,
                                    tiled_sender_idx,
                                    tiled_hidden_idx);
        if (tiled_hidden_idx.size(0) == 0 || tiled_sender_idx.size(0) == 0) {
            continue; // no rays found after glia intersection
        }
        WColIdx = torch::cat({WColIdx, tiled_sender_idx}, 0); // WColIdx is the sender neuron index
        WRowIdx = torch::cat({WRowIdx, tiled_hidden_idx}, 0); // WRowIdx is the hidden neuron index
        dedup_and_sort(WRowIdx, WColIdx);

        if (WRowIdx.size(0) > round_prev_con_num) {
            round_prev_con_num = WRowIdx.size(0);
            same_counter = 0;
        } else
            same_counter++;
        if (same_counter > MAX_SAME_COUNTER) {
            break;
        } // if we have not found new connections for some (default 5) rounds, we can stop
    }
    // deduplicate and sort the rays
    if (prev_WRowIdx.has_value() && prev_WColIdx.has_value()) {
        assert(prev_WColIdx.value().size(0) == prev_WRowIdx.value().size(0));

        WRowIdx = torch::cat({prev_WRowIdx.value(), WRowIdx}, 0);
        WColIdx = torch::cat({prev_WColIdx.value(), WColIdx}, 0);
        dedup_and_sort(WRowIdx, WColIdx);
    } else {
        dedup_and_sort(WRowIdx, WColIdx);
    }
    return {WRowIdx, WColIdx};
}