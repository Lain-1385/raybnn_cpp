#pragma once

#include <cstddef>
#include <cstdint>
#include <torch/torch.h>

struct modeldata {
    int64_t neuron_size;
    int64_t input_size;
    int64_t output_size;
    int64_t proc_num;
    int64_t active_size;
    int64_t batch_size;
    int64_t ray_input_connection_num;
    size_t ray_max_rounds;
    bool ray_glia_intersect;
    bool ray_neuron_intersect;
    float neuron_rad;
    float time_step;
    float nration;
    float neuron_std;
    float sphere_rad;
    float con_rad;
};

class raytrace {
public:
    static std::pair<torch::Tensor, torch::Tensor>
    filter_rays(const float con_rad, const torch::Tensor &target_pos, const torch::Tensor &input_pos, const torch::Tensor &input_idx);

    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rays_from_neuronsA_to_neuronsB(const float con_rad,
                                                                                                                 const torch::Tensor &pos_A,
                                                                                                                 const torch::Tensor &pos_B,
                                                                                                                 const torch::Tensor &idx_A,
                                                                                                                 const torch::Tensor &idx_B);

    static torch::Tensor line_sphere_intersect(const torch::Tensor &line_start,
                                               const torch::Tensor &line_end,
                                               const torch::Tensor &block_cells,
                                               const torch::Tensor &block_radius);

    static void line_sphere_intersect_batch(const int64_t batch_size,
                                            const int64_t max_allowed_hits,
                                            const torch::Tensor &block_cells,
                                            const torch::Tensor &block_radius,
                                            torch::Tensor &line_start,
                                            torch::Tensor &line_end,
                                            torch::Tensor &index_start,
                                            torch::Tensor &index_end);
    std::tuple<torch::Tensor, torch::Tensor> raytrace_distance_limited(const modeldata &model_info,
                                                                       const torch::Tensor &glia_pos,
                                                                       const torch::Tensor &sender_pos,
                                                                       const torch::Tensor &receiver_pos,
                                                                       const std::optional<torch::Tensor> &prev_WRowIdx = std::nullopt,
                                                                       const std::optional<torch::Tensor> &prev_WColIdx = std::nullopt);
};