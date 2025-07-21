#pragma once

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
    int64_t ray_max_rounds;
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
private:
    struct modeldata;

public:
    void raytrace3();
    static std::pair<torch::Tensor, torch::Tensor>
    filter_rays(const float con_rad, torch::Tensor &target_pos, torch::Tensor &input_pos, torch::Tensor &input_idx);
    static torch::Tensor rays_from_neuronsA_to_neuronsB(const float con_rad, const torch::Tensor &pos_A, const torch::Tensor &pos_B);

    static torch::Tensor
    line_sphere_intersect(const torch::Tensor &line_start, const torch::Tensor &line_end, const torch::Tensor &block_cells, const float block_radius);
};