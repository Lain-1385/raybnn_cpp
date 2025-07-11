#pragma once

#include <c10/core/DeviceType.h>
#include <cassert>
#include <torch/torch.h>

class cells {
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

public:
    torch::Tensor sphere_even(int64_t nums, float sphere_radius) const;
    // torch::Tensor sphere_random( int64_t &nums,  float &sphere_radius) const;
    void specify_tensor_options(const torch::TensorOptions &options);

    torch::Tensor ball_random(int64_t nums, float sphere_radius) const;

    torch::Tensor find_in_cube(const torch::Tensor &points, const torch::Tensor &pivot, float length) const;

    torch::Tensor select_overlap(const torch::Tensor &points, float neuron_rad) const;

    torch::Tensor check_all_collision_minibatch(const torch::Tensor &cell_pos, const float sphere_rad, const float neuron_rad) const;

    torch::Tensor generate_pivot_tensor(float sphere_rad, float step) const;
};