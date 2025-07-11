#include "cells.hpp"
#include <c10/core/ScalarType.h>
#include <cassert>
#include <cmath>
#include <iostream>

using namespace torch;

constexpr float NEURON_RAD_FACTOR = 1.1f;
constexpr float TARGET_DENSITY = 3500.0f;

void cells::specify_tensor_options(const TensorOptions &options) { opts = options; }

/*
Creates input neurons on the surface of a sphere with even neuron position assignment

Inputs
sphere_radius:   3D Sphere Radius
nums:   Number of input neurons

Outputs:
The 3D position of neurons on the surface of a 3D sphere [N,3]
*/
Tensor cells::sphere_even(int64_t nums, float sphere_radius) const {

    Tensor one = tensor(1.0f, opts);
    Tensor two = tensor(2.0f, opts);
    Tensor nums_tensor = tensor(static_cast<float>(nums), opts);

    Tensor idx = torch::arange(0, nums, opts).view({nums});

    Tensor phi = torch::acos(one - two * idx / nums_tensor);
    const float golden_angle = M_PI * (3.0f - std::sqrt(5.0f));
    Tensor theta = tensor(golden_angle, opts) * idx;

    Tensor x = sphere_radius * torch::sin(phi) * torch::cos(theta);
    Tensor y = sphere_radius * torch::sin(phi) * torch::sin(theta);
    Tensor z = sphere_radius * torch::cos(phi);

    Tensor points = torch::stack({x, y, z}, 1);
    std::cout << "points shape: " << points.sizes() << std::endl; // should be [N, 3]
    return points;
}

// Creates input neurons on the surface of a sphere with random neuron position assignment
// Tensor cells::sphere_random(const int64_t &nums, const float &sphere_radius) const {}

/*
Creates hidden neurons&glial in the ball with random neuron position assignment

Inputs
sphere_radius:   3D Sphere Radius
nums:   target Number of hidden neurons&glial

Outputs:
The 3D position of neurons in the volume of a 3D sphere [N,3]
*/
Tensor cells::ball_random(int64_t nums, float sphere_radius) const {
    std::vector<int64_t> shape = {nums};

    Tensor r = torch::rand(shape, opts);
    r = torch::pow(r, tensor(1.0f / 3.0f, opts)) * tensor(sphere_radius, opts);

    Tensor theta = torch::rand(shape, opts) * tensor(2.0f * M_PI, opts);
    Tensor phi = torch::rand(shape, opts) * tensor(M_PI, opts);
    Tensor x = r * torch::sin(phi) * torch::cos(theta);
    Tensor y = r * torch::sin(phi) * torch::sin(theta);
    Tensor z = r * torch::cos(phi);
    Tensor points = torch::stack({x, y, z}, 1);
    std::cout << "points shape: " << points.sizes() << std::endl; // should be [N, 3]

    return points;
}

// Finds the indices of near points to pivot within a given cube length
// Inputs:
// points: Tensor of points in 3D space [N, 3]
// pivot: Tensor of the pivot point in 3D space [3]
// length: The maximum distance in cube from the pivot to consider a point as "near"
// Outputs:
// A tensor containing the indices of points that are within the specified length from the pivot [M]
Tensor cells::find_in_cube(const Tensor &points, const Tensor &pivot, float length) const {
    Tensor distances = points - pivot;                     // [N, 3]
    Tensor mask = (distances < length) & (distances >= 0); // [N, 3]
    mask = mask.all(1);                                    //[N]
    return mask.nonzero().squeeze(1);                      // nonzero will return [M,1] shape, squeeze will convert it to [M]
    // std::cout << "indicesshape: " << indices.sizes() << std::endl; // should be [N]
}

// selects overlapping points within in given postions and radius
// Inputs:
// points: Tensor of points in 3D space [N, 3]
// neuron_rad: The radius of the neuron to consider for overlap
// Outputs:
// A tensor containing the indices of points that are overlapping [M]
Tensor cells::select_overlap(const Tensor &points, float radius) const {
    Tensor distances = points.unsqueeze(0) - points.unsqueeze(1); // [1, N, 3] - [N, 1, 3] = [N, N, 3]
    distances = torch::norm(distances, 2, 2);                     // [N, N]

    distances.fill_diagonal_(radius * 2); // fill_diagonal_ may not be avaiable in old version
    Tensor mask = distances < radius;     // [N, N]
    mask = mask.any(1);                   // [N]
    return mask.nonzero().squeeze(1);     // [N]
}

// build in function to set the diagonal of a tensor to a specific value
void set_diag(torch::Tensor &tensor, float value) {
    Tensor mask = torch::eye(tensor.size(0), tensor.options());
    tensor.masked_fill_(mask.to(torch::kByte), value);
}

// Detects cell collisions in minibatch, where groups/minibatches of cells are checked
// Inputs:
// cell_pos: Tensor of points in 3D space [N, 3]
// Outputs:
// positions of non colliding cells [N, 3]
Tensor cells::check_all_collision_minibatch(const Tensor &cell_pos, const float sphere_rad, const float neuron_rad) const {
    assert(sphere_rad > 0 && neuron_rad > 0);
    assert(cell_pos.dim() == 2 && cell_pos.size(1) == 3);

    float step = (4.0f / 3.0f) * M_PI * sphere_rad * sphere_rad * sphere_rad * TARGET_DENSITY / cell_pos.size(0);
    float cube_size = 2.05f * neuron_rad * 1.1f + step;

    Tensor mask = torch::ones({cell_pos.size(0)}, opts).to(torch::kBool); // [N]
    Tensor pivots = generate_pivot_tensor(sphere_rad, step);              // [N, 3]

    for (int i = 0; i < pivots.size(0); ++i) {
        Tensor indices_cur = find_in_cube(cell_pos, pivots[i], cube_size);
        if (indices_cur.size(0) < 2) {
            continue; // No points found in this cube, skip to next pivot
        }
        Tensor points_in_cube = cell_pos.index_select(0, indices_cur);           // [M, 3]
        Tensor indices_overlap_cur = select_overlap(points_in_cube, neuron_rad); // [M]
        if (indices_overlap_cur.size(0) < 1) {
            continue;
        }
        indices_cur = indices_cur.index_select(0, indices_overlap_cur); // [M]
        mask.index_put_({indices_cur}, false);
    }
    // Return only non-colliding points
    return cell_pos.index_select(0, mask.nonzero().squeeze(1)); //
}

// Generate a [N, 3] tensor of pivot positions covering [-sphere_rad, sphere_rad]^3
// with given step size. Fully vectorized and parallelizable.
Tensor cells::generate_pivot_tensor(float sphere_rad, float step) const {
    // 1e-5 to avoid floating point precision issues at the boundary
    auto range = torch::arange(-sphere_rad, sphere_rad + 1e-5, step, opts); // [S]

    auto grids = torch::meshgrid({range, range, range}, "ij"); // [S, S, S] each
    auto grid_x = grids[0];
    auto grid_y = grids[1];
    auto grid_z = grids[2];

    // Stack and reshape into [N, 3] pivot positions
    torch::Tensor pivots = torch::stack({grid_x, grid_y, grid_z}, 3).reshape({-1, 3}); // [S³, 3]

    return pivots;
}
