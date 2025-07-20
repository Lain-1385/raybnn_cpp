// test_find_near.cpp
#include "cells/cells.hpp"
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

cells c;
auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
TEST_CASE("test_sphere_even", "[sphere]") {

    torch::Tensor points = c.sphere_even(100, 1.0);
    REQUIRE(points.sizes() == std::vector<int64_t>{100, 3});
    // torch::save(points, "sphere_even_100.pt");

    torch::Tensor points2 = c.ball_random(100, 1.0);
    REQUIRE(points2.sizes() == std::vector<int64_t>{100, 3});
    // torch::save(points2, "ball_random_100.pt");
}

TEST_CASE("find_near returns correct shape and values", "[find_near]") {
    torch::Tensor points = torch::tensor({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {2.0, 2.0, 2.0}, {10.0, 10.0, 10.0}, {0.5, 0.5, 0.5}}, opts);

    torch::Tensor pivot = torch::tensor({1.0, 2.0, 2.0}, opts);

    float length = 2.0;

    torch::Tensor result = c.find_in_cube(points, pivot, length);
    REQUIRE(result.dim() == 1);
    std::cout << "result: " << result << std::endl;
    REQUIRE(result.size(0) == 2);
}

TEST_CASE("find_near with 0", "[find_near]") {

    torch::Tensor points = torch::tensor({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {2.0, 2.0, 2.0}, {10.0, 10.0, 10.0}, {0.5, 0.5, 0.5}}, opts);

    torch::Tensor pivot = torch::tensor({1.0, 2.0, 2.0}, opts);
    float length = 0.1;

    torch::Tensor result = c.find_in_cube(points, pivot, length);

    std::cout << "result: " << result << std::endl;
    REQUIRE(result.size(0) == 0);
}

TEST_CASE("select_overlap normal", "[select_overlap]") {
    torch::Tensor points = torch::tensor({{1.0, 2.0, 3.0}, {2.0, 2.0, 2.0}, {10.0, 10.0, 10.0}, {0.5, 0.5, 0.5}}, opts);

    float radius = 10.0f;

    torch::Tensor result = c.select_overlap(points, radius);
    std::cout << "result: " << result << std::endl;
    REQUIRE(result.size(0) == 3);
}

TEST_CASE("generate_pivots", "[generate_pivot_tensor]") {
    torch::Tensor pivots = c.generate_pivot_tensor(5.0f, 1.0f);
    REQUIRE(pivots.dim() == 2);
    REQUIRE(pivots.size(0) == 11 * 11 * 11);
    REQUIRE(pivots.size(1) == 3);

    // std::cout << "pivots: " << pivots << std::endl;
}

TEST_CASE("test_check_all_collision_minibatch", "[check_all_collision_minibatch]") {
    auto range = torch::arange(-1, 2, opts);
    auto grids = torch::meshgrid({range, range, range}, "ij");
    auto grid = torch::stack(grids, -1).reshape({-1, 3});
    auto one_more = torch::tensor({{0.5, 0.5, 0.5}}, opts);
    grid = torch::cat({grid, one_more}, 0); // Add one more point to test collision

    torch::Tensor result = c.check_all_collision_minibatch(grid, 1.0f, 0.9f);
    std::cout << "result: " << result << std::endl;
}

TEST_CASE("split 9 neurons", "[split_into_glia_neuron]") {
    auto points = torch::arange(27, opts).reshape({9, 3});
    auto [neuron_pos, glia_pos] = c.split_into_glia_neuron(0.5f, points);

    REQUIRE(neuron_pos.size(0) == 4);
    REQUIRE(glia_pos.size(0) == 5);
    std::cout << "neuron_pos: " << neuron_pos << std::endl;
    std::cout << "glia_pos: " << glia_pos << std::endl;
}