#include "raytrace/raytrace.hpp"
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("dedup_and_sort", "[dedup_and_sort]") {
    torch::Tensor WRowIdx = torch::tensor({0, 9, 0, 1, 2, 0, 1, 2}, torch::dtype(torch::kLong));
    torch::Tensor WColIdx = torch::tensor({9, 9, 1, 2, 3, 1, 2, 3}, torch::dtype(torch::kLong));
    torch::Tensor indices = torch::stack({WRowIdx, WColIdx}, 1);
    std::cout << "Indices: " << indices << std::endl;
    // WARNING:my libtorch has no unique function, so we use torch::_unique
    auto [unique_indices, temp] = torch::_unique(indices, true, true);
    std::cout << "Unique Indices: " << unique_indices << std::endl;
    std::cout << "Temp: " << temp << std::endl;
}
