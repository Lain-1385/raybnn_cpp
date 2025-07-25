#include "raytrace/raytrace.hpp"
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

inline void dedup_and_sort(torch::Tensor &WRowIdx, torch::Tensor &WColIdx) {
    int64_t max_col = WColIdx.max().item<int64_t>() + 1; // Ensure we have a valid max column index
    torch::Tensor hash = WRowIdx * max_col + WColIdx;
    auto [unique_hash, _] = torch::_unique(hash, true, false);
    std::cout << "unique_hash dtype: " << unique_hash.dtype() << std::endl;
    WRowIdx = torch::div(unique_hash, max_col, "floor");
    std::cout << "WRowIdx dtype: " << WRowIdx.dtype() << std::endl;
    WColIdx = unique_hash % max_col;
}

TEST_CASE("dedup_and_sort", "[dedup_and_sort]") {
    torch::Tensor WRowIdx = torch::tensor({0, 9, 0, 1, 2, 0, 1, 2}, torch::dtype(torch::kLong));
    torch::Tensor WColIdx = torch::tensor({9, 9, 1, 2, 3, 1, 2, 3}, torch::dtype(torch::kLong));
    constexpr int64_t MAX_COL = 10000;

    dedup_and_sort(WRowIdx, WColIdx);
    std::cout << "WRowIdx: " << WRowIdx << std::endl;
    std::cout << "WColIdx: " << WColIdx << std::endl;
}
