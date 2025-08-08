#include "sparse.hpp"
#include <ATen/core/TensorBody.h>

using namespace torch;

// find the element and output its index
//  WRowIdxCOO [M]
//  target_rows [N]
torch::Tensor sparse::COO_find(const torch::Tensor &WRowIdxCOO, const torch::Tensor &target_rows) {
    // TORCH_CHECK(WRowIdxCOO.device() == target_rows.device(), "Device mismatch");
    // TORCH_CHECK(WRowIdxCOO.dtype() == target_rows.dtype(), "Dtype mismatch");

    Tensor target_rows_unsq = target_rows.unsqueeze(1);      // [N, 1]
    Tensor WRowIdxCOO_unsq = WRowIdxCOO.unsqueeze(0);        // [1, M]
    Tensor eq_result = target_rows_unsq.eq(WRowIdxCOO_unsq); // [N, M] bool

    Tensor any_result = eq_result.any(1); // [N] bool

    return any_result;
}

Tensor sparse::COO_find_batch(const Tensor &WRowIdxCOO, const Tensor &target_rows, int64_t batch_size) {
    int64_t total_size = target_rows.size(0);
    int64_t i = 0;
    std::vector<Tensor> idx_list;

    while (i < total_size) {
        int64_t startseq = i;
        int64_t endseq = std::min(i + batch_size - 1, total_size - 1);

        Tensor inputarr = target_rows.slice(0, startseq, endseq + 1);

        Tensor idx = sparse::COO_find(WRowIdxCOO, inputarr);

        Tensor valid_idx = idx.masked_select(idx >= 0);
        if (valid_idx.size(0) > 0) {
            idx_list.push_back(valid_idx);
        }

        i += batch_size;
    }

    Tensor total_idx;
    if (!idx_list.empty()) {
        total_idx = torch::cat(idx_list, 0);
        total_idx = std::get<0>(torch::sort(total_idx, /*dim=*/0, /*descending=*/true));
    } else {
        total_idx = torch::empty({0}, torch::kInt64);
    }

    return total_idx;
}

// arr [M], ranging from 0 to neuron_size
// output index for unique value
torch::Tensor sparse::find_unique(const torch::Tensor &arr, int64_t neuron_size) {
    Tensor table = torch::zeros({neuron_size}, torch::TensorOptions().dtype(torch::kBool).device(arr.device()));

    table.index_put_({arr}, true);

    auto unique_idx = torch::nonzero(table).squeeze();

    return unique_idx;
}
