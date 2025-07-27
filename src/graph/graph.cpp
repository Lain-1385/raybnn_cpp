#include "graph.hpp"

using namespace torch;

Tensor COO_find_batch(Tensor COO_weight, Tensor idx);

// neuron_idx_in [N]
torch::Tensor RayBNNGraph::traverse_forward(torch::Tensor neuron_idx_in, int64_t depth) {
    for (int64_t cur_depth = 0; cur_depth < depth; ++cur_depth) {
        torch::Tensor idx_temp = COO_find_batch(this->WColIdx_, neuron_idx_in);
        neuron_idx_in = this->WRowIdx_.index_select(0, idx_temp);
    }
    return neuron_idx_in;
}
