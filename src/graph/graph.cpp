#include "graph.hpp"
#include "sparse.hpp"
using namespace torch;

constexpr int64_t COO_FIND_LIMIT = 150000000;

// neuron_idx_in [N]
torch::Tensor RayBNNGraph::traverse_forward(torch::Tensor &neuron_idx_in, int64_t depth, int64_t neuron_size) {
    torch::Tensor out_idx = neuron_idx_in.clone();

    int64_t COO_batch_size = 1 + (COO_FIND_LIMIT / this->WColIdx_.size(0));

    for (int64_t cur_depth = 0; cur_depth < depth; ++cur_depth) {
        torch::Tensor valsel = sparse::COO_find_batch(this->WColIdx_, out_idx, COO_batch_size);
        if (valsel.size(0) == 0) {
            break;
        }
        out_idx = this->WRowIdx_.index_select(0, valsel);

        out_idx = sparse::find_unique(out_idx, neuron_size);

        if (out_idx.size(0) == 0) {
            break;
        }
    }
    return out_idx;
}

torch::Tensor RayBNNGraph::traverse_backward(torch::Tensor &neuron_idx_in, int64_t depth, int64_t neuron_size) {
    torch::Tensor out_idx = neuron_idx_in.clone();

    int64_t COO_batch_size = 1 + (COO_FIND_LIMIT / this->WColIdx_.size(0));

    for (int64_t cur_depth = 0; cur_depth < depth; ++cur_depth) {

        torch::Tensor valsel = sparse::COO_find_batch(this->WRowIdx_, out_idx, COO_batch_size);
        if (valsel.size(0) == 0) {
            break;
        }
        // find corresponding index in the WColIdx
        out_idx = this->WColIdx_.index_select(0, valsel);

        out_idx = sparse::find_unique(out_idx, neuron_size);

        if (out_idx.size(0) == 0) {
            break;
        }
    }
    return out_idx;
}

bool RayBNNGraph::check_connected(const torch::Tensor &in_idx, const torch::Tensor &out_idx, int64_t neuron_size, int64_t depth) {
    bool connected = true;

    int64_t in_num = in_idx.size(0);
    int64_t out_num = out_idx.size(0);

    int64_t COO_batch_size = 1 + (COO_FIND_LIMIT / out_num);

    for (int64_t i = 0; i < in_num; ++i) {

        torch::Tensor input_idx = in_idx[i].unsqueeze(0);

        torch::Tensor temp_out_idx = this->traverse_forward(input_idx, depth, neuron_size);

        torch::Tensor detect_out_idx = sparse::COO_find_batch(out_idx, temp_out_idx, COO_batch_size);

        if (detect_out_idx.size(0) < out_num) {
            connected = false;
            break;
        }
    }

    return connected;
}

void RayBNNGraph::delete_loops(const torch::Tensor &last_idx,
                               const torch::Tensor &first_idx,
                               int64_t neuron_size,
                               int64_t depth,
                               torch::Tensor &WValues,
                               torch::Tensor &WRowIdxCOO,
                               torch::Tensor &WColIdx) {
    using namespace torch;

    Tensor cur_idx = last_idx.clone();
    Tensor filter_idx = torch::cat({first_idx, last_idx}, 0);

    Tensor delWRowIdxCOO = torch::empty({0}, WRowIdxCOO.options());
    Tensor delWColIdx = torch::empty({0}, WColIdx.options());

    int64_t COO_batch_size = 1 + (COO_FIND_LIMIT / filter_idx.size(0));

    for (int64_t j = 0; j < depth; ++j) {
        int64_t cur_num = cur_idx.size(0);

        if (j == depth - 1) {
            filter_idx = filter_idx.slice(0, first_idx.size(0), filter_idx.size(0));
        }

        Tensor next_idx = torch::empty({0}, cur_idx.options());

        for (int64_t i = 0; i < cur_num; ++i) {
            Tensor input_idx = cur_idx[i].unsqueeze(0);

            Tensor temp_first_idx = this->traverse_backward(input_idx, 1, neuron_size);

            if (temp_first_idx.size(0) == 0)
                continue;

            COO_batch_size = 1 + (COO_FIND_LIMIT / temp_first_idx.size(0));
            Tensor detect_first_idx = sparse::COO_find_batch(filter_idx, temp_first_idx, COO_batch_size);

            if (detect_first_idx.size(0) > 0) {
                Tensor con_first_idx = temp_first_idx.index_select(0, detect_first_idx);

                Tensor tiled_input_idx = input_idx.expand({con_first_idx.size(0)});
                delWRowIdxCOO = torch::cat({delWRowIdxCOO, tiled_input_idx}, 0);
                delWColIdx = torch::cat({delWColIdx, con_first_idx}, 0);

                Tensor mask = torch::ones(temp_first_idx.sizes(), torch::kBool).to(temp_first_idx.device());
                mask.index_put_({detect_first_idx}, false);
                Tensor tempidx = mask.nonzero().squeeze();
                if (tempidx.size(0) > 0) {
                    temp_first_idx = temp_first_idx.index_select(0, tempidx);
                } else {
                    continue;
                }
            }
            next_idx = torch::cat({next_idx, temp_first_idx}, 0);
            next_idx = sparse::find_unique(next_idx, neuron_size);
        }
        cur_idx = next_idx.clone();
        filter_idx = torch::cat({next_idx, filter_idx}, 0);
        filter_idx = sparse::find_unique(filter_idx, neuron_size);
    }

    if (delWRowIdxCOO.size(0) > 0)
        delWRowIdxCOO = delWRowIdxCOO.slice(0, 1, delWRowIdxCOO.size(0));
    if (delWColIdx.size(0) > 0)
        delWColIdx = delWColIdx.slice(0, 1, delWColIdx.size(0));

    Tensor gidx1 = get_global_weight_idx(neuron_size, WRowIdxCOO, WColIdx);
    Tensor gidx2 = get_global_weight_idx(neuron_size, delWRowIdxCOO, delWColIdx);

    auto gidx1_cpu = gidx1.to(torch::kCPU);
    auto gidx2_cpu = gidx2.to(torch::kCPU);
    auto WValues_cpu = WValues.to(torch::kCPU);
    auto WRowIdxCOO_cpu = WRowIdxCOO.to(torch::kCPU);
    auto WColIdx_cpu = WColIdx.to(torch::kCPU);

    std::unordered_map<uint64_t, float> join_WValues;
    std::unordered_map<uint64_t, int32_t> join_WColIdx;
    std::unordered_map<uint64_t, int32_t> join_WRowIdxCOO;

    auto gidx1_acc = gidx1_cpu.data_ptr<uint64_t>();
    auto gidx2_acc = gidx2_cpu.data_ptr<uint64_t>();
    auto WValues_acc = WValues_cpu.data_ptr<float>();
    auto WRowIdxCOO_acc = WRowIdxCOO_cpu.data_ptr<int32_t>();
    auto WColIdx_acc = WColIdx_cpu.data_ptr<int32_t>();

    for (int64_t qq = 0; qq < gidx1_cpu.size(0); ++qq) {
        uint64_t cur_gidx = gidx1_acc[qq];
        join_WValues[cur_gidx] = WValues_acc[qq];
        join_WColIdx[cur_gidx] = WColIdx_acc[qq];
        join_WRowIdxCOO[cur_gidx] = WRowIdxCOO_acc[qq];
    }
    for (int64_t qq = 0; qq < gidx2_cpu.size(0); ++qq) {
        uint64_t cur_gidx = gidx2_acc[qq];
        join_WValues.erase(cur_gidx);
        join_WColIdx.erase(cur_gidx);
        join_WRowIdxCOO.erase(cur_gidx);
    }

    std::vector<uint64_t> gidx3;
    for (const auto &kv : join_WValues)
        gidx3.push_back(kv.first);
    std::sort(gidx3.begin(), gidx3.end());

    std::vector<float> WValues_vec;
    std::vector<int32_t> WColIdx_vec;
    std::vector<int32_t> WRowIdxCOO_vec;
    for (auto qq : gidx3) {
        WValues_vec.push_back(join_WValues[qq]);
        WColIdx_vec.push_back(join_WColIdx[qq]);
        WRowIdxCOO_vec.push_back(join_WRowIdxCOO[qq]);
    }

    WValues = torch::from_blob(WValues_vec.data(), {(int64_t)WValues_vec.size(), 1}, torch::TensorOptions().dtype(WValues.dtype())).clone();
    WColIdx = torch::from_blob(WColIdx_vec.data(), {(int64_t)WColIdx_vec.size(), 1}, torch::TensorOptions().dtype(WColIdx.dtype())).clone();
    WRowIdxCOO =
        torch::from_blob(WRowIdxCOO_vec.data(), {(int64_t)WRowIdxCOO_vec.size(), 1}, torch::TensorOptions().dtype(WRowIdxCOO.dtype())).clone();
}