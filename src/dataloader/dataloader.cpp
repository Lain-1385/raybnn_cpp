#include "dataloader.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <vector>

// Load a CSV file into a torch tensor (float32)
torch::Tensor load_csv2tensor(const std::string &file_path) {
    std::ifstream file(file_path);
    std::string line, cell;
    std::vector<float> values;
    size_t num_rows = 0;
    size_t num_cols = 0;

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        size_t current_cols = 0;

        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stof(cell));
            ++current_cols;
        }

        if (num_cols == 0) {
            num_cols = current_cols;
        } else if (current_cols != num_cols) {
            throw std::runtime_error("Inconsistent number of columns in CSV file.");
        }

        ++num_rows;
    }

    return torch::from_blob(values.data(), {static_cast<long>(num_rows), static_cast<long>(num_cols)}, torch::kFloat32).clone();
}