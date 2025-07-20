#include "dataloader.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <vector>

std::vector<float> load_csv_to_vector(const std::string &file_path, const char &delimiter) {

    std::ifstream file(file_path);
    std::vector<float> data;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream line_stream(line);
        std::string value;
        while (std::getline(line_stream, value, delimiter)) {
            data.push_back(std::stof(value));
        }
    }
    return data;
}

// Load a CSV file into a torch tensor (float32)
torch::Tensor load_csv_to_tensor(const std::string &file_path, const char &delimiter, const torch::TensorOptions &ops) {
    std::vector<float> data = load_csv_to_vector(file_path, delimiter);
    auto tensor = torch::from_blob(data.data(), {static_cast<long>(data.size())}, ops);
    return tensor.clone(); // Clone to ensure the tensor is not referencing the original data
}