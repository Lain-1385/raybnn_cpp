#pragma once

#include <string>
#include <torch/torch.h>

std::vector<float> load_csv_to_vector(const std::string &file_path, const char &delimiter = ',');
torch::Tensor load_csv_to_tensor(const std::string &file_path,
                                 const char &delimiter = ',',
                                 const torch::TensorOptions &ops = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
