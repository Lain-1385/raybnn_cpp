#pragma once

#include <string>
#include <torch/torch.h>

torch::Tensor load_csv2tensor(const std::string &file_path);