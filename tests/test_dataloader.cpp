#include "dataloader/dataloader.hpp"
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("load_files_to_vectorr", "[load_files_to_vector]") {
    // Test loading a CSV file into a tensor
    std::string file_path = "/home/tyler/raybnn_cpp/third_party/mnist_test.csv";
    auto data = load_csv_to_tensor(file_path);
    data = data.view({10000, -1});
    torch::Tensor features = data.index({"...", torch::indexing::Slice(0, 784)}); // [10000, 784]
    torch::Tensor labels = data.index({"...", torch::indexing::Slice(784, 785)}); // [10000, 1]
    std::cout << "Features shape: " << features.sizes() << std::endl;
    std::cout << "Labels shape: " << labels.sizes() << std::endl;
    std::cout << "Features[10]: " << features.index({10}) << std::endl;
    // std::cout << "Labels " << labels.index({torch::indexing::Slice{0, 50}, "..."}) << std::endl;
}
