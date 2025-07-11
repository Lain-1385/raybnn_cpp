cmake_minimum_required(VERSION 3.31)  


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED 23)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Torch_DIR "$ENV{HOME}/libtorch/share/cmake/Torch")
project(raybnn_cpp)

find_package(Torch REQUIRED)

add_subdirectory(src)

add_executable(MyApp main.cpp)

target_link_libraries(MyApp "${TORCH_LIBRARIES}")

add_subdirectory(tests)