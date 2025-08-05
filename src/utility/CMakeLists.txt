add_library(utility STATIC
    utility.cpp
)

target_include_directories(utility PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(TORCH_CUDA_FOUND)
  add_definitions(-DUSE_CUDA)
endif()

target_link_libraries(utility "${TORCH_LIBRARIES}")