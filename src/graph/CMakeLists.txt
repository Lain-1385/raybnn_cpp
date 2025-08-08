add_library(graph STATIC
    graph.cpp
)

target_include_directories(graph PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/sparse
)

target_link_libraries(graph "${TORCH_LIBRARIES}")