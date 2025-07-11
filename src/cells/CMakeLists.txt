
add_library(cells STATIC
    cells.cpp
    
)

target_include_directories(cells PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party
)

target_link_libraries(cells "${TORCH_LIBRARIES}")