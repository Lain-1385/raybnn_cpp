add_library(sparse STATIC
    sparse.cpp
)

target_include_directories(sparse PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(sparse "${TORCH_LIBRARIES}")