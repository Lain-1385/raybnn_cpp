add_library(utility STATIC
    utility.cpp
)

target_include_directories(utility PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(utility "${TORCH_LIBRARIES}")