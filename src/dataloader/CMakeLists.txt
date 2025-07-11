
add_library(dataloader STATIC
    dataloader.cpp
    
)

target_include_directories(dataloader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(dataloader "${TORCH_LIBRARIES}")