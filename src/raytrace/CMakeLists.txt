add_library(raytrace STATIC
    raytrace.cpp
    
)

target_include_directories(raytrace PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(raytrace "${TORCH_LIBRARIES}")