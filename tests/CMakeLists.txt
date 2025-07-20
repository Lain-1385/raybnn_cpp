file(GLOB TEST_SOURCES "*.cpp")

find_package(Catch2 3 REQUIRED)

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)

    add_executable(${test_name} ${test_src})

    # 链接项目代码库 + Catch2 + LibTorch
    target_link_libraries(${test_name}
        PRIVATE
            cells                      
            dataloader
            Catch2::Catch2WithMain    # 提供 main() 和 Catch2 实现
            ${TORCH_LIBRARIES}
    )

    # 设置 include 路径（根据你项目结构调整）
    target_include_directories(${test_name}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/third_party
    )

    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()