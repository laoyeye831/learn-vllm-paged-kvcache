cmake_minimum_required(VERSION 3.20)
project(KV_Cache LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 单配置生成器(Ninja/Makefile)未指定时默认 Debug
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# ========== 核心代码：静态库，不含 main ==========
add_library(KV_Cache_lib STATIC
    allocator.cpp
    manager.cpp
    scheduler.cpp
    other.cpp
)
target_include_directories(KV_Cache_lib PUBLIC ${PROJECT_SOURCE_DIR})

if(MSVC)
    target_compile_options(KV_Cache_lib PUBLIC /utf-8)
endif()

# ========== 测试：每个 cpp 自带 main ==========
enable_testing()

foreach(test_name IN ITEMS kvcache_phase1_3_test scheduler_test random_pressure_test scheduler_thread_test thread_stress_test)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE KV_Cache_lib)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
