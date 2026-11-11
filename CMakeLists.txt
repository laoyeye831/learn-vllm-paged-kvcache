cmake_minimum_required(VERSION 3.20)
project(KV_Cache LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 单配置生成器(Ninja/Makefile)未指定时默认 Debug
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# ========== ThreadSanitizer（仅 GCC/Clang） ==========
option(KV_CACHE_ENABLE_TSAN "Build with ThreadSanitizer" OFF)

if(KV_CACHE_ENABLE_TSAN)
    if(MSVC)
        message(FATAL_ERROR "KV_CACHE_ENABLE_TSAN requires GCC or Clang; MSVC has no ThreadSanitizer")
    endif()
    message(STATUS "ThreadSanitizer enabled")
    add_compile_options(-fsanitize=thread -g -O1 -fno-omit-frame-pointer)
    add_link_options(-fsanitize=thread)
endif()

# ========== 核心代码：静态库，不含 main ==========
add_library(KV_Cache_lib STATIC
    src/allocator.cpp
    src/manager.cpp
    src/scheduler.cpp
    src/other.cpp
)
target_include_directories(KV_Cache_lib PUBLIC ${PROJECT_SOURCE_DIR}/include)

if(MSVC)
    target_compile_options(KV_Cache_lib PUBLIC /utf-8)
else()
    target_compile_options(KV_Cache_lib PRIVATE -Wall -Wextra)
endif()

# ========== 测试：每个 cpp 自带 main ==========
enable_testing()

foreach(test_name IN ITEMS kvcache_phase1_3_test scheduler_test random_pressure_test scheduler_thread_test thread_stress_test)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE KV_Cache_lib)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# TSan 插桩会让计时类断言失真（开销可达 5-15 倍），单独打标签便于按需排除
if(KV_CACHE_ENABLE_TSAN)
    set_tests_properties(scheduler_thread_test thread_stress_test
        PROPERTIES LABELS concurrency TIMEOUT 900)
endif()

# ========== 示例：可执行 demo，不注册为测试 ==========
option(KV_CACHE_BUILD_EXAMPLES "Build examples/demo.cpp" ON)

if(KV_CACHE_BUILD_EXAMPLES)
    add_executable(kvcache_demo examples/demo.cpp)
    target_link_libraries(kvcache_demo PRIVATE KV_Cache_lib)
    # 冒烟测试：demo 必须能跑完（覆盖调度器终止性与停机路径）
    add_test(NAME kvcache_demo COMMAND kvcache_demo)
    set_tests_properties(kvcache_demo PROPERTIES LABELS demo TIMEOUT 120)
endif()
