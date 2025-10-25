cmake_minimum_required(VERSION 3.15...3.26)
project(
  "${SKBUILD_PROJECT_NAME}"
  LANGUAGES CXX
  VERSION "${SKBUILD_PROJECT_VERSION}")

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_mvhg 
                    src/pybind11.cpp 
                    src/mvhg.cpp)

target_include_directories(_mvhg
                           PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_definitions(_mvhg
                           PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _mvhg LIBRARY DESTINATION mvhg)

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(STATUS "Enabling performance optimizations for ${CMAKE_BUILD_TYPE} build...")

    # Add compiler-specific optimization flags
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(_mvhg PRIVATE
            -O3                 # Full optimization
            -march=native       # Use all CPU features available on this machine
            -funroll-loops      # Unroll loops for performance
            -fomit-frame-pointer
        )
    elseif(MSVC)
        target_compile_options(_mvhg PRIVATE
            /O2                 # Optimize for speed
            /fp:fast            # Fast math
            /openmp             # Enable OpenMP
        )
    endif()
endif()
