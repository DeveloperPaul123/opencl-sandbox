cmake_minimum_required(VERSION 3.12.0)

project(opencl-sycl-test)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCL REQUIRED)

set(project_sources
    src/main.cpp)

add_executable(${PROJECT_NAME} ${project_sources})

target_compile_definitions(${PROJECT_NAME} 
    PUBLIC 
    __CL_ENABLE_EXCEPTIONS
    OPEN_CL_KERNELS_DIR="${PROJECT_SOURCE_DIR}/cl")
target_link_libraries(${PROJECT_NAME} PUBLIC OpenCL::OpenCL)