cmake_minimum_required(VERSION 3.16)
project(birdseye_view CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

add_executable(birdseye main.cpp)
target_include_directories(birdseye PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(birdseye PRIVATE ${OpenCV_LIBS})