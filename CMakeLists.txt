cmake_minimum_required(VERSION 3.24)
project(WinterTask)

# find required opencv
find_package(OpenCV REQUIRED)
# directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS})


set(CMAKE_CXX_STANDARD 17)

add_executable(WinterTask main.cpp task.cpp task.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
