cmake_minimum_required(VERSION 3.24)
project(WinterTask)

# find required opencv
find_package(OpenCV REQUIRED)
# directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找 OpenCV 目录下的所有源文件将名称保存到 DIR_OPENCV_SRCS 变量
aux_source_directory(./OpenCV DIR_OPENCV_SRCS)

set(CMAKE_CXX_STANDARD 17)

add_executable(WinterTask main.cpp ${DIR_OPENCV_SRCS})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
