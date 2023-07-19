cmake_minimum_required(VERSION 3.16)
project(HumanGazeEstimator)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCES src/main.cpp src/daugman.cpp src/daugman.h src/metrics.cpp src/metrics.h src/detector.cpp src/detector.h)
add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
