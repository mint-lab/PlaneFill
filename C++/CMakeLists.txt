CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(plane_fill VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_BUILD_TYPE Debug)

find_package( OpenCV REQUIRED )

add_subdirectory(cnpy)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/cnpy)

link_directories(${OpenCV_LIBRARY_DIRS})

ADD_EXECUTABLE(${PROJECT_NAME} main.cpp plane_fill.cpp utils.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBRARIES} cnpy)
