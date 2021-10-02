cmake_minimum_required(VERSION 3.5)

project(yolov5-pplnn)
set(CMAKE_CXX_FLAGS "-std=c++14 -g -pthread ${CMAKE_CXX_FLAGS}")

#Opencv
set(OPENCV_LIBS "")
if(OpenCV_FOUND)
    message(STATUS "OpenCV3 Found (${OpenCV_CONFIG_PATH})")
    list(APPEND OPENCV_LIBS ${OpenCV_LIBS})
endif()
if(NOT OpenCV_FOUND)
    # OpenCV 2
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
    message(STATUS "OpenCV2 Found (${OpenCV_CONFIG_PATH})")
    list(APPEND OPENCV_LIBS ${OpenCV_LIBS})
endif()
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV libs: (${OPENCV_LIBS})")
else()
    message(WARNING "Not find OpenCV. project maybe invalid ")
endif()

file(
    GLOB
    yolov5_source
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/lib)

add_executable(test_yolov5
               ${CMAKE_CURRENT_SOURCE_DIR}/test/test_yolov5.cpp
               ${yolov5_source})

target_link_libraries(
    test_yolov5
    pplnn_static
    pplcommon_static
    PPLKernelX86
    protobuf
    ${OPENCV_LIBS}
)
