cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0146 OLD)
project(yolov11-12-opencv-cuda DESCRIPTION "Real time Object Detection using YOLOv11/12, OpenCV with CUDA backend" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

find_package(OpenCV REQUIRED)

set(SOURCES
    src/main.cpp
    src/Inference.cpp
)

add_executable(yolo-inference ${SOURCES})

target_include_directories(yolo-inference
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(yolo-inference
    PRIVATE
        ${OpenCV_LIBS}
)

target_compile_options(yolo-inference PRIVATE
    $<$<CONFIG:Debug>:-g -O0 -Wall -Wextra -Wpedantic -Werror -Wimplicit-fallthrough -Wsign-conversion>
    $<$<CONFIG:Release>:-O3>
)

# Doxygen Documentation
option(BUILD_DOCS "Build Doxygen documentation" OFF)
find_package(Doxygen QUIET)
if(BUILD_DOCS AND DOXYGEN_FOUND)
    add_custom_target(doc_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
elseif(NOT BUILD_DOCS)
    message(STATUS "BUILD_DOCS is OFF.")
else()
    message(STATUS "Doxygen not found. Documentation target will not be available.")
endif()
