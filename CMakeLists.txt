cmake_minimum_required(VERSION 3.10)
project(PCClassify VERSION 1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif()

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if((CMAKE_CXX_COMPILER_ID MATCHES "Clang") OR (CMAKE_CXX_COMPILER_ID MATCHES "GNU"))
add_compile_options(-march=native)
endif()

find_package(OpenMP REQUIRED)
find_package(Boost COMPONENTS iostreams serialization)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(vendor/ethz)

set(SOURCES common.cpp scale.cpp point_io.cpp randomforest.cpp features.cpp color.cpp labels.cpp)

add_library(pcc OBJECT ${SOURCES})

add_executable(pctrain pctrain.cpp)
add_executable(pcclassify pcclassify.cpp)

target_link_libraries(pcc Eigen3::Eigen OpenMP::OpenMP_CXX Boost::iostreams Boost::serialization)
target_link_libraries(pctrain pcc)
target_link_libraries(pcclassify pcc)

install(TARGETS pctrain RUNTIME DESTINATION bin)
install(TARGETS pcclassify RUNTIME DESTINATION bin)
