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

set(LIGHTGBM_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/local")
include(ExternalProject)
ExternalProject_Add(lightgbm
      GIT_REPOSITORY  https://github.com/microsoft/LightGBM
      GIT_TAG e4231205a3bac13662a81db9433ddaea8924fbce
      GIT_SHALLOW ON
      CMAKE_ARGS -DBUILD_STATIC_LIB=ON 
                  -DCMAKE_INSTALL_PREFIX=${LIGHTGBM_PREFIX}
)
add_library(lib_lightgbm SHARED IMPORTED)

set(LIGHTGBM_INCLUDE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lightgbm-prefix/src/lightgbm/include")

# TODO: test on Windows, macOS
set(LIGHTGBM_LIB_PATH "${LIGHTGBM_PREFIX}/lib/lib_lightgbm.a")
add_definitions(-DMM_MALLOC)

if (NOT EXISTS ${LIGHTGBM_INCLUDE_DIRECTORY})
file(MAKE_DIRECTORY ${LIGHTGBM_INCLUDE_DIRECTORY})
endif()
if (NOT EXISTS ${LIGHTGBM_LIB_PATH})
file(WRITE ${LIGHTGBM_LIB_PATH} "")
endif()
set_target_properties(lib_lightgbm PROPERTIES
  IMPORTED_LOCATION ${LIGHTGBM_LIB_PATH}
  INTERFACE_INCLUDE_DIRECTORIES ${LIGHTGBM_INCLUDE_DIRECTORY}
)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(vendor/ethz)
include_directories(${LIGHTGBM_INCLUDE_DIRECTORY})

set(SOURCES common.cpp scale.cpp point_io.cpp randomforest.cpp gbm.cpp features.cpp color.cpp labels.cpp)

add_library(pcc OBJECT ${SOURCES})
add_dependencies(pcc lightgbm)

add_executable(pctrain pctrain.cpp)
add_executable(pcclassify pcclassify.cpp)

target_link_libraries(pcc Eigen3::Eigen OpenMP::OpenMP_CXX Boost::iostreams Boost::serialization lib_lightgbm)
target_link_libraries(pctrain pcc)
target_link_libraries(pcclassify pcc)

install(TARGETS pctrain RUNTIME DESTINATION bin)
install(TARGETS pcclassify RUNTIME DESTINATION bin)
