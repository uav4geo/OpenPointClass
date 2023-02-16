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

# find_package(CGAL REQUIRED)
find_package(PDAL REQUIRED)
#find_package(Eigen REQUIRED)
find_package(TBB REQUIRED)
find_package(Boost COMPONENTS system serialization)

# add_definitions(-DCGAL_EIGEN3_ENABLED=1)
add_definitions(-DCGAL_LINKED_WITH_TBB=1)

include_directories(${PDAL_INCLUDE_DIR})
include_directories(${PDAL_INCLUDE_DIR}/pdal)
include_directories(vendor/CGAL)
include_directories(vendor/json)

set(LINK_LIBRARIES ${PDAL_LIBRARIES} TBB::tbb Boost::system Boost::serialization)
add_executable(pctrain pctrain.cpp)
add_executable(pcclassify pcclassify.cpp)

target_link_libraries(pctrain ${LINK_LIBRARIES})
target_link_libraries(pcclassify ${LINK_LIBRARIES})

install(TARGETS pctrain RUNTIME DESTINATION bin)
install(TARGETS pcclassify RUNTIME DESTINATION bin)
