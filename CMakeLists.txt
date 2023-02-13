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

#find_package(OpenMP REQUIRED)
find_package(CGAL REQUIRED)
find_package(PDAL REQUIRED)
find_package(Boost COMPONENTS system serialization)

include_directories(${PDAL_INCLUDE_DIR})
include_directories(${PDAL_INCLUDE_DIR}/pdal)

set(LINK_LIBRARIES CGAL::CGAL ${PDAL_LIBRARIES} Boost::system Boost::serialization)
add_executable(pctrain pctrain.cpp)

target_link_libraries(pctrain ${LINK_LIBRARIES})

# if(OpenMP_CXX_FOUND)
#     target_link_libraries(FPCFilter PUBLIC OpenMP::OpenMP_CXX)
# endif()


install(TARGETS pctrain RUNTIME DESTINATION bin)
