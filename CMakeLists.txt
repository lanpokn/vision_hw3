cmake_minimum_required(VERSION 2.8)
set(CMAKE_BUILD_TYPE "Release")

project(steroVision)
set(CMAKE_BUILD_TYPE "Debug")
#设置使用C++11编译
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#设定可执行二进制文件的目录
SET( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) 
	
#设定存放编译出来的库文件的目录
SET( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib) 
#并且把该目录设为连接目录
LINK_DIRECTORIES( ${PROJECT_SOURCE_DIR}/lib)

# find_package(Pangolin REQUIRED)
# include_directories(${Pangolin_INCLUDE_DIRS})

include_directories("/usr/include/eigen3")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# 增加PCL库的依赖
FIND_PACKAGE( PCL REQUIRED COMPONENTS common io )
ADD_DEFINITIONS( ${PCL_DEFINITIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS}  )
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )
# list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

#add library
INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/include)

# #add executable

# add_executable(pointcloud src/pointcloud.cpp)
# target_link_libraries(pointcloud ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

# add_executable(test01 src/test01.cpp)
# target_link_libraries(test01 ${OpenCV_LIBS})

# add_executable(sgbm src/sgbm.cpp)
# target_link_libraries(sgbm ${OpenCV_LIBS})

# ADD_EXECUTABLE( generate_pointcloud generatePointCloud.cpp )
# TARGET_LINK_LIBRARIES( generate_pointcloud 
#     ${OpenCV_LIBS} 
#     ${PCL_LIBRARIES} )

ADD_LIBRARY( slambase src/slamBase.cpp )
TARGET_LINK_LIBRARIES( slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )

ADD_EXECUTABLE( detectFeatures src/detectFeatures.cpp )
TARGET_LINK_LIBRARIES( detectFeatures 
    slambase
    ${OpenCV_LIBS} 
    ${PCL_LIBRARIES} )

