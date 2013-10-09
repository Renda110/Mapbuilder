cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()
rosbuild_genmsg()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

if (CMAKE_BUILD_TOOL MATCHES "make")
    add_definitions( -Wno-write-strings -Wno-reorder -Wno-unused-variable -Wno-unused-function -Wno-deprecated-declarations -Wno-unused-but-set-variable -Wno-empty-body)
endif (CMAKE_BUILD_TOOL MATCHES "make")

rosbuild_add_boost_directories()

#include_directories(/usr/X11R6/include/ /usr/local/include/)

FIND_PACKAGE( OpenGL REQUIRED )

FIND_PACKAGE( OpenCV REQUIRED core imgproc highgui legacy calib3d features2d)

FIND_PACKAGE( MRPT REQUIRED base slam obs vision opengl maps bayes gui)
#remove -dbg from /usr/share/mrpt/MRPTConfig.cmake

#include_directories(lib/CSparse/Include)
#include_directories(lib/CSparse/Lib)
include_directories(/usr/local/include/lcm)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(/home/enda/fuerte_workspace/sandbox/mapbuilder/include)

#LIBS+= -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_legacy -lopencv_calib3d -lopencv_features2d
#LIBS+= -lglut -lGL -lGLU -lGLEW
#LIBS+= -lcsparse -luuid -llcm

#link_directories(/usr/X11R6/lib)
SET(EXTRA_LIBS ${MRPT_LIBS} uuid GL X11 GLU glut lcm )

set(mapbuilder_SOURCES
    src/Mapping.cpp
    src/Utm.cpp
    src/csparse.cpp
    src/simplelog.cpp
    src/spa2d.cpp
    src/MapBuilder.cpp
    #src/displaywidget.cpp
    #src/optimizerthread.cpp
    #src/renderthread.cpp
    #src/displaythread.cpp
    #src/logplaythread.cpp
    #src/matcherthread.cpp
    src/main.cpp
)

rosbuild_add_executable(mapbuilder
    ${mapbuilder_SOURCES})
rosbuild_add_boost_directories()
target_link_libraries(mapbuilder  ${MRPT_LIBS} ${OpenCV_LIBS} uuid z X11 m GL GLU glut lcm csparse GLEW )

#rosbuild_add_executable(${PROJECT_NAME} src/main.cpp)
#target_link_libraries(${PROJECT_NAME} X11 m GL GLU glut lcm)
#target_link_libraries(opengl_test /usr/lib/libGL.so)
#target_link_libraries(opengl_test /usr/lib/libX11.a)
#target_link_libraries(magic /usr/lib/libX11.so)


