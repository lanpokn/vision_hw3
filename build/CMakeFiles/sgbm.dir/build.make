# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lanpokn/Documents/2021/robot_vision/hw3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lanpokn/Documents/2021/robot_vision/hw3/build

# Include any dependencies generated for this target.
include CMakeFiles/sgbm.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sgbm.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sgbm.dir/flags.make

CMakeFiles/sgbm.dir/src/sgbm.cpp.o: CMakeFiles/sgbm.dir/flags.make
CMakeFiles/sgbm.dir/src/sgbm.cpp.o: ../src/sgbm.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lanpokn/Documents/2021/robot_vision/hw3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sgbm.dir/src/sgbm.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sgbm.dir/src/sgbm.cpp.o -c /home/lanpokn/Documents/2021/robot_vision/hw3/src/sgbm.cpp

CMakeFiles/sgbm.dir/src/sgbm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sgbm.dir/src/sgbm.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lanpokn/Documents/2021/robot_vision/hw3/src/sgbm.cpp > CMakeFiles/sgbm.dir/src/sgbm.cpp.i

CMakeFiles/sgbm.dir/src/sgbm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sgbm.dir/src/sgbm.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lanpokn/Documents/2021/robot_vision/hw3/src/sgbm.cpp -o CMakeFiles/sgbm.dir/src/sgbm.cpp.s

CMakeFiles/sgbm.dir/src/sgbm.cpp.o.requires:

.PHONY : CMakeFiles/sgbm.dir/src/sgbm.cpp.o.requires

CMakeFiles/sgbm.dir/src/sgbm.cpp.o.provides: CMakeFiles/sgbm.dir/src/sgbm.cpp.o.requires
	$(MAKE) -f CMakeFiles/sgbm.dir/build.make CMakeFiles/sgbm.dir/src/sgbm.cpp.o.provides.build
.PHONY : CMakeFiles/sgbm.dir/src/sgbm.cpp.o.provides

CMakeFiles/sgbm.dir/src/sgbm.cpp.o.provides.build: CMakeFiles/sgbm.dir/src/sgbm.cpp.o


# Object files for target sgbm
sgbm_OBJECTS = \
"CMakeFiles/sgbm.dir/src/sgbm.cpp.o"

# External object files for target sgbm
sgbm_EXTERNAL_OBJECTS =

../bin/sgbm: CMakeFiles/sgbm.dir/src/sgbm.cpp.o
../bin/sgbm: CMakeFiles/sgbm.dir/build.make
../bin/sgbm: /usr/local/lib/libopencv_stitching.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_superres.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_videostab.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_aruco.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_bgsegm.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_bioinspired.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_ccalib.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_dpm.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_face.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_freetype.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_fuzzy.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_hdf.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_hfs.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_img_hash.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_line_descriptor.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_optflow.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_reg.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_rgbd.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_saliency.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_stereo.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_structured_light.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_surface_matching.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_tracking.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_xfeatures2d.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_ximgproc.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_xobjdetect.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_xphoto.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_shape.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_highgui.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_videoio.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_viz.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_video.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_datasets.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_plot.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_text.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_dnn.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_ml.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_imgcodecs.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_objdetect.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_calib3d.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_features2d.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_flann.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_photo.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_imgproc.so.3.4.11
../bin/sgbm: /usr/local/lib/libopencv_core.so.3.4.11
../bin/sgbm: CMakeFiles/sgbm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lanpokn/Documents/2021/robot_vision/hw3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/sgbm"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sgbm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sgbm.dir/build: ../bin/sgbm

.PHONY : CMakeFiles/sgbm.dir/build

CMakeFiles/sgbm.dir/requires: CMakeFiles/sgbm.dir/src/sgbm.cpp.o.requires

.PHONY : CMakeFiles/sgbm.dir/requires

CMakeFiles/sgbm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sgbm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sgbm.dir/clean

CMakeFiles/sgbm.dir/depend:
	cd /home/lanpokn/Documents/2021/robot_vision/hw3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lanpokn/Documents/2021/robot_vision/hw3 /home/lanpokn/Documents/2021/robot_vision/hw3 /home/lanpokn/Documents/2021/robot_vision/hw3/build /home/lanpokn/Documents/2021/robot_vision/hw3/build /home/lanpokn/Documents/2021/robot_vision/hw3/build/CMakeFiles/sgbm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sgbm.dir/depend
