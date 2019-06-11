# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
project(dewarp)
set(CMAKE_VERBOSE_MAKEFILE on)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nlopt-2.6.1)
add_library(lib_opencv SHARED IMPORTED)
add_library(lib_nlopt SHARED IMPORTED)
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libopencv_java3.so)
set_target_properties(lib_nlopt PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libnlopt.so)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
#set(SOURCES native-lib.cpp)
add_library( # Sets the name of the library.
        preprocess
        # Sets the library as a shared library.
        SHARED
        # Provides a relative path to your source file(s).
        ${CMAKE_CURRENT_SOURCE_DIR}/page_dewarp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/C_preprocess.cpp
        )

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

#find_library( # Sets the name of the path variable.
#        log-lib

# Specifies the name of the NDK library that
# you want CMake to locate.
#        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        preprocess
        lib_opencv
        lib_nlopt
        # Links the target library to the log library
        # included in the NDK.
        android
        log)