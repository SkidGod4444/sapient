cmake_minimum_required(VERSION 3.4.1)
project(SapientReactNative)

set (CMAKE_VERBOSE_MAKEFILE ON)

add_library(
    openhorizon-sapient-react-native    STATIC
    ../cpp/SapientReactNativeImpl.cpp
)

set_target_properties(
    openhorizon-sapient-react-native PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_include_directories(
    openhorizon-sapient-react-native    PUBLIC
    ../cpp
)

target_link_libraries(
    openhorizon-sapient-react-native    jsi
    reactnative
    react_codegen_SapientReactNativeSpec
)
