cmake_minimum_required(VERSION 3.4.1)
project(SapientReactNative)

set (CMAKE_VERBOSE_MAKEFILE ON)

# The module impl + JSI bindings are uniffi-bindgen-react-native output
# (the scaffold's demo impl was removed).
add_library(
    openhorizon-sapient-react-native    STATIC
    ../cpp/openhorizon-sapient-react-native.cpp
    ../cpp/generated/sapient_ffi.cpp
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
