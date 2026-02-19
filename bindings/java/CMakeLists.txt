cmake_minimum_required(VERSION 3.16)
project(ultrafast_secp256k1_jni C)

find_package(JNI REQUIRED)

# Generate JNI header from Java class
# javac -h jni/ src/main/java/com/ultrafast/secp256k1/Secp256k1.java

add_library(ultrafast_secp256k1_jni SHARED jni/secp256k1_jni.c)
target_include_directories(ultrafast_secp256k1_jni PRIVATE
    ${JNI_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../c_api
    ${CMAKE_CURRENT_SOURCE_DIR}/jni
)
target_link_libraries(ultrafast_secp256k1_jni PRIVATE ultrafast_secp256k1)

# Install
install(TARGETS ultrafast_secp256k1_jni
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)
