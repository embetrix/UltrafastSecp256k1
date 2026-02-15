# Examples CMakeLists.txt

# ESP32 / STM32 examples are separate projects (ESP-IDF / STM32CubeMX)
# See examples/esp32_test/ and examples/stm32_test/

# ── Desktop example ──────────────────────────────────────────────────────────
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "xtensa|Xtensa" AND NOT SECP256K1_PLATFORM_STM32)
    add_executable(example_basic_usage basic_usage/main.cpp)
    target_link_libraries(example_basic_usage PRIVATE fastsecp256k1)
    message(STATUS "Examples: basic_usage enabled")
endif()
