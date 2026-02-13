/**
 * UltrafastSecp256k1 - ESP32 Integration Test
 *
 * Testing real secp256k1 library on ESP32 using the library's Selftest
 */

#include <stdio.h>
#include "esp_chip_info.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// Include real secp256k1 library
#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/selftest.hpp"

using namespace secp256k1::fast;

// Helper to get chip model name
static const char* get_chip_model_name(esp_chip_model_t model) {
    switch (model) {
        case CHIP_ESP32:   return "ESP32";
        case CHIP_ESP32S2: return "ESP32-S2";
        case CHIP_ESP32S3: return "ESP32-S3";
        case CHIP_ESP32C3: return "ESP32-C3";
        case CHIP_ESP32C2: return "ESP32-C2";
        case CHIP_ESP32C6: return "ESP32-C6";
        case CHIP_ESP32H2: return "ESP32-H2";
        default:           return "Unknown";
    }
}

extern "C" void app_main() {
    vTaskDelay(pdMS_TO_TICKS(1000));

    printf("\n");
    printf("============================================================\n");
    printf("   UltrafastSecp256k1 - ESP32 Library Test\n");
    printf("============================================================\n");
    printf("\n");

    // Platform information
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);

    printf("Platform Information:\n");
    printf("  Chip Model:   %s\n", get_chip_model_name(chip_info.model));
    printf("  Cores:        %d\n", chip_info.cores);
    printf("  Revision:     %d.%d\n", chip_info.revision / 100, chip_info.revision % 100);
    printf("  Free Heap:    %lu bytes\n", (unsigned long)esp_get_free_heap_size());
    printf("  Build:        32-bit Portable (no __int128)\n");
    printf("\n");

    // Run the real library self-test
    printf("Running SECP256K1 Library Self-Test...\n");
    printf("(This may take a few seconds on ESP32)\n\n");

    bool test_passed = Selftest(true);  // verbose = true

    printf("\n");
    if (test_passed) {
        printf("============================================================\n");
        printf("   SUCCESS: All library tests passed on ESP32!\n");
        printf("============================================================\n");
    } else {
        printf("============================================================\n");
        printf("   FAILURE: Some tests failed. Check output above.\n");
        printf("============================================================\n");
    }

    // Simple performance benchmark
    printf("\n");
    printf("==============================================\n");
    printf("  Basic Performance Benchmark\n");
    printf("==============================================\n");

    const int iterations = 1000;
    FieldElement a = FieldElement::from_limbs({0x12345678, 0xABCDEF01, 0x11223344, 0x55667788});
    FieldElement b = FieldElement::from_limbs({0x87654321, 0xFEDCBA98, 0x99AABBCC, 0xDDEEFF00});

    // Field Multiplication
    {
        int64_t start = esp_timer_get_time();
        FieldElement result = a;
        for (int i = 0; i < iterations; i++) {
            result = result * b;
        }
        int64_t elapsed = esp_timer_get_time() - start;
        printf("  Field Mul:    %5lld ns/op\n", (elapsed * 1000) / iterations);
        (void)result; // Prevent optimization
    }

    // Field Squaring
    {
        int64_t start = esp_timer_get_time();
        FieldElement result = a;
        for (int i = 0; i < iterations; i++) {
            result = result.square();
        }
        int64_t elapsed = esp_timer_get_time() - start;
        printf("  Field Square: %5lld ns/op\n", (elapsed * 1000) / iterations);
        (void)result;
    }

    // Field Addition
    {
        int64_t start = esp_timer_get_time();
        FieldElement result = a;
        for (int i = 0; i < iterations; i++) {
            result = result + b;
        }
        int64_t elapsed = esp_timer_get_time() - start;
        printf("  Field Add:    %5lld ns/op\n", (elapsed * 1000) / iterations);
        (void)result;
    }

    // Scalar Multiplication (if tests passed)
    if (test_passed) {
        printf("\n  Scalar Mul benchmark (10 iterations):\n");
        int64_t start = esp_timer_get_time();
        Point G = Point::generator();
        Point result = G;
        Scalar k = Scalar::from_hex("0000000000000000000000000000000000000000000000000000000000000005");
        for (int i = 0; i < 10; i++) {
            result = G.scalar_mul(k);
        }
        int64_t elapsed = esp_timer_get_time() - start;
        printf("  Scalar*G:     %5lld us/op\n", elapsed / 10);
        (void)result;
    }

    printf("\n");
    printf("============================================================\n");
    printf("   UltrafastSecp256k1 on ESP32 - Test Complete\n");
    printf("============================================================\n");

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
