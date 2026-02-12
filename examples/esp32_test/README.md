# ESP32 secp256k1 Test

This example tests portable secp256k1 field arithmetic on ESP32.

## Requirements

- ESP-IDF v5.4+ installed
- ESP32 or ESP32-C3/C6 board

## Build & Flash

### Option 1: CLion with ESP-IDF plugin

1. Open this folder in CLion
2. Configure ESP-IDF path in Settings
3. Select target (esp32 / esp32c3 / esp32s3)
4. Build and Flash

### Option 2: Command Line

```bash
# Set up ESP-IDF environment
# Windows:
C:\Espressif\frameworks\esp-idf-v5.5.1\export.bat

# Linux/Mac:
. ~/esp/esp-idf/export.sh

# Build
idf.py build

# Flash (replace COM3 with your port)
idf.py -p COM3 flash

# Monitor output
idf.py -p COM3 monitor
```

## Expected Output

```
I (xxx) secp256k1: ╔══════════════════════════════════════════════════════════╗
I (xxx) secp256k1: ║   UltrafastSecp256k1 - ESP32 Benchmark                   ║
I (xxx) secp256k1: ╚══════════════════════════════════════════════════════════╝
I (xxx) secp256k1: 
I (xxx) secp256k1: === Benchmark Results ===
I (xxx) secp256k1: Field Mul: XXXX ns/op (10000 iterations)
I (xxx) secp256k1: Field Sqr: XXXX ns/op (10000 iterations)
I (xxx) secp256k1: Field Add: XXX ns/op (10000 iterations)
```

## Notes

- This uses portable C++ code (no assembly)
- ESP32 is 32-bit, much slower than x86-64/RISC-V64
- For best embedded performance, use ESP32-C3/C6 (RISC-V)

