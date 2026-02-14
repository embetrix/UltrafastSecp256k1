# CLion + ESP-IDF სრული Development გარემო

## წინაპირობები

1. **ESP-IDF 5.5.1** დაინსტალირებული (`C:\Espressif\`)
2. **CLion 2024.x** ან უფრო ახალი

---

## ნაბიჯი 1: CLion ESP-IDF პლაგინის დაყენება

1. **File → Settings → Plugins**
2. Marketplace-ში მოძებნე: **"ESP-IDF"**
3. დააინსტალირე და გადატვირთე CLion

---

## ნაბიჯი 2: ESP-IDF კონფიგურაცია CLion-ში

1. **File → Settings → Languages & Frameworks → ESP-IDF**
2. შეავსე:
   - **ESP-IDF Path:** `C:\Espressif\frameworks\esp-idf-v5.5.1`
   - **Python:** `C:\Espressif\python_env\idf5.5_py3.11_env\Scripts\python.exe`
   - **Tools Path:** `C:\Espressif`

3. **Apply** და **OK**

---

## ნაბიჯი 3: პროექტის გახსნა

1. **File → Open**
2. აირჩიე: `D:\Dev\Secp256K1\libs\UltrafastSecp256k1\examples\esp32_test`
3. CLion იპოვის CMakeLists.txt და დაიწყებს კონფიგურაციას

---

## ნაბიჯი 4: Target Device კონფიგურაცია

1. **Run → Edit Configurations**
2. დააჭირე **+** → **ESP-IDF**
3. შეავსე:
   - **Name:** `ESP32-S3 Flash & Monitor`
   - **Target:** `esp32s3`
   - **Serial Port:** `COM3`
   - **Flash:** ✓
   - **Monitor:** ✓
   - **Baud rate:** `115200`

---

## გამოყენება

| მოქმედება | როგორ |
|-----------|-------|
| **Build** | `Ctrl+F9` ან 🔨 ღილაკი |
| **Flash** | აირჩიე configuration → `Shift+F10` |
| **Monitor** | ავტომატურად იხსნება flash-ის შემდეგ |
| **Debug** | `Shift+F9` (JTAG საჭიროა) |

---

## Serial Monitor CLion-ში

1. **View → Tool Windows → Serial Monitor**
2. Port: `COM3`
3. Baud: `115200`
4. **Connect**

---

## Troubleshooting

### "IDF_PATH not found"
- Settings → Languages & Frameworks → ESP-IDF → შეამოწმე paths

### "Cannot open COM port"
- დახურე სხვა პროგრამები (Arduino IDE, PuTTY)
- შეამოწმე Device Manager-ში COM პორტი

### Build errors
- Terminal-ში გაუშვი: `idf.py fullclean`
- თავიდან დაბილდე

---

## ალტერნატივა: ESP-IDF CMD + CLion

თუ პლაგინი არ მუშაობს:

1. გახსენი **ESP-IDF 5.5.1 PowerShell** (Start მენიუდან)
2. გაუშვი:
   ```cmd
   cd D:\Dev\Secp256K1\libs\UltrafastSecp256k1\examples\esp32_test
   clion .
   ```

ეს გახსნის CLion-ს სწორი ESP-IDF გარემოთი.
