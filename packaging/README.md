# Linux Distribution Packaging

This directory contains packaging files for building native packages
on various Linux distributions.

## Debian / Ubuntu (.deb)

```bash
# Install build dependencies
sudo apt install debhelper cmake ninja-build g++ pkg-config

# Build package from source tarball
dpkg-buildpackage -us -uc -b
# — or use CPack —
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DSECP256K1_BUILD_SHARED=ON -DSECP256K1_INSTALL=ON
cmake --build build
cd build && cpack -G DEB
```

Produces:
- `libsecp256k1-fast3_<ver>_<arch>.deb` — shared library
- `libsecp256k1-fast-dev_<ver>_<arch>.deb` — headers + static lib + cmake/pkgconfig

## Fedora / RHEL / CentOS (.rpm)

```bash
# Install build dependencies
sudo dnf install cmake ninja-build gcc-c++ rpm-build

# Build RPM from spec
rpmbuild -ba packaging/rpm/libsecp256k1-fast.spec
# — or use CPack —
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DSECP256K1_BUILD_SHARED=ON -DSECP256K1_INSTALL=ON
cmake --build build
cd build && cpack -G RPM
```

## Arch Linux (AUR)

```bash
# From the packaging/arch/ directory:
cd packaging/arch
makepkg -si
```

The `PKGBUILD` downloads the source tarball, builds with CMake+Ninja,
runs tests, and installs to `/usr`.

## Generic install (any distro)

```bash
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DSECP256K1_BUILD_SHARED=ON \
    -DSECP256K1_INSTALL=ON \
    -DSECP256K1_INSTALL_PKGCONFIG=ON \
    -DSECP256K1_USE_ASM=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
sudo cmake --install build
sudo ldconfig
```

After install, applications can find the library via:
- **pkg-config**: `pkg-config --cflags --libs secp256k1-fast`
- **CMake**: `find_package(secp256k1-fast 3 REQUIRED COMPONENTS CPU)`

## Package naming convention

| Distro | Runtime | Development |
|--------|---------|-------------|
| Debian/Ubuntu | `libsecp256k1-fast3` | `libsecp256k1-fast-dev` |
| Fedora/RHEL | `libsecp256k1-fast` | `libsecp256k1-fast-devel` |
| Arch | `libsecp256k1-fast` | (included in main package) |
