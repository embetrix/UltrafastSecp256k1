// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "UltrafastSecp256k1",
    platforms: [.macOS(.v12), .iOS(.v15)],
    products: [
        .library(name: "UltrafastSecp256k1", targets: ["UltrafastSecp256k1"]),
    ],
    targets: [
        .systemLibrary(
            name: "CUltrafastSecp256k1",
            path: "Sources/CUltrafastSecp256k1",
            pkgConfig: "ultrafast_secp256k1",
            providers: [
                .brew(["ultrafast_secp256k1"]),
            ]
        ),
        .target(
            name: "UltrafastSecp256k1",
            dependencies: ["CUltrafastSecp256k1"],
            path: "Sources/UltrafastSecp256k1"
        ),
    ]
)
