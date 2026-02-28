window.BENCHMARK_DATA = {
  "lastUpdate": 1772310204848,
  "repoUrl": "https://github.com/embetrix/UltrafastSecp256k1",
  "entries": {
    "UltrafastSecp256k1 Performance": [
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "ddc9dcf1e1e5e4c7631e42e01aeacd4b4a262ac2",
          "message": "Merge dev: BIP-32 TV1-TV5 vectors, nightly workflows, public key decompression fix",
          "timestamp": "2026-02-24T07:48:30+04:00",
          "tree_id": "f34382d237b7a6d969ce851b1a111a5f92a325db",
          "url": "https://github.com/embetrix/UltrafastSecp256k1/commit/ddc9dcf1e1e5e4c7631e42e01aeacd4b4a262ac2"
        },
        "date": 1771917638552,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 27,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 22,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 3,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 3,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 281,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 17000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 131,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "Vano Chkheidze",
            "username": "shrec"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "90477f4db2cc7c16d439d27b2de6d070f87ba196",
          "message": "ci: make codecov/patch informational (non-blocking) (#69)\n\ncodecov/patch reports 0.00% on diffs that touch only non-source\nfiles (YAML, config) or excluded directories, since there are\nzero coverable lines. This blocks the status check even though\nthe project-level coverage gate still protects overall health.\n\nSet informational: true so codecov/patch always reports a\nneutral status instead of failure.",
          "timestamp": "2026-02-28T12:53:18+04:00",
          "tree_id": "98c7df4fcc3e003ac84cafce260ad7446af9ffe2",
          "url": "https://github.com/embetrix/UltrafastSecp256k1/commit/90477f4db2cc7c16d439d27b2de6d070f87ba196"
        },
        "date": 1772310203801,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 27,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 22,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 3,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 3,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 282,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 131,
            "unit": "ns"
          }
        ]
      }
    ]
  }
}