window.BENCHMARK_DATA = {
  "lastUpdate": 1771582600455,
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
          "id": "cc20253ac3809fab0fa2fd1c475cc6252a81f4ab",
          "message": "docs: auto-inject version from VERSION.txt into Doxyfile\n\nPROJECT_NUMBER was hardcoded 3.0.0. Now uses 0.0.0-dev placeholder,\ndocs.yml injects actual version from VERSION.txt before doxygen runs.",
          "timestamp": "2026-02-20T04:53:24+04:00",
          "tree_id": "a31c851ccd5b4790105178063bb2e71b541eadfb",
          "url": "https://github.com/embetrix/UltrafastSecp256k1/commit/cc20253ac3809fab0fa2fd1c475cc6252a81f4ab"
        },
        "date": 1771582600055,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 256,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 147,
            "unit": "ns"
          }
        ]
      }
    ]
  }
}