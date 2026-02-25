#!/usr/bin/env python3
"""Convert cppcheck XML output to SARIF 2.1.0 format.

Usage:  python3 xml2sarif.py [input.xml] [output.sarif]
Defaults: cppcheck-results.xml -> cppcheck-results.sarif
"""
import xml.etree.ElementTree as ET
import json
import sys


def convert(xml_path, sarif_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    results = []
    for error in root.iter("error"):
        for loc in error.findall("location"):
            results.append(
                {
                    "ruleId": error.get("id", "unknown"),
                    "level": (
                        "warning"
                        if error.get("severity") != "error"
                        else "error"
                    ),
                    "message": {"text": error.get("msg", "")},
                    "locations": [
                        {
                            "physicalLocation": {
                                "artifactLocation": {
                                    "uri": loc.get("file", "")
                                },
                                "region": {
                                    "startLine": int(loc.get("line", 1))
                                },
                            }
                        }
                    ],
                }
            )

    sarif = {
        "$schema": (
            "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/"
            "main/sarif-2.1/schema/sarif-schema-2.1.0.json"
        ),
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {"name": "Cppcheck", "version": "2.x"}
                },
                "results": results,
            }
        ],
    }
    with open(sarif_path, "w") as f:
        json.dump(sarif, f, indent=2)
    print(f"Converted {len(results)} findings to SARIF")


if __name__ == "__main__":
    xml_in = sys.argv[1] if len(sys.argv) > 1 else "cppcheck-results.xml"
    sarif_out = sys.argv[2] if len(sys.argv) > 2 else "cppcheck-results.sarif"
    convert(xml_in, sarif_out)
