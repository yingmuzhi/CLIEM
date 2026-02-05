'''
Author: yingmuzhi cyxscj@126.com
Date: 2025-10-21 15:59:53
LastEditors: yingmuzhi cyxscj@126.com
LastEditTime: 2025-10-21 16:02:48
FilePath: /20251010_projection/code/test_d.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import csv
from typing import List, Dict


def add_offset_to_dy(csv_path: str, offset: float = 120.0) -> None:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames: List[str] = reader.fieldnames or []
        if "dY" not in fieldnames:
            raise KeyError("Column 'dy' not found in CSV header")
        rows: List[Dict[str, str]] = []
        for row in reader:
            value_str = row.get("dY", "")
            try:
                value_float = float(value_str)
            except ValueError:
                # Keep non-numeric values unchanged
                rows.append(row)
                continue
            row["dY"] = str(value_float + offset)
            rows.append(row)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    CSV_PATH = "/Volumes/T7/20251010_projection/src5/align/part1/Results.csv"
    add_offset_to_dy(CSV_PATH, 120.0)


