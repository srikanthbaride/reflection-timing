import csv, json, os, time
from typing import Dict, Any

class CSVLogger:
    def __init__(self, csv_path: str):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.csv_path = csv_path
        self.fieldnames = None

    def log(self, row: Dict[str, Any]):
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
            write_header = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames)
                if write_header:
                    w.writeheader()
                w.writerow(row)
        else:
            with open(self.csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames)
                w.writerow(row)
