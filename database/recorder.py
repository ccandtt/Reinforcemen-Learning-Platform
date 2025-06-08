# rl_platform/database/recorder.py

import csv
import json
import os

class Recorder:
    def __init__(self, save_dir: str = "results"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.records = []

    def log(self, episode: int, reward: float, extra: dict = None):
        record = {"episode": episode, "reward": reward}
        if extra:
            record.update(extra)
        self.records.append(record)

    def save_csv(self, filename="records.csv"):
        if not self.records:
            print("No records to save.")
            return

        keys = self.records[0].keys()
        path = os.path.join(self.save_dir, filename)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.records)
        print(f"[Recorder] Saved CSV to {path}")

    def save_json(self, filename="records.json"):
        path = os.path.join(self.save_dir, filename)
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)
        print(f"[Recorder] Saved JSON to {path}")
