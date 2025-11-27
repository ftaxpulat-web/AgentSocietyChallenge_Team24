import json
from itertools import islice
from pathlib import Path

SRC = Path("data")   # your current processed dir
DST = Path("tiny_data")
DST.mkdir(exist_ok=True)

def sample_json_lines(src_file, dst_file, n_lines):
    with open(src_file, "r", encoding="utf-8") as fin, \
         open(dst_file, "w", encoding="utf-8") as fout:
        for line in islice(fin, n_lines):
            fout.write(line)

# e.g. 1000 users, 1000 items, 5000 reviews
sample_json_lines(SRC / "user.json",   DST / "user.json",   1000)
sample_json_lines(SRC / "item.json",   DST / "item.json",   1000)
sample_json_lines(SRC / "review.json", DST / "review.json", 5000)
