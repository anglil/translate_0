import os
import sys

directory = sys.argv[1]
file_prefix = sys.argv[2]

best_ppl = 9e+99
best_file = None
for filename in os.listdir(directory):
    if filename.startswith(file_prefix):
        tmp = filename.split("_")[-1]
        ppl = ".".join(tmp.split(".")[:-1])
        ppl = float(ppl)
        if ppl < best_ppl:
            best_ppl = ppl
            best_file = filename
print(best_file)
