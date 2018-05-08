import os 
import sys

model_dir = sys.argv[1]
ckpt = int(sys.argv[2])

min_dist = 1000000
ckpt_final = -1
for filename in os.listdir(model_dir):
    if filename.startswith("model.ckpt-") and filename.endswith(".index"):
        ckpt_cur = int(filename.split('.index')[0][11:])
        if abs(ckpt-ckpt_cur)<min_dist:
            ckpt_final = ckpt_cur
            min_dist = abs(ckpt-ckpt_cur)

print(ckpt_final)



