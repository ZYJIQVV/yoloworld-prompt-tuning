import os
from tqdm import tqdm

lbl_rt = '/data/zyj-ll/data/Object365/yolo/val/labels'
lbls = os.listdir(lbl_rt)
cats = []
for lbl in tqdm(lbls):
    lbl_path = os.path.join(lbl_rt, lbl)
    with open(lbl_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        cats.append(int(line[0]))
cats = sorted(list(set(cats)))
print(cats)