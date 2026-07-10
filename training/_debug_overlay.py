import sys
from pathlib import Path

import cv2
import numpy as np

d = Path(sys.argv[1])
name = sys.argv[2] if len(sys.argv) > 2 else "img_000000"
img = cv2.imread(str(d / "images" / f"{name}.jpg"))
h, w = img.shape[:2]
colors = {0: (0, 180, 0), 1: (0, 140, 255), 2: (255, 80, 0), 3: (0, 0, 220)}
for line in (d / "labels" / f"{name}.txt").read_text().splitlines():
    p = line.split()
    cls = int(p[0])
    pts = np.array(p[1:], dtype=float).reshape(4, 2)
    pts[:, 0] *= w
    pts[:, 1] *= h
    cv2.polylines(img, [pts.astype(int)], True, colors[cls], 2)
out = d / f"_debug_{name}.png"
cv2.imwrite(str(out), img)
print("saved", out, img.shape)
