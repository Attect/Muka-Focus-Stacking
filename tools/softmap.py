"""Map where our output is soft relative to what the stack can actually deliver.

For every block of the frame we compare the edge acutance of our result against
the maximum, over all raw frames, of the same statistic in the same block. No
reference image is involved, so the answer is independent of whatever the
reference export did in post production.

Reports the worst blocks with their coordinates, and writes a heat map.
"""
import os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("STACK_DIR", "stack")
AF_X, AF_Y = 1288, 488
CV_X, CV_Y = 20, 13
S = 2          # downsample factor for speed
BLOCK = 64     # block size on the downsampled grid
files = sorted(f for f in os.listdir(D) if f.upper().startswith("DSC") and f.upper().endswith(".JPG"))

af_im = Image.open(os.path.join(D, "AF导出.png")).convert("RGB")
CW, CH = af_im.width, af_im.height


def acutance_map(a, q=0.97):
    """Local edge steepness per block: mean of the strongest gradients."""
    g = a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    gy, gx = np.gradient(g.astype(np.float32))
    m = np.hypot(gx, gy)
    h, w = m.shape[0] // BLOCK * BLOCK, m.shape[1] // BLOCK * BLOCK
    m = m[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK)
    return np.quantile(m, q, axis=(1, 3)), (h, w)


best = None
for i, f in enumerate(files):
    a = np.asarray(Image.open(os.path.join(D, f)).convert("RGB").crop(
        (AF_X + CV_X, AF_Y + CV_Y, AF_X + CV_X + CW, AF_Y + CV_Y + CH)), dtype=np.float32)
    a = a[::S, ::S]
    v, shape = acutance_map(a)
    best = v if best is None else np.maximum(best, v)
    if (i + 1) % 10 == 0:
        print("  %d/%d" % (i + 1, len(files)))

im = Image.open(r"B:\FocusMerge\out\final.png").convert("RGB")
o = np.asarray(im.crop((AF_X, AF_Y, AF_X + CW, AF_Y + CH)), dtype=np.float32)[::S, ::S]
our, _ = acutance_map(o)
af = np.asarray(af_im, dtype=np.float32)[::S, ::S]
afv, _ = acutance_map(af)

ratio = our / (best + 1e-6)
print("\nour acutance / achievable (max over frames):")
print("  mean %.3f  median %.3f  p10 %.3f" % (ratio.mean(), np.median(ratio), np.percentile(ratio, 10)))

flat = ratio.ravel()
order = np.argsort(flat)
nb = ratio.shape[1]
print("\nworst %d blocks:" % 20)
print("%-10s %-10s %8s %8s %8s" % ("ref-x", "ref-y", "ours/ach", "ours/AF", "ach"))
rows = []
for k in order[:400]:
    by, bx = divmod(int(k), nb)
    x, y = bx * BLOCK * S, by * BLOCK * S
    if any(abs(x - px) < 300 and abs(y - py) < 300 for px, py in rows):
        continue
    rows.append((x, y))
    print("%-10d %-10d %8.3f %8.3f %8.2f"
          % (x, y, ratio[by, bx], our[by, bx] / (afv[by, bx] + 1e-6), best[by, bx]))
    if len(rows) >= 12:
        break

# heat map of the ratio, stretched for visibility
hm = np.clip(ratio * 128, 0, 255).astype(np.uint8)
Image.fromarray(hm).resize((CW // S, CH // S)).save(r"B:\FocusMerge\out\softmap.png")
print("\nwrote out/softmap.png (dark = soft relative to what is achievable)")
