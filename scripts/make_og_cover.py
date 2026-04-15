"""Generate a 1200x630 OG cover image from computed eigenfunctions."""
import os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "public" / "files" / "eigenfunctions" / "img"
OUT = ROOT / "public" / "files" / "eigenfunctions" / "og-cover.jpg"

# Pick a set of ready JPGs (prefer Dirichlet higher modes for visual interest)
candidate_ids = [
    "disk", "square", "l-shape", "cardioid", "reuleaux-triangle", "trapezoid-iso",
    "regpoly-8", "isotri-30", "stadium-1", "robnik-0.3", "thintri-0.2", "annulus-0.5",
    "ellipse-3", "sector-1.0471975511965976", "pacman-1.5707963267948966",
    "rhombus-30", "dumbbell-0.1", "sinai-0.3", "30-60-90-triangle", "right-isoceles-triangle",
]
picks = []
for d in candidate_ids:
    for k in (5, 4, 6, 3, 2, 1):
        p = IMG / f"{d}_dirichlet_{k}.jpg"
        if p.exists():
            picks.append(p); break
    if len(picks) >= 12: break

fig, axes = plt.subplots(3, 6, figsize=(12, 6.3), dpi=100, facecolor="white")
for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

# Pad picks up to 18 with repeats
while len(picks) < 18 and picks:
    picks.append(picks[len(picks) % len(picks) if picks else 0])

for ax, p in zip(axes.flat, picks):
    img = mpimg.imread(str(p))
    ax.imshow(img)

fig.suptitle("Eigenfunction database (β)", fontsize=22, y=0.97)
fig.text(0.5, 0.04,
         "Dirichlet / Neumann / Robin / Steklov Laplacian  ·  ryendo.github.io",
         ha="center", fontsize=12, color="#444")
fig.tight_layout(rect=(0, 0.05, 1, 0.94))
fig.savefig(OUT, dpi=100, bbox_inches="tight", pad_inches=0.1, facecolor="white")
print("wrote", OUT)
