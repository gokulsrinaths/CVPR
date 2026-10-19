import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ex = json.load(open("teaser_example.json"))
BLUE, ORANGE, GRAY, DARK = "#0072B2", "#D55E00", "#8A8F98", "#222222"
fig = plt.figure(figsize=(7.4, 3.25))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 44)
ax.axis("off")

# left card: the question and its options
ax.add_patch(FancyBboxPatch((1, 11.5), 60, 31.5, boxstyle="round,pad=0.2,rounding_size=1.2", fc="#FFFFFF", ec=GRAY, lw=0.9))
ax.text(2.5, 40.6, "A held-out Med-CMR question (text only; the scan is not shown)", fontsize=7.4, weight="bold", color=DARK, va="center")
q = textwrap.fill(ex["question"], 74)
ax.text(2.5, 36.0, q, fontsize=6.6, color=DARK, va="center", linespacing=1.15)
maxlen = max(len(o) for o in ex["options"])
y0 = 29.2
for k, o in enumerate(ex["options"]):
    y = y0 - k * 3.6
    correct = "ABCDE"[k] == ex["gold"]
    if correct:
        ax.add_patch(Rectangle((1.7, y - 1.6), 58.6, 3.3, fc="#E8F2FA", ec=BLUE, lw=1.0))
    ax.text(2.6, y, "ABCDE"[k], fontsize=6.8, weight="bold", color=BLUE if correct else DARK, va="center")
    ax.text(5.0, y, textwrap.shorten(o, 56, placeholder="..."), fontsize=6.3, color=DARK, va="center")
    bw = 8.5 * len(o) / maxlen
    ax.add_patch(Rectangle((49.2, y - 0.8), bw, 1.6, fc=BLUE if correct else "#C9CDD3", ec="none"))
    ax.text(49.2 + bw + 0.5, y, f"{len(o)}", fontsize=5.4, color=DARK, va="center")
ax.text(49.2, 32.2, "characters", fontsize=5.4, color=GRAY, va="center")
ax.text(33.5, y0 - 4 * 3.6, "correct, and the shortest", fontsize=5.8, color=BLUE, va="center", ha="left", style="italic")

# right card: the model's answers
ax.add_patch(FancyBboxPatch((64, 11.5), 35, 31.5, boxstyle="round,pad=0.2,rounding_size=1.2", fc="#FFFFFF", ec=GRAY, lw=0.9))
ax.text(65.5, 40.6, "Answer of a fine-tuned Qwen-3B", fontsize=7.4, weight="bold", color=DARK, va="center")
for i, (arm, lab, col, fc) in enumerate((("base", "true image", GRAY, "#D9DCE1"), ("black", "black image", DARK, "#000000"))):
    yy = 33.2 - i * 10.8
    ax.add_patch(Rectangle((66, yy - 4.2), 8.5, 8.4, fc=fc, ec=GRAY, lw=0.6))
    if arm == "base":
        ax.text(70.25, yy, "scan\n(not shown)", fontsize=4.6, color="#555555", ha="center", va="center")
    ax.text(76.5, yy + 2.2, lab, fontsize=6.4, color=DARK, va="center")
    p = ex["pred"][arm]
    ax.text(76.5, yy - 1.2, f"answer {p['pred']}", fontsize=8.0, weight="bold", color=BLUE, va="center")
    ax.text(76.5, yy - 3.6, f"confidence {100 * p['top_prob']:.1f}%", fontsize=5.8, color=DARK, va="center")
ax.text(81.5, 13.4, "same answer, same confidence", fontsize=6.0, color=ORANGE, va="center", ha="center", style="italic")

# stat strip
tiles = [("91.9%", "of Med-CMR questions: the correct\noption is the shortest"), ("96%", "overall\naccuracy"), ("68-73%", "where the shortest option is wrong\n(our models; released system 84%)")]
xs = [1, 36, 54]
ws = [33, 16, 45]
offs = [14.0, 9.2, 16.0]
for (big, small), x, w, off in zip(tiles, xs, ws, offs):
    ax.add_patch(FancyBboxPatch((x, 0.8), w, 8.4, boxstyle="round,pad=0.15,rounding_size=1.0", fc="#F4F5F7", ec="none"))
    ax.text(x + 1.5, 5.0, big, fontsize=11.5, weight="bold", color=ORANGE if big.startswith("68") else BLUE, va="center")
    ax.text(x + off, 5.0, small, fontsize=5.8, color=DARK, va="center", linespacing=1.1)
fig.savefig("paper_v6/fig/f1_teaser.png", dpi=320)
print("saved")
