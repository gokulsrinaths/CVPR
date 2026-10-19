import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

d = json.load(open("a36_blind_summary.json"))["benchmarks"]
ORDER = [("Med-CMR", "Med-CMR"), ("OmniMedVQA", "OmniMedVQA"), ("PMC-VQA", "PMC-VQA"), ("A-OKVQA", "A-OKVQA")]
BLUE, ORANGE, DARK, GRAY = "#0072B2", "#D55E00", "#222222", "#8A8F98"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.4})
fig, ax = plt.subplots(figsize=(6.9, 2.4))
for i, (key, label) in enumerate(ORDER):
    y = len(ORDER) - 1 - i
    v = d[key]
    chance, ceil, black, true = v["chance"], v["B_opt"], v["ft_black"], v["ft_true"]
    lo = max(ceil, black)
    ax.plot([0, 100], [y, y], color="#E3E5E8", lw=0.8, zorder=1)
    if true > lo:
        ax.add_patch(plt.Rectangle((lo, y - 0.19), true - lo, 0.38, fc=BLUE, alpha=0.28, ec=BLUE, lw=0.8, zorder=2))
    ax.plot(chance, y, "o", ms=5.2, mfc="white", mec=GRAY, mew=1.1, zorder=4)
    ax.plot(ceil, y + 0.15, "D", ms=4.6, color=ORANGE, zorder=5)
    ax.plot(black, y - 0.15, "s", ms=4.4, color=DARK, zorder=5)
    ax.plot(true, y, "o", ms=5.4, color=BLUE, zorder=6)
    h = v["visual_headroom"]
    ax.text(101.5, y, f"{h + 1e-9:+.1f}".replace("-", "−"), va="center", ha="left", fontsize=8.2, weight="bold", color=BLUE if h > 1 else DARK)
ax.text(101.5, len(ORDER) - 0.38, "headroom\n(points)", va="bottom", ha="left", fontsize=6.6, color=DARK, linespacing=1.05)
ax.set_yticks(range(len(ORDER)))
ax.set_yticklabels([l for _, l in ORDER][::-1])
ax.set_xlim(0, 100)
ax.set_ylim(-0.6, len(ORDER) - 0.35)
ax.set_xlabel("accuracy (%)", fontsize=7.6)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", labelsize=7.2, width=0.6)
handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=GRAY, mew=1.1, ms=5, label="chance"),
           Line2D([], [], marker="D", ls="", color=ORANGE, ms=4.6, label="blind ceiling (option text only)"),
           Line2D([], [], marker="s", ls="", color=DARK, ms=4.4, label="black image"),
           Line2D([], [], marker="o", ls="", color=BLUE, ms=5.4, label="true image"),
           Patch(fc=BLUE, alpha=0.28, ec=BLUE, label="visual headroom")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.46, 1.17), ncol=5, frameon=False, fontsize=6.9, handletextpad=0.3, columnspacing=1.1)
fig.subplots_adjust(left=0.13, right=0.915, top=0.84, bottom=0.2)
fig.savefig("paper_v6/fig/f2_headroom.png", dpi=400)
print("saved")
