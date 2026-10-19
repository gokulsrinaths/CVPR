import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

D = json.load(open("phase_data.json"))
LABELS = {"blur32": "blur 32", "blur16": "blur 16", "blur8": "blur 8", "blur4": "blur 4", "blur2": "blur 2", "base": "intact"}
fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9), gridspec_kw={"wspace": 0.28})
norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=40)
for ax, name in zip(axes, ("PMC-VQA", "OmniMedVQA", "A-OKVQA")):
    d = D[name]
    doses, levels = d["doses"], d["levels"]
    M = np.array([[d["mean_g"][f"{x}|{lv}"] for x in doses] for lv in levels])
    V = np.array([[d["veto_all"][f"{x}|{lv}"] for x in doses] for lv in levels])
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="RdYlBu_r" if False else "viridis", norm=norm)
    for i in range(len(levels)):
        for j in range(len(doses)):
            ax.text(j, i, f"{abs(M[i, j]) if abs(M[i, j]) < 0.5 else M[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color="white" if M[i, j] < 22 else "black")
            if V[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="red", lw=1.6))
    ax.set_xticks(range(len(doses)))
    ax.set_xticklabels([f"{x:g}" for x in doses], fontsize=7)
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels([LABELS[l] for l in levels], fontsize=7)
    ax.set_xlabel("shortcut strength $d$", fontsize=8)
    ax.set_title(name, fontsize=9)
axes[0].set_ylabel("visual strength", fontsize=8)
cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.015)
cb.set_label("image benefit (points)", fontsize=7)
cb.ax.tick_params(labelsize=6)
fig.savefig("paper_v6/fig/f12_phase.png", dpi=300, bbox_inches="tight")
print("saved")
