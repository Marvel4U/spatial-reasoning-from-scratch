import numpy as np, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
z = np.load("data/amsterdam/de_pijp/crops/v4/district_labels_1m.npz"); building = z["surface"] == 3
s = np.load(sys.argv[1]); out = sys.argv[2]
N = len(s["origin"]); cols = 10; rows = (N + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.75 * rows))
for k, ax in enumerate(axes.ravel()):
    if k >= N: ax.axis("off"); continue
    dx, dy = s["origin"][k]; m = building[dy:dy + 256, dx:dx + 256]
    ax.imshow(m, cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
    a, b = s["markers"][k]; nv = s["n_verts"][k]; p = s["poly"][k, :nv]
    ax.plot([a[0], b[0]], [a[1], b[1]], "c--", lw=0.8); ax.plot(p[:, 0], p[:, 1], "r-", lw=1.3)
    ax.plot(*a, "go", ms=4); ax.plot(*b, "bo", ms=4)
    ax.set_title(f"#{k}  {s['straight_px'][k]:.0f} m  x{s['detour_ratio'][k]:.2f}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"{sys.argv[1]}: {N} samples, cyan = straight segment (c4a), red = shortest path around buildings (c4b)", fontsize=11)
fig.tight_layout(); fig.savefig(out, dpi=100); print("saved", out)
