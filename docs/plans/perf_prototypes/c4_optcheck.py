"""Compare stored visibility-graph path lengths with a dense 16-connected grid Dijkstra on the same
building mask. Grid lengths overestimate the Euclidean optimum by at most ~2.8 %; a stored path
longer than the grid path means the visibility graph missed a corner."""
import numpy as np, sys
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
sys.path.insert(0, ".")
from spatial_data import c4_routes as R
z = np.load("data/amsterdam/de_pijp/crops/v4/district_labels_1m.npz"); building = z["surface"] == 3
s = np.load(sys.argv[1])
moves = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if (dx, dy) != (0, 0) and np.gcd(abs(dx), abs(dy)) == 1]
N = 256
def grid_len(mask, a, b):
    free = ~mask; idx = np.arange(N * N).reshape(N, N)
    rows, cols, w = [], [], []
    for dx, dy in moves:
        ys, xs = np.mgrid[0:N, 0:N]
        ok = (xs + dx >= 0) & (xs + dx < N) & (ys + dy >= 0) & (ys + dy < N)
        src = idx[ok]; dst = idx[(ys + dy)[ok], (xs + dx)[ok]]
        good = free.ravel()[src] & free.ravel()[dst]
        # knight moves must not cut a building corner: require the two intermediate pixels free too
        if abs(dx) + abs(dy) == 3:
            mx, my = xs[ok] + np.sign(dx) * (abs(dx) == 2), ys[ok] + np.sign(dy) * (abs(dy) == 2)
            mx2, my2 = xs[ok] + np.sign(dx), ys[ok] + np.sign(dy)
            good &= free[my, mx] & free[my2, mx2]
        elif abs(dx) + abs(dy) == 2:
            good &= free[ys[ok], xs[ok] + dx] & free[ys[ok] + dy, xs[ok]]
        rows.append(src[good]); cols.append(dst[good]); w.append(np.full(good.sum(), np.hypot(dx, dy)))
    G = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(N * N, N * N)).tocsr()
    ia = int(a[1]) * N + int(a[0]); ib = int(b[1]) * N + int(b[0])
    d = dijkstra(G, indices=ia, min_only=True)
    return d[ib]
worst = []
for i in range(len(s["origin"])):
    dx, dy = s["origin"][i]; m = building[dy:dy + N, dx:dx + N]
    a, b = s["markers"][i]
    gl = grid_len(m, a, b); vl = float(s["path_px"][i])
    worst.append((vl / gl, i, vl, gl, float(s["detour_ratio"][i])))
worst.sort(reverse=True)
print("stored/grid length ratio (>1.03 means the visibility path is NOT optimal):")
for r, i, vl, gl, dr in worst[:8]: print(f"  sample {i:3d}: stored {vl:6.1f}  grid {gl:6.1f}  ratio {r:.3f}  detour {dr:.2f}")
rs = np.array([w[0] for w in worst]); print(f"  n={len(rs)}  median {np.median(rs):.3f}  share >1.03: {(rs>1.03).mean():.2f}")
