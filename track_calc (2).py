#!/usr/bin/env python3
"""
Extract a track centerline from a top-down track image (e.g. the in-game track map)
and compute length, corners and an ordered point list.

Install:  pip install opencv-python numpy scipy scikit-image
Usage:    python track_calc.py track.png
          python track_calc.py track.png --real-length-m 9370        (scale from the length the game shows)
          python track_calc.py track.png --studs-per-pixel 2.5       (or set the scale yourself)

Outputs:  track.json (points in studs, x/z plane) and track_preview.png

v2: finds the track by color (works on the multi-color sector maps), ignores the legend,
    and follows the line with a direction-aware tracer, so gaps (start line, number
    circles) and places where the track crosses itself are handled.
"""
import argparse
import json

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

METERS_PER_STUD = 0.28  # commonly used Roblox conversion


# ---------------------------------------------------------------- mask
def find_panel(img):
    """Crop to the biggest dark rectangle (the map panel). Falls back to the whole image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = (gray < 60).astype(np.uint8) * 255
    n, _, st, _ = cv2.connectedComponentsWithStats(dark)
    if n < 2:
        return 0, 0, img.shape[1], img.shape[0]
    i = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    x, y, w, h, area = st[i]
    if area < 0.15 * img.shape[0] * img.shape[1]:
        return 0, 0, img.shape[1], img.shape[0]
    return int(x), int(y), int(w), int(h)


def make_mask(img, mode, color, tol, pick=None, ignore=None):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if mode == "asphalt":  # gray road on grass: match the road's hue / saturation / brightness
        if pick:
            x, y = int(pick[0]), int(pick[1])
            ref = np.median(hsv[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1, 3), axis=0)
        else:
            ref = np.array([108.0, 68.0, 165.0])  # the gray-blue road of the F1 map
        H, S, V = (hsv[..., i].astype(float) for i in range(3))
        dh = np.minimum(np.abs(H - ref[0]), 180 - np.abs(H - ref[0]))
        m = (dh <= 15) & (S >= ref[1] - 38) & (S <= ref[1] + 30) & (V >= ref[2] - 80) & (V <= ref[2] + 50)
    elif color:
        r, g, b = color
        diff = np.abs(img.astype(np.int16) - np.array([b, g, r], np.int16)).max(axis=2)
        m = diff <= tol
    elif mode == "bright":
        m = hsv[..., 2] > 150
    elif mode == "dark":
        m = hsv[..., 2] < 90
    else:  # "color": saturated pixels (pink / yellow / blue sector lines)
        m = (hsv[..., 1] > 120) & (hsv[..., 2] > 120)
    mask = m.astype(np.uint8) * 255
    for poly in ignore or []:  # areas to cut out (e.g. an old, unused part of the circuit)
        cv2.fillPoly(mask, [np.array(poly, np.int32).reshape(-1, 1, 2)], 0)

    # how thick is the drawn line? (half-width in pixels, so everything below scales with image size)
    # (first drop hairline specks / outlines 1-2 px wide so they don't fool the estimate)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if clean.any():
        mask = clean
    dt0 = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    sk0 = skeletonize(mask > 0)
    r = float(np.median(dt0[sk0])) if sk0.any() else 3.0
    r = max(r, 1.5)
    ksz = int(5 * r) | 1

    # remove thick blobs (legend boxes, star ratings...). Track lines are thin.
    big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    thick = cv2.morphologyEx(mask, cv2.MORPH_OPEN, big)
    thick = cv2.dilate(thick, big)
    mask[thick > 0] = 0

    kc = int(max(3, round(1.6 * r))) | 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # drop small specks, keep every sizeable piece (the line is cut up by circles / start line)
    n, labels, st, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        raise SystemExit("No track found. Try --mode bright/dark or --color R,G,B --tol 25.")
    keep = np.zeros_like(mask)
    biggest = st[1:, cv2.CC_STAT_AREA].max()
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 0.04 * biggest:
            keep[labels == i] = 255
    return keep, r


# ---------------------------------------------------------------- tracer
def color_fields(img, mask, mode, r=3.2):
    """One blurred field per color family on the track, so the tracer can stay on its own color."""
    if mode != "color":
        return [np.ones(mask.shape, np.float32)], None
    hue = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 0].astype(int)
    bins = (hue // 15) % 12
    on = mask > 0
    fields = []
    for b in range(12):
        m = on & (bins == b)
        if m.sum() > 0.05 * on.sum():
            f = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8 * r)
            fields.append(f)
    if not fields:
        return [np.ones(mask.shape, np.float32)], None
    return fields, None


def pick_start(skel, near=None, r=3.2):
    """A straight bit of skeleton to start from, and its heading (radians)."""
    ys, xs = np.nonzero(skel)
    pts = np.column_stack([xs, ys]).astype(float)
    tree = cKDTree(pts)
    if near is not None:
        idxs = [int(np.argmin(np.hypot(pts[:, 0] - near[0], pts[:, 1] - near[1])))]
    else:
        idxs = range(0, len(pts), max(1, len(pts) // 400))
    best, best_score, best_th = None, -1, 0.0
    for i in idxs:
        nb = pts[tree.query_ball_point(pts[i], 3.75 * r)]
        if len(nb) < 8:
            continue
        c = nb - nb.mean(axis=0)
        w, v = np.linalg.eigh(c.T @ c)
        score = w[1] / (w[0] + 1e-6)
        if near is not None or score > best_score:
            best, best_score = pts[i], score
            best_th = float(np.arctan2(v[1, 1], v[0, 1]))
    if best is None:
        raise SystemExit("Could not find a usable start point on the track.")
    return best, best_th


def trace_loop(mask, fields=None, start_hint=None, r=3.2, max_turn=40.0):
    k = r / 3.2  # all distances below were tuned at a line half-width of ~3.2 px
    step, look, max_gap, far = 3.0 * k, 9.0 * k, 45.0 * k, 38.0 * k
    """Follow the line around the loop, keeping heading, so crossings and gaps are fine."""
    G = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 0.8 * r)
    G /= G.max()
    skel = skeletonize(mask > 0)
    start, th = pick_start(skel, start_hint, r)
    fields = fields or [G]
    gmax = max(f.max() for f in fields)
    fields = [f / gmax for f in fields]
    cur = int(np.argmax([map_coordinates(f, [[start[1]], [start[0]]], order=1)[0] for f in fields]))

    deltas = np.radians(np.linspace(-max_turn, max_turn, 17))
    pos, th0 = start.copy(), th
    path = [pos.copy()]
    traveled, gap = 0.0, 0.0
    for _ in range(200000):
        cand = th + deltas
        px = pos[0] + look * np.cos(cand)
        py = pos[1] + look * np.sin(cand)
        vals = map_coordinates(G, [py, px], order=1, mode="constant")
        same = map_coordinates(fields[cur], [py, px], order=1, mode="constant")
        # long look-ahead on our own color: carries us through places hidden under number circles
        fx = pos[0] + far * np.cos(cand)
        fy = pos[1] + far * np.sin(cand)
        same_far = map_coordinates(fields[cur], [fy, fx], order=1, mode="constant")
        score = 0.3 * vals + 3.0 * same + 3.0 * same_far - 0.2 * np.abs(deltas) / np.radians(max_turn)
        j = int(np.argmax(score))
        if vals[j] < 0.12:  # no line ahead: bridge the gap straight
            gap += step
            if gap > max_gap:
                raise SystemExit(f"Lost the track near pixel ({pos[0]:.0f}, {pos[1]:.0f}). "
                                 "Check the preview, or try another --mode / --color.")
        else:
            gap = 0.0
            th = 0.5 * th + 0.5 * cand[j]
        pos = pos + step * np.array([np.cos(th), np.sin(th)])
        if gap == 0.0:  # re-center on the line
            nrm = np.array([-np.sin(th), np.cos(th)])
            offs = np.arange(-4, 5, dtype=float) * k
            pv = map_coordinates(G, [pos[1] + offs * nrm[1], pos[0] + offs * nrm[0]],
                                 order=1, mode="constant")
            pv = pv * np.exp(-(offs / (3.0 * k)) ** 2)
            if pv.sum() > 0.2:
                pos = pos + nrm * 0.6 * (offs * pv).sum() / pv.sum()
        here = [map_coordinates(f, [[pos[1]], [pos[0]]], order=1)[0] for f in fields]
        best = int(np.argmax(here))
        if here[best] > 0.3 and here[best] > here[cur] + 0.2:  # sector color change
            cur = best
        path.append(pos.copy())
        traveled += step
        if traveled > 30 * step and np.hypot(*(pos - start)) < 1.6 * step:
            d = np.arctan2(np.sin(th - th0), np.cos(th - th0))
            if abs(d) < np.radians(60):
                return np.array(path), skel
    raise SystemExit("The line never closed into a loop. Check the preview image.")



# ---------------------------------------------------------------- fragment linking
from collections import deque

N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _bfs(nodes, src):
    prev, q, last = {src: None}, deque([src]), src
    while q:
        u = q.popleft()
        last = u
        for dx, dy in N8:
            v = (u[0] + dx, u[1] + dy)
            if v in nodes and v not in prev:
                prev[v] = u
                q.append(v)
    return last, prev


def _longest_path(nodes):
    """Main run of a skeleton piece (ignores tiny side spurs)."""
    a, _ = _bfs(nodes, next(iter(nodes)))
    b, prev = _bfs(nodes, a)
    path, u = [], b
    while u is not None:
        path.append(u)
        u = prev[u]
    return np.array(path, dtype=float)  # (x, y)


def split_by_color(img, mask):
    """Assign every track pixel to one of the dominant color families (pink / yellow / blue...)."""
    hue = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 0].astype(int)
    bins = (hue // 6) % 30
    on = mask > 0
    counts = np.bincount(bins[on], minlength=30)
    dom = [b for b in range(30) if counts[b] > 0.04 * on.sum()]
    if len(dom) < 2:
        return [on]
    d = np.array([[min((b - c) % 30, (c - b) % 30) for c in dom] for b in range(30)])
    cls = np.argmin(d, axis=1)[bins]
    return [on & (cls == i) for i in range(len(dom))]


def build_fragments(classes, r):
    min_len = max(8, int(2.5 * r))
    kc = int(max(3, round(1.6 * r))) | 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))
    frags = []
    for ci, m in enumerate(classes):
        m8 = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k)
        sk = skeletonize(m8 > 0)
        n, lab = cv2.connectedComponents(sk.astype(np.uint8), connectivity=8)
        for j in range(1, n):
            ys, xs = np.nonzero(lab == j)
            if len(xs) < min_len:
                continue
            pts = _longest_path(set(zip(xs.tolist(), ys.tolist())))
            if len(pts) >= min_len:
                frags.append({"pts": pts, "color": ci})
    return frags


def _outward(pts, side, n=10):
    n = min(n, len(pts) - 1)
    a, b = (pts[0], pts[min(n, len(pts) - 1)]) if side == 0 else (pts[-1], pts[max(-1 - n, -len(pts))])
    v = a - b
    return v / (np.hypot(*v) + 1e-9)


def link_fragments(frags, r, mismatch=None, angle_w=0.15, reach=22.0):
    max_link = reach * r
    mismatch = mismatch or 12.0 * r
    n_dir = max(6, int(3 * r))
    """Pair up fragment ends into one closed loop; same color + straight continuation is preferred."""
    ends = []
    for fi, f in enumerate(frags):
        for side in (0, 1):
            ends.append((fi, side, f["pts"][0 if side == 0 else -1], _outward(f["pts"], side, n_dir), f["color"]))
    cands = []
    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            fi, _, pi, oi, ci = ends[i]
            fj, _, pj, oj, cj = ends[j]
            if fi == fj:
                continue
            d = float(np.hypot(*(pi - pj)))
            if d > max_link:
                continue
            ang = np.degrees(np.arccos(np.clip(np.dot(oi, -oj), -1, 1)))
            cands.append((d + mismatch * (ci != cj) + angle_w * ang * 1.0, i, j))
    cands.sort()
    link, parent = {}, list(range(len(frags)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    joined = 0
    for c, i, j in cands:
        if i in link or j in link:
            continue
        fi, fj = ends[i][0], ends[j][0]
        if find(fi) == find(fj):
            if joined != len(frags) - 1:  # only close the loop when everything is chained
                continue
        else:
            parent[find(fi)] = find(fj)
            joined += 1
        link[i] = j
        link[j] = i
        if joined == len(frags) - 1 and find(fi) == find(fj) and len(link) == 2 * (len(frags) - 1) + 2:
            break
    return ends, link


def trace_fragments(frags, ends, link):
    """Walk the linked fragments in order and return one closed point list."""
    idx = {(e[0], e[1]): k for k, e in enumerate(ends)}
    fi, enter = 0, 0
    pts, used = [], set()
    while fi not in used:
        used.add(fi)
        p = frags[fi]["pts"]
        pts.append(p if enter == 0 else p[::-1])
        exit_side = 1 - enter
        k = link.get(idx[(fi, exit_side)])
        if k is None:
            raise SystemExit("Track pieces did not link into a loop (a gap is too big). "
                             "Check the preview, or try --method trace.")
        fi, enter = ends[k][0], ends[k][1]
    if len(used) < len(frags):
        left = sum(len(frags[i]["pts"]) for i in range(len(frags)) if i not in used)
        if left > 40:
            print(f"Note: {len(frags) - len(used)} line pieces were not part of the loop and were ignored.")
    return np.vstack(pts + [pts[0][:1]])



# ---------------------------------------------------------------- skeleton-graph method
import heapq
import itertools


def _matchings(items):
    if not items:
        yield []
        return
    a = items[0]
    for k in range(1, len(items)):
        for rest in _matchings(items[1:k] + items[k + 1:]):
            yield [(a, items[k])] + rest


_DBG = {}


def graph_loop(mask, r):
    """Single-color tracks. Split the skeleton at junctions into arms, then walk one closed loop:
    where two strands run together (the skeleton shows one arm) that arm is used twice, and at every
    junction the arms are paired so the track bends as little as possible overall."""
    skel = skeletonize(mask > 0)
    nb = cv2.filter2D(skel.astype(np.uint8), -1, np.ones((3, 3), np.float32),
                      borderType=cv2.BORDER_CONSTANT) - 1
    junc = skel & (nb >= 3)

    if not junc.any():  # a plain ring: cut it open and walk it
        ys, xs = np.nonzero(skel)
        nodes = set(zip(xs.tolist(), ys.tolist()))
        p = next(iter(nodes))
        for dx, dy in N8 + [(0, 0)]:
            nodes.discard((p[0] + dx, p[1] + dy))
        path = _longest_path(nodes)
        return np.vstack([path, path[:1]])

    rad = max(2, int(2.0 * r))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    ncl, cl = cv2.connectedComponents(cv2.dilate(junc.astype(np.uint8), ker), connectivity=8)

    n, lab = cv2.connectedComponents((skel & (cl == 0)).astype(np.uint8), connectivity=8)
    arms = []
    for j in range(1, n):
        ys, xs = np.nonzero(lab == j)
        if len(xs) < 3:
            continue
        path = _longest_path(set(zip(xs.tolist(), ys.tolist())))
        ends = []
        for pt in (path[0], path[-1]):
            x, y = int(pt[0]), int(pt[1])
            ends.append(int(cl[max(0, y - 2):y + 3, max(0, x - 2):x + 3].max()))
        arms.append({"pts": path, "cl": ends})

    # tiny loops that start and end at the same junction are noise (grandstand edges, curbs...)
    arms = [a for a in arms if not (a["cl"][0] == a["cl"][1] and len(a["pts"]) < 8 * r)]
    # prune dead ends (skeleton spurs, roads into buildings...): the lap never goes down a dead end
    while True:
        cnt = {}
        for a in arms:
            for c in a["cl"]:
                cnt[c] = cnt.get(c, 0) + 1
        keep = [a for a in arms if 0 not in a["cl"] and all(cnt[c] > 1 for c in a["cl"])]
        if len(keep) == len(arms):
            break
        arms = keep
    if not arms:
        raise SystemExit("No closed loop found in the road. Check the mask / try --crop or --ignore.")
    _DBG["arms"], _DBG["cl"] = arms, cl

    # --- arms that must be used twice (two strands merged into one skeleton arm)
    deg = {}
    for a in arms:
        for c in a["cl"]:
            deg[c] = deg.get(c, 0) + 1
    odd = [c for c in deg if deg[c] % 2]
    double = set()
    if odd:
        dist = {}
        for src in odd:  # Dijkstra over the cluster graph, remembering which arms were used
            best, heap = {src: (0.0, [])}, [(0.0, src, [])]
            while heap:
                d, u, via = heapq.heappop(heap)
                if best[u][0] < d:
                    continue
                for i, a in enumerate(arms):
                    for sd in (0, 1):
                        if a["cl"][sd] == u and a["cl"][1 - sd] != u:
                            v, nd = a["cl"][1 - sd], d + len(a["pts"])
                            if v not in best or nd < best[v][0]:
                                best[v] = (nd, via + [i])
                                heapq.heappush(heap, (nd, v, via + [i]))
            dist[src] = best
        pick = None
        for m in _matchings(odd):
            cost = sum(dist[u].get(v, (1e18,))[0] for u, v in m)
            if pick is None or cost < pick[0]:
                pick = (cost, m)
        for u, v in pick[1]:
            for i in dist[u][v][1]:
                double ^= {i}
    inst = list(arms) + [arms[i] for i in sorted(double)]

    # --- pair the arm ends at every junction
    n_dir = max(5, int(3 * r))

    def head(a, side):  # direction of travel when arriving at that end
        p = a["pts"]
        k = min(n_dir, len(p) - 1)
        v = (p[-1] - p[-1 - k]) if side == 1 else (p[0] - p[k])
        return v / (np.hypot(*v) + 1e-9)

    options = []  # per junction: list of (bend cost, pairs)
    for c in sorted({c for a in inst for c in a["cl"]}):
        items = [(i, sd) for i, a in enumerate(inst) for sd in (0, 1) if a["cl"][sd] == c]
        if len(items) % 2:
            raise SystemExit("Could not make sense of a junction in the track. Check the preview / mask.")
        hd = {it: head(inst[it[0]], it[1]) for it in items}
        opts = []
        for m in _matchings(items):
            cost = sum(float(np.degrees(np.arccos(np.clip(np.dot(hd[u], -hd[v]), -1, 1)))) for u, v in m)
            opts.append((cost, m))
        options.append(sorted(opts, key=lambda t: t[0])[:6])

    combos = sorted(itertools.product(*options), key=lambda cs: sum(c[0] for c in cs))[:20000]
    for combo in combos:
        link = {}
        for _, m in combo:
            for u, v in m:
                link[u], link[v] = v, u
        order, cur, enter = [], 0, 0
        while True:
            order.append((cur, enter))
            nxt = link.get((cur, 1 - enter))
            if nxt is None:
                break
            cur, enter = nxt
            if cur == 0 and enter == 0:
                break
            if len(order) > len(inst):
                break
        if len(order) == len(inst) and nxt == (0, 0):
            pts = [inst[i]["pts"] if e == 0 else inst[i]["pts"][::-1] for i, e in order]
            return np.vstack(pts + [pts[0][:1]])
    raise SystemExit("The track pieces did not close into one loop. Check the preview / mask.")


# ---------------------------------------------------------------- geometry
def fit_spline(xy, smooth, n_out, r=3.0):
    xy = xy[:-1]
    tck, _ = splprep([xy[:, 0], xy[:, 1]], s=smooth * (r / 3.0) ** 2 * len(xy), per=True)
    u_dense = np.linspace(0, 1, 20000, endpoint=False)
    x, y = splev(u_dense, tck)
    seg = np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]]))
    arc = np.r_[0, np.cumsum(seg)]
    length = arc[-1]
    targets = np.linspace(0, length, n_out, endpoint=False)
    u = np.interp(targets, arc, np.r_[u_dense, 1.0])
    x, y = splev(u, tck)
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    curv = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)  # 1 / pixels
    return np.column_stack([x, y]), curv, length


def find_corners(curv, radius_px):
    flag = np.abs(curv) > 1.0 / radius_px
    if flag.all() or not flag.any():
        return []
    shift = int(np.argmin(flag))
    idx = (np.arange(len(curv)) + shift) % len(curv)
    corners, run = [], []
    for i in idx:
        if flag[i]:
            run.append(i)
        elif run:
            corners.append(run)
            run = []
    if run:
        corners.append(run)
    return corners



def save_debug(img, mask, frags, out):
    """Picture of what the script found, saved when it fails so you can see why."""
    vis = (img * 0.35).astype(np.uint8)
    vis[mask > 0] = (200, 200, 200)
    rng = np.random.RandomState(3)
    for f in frags or []:
        col = tuple(int(c) for c in rng.randint(60, 255, 3))
        cv2.polylines(vis, [f["pts"].round().astype(np.int32).reshape(-1, 1, 2)], False, col, 2)
        for e in (f["pts"][0], f["pts"][-1]):
            cv2.circle(vis, tuple(e.round().astype(int)), 5, (255, 255, 255), 1)
    cv2.imwrite(f"{out}_debug.png", vis)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--studs-per-pixel", type=float, default=None)
    p.add_argument("--real-length-m", type=float,
                   help="real track length in meters (e.g. 9370 for 9.37 km); sets the scale")
    p.add_argument("--meters-per-stud", type=float, default=METERS_PER_STUD)
    p.add_argument("--mode", choices=["auto", "color", "bright", "dark", "asphalt"], default="auto",
                   help="auto (default) picks color / dark / bright for you. Use asphalt for a real game screenshot (gray road on grass)")
    p.add_argument("--color", type=lambda s: tuple(int(v) for v in s.split(",")),
                   help="track RGB, e.g. 232,62,168 (overrides --mode)")
    p.add_argument("--tol", type=int, default=25)
    p.add_argument("--crop", type=lambda s: tuple(int(v) for v in s.split(",")),
                   help="x,y,w,h of the map area (default: auto-detect the dark panel)")
    p.add_argument("--pick", type=lambda s: tuple(float(v) for v in s.split(",")),
                   help="x,y of a point ON the road (asphalt mode: matches that road color)")
    p.add_argument("--ignore", action="append", default=[],
                   help='area to cut out, as "x,y;x,y;x,y..." in ORIGINAL image pixels (can repeat)')
    p.add_argument("--points", type=int, default=400)
    p.add_argument("--smooth", type=float, default=1.0)
    p.add_argument("--corner-radius", type=float, default=None,
                   help="studs; turns tighter than this count as corners (default: track length / 60)")
    p.add_argument("--start", type=lambda s: tuple(float(v) for v in s.split(",")),
                   help="start/finish x,y in the ORIGINAL image; snaps to the nearest line point")
    p.add_argument("--method", choices=["auto", "link", "graph", "trace"], default="auto",
                   help="link = join colored pieces (color maps); graph = split at junctions (single color); trace = follow the line")
    p.add_argument("--out", default="track")
    a = p.parse_args()

    raw = cv2.imread(a.image, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise SystemExit(f"Could not read {a.image}")
    if raw.ndim == 2:
        full = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    elif raw.shape[2] == 4:  # transparent PNG: lay it on white (otherwise transparent = black)
        al = raw[..., 3:4].astype(np.float32) / 255.0
        full = (raw[..., :3] * al + 255.0 * (1 - al)).astype(np.uint8)
    else:
        full = raw
    x0, y0, w, h = a.crop if a.crop else find_panel(full)
    img = full[y0:y0 + h, x0:x0 + w]

    mode = a.mode
    if mode == "auto":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        colorful = ((hsv[..., 1] > 120) & (hsv[..., 2] > 120)).mean()
        mode = "color" if colorful > 0.003 else ("dark" if np.median(hsv[..., 2]) > 128 else "bright")
        print(f"Detected a {mode} track" if mode != "color" else "Detected a colored track map")
    polys = []
    for txt in a.ignore:
        poly = [tuple(float(v) for v in pt.split(",")) for pt in txt.split(";") if pt.strip()]
        polys.append([(px - x0, py - y0) for px, py in poly])
    pick = (a.pick[0] - x0, a.pick[1] - y0) if a.pick else None
    mask, r = make_mask(img, mode, a.color, a.tol, pick, polys)
    hint = (a.start[0] - x0, a.start[1] - y0) if a.start else None
    method = a.method if a.method != "auto" else ("link" if (mode == "color" and not a.color) else "graph")
    frags = None
    try:
        if method == "link":
            frags = build_fragments(split_by_color(img, mask), r)
            path = None
            for reach in (22.0, 50.0):  # first normal gaps, then bigger ones
                try:
                    ends, link = link_fragments(frags, r, reach=reach)
                    path = trace_fragments(frags, ends, link)
                    break
                except SystemExit:
                    continue
            if path is None:
                path = graph_loop(mask, r)  # last resort: ignore colors, split at junctions
        elif method == "graph":
            path = graph_loop(mask, r)
        else:
            fields, _ = color_fields(img, mask, 'color' if (mode == 'color' and not a.color) else 'other', r)
            path, skel = trace_loop(mask, fields, hint, r)
    except SystemExit as e:
        save_debug(img, mask, frags, a.out)
        raise SystemExit(f"{e}\nSaved {a.out}_debug.png: it shows what the script found. Send it to me.")
    pts, curv, length_px = fit_spline(path, a.smooth, a.points, r)

    if a.start:
        i = int(np.argmin(np.hypot(pts[:, 0] - hint[0], pts[:, 1] - hint[1])))
        pts, curv = np.roll(pts, -i, axis=0), np.roll(curv, -i)

    if a.real_length_m:
        s = (a.real_length_m / a.meters_per_stud) / length_px
    else:
        s = a.studs_per_pixel or 1.0

    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    xi = pts[:, 0].round().astype(int).clip(0, dt.shape[1] - 1)
    yi = pts[:, 1].round().astype(int).clip(0, dt.shape[0] - 1)
    width_px = 2 * dt[yi, xi]

    radius_studs = np.where(np.abs(curv) > 1e-9, s / np.abs(curv + 1e-12), np.inf)
    corner_r = a.corner_radius if a.corner_radius else (length_px * s) / 60.0
    corners = find_corners(curv, corner_r / s)
    step = length_px / len(pts) * s

    corner_data = [{
        "corner": n,
        "start_distance": round(run[0] * step, 1),
        "end_distance": round(run[-1] * step, 1),
        "min_radius": round(float(np.min(radius_studs[run])), 1),
        "direction": "right" if np.mean(curv[run]) > 0 else "left",  # image y points down
    } for n, run in enumerate(corners, 1)]

    result = {
        "length_studs": round(length_px * s, 1),
        "length_m": round(length_px * s * a.meters_per_stud, 1),
        "studs_per_pixel": s,
        "corner_count": len(corner_data),
        "corners": corner_data,
        "note": "width is the drawn line width in the image, not necessarily the real road width",
        # Roblox: x = image x, z = image y (origin = top-left of the cropped map)
        "points": [{"x": round(float(x) * s, 2), "z": round(float(y) * s, 2)} for x, y in pts],
    }
    with open(f"{a.out}.json", "w") as f:
        json.dump(result, f, indent=2)

    prev = img.copy()
    cv2.polylines(prev, [pts.round().astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    for run in corners:
        for i in run:
            cv2.circle(prev, tuple(pts[i].round().astype(int)), 3, (0, 0, 255), -1)
    cv2.circle(prev, tuple(pts[0].round().astype(int)), 8, (255, 255, 255), -1)
    cv2.imwrite(f"{a.out}_preview.png", prev)

    print(f"Length: {result['length_studs']} studs ({result['length_m']} m) | corners: {len(corner_data)}")
    if not (a.real_length_m or a.studs_per_pixel):
        print("Scale not set: lengths are in pixels. Use --real-length-m or --studs-per-pixel.")
    print(f"Wrote {a.out}.json and {a.out}_preview.png")


if __name__ == "__main__":
    main()
