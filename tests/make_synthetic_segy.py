#!/usr/bin/env python3
"""Write a small synthetic post-stack SEG-Y with known contents, for testing the reader.

Geometry is a regular inline/crossline grid with a fraction of positions
removed (like the gaps in the F3 export) and a few dead (all-zero) traces.
The true samples are saved to an .npz next to the SEG-Y for comparison.
"""
import argparse

import numpy as np


def ieee_to_ibm(x):
    """float -> IBM System/360 float bit patterns (uint32)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.uint32)
    nz = x != 0
    a = np.abs(x[nz])
    e = np.floor(np.log(a) / np.log(16.0)).astype(np.int64) + 1   # a = m * 16^e, m in [1/16, 1)
    m = a / 16.0 ** e
    lo = m < 1 / 16                                               # guard against log rounding
    e[lo] -= 1; m[lo] *= 16
    hi = m >= 1
    e[hi] += 1; m[hi] /= 16
    mant = np.round(m * 2 ** 24).astype(np.int64)
    ov = mant >= 2 ** 24
    mant[ov] //= 16; e[ov] += 1
    sign = (x[nz] < 0).astype(np.int64)
    out[nz] = ((sign << 31) | ((e + 64) << 24) | mant).astype(np.uint32)
    return out


def ricker(t, f):
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="output .sgy path")
    ap.add_argument("--il", default="100,139", help="first,last inline")
    ap.add_argument("--xl", default="300,359", help="first,last crossline")
    ap.add_argument("--ns", type=int, default=250)
    ap.add_argument("--dt-ms", type=float, default=4.0)
    ap.add_argument("--format", type=int, choices=[1, 5], default=1, help="1 = IBM, 5 = IEEE")
    ap.add_argument("--missing", type=float, default=0.05, help="fraction of positions dropped")
    ap.add_argument("--dead", type=int, default=10, help="number of all-zero traces")
    ap.add_argument("--dups", type=int, default=0, help="number of duplicated traces")
    ap.add_argument("--gain-patch", default="",
                    help="il0,il1,xl0,xl1,factor: scale a block of traces (e.g. 120,129,400,419,3)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    il0, il1 = map(int, args.il.split(","))
    xl0, xl1 = map(int, args.xl.split(","))
    IL, XL = np.meshgrid(np.arange(il0, il1 + 1), np.arange(xl0, xl1 + 1), indexing="ij")
    il, xl = IL.ravel(), XL.ravel()
    keep = rng.random(il.size) >= args.missing
    il, xl = il[keep], xl[keep]
    n = il.size

    # Dipping reflectors: Ricker wavelets whose time shifts with inline and crossline.
    t = np.arange(args.ns) * args.dt_ms / 1000.0
    data = np.zeros((n, args.ns))
    for t0, dip_il, dip_xl, amp in [(0.2, 0.002, 0.001, 1.0), (0.5, -0.001, 0.0015, -0.7),
                                    (0.8, 0.0005, -0.002, 0.5)]:
        tt = t0 + dip_il * (il - il0) + dip_xl * (xl - xl0)
        data += amp * ricker(t[None, :] - tt[:, None], 30.0)
    data += 0.05 * rng.standard_normal(data.shape)
    data *= 1000.0

    # Optional gain patch: a block of traces scaled by a constant, like a gain or
    # acquisition problem. Applied before the dead traces so those stay at zero.
    patch = np.array([], dtype=np.int64)
    if args.gain_patch:
        pil0, pil1, pxl0, pxl1, factor = args.gain_patch.split(",")
        in_patch = ((il >= int(pil0)) & (il <= int(pil1)) &
                    (xl >= int(pxl0)) & (xl <= int(pxl1)))
        data[in_patch] *= float(factor)
        patch = np.flatnonzero(in_patch)

    dead = rng.choice(n, size=min(args.dead, n), replace=False)
    data[dead] = 0.0

    # Optional duplicates: copies of existing traces appended at the end of the
    # file, so each duplicate has a later trace sequence number than its original.
    dup_src = np.array([], dtype=np.int64)
    if args.dups:
        dup_src = np.sort(rng.choice(n, size=args.dups, replace=False))
        il, xl = np.concatenate([il, il[dup_src]]), np.concatenate([xl, xl[dup_src]])
        data = np.concatenate([data, data[dup_src]])
        n = il.size
    data = data.astype(np.float32)

    # ---- file header
    text = "".join(f"{('C%2d SYNTHETIC TEST SURVEY' % (i + 1)) if i == 0 else ('C%2d' % (i + 1)):<80}"
                   for i in range(40))
    binh = np.zeros(400, dtype=np.uint8)

    def put(buf, byte1, val, fmt):
        b = np.array([val], dtype=fmt).view(np.uint8)
        buf[byte1 - 1:byte1 - 1 + b.size] = b

    put(binh, 3217 - 3200, int(args.dt_ms * 1000), ">u2")
    put(binh, 3221 - 3200, args.ns, ">u2")
    put(binh, 3225 - 3200, args.format, ">i2")
    put(binh, 3501 - 3200, 0x0100, ">u2")     # revision 1.0
    put(binh, 3503 - 3200, 1, ">i2")          # fixed-length traces
    put(binh, 3505 - 3200, 0, ">i2")          # no extended textual headers

    # ---- traces
    th = np.zeros((n, 240), dtype=np.uint8)

    def put_col(byte1, vals, fmt):
        b = np.asarray(vals, dtype=fmt).reshape(n, 1).view(np.uint8)
        th[:, byte1 - 1:byte1 - 1 + b.shape[1]] = b

    x = 600000.0 + 25.0 * (xl - xl0)
    y = 6000000.0 + 25.0 * (il - il0)
    put_col(1, np.arange(1, n + 1), ">i4")
    put_col(71, np.full(n, -100), ">i2")      # coordinates stored x100
    put_col(115, np.full(n, args.ns), ">u2")
    put_col(117, np.full(n, int(args.dt_ms * 1000)), ">u2")
    put_col(181, np.round(x * 100), ">i4")
    put_col(185, np.round(y * 100), ">i4")
    put_col(189, il, ">i4")
    put_col(193, xl, ">i4")

    body = ieee_to_ibm(data).astype(">u4") if args.format == 1 else data.astype(">f4")

    with open(args.out, "wb") as f:
        f.write(text.encode("cp037"))
        f.write(binh.tobytes())
        for i in range(n):
            f.write(th[i].tobytes())
            f.write(body[i].tobytes())

    np.savez(args.out + ".truth.npz", il=il, xl=xl, x=x, y=y, data=data, dead=dead,
             dup_src=dup_src, patch=patch, il_range=[il0, il1], xl_range=[xl0, xl1],
             gain_patch=args.gain_patch)
    n_grid = (il1 - il0 + 1) * (xl1 - xl0 + 1)
    n_pos = n - len(dup_src)
    print(f"wrote {args.out}: {n} traces at {n_pos} of {n_grid} positions "
          f"({n_grid - n_pos} missing), {len(dead)} dead, {len(dup_src)} duplicates, "
          f"{len(patch)} in gain patch, ns={args.ns}, dt={args.dt_ms} ms, format={args.format}")


if __name__ == "__main__":
    main()
