#!/usr/bin/env python3
"""
RMS difference between axial surface models vs fault discretisation.

Sweeps npts (number of resampled nodes) from 3 (sharp, 2 segments) to 1001
(smooth, 1000 segments).  For each, n_locked = (npts-1)//2 so the lock/creep
transition stays near x ≈ 50 km.

Outputs: axial_rms_convergence.png
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["figure.dpi"] = 150
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


# ═══════════════════════════════════════════════════════════════════════
# Edge dislocation kernels
# ═══════════════════════════════════════════════════════════════════════

def u2_edge(x_obs, slip, delta, d, x_ref, orient):
    if d <= 0:
        eps = 1e-12
        return -(slip / np.pi) * np.sin(delta) * np.arctan2(
            orient * (x_obs - x_ref), eps)
    zeta = orient * (x_obs - x_ref) / d
    return -(slip / np.pi) * (
        np.sin(delta) * np.arctan(zeta)
        + (np.cos(delta) + zeta * np.sin(delta)) / (1.0 + zeta**2))


def u2_segment(x_obs, x1, z1, x2, z2, signed_slip):
    d1, d2 = -z1, -z2
    dx, dd = x2 - x1, d2 - d1
    delta = np.arctan2(abs(dd), max(abs(dx), 1e-12))
    orient = np.sign(dx) if abs(dx) > 0 else 1.0
    if d1 <= d2:
        x_top, d_top, x_bot, d_bot = x1, d1, x2, d2
    else:
        x_top, d_top, x_bot, d_bot = x2, d2, x1, d1
    return (u2_edge(x_obs, signed_slip, delta, d_top, x_top, orient)
            + u2_edge(x_obs, -signed_slip, delta, d_bot, x_bot, orient))


# ═══════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════════

def segment_angles(xnode, znode):
    return np.arctan2(np.diff(znode), np.diff(xnode))


def compute_axial_surface_intersections(xnode, znode, theta):
    nseg = theta.size
    x_axial = np.empty(nseg - 1, dtype=float)
    for j in range(1, nseg):
        g = 0.5 * (theta[j - 1] + theta[j] + np.pi)
        tg = np.tan(g)
        x_axial[j - 1] = (np.nan if np.isclose(tg, 0.0, atol=1e-12)
                           else xnode[j] - znode[j] / tg)
    return x_axial


# ═══════════════════════════════════════════════════════════════════════
# C2 quintic smoothing + arc-length resampling
# ═══════════════════════════════════════════════════════════════════════

def _quintic_patch_coeffs(xL, zL, mL, xR, zR, mR):
    h = xR - xL
    a0, a1 = zL, mL * h
    D0 = zR - zL - a1
    D1 = (mR - mL) * h
    A = np.array([[1, 1, 1], [3, 4, 5], [6, 12, 20]], dtype=float)
    a3, a4, a5 = np.linalg.solve(A, [D0, D1, 0.0])
    return np.array([a0, a1, 0.0, a3, a4, a5]), h


def _eval_quintic(x, xL, coeffs, h):
    t = (x - xL) / h
    a0, a1, a2, a3, a4, a5 = coeffs
    return (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t + a0)


def build_c2_fault_model(x, z, w=5.0):
    x, z = np.asarray(x, float), np.asarray(z, float)
    m_seg = np.diff(z) / np.diff(x)
    n = len(x)
    wv = np.zeros(n)
    for i in range(1, n - 1):
        wv[i] = min(float(w), 0.45 * (x[i] - x[i-1]), 0.45 * (x[i+1] - x[i]))
    patches = {}
    for i in range(1, n - 1):
        if wv[i] <= 0:
            continue
        xL, xR = x[i] - wv[i], x[i] + wv[i]
        mL, mR = m_seg[i-1], m_seg[i]
        zL = z[i] + mL * (xL - x[i])
        zR = z[i] + mR * (xR - x[i])
        coeffs, h = _quintic_patch_coeffs(xL, zL, mL, xR, zR, mR)
        patches[i] = (xL, coeffs, h)
    pieces = []
    for seg in range(n - 1):
        lc = wv[seg] if seg > 0 else 0.0
        rc = wv[seg + 1] if seg + 1 < n - 1 else 0.0
        xL, xR = x[seg] + lc, x[seg + 1] - rc
        if xR > xL:
            pieces.append(("line", xL, xR, seg))
        v = seg + 1
        if 1 <= v <= n - 2 and wv[v] > 0:
            pieces.append(("patch", x[v] - wv[v], x[v] + wv[v], v))
    return dict(x=x, z=z, m_seg=m_seg, pieces=pieces, patches=patches)


def eval_c2_fault(xq, model, extrapolate=True):
    x, z = model["x"], model["z"]
    m_seg, pieces, patches = model["m_seg"], model["pieces"], model["patches"]
    xq = np.asarray(xq, float)
    zq = np.full_like(xq, np.nan)
    for kind, xL, xR, idx in pieces:
        mask = (xq >= xL) & (xq <= xR)
        if not np.any(mask):
            continue
        if kind == "line":
            zq[mask] = z[idx] + m_seg[idx] * (xq[mask] - x[idx])
        else:
            xL0, coeffs, h = patches[idx]
            zq[mask] = _eval_quintic(xq[mask], xL0, coeffs, h)
    if extrapolate:
        zq = np.where(np.isnan(zq) & (xq < x[0]),  z[0]  + m_seg[0]  * (xq - x[0]),  zq)
        zq = np.where(np.isnan(zq) & (xq > x[-1]), z[-1] + m_seg[-1] * (xq - x[-1]), zq)
    return zq


def resample_equal_arclength(x_dense, z_dense, npts):
    ds = np.sqrt(np.diff(x_dense)**2 + np.diff(z_dense)**2)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    L = s[-1]
    s_uniform = np.linspace(0, L, npts)
    return np.interp(s_uniform, s, x_dense), np.interp(s_uniform, s, z_dense)


# ═══════════════════════════════════════════════════════════════════════
# Dislocation network with axial_mode
# ═══════════════════════════════════════════════════════════════════════

def build_dislocation_network(xnode, znode, x_axial, theta, slip_fault,
                              axial_mode="half"):
    nseg = theta.size
    slip_fault = np.asarray(slip_fault, float)
    segments = []

    for i in range(nseg):
        s = slip_fault[i]
        segments.append(dict(x1=xnode[i], z1=znode[i],
                             x2=xnode[i+1], z2=znode[i+1], slip=-s))

    for j in range(1, nseg):
        xa = x_axial[j - 1]
        if not np.isfinite(xa):
            continue
        dtheta = theta[j] - theta[j - 1]

        if axial_mode == "hard":
            slip_local = slip_fault[j - 1] if (slip_fault[j-1] > 0 and slip_fault[j] > 0) else 0.0
        elif axial_mode == "half":
            slip_local = 0.5 * (slip_fault[j - 1] + slip_fault[j])
        elif axial_mode == "full":
            slip_local = max(slip_fault[j - 1], slip_fault[j])
        else:
            raise ValueError(f"Unknown axial_mode: {axial_mode!r}")

        sf = 2.0 * slip_local * np.sin(0.5 * dtheta)
        segments.append(dict(x1=xa, z1=0.0,
                             x2=xnode[j], z2=znode[j], slip=-sf))

    return segments


def elastic_u2(x_obs, segments):
    U2 = np.zeros_like(x_obs)
    for seg in segments:
        U2 += u2_segment(x_obs, seg['x1'], seg['z1'],
                         seg['x2'], seg['z2'], seg['slip'])
    return U2


def structural_velocity_v(x_obs, theta, x_axial, slip, x_start):
    x_obs = np.asarray(x_obs, float)
    x_bounds = x_axial[np.isfinite(x_axial)]
    dom = np.clip(np.searchsorted(x_bounds, x_obs, side='right'), 0, theta.size - 1)
    v = -slip[dom] * np.sin(theta[dom])
    v = v.astype(float, copy=True)
    v[x_obs < x_start] = 0.0
    return v


# ═══════════════════════════════════════════════════════════════════════
# Sweep
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    xctrl = [0, 50, 100]
    zctrl = [0, -20, -30]
    w_blend = 8
    structuralslip = 20.0
    sliprate = 20.0
    x_obs = np.linspace(1e-1, 100.0, 2000)

    npts_list = [3, 5, 7, 11, 21, 51, 101, 201, 501, 1001]

    rms_coseis = []
    rms_inter  = []

    for npts_i in npts_list:
        mdl = build_c2_fault_model(xctrl, zctrl, w=w_blend)
        xd = np.linspace(xctrl[0], xctrl[-1], 5000)
        zd = eval_c2_fault(xd, mdl)
        xn, zn = resample_equal_arclength(xd, zd, npts_i)

        ns = len(xn) - 1
        th = segment_angles(xn, zn)
        xa = compute_axial_surface_intersections(xn, zn, th)

        nl = (npts_i - 1) // 2
        nl = max(0, min(nl, ns))
        sf = np.zeros(ns)
        sf[:nl] = sliprate

        ss = structuralslip * np.ones(ns)
        v_str = structural_velocity_v(x_obs, th, xa, ss, x_start=xn[0])

        vel = {}
        for m in ["hard", "full"]:
            segs = build_dislocation_network(xn, zn, xa, th, sf, axial_mode=m)
            U2 = elastic_u2(x_obs, segs)
            vel[m] = dict(coseis=U2, inter=v_str - U2)

        rms_coseis.append(
            np.sqrt(np.mean((vel["full"]["coseis"] - vel["hard"]["coseis"])**2)))
        rms_inter.append(
            np.sqrt(np.mean((vel["full"]["inter"] - vel["hard"]["inter"])**2)))

        print(f"npts={npts_i:5d}, nseg={ns:4d}, n_locked={nl:4d}, "
              f"lock_x={xn[nl]:.1f} km, "
              f"RMS(co)={rms_coseis[-1]:.4e}, RMS(in)={rms_inter[-1]:.4e}")

    # ── Plot ────────────────────────────────────────────────────────────
    nseg_arr = np.array([n - 1 for n in npts_list])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.5),
                                   sharey=True, layout="constrained")

    for ax, data, title in [(ax1, rms_coseis, 'Coseismic'),
                             (ax2, rms_inter, 'Interseismic')]:
        ax.loglog(nseg_arr, data, 'o', color='k', ms=4, zorder=2)
        ax.loglog(nseg_arr, data, '-', color='firebrick', lw=1.2, zorder=1)
        ax.set_xlabel('Number of fault segments')
        ax.set_title(title)

    ax1.set_ylabel('RMS difference (mm/yr)')

    outpath = "/Users/braydennoh/Research/ThrustFault/2.28/himalaya/axial_rms_convergence.pdf"
    plt.savefig(outpath, bbox_inches="tight")
    print(f"\nSaved: {outpath}")
    plt.show()
