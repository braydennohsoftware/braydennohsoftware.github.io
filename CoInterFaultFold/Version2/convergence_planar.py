#!/usr/bin/env python3
"""
Planar 30-degree thrust fault: elastic → structural convergence as D → ∞.

No folds, no kinks — just a single planar fault from the surface to depth D.
Shows O(1/D) convergence of elastic to the structural step function.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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


def u2_segment(x_obs, x1, z1, x2, z2, slip_mag, slip_sign=-1.0):
    d1, d2 = -z1, -z2
    dx, dd = x2 - x1, d2 - d1
    delta = np.arctan2(abs(dd), max(abs(dx), 1e-12))
    orient = np.sign(dx) if abs(dx) > 0 else 1.0
    if d1 <= d2:
        x_top, d_top, x_bot, d_bot = x1, d1, x2, d2
    else:
        x_top, d_top, x_bot, d_bot = x2, d2, x1, d1
    s_top = slip_sign * slip_mag
    s_bot = -s_top
    return (u2_edge(x_obs, s_top, delta, d_top, x_top, orient)
            + u2_edge(x_obs, s_bot, delta, d_bot, x_bot, orient))


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    dip_deg = 30.0
    dip_rad = np.radians(dip_deg)
    sliprate = 1.0

    x_obs = np.linspace(1e-1, 100.0, 1000)
    x_fw_ref = np.array([-1.0])

    v_struct = sliprate * np.sin(dip_rad) * np.ones_like(x_obs)

    # ── Depth sweep ───────────────────────────────────────────────
    depths = np.logspace(np.log10(5), np.log10(100000), 40)
    rms_uz = np.zeros(len(depths))

    for k, D in enumerate(depths):
        x_deep = D / np.tan(dip_rad)
        z_deep = -D
        U2 = u2_segment(x_obs, 0, 0, x_deep, z_deep, sliprate, -1.0)
        U2_ref = u2_segment(x_fw_ref, 0, 0, x_deep, z_deep, sliprate, -1.0)
        U2 = U2 - U2_ref[0]
        rms_uz[k] = np.sqrt(np.mean((U2 - v_struct)**2))

    # ═══════════════════════════════════════════════════════════════
    # Plot: signal (left), RMS (right)
    # ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.3), layout="constrained")

    # ── Left: vertical velocity profiles at selected depths ──────
    ax = axes[0]
    depth_show = [1, 3, 5, 10, 20, 50, 100, 200, 500]
    cmap = plt.cm.viridis
    norm = mpl.colors.LogNorm(vmin=depth_show[0], vmax=depth_show[-1])

    for D_show in depth_show:
        x_deep = D_show / np.tan(dip_rad)
        U2 = u2_segment(x_obs, 0, 0, x_deep, -D_show, sliprate, -1.0)
        U2_ref = u2_segment(x_fw_ref, 0, 0, x_deep, -D_show, sliprate, -1.0)
        U2 = U2 - U2_ref[0]
        ax.plot(x_obs, U2, lw=0.7, color=cmap(norm(D_show)))

    ax.axhline(v_struct[0], color="firebrick", lw=1.2, ls="--",
               label=r"Structural $s\sin\delta$")
    ax.axhline(0, color="k", lw=0.3)
    ax.set_xlabel("$x$ (km)")
    ax.set_ylabel("Vertical velocity ($u_z / s$)")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=7, frameon=False)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, aspect=25)
    cb.set_label("Fault depth $D$ (km)")

    # ── Right: RMS convergence (log-log), vertical only ──────────
    ax = axes[1]
    ax.loglog(depths, rms_uz, "o", color="k", ms=3, zorder=2)
    ax.loglog(depths, rms_uz, "-", color="firebrick", lw=1.2, zorder=1,
              label="Vertical $u_z$")
    D_ref = np.array([depths[0], depths[-1]])
    scale = rms_uz[0] * depths[0]
    ax.loglog(D_ref, scale / D_ref, "k--", lw=1.2, alpha=0.5,
              label=r"$\propto 1/D$")
    ax.set_xlabel("Fault depth $D$ (km)")
    ax.set_ylabel("RMS(elastic $-$ structural)")
    ax.legend(fontsize=7, frameon=False)

    outpath = "/Users/braydennoh/Research/ThrustFault/2.28/planarfault/convergence_planar.pdf"
    plt.savefig(outpath, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.show()
