#!/usr/bin/env python3
"""Discrete fault-bend fold model (piecewise linear, no smoothing).

Uses the same control points as the C2 model but without quintic-patch
smoothing.  Each segment is a straight dislocation, and each bend vertex
has a concentrated fold dislocation along its axial surface.
"""

import sys
sys.path.insert(0, '/Users/braydennoh/Research/ThrustFault/3.2')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from continuous import (u1_segment, u2_segment, CALTECH_ORANGE)

mpl.rcParams["figure.dpi"] = 150
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 10,
})


def particle_path_discrete(z0, xctrl, zctrl, theta_seg, x_start=200.0):
    """Particle path using exact discrete geometry (no interpolation).

    At each step, axial surface positions are computed analytically
    from the vertex coordinates and bisector angles.
    """
    if z0 >= -0.01:
        return np.array([x_start, -5]), np.array([0.0, 0.0])

    n_vert = len(xctrl)
    n_seg = n_vert - 1
    # Precompute bisector angle at each interior vertex
    theta_avg = [0.5 * (theta_seg[i - 1] + theta_seg[i])
                 for i in range(1, n_vert - 1)]

    x_curr, z_curr = x_start, z0
    path_x = [x_curr]
    path_z = [z_curr]
    dx = -0.20

    for _ in range(6000):
        if x_curr < -5 or z_curr >= 0.0:
            break

        # Find nearest axial surface to the RIGHT at current depth
        best_xa = np.inf
        best_seg = n_seg - 1  # default: last segment
        for j, i in enumerate(range(1, n_vert - 1)):
            if zctrl[i] >= z_curr:  # vertex at or above particle
                continue
            ta = theta_avg[j]
            ct = np.cos(ta)
            if abs(ct) < 1e-10:
                continue
            t_p = (z_curr - zctrl[i]) / ct
            xa = xctrl[i] + t_p * (-np.sin(ta))
            if xa > x_curr and xa < best_xa:
                best_xa = xa
                best_seg = i - 1

        theta = theta_seg[best_seg]
        dz = dx * np.clip(np.tan(theta), -50, 50)
        x_new = x_curr + dx
        z_new = z_curr + dz

        if z_new >= 0.0 and abs(dz) > 1e-15:
            frac = -z_curr / dz
            path_x.append(x_curr + frac * dx)
            path_z.append(0.0)
            break

        x_curr, z_curr = x_new, z_new
        path_x.append(x_curr)
        path_z.append(z_curr)

    return np.array(path_x), np.array(path_z)


def structural_velocity_discrete(x_obs, xctrl, zctrl, theta_seg, s0):
    """Structural velocity for discrete fault-bend fold.

    Zones separated by axial surface projections at each interior vertex.
    """
    x_obs = np.asarray(x_obs, float)
    n = len(xctrl)

    # Axial surface x-positions at the free surface for interior vertices
    ax_x = []
    for i in range(1, n - 1):
        theta_avg = 0.5 * (theta_seg[i - 1] + theta_seg[i])
        ct = np.cos(theta_avg)
        t = -zctrl[i] / ct
        x_s = xctrl[i] + t * (-np.sin(theta_avg))
        ax_x.append(x_s)
    ax_x = np.sort(ax_x)

    # Zone index via searchsorted: zone j -> theta_seg[j]
    zone = np.searchsorted(ax_x, x_obs, side='right')
    theta_at_obs = theta_seg[zone]

    u = s0 * np.cos(theta_at_obs)
    v = -s0 * np.sin(theta_at_obs)

    # Zero out before tip
    mask = x_obs < xctrl[0]
    u[mask] = 0.0
    v[mask] = 0.0
    return u, v


def coseismic_discrete(x_obs, xctrl, zctrl, theta_seg, s0,
                       locked_segs, locked_verts):
    """Coseismic from discrete fault segments + fold at bend vertices."""
    x_obs = np.asarray(x_obs, float)
    U1 = np.zeros_like(x_obs)
    U2 = np.zeros_like(x_obs)

    # Fault segment contributions
    for i in locked_segs:
        U1 += u1_segment(x_obs, xctrl[i], zctrl[i],
                         xctrl[i + 1], zctrl[i + 1], -s0)
        U2 += u2_segment(x_obs, xctrl[i], zctrl[i],
                         xctrl[i + 1], zctrl[i + 1], -s0)

    # Fold contributions at each locked bend vertex
    for vi in locked_verts:
        delta_theta = theta_seg[vi] - theta_seg[vi - 1]
        theta_avg = 0.5 * (theta_seg[vi - 1] + theta_seg[vi])
        ct = np.cos(theta_avg)
        t = -zctrl[vi] / ct
        x_ax = xctrl[vi] + t * (-np.sin(theta_avg))

        fs = s0 * delta_theta
        U1 += u1_segment(x_obs, x_ax, 0.0, xctrl[vi], zctrl[vi], +fs)
        U2 += u2_segment(x_obs, x_ax, 0.0, xctrl[vi], zctrl[vi], -fs)

    return U1, U2


if __name__ == "__main__":

    s0 = 1.0
    x_plot = 200.0

    # Control points (same as continuous)
    xctrl = np.array([0, 6, 83, 112, 200], dtype=float)
    zctrl = np.array([-0.01, -3, -12, -21, -29], dtype=float)

    n_seg = len(xctrl) - 1
    theta_seg = np.array([
        np.arctan2(zctrl[i + 1] - zctrl[i], xctrl[i + 1] - xctrl[i])
        for i in range(n_seg)])

    # Locking at last interior vertex (x=112, z=-21)
    lock_vertex = 3
    x_lock = xctrl[lock_vertex]
    z_lock = zctrl[lock_vertex]

    locked_segs = [0, 1, 2]
    locked_verts = [1, 2, 3]

    print("Discrete fault-bend fold model")
    for i in range(n_seg):
        print(f"  Seg {i}: ({xctrl[i]:.0f},{zctrl[i]:.2f}) -> "
              f"({xctrl[i+1]:.0f},{zctrl[i+1]:.2f}), "
              f"theta={np.degrees(theta_seg[i]):.1f} deg")
    print(f"Lock at vertex {lock_vertex}: ({x_lock:.0f}, {z_lock:.1f})")

    # Observation points
    x_obs = np.linspace(0.3, x_plot, 600)

    # Velocities
    u_kin, v_kin = structural_velocity_discrete(
        x_obs, xctrl, zctrl, theta_seg, s0)
    u_co, v_co = coseismic_discrete(
        x_obs, xctrl, zctrl, theta_seg, s0, locked_segs, locked_verts)
    u_inter = u_kin - u_co
    v_inter = v_kin - v_co

    # ---- Particle paths (exact discrete geometry) ----
    z_fault_right = zctrl[-1]  # depth at rightmost control point
    z_particles = np.linspace(-0.5, z_fault_right, 21)

    print("Computing particle paths...")
    paths = []
    for z0 in z_particles:
        xp, zp = particle_path_discrete(z0, xctrl, zctrl, theta_seg,
                                         x_start=x_plot)
        paths.append((xp, zp))

    x_grid = np.linspace(-5, x_plot, 800)
    z_on_grid = []
    for xp, zp in paths:
        xp_inc = xp[::-1]
        zp_inc = zp[::-1]
        z_interp = np.interp(x_grid, xp_inc, zp_inc,
                             left=0.0, right=zp_inc[-1])
        z_interp = np.minimum(z_interp, 0.0)
        z_on_grid.append(z_interp)

    # ---- Plotting ----
    fig, axes = plt.subplots(3, 1, figsize=(6, 5),
                              gridspec_kw={"height_ratios": [1, 1, 1.2],
                                           "hspace": 0.05},
                              sharex=True,
                              layout="constrained")

    # Top: vertical velocity
    ax_v = axes[0]
    ax_v.plot(x_obs, v_kin, 'k', lw=1.2, label='Structural')
    ax_v.plot(x_obs, v_co, 'steelblue', lw=1.2, label='Coseismic')
    ax_v.plot(x_obs, v_inter, 'firebrick', lw=1.2, label='Interseismic')
    ax_v.set_ylabel(r'$v_z\;/\;s$')
    ax_v.legend(fontsize=7, frameon=False, loc='lower left')
    ax_v.set_title('Vertical velocity')

    # Middle: horizontal velocity
    ax_h = axes[1]
    ax_h.plot(x_obs, u_kin, 'k', lw=1.2, label='Structural')
    ax_h.plot(x_obs, u_co, 'steelblue', lw=1.2, label='Coseismic')
    ax_h.plot(x_obs, u_inter, 'firebrick', lw=1.2, label='Interseismic')
    ax_h.set_ylabel(r'$v_x\;/\;s$')
    ax_h.legend(fontsize=7, frameon=False, loc='lower left')
    ax_h.set_title('Horizontal velocity')

    # Bottom: fault geometry with particle layering
    ax_g = axes[2]

    # Alternating fills
    for i in range(len(z_on_grid) - 1):
        color = CALTECH_ORANGE if i % 2 == 0 else 'white'
        ax_g.fill_between(x_grid, z_on_grid[i], z_on_grid[i + 1],
                          color=color, alpha=1.0, zorder=1, edgecolor='none')

    # Particle boundary lines
    for xp, zp in paths:
        ax_g.plot(xp, zp, 'k', lw=0.4, zorder=3)

    # Draw axial surfaces only at the 3 bend vertices
    for i in range(1, len(xctrl) - 1):
        theta_avg = 0.5 * (theta_seg[i - 1] + theta_seg[i])
        ct = np.cos(theta_avg)
        t = -zctrl[i] / ct
        x_ax = xctrl[i] + t * (-np.sin(theta_avg))
        ax_g.plot([xctrl[i], x_ax], [zctrl[i], 0], color='grey',
                  lw=0.8, alpha=0.6, zorder=2)

    # Draw fault: locked and creeping as connected polylines
    li = lock_vertex  # index of locking vertex
    ax_g.plot(xctrl[:li + 1], zctrl[:li + 1],
              color='steelblue', lw=1.2, zorder=4, label='Locked',
              solid_joinstyle='round', solid_capstyle='round')
    ax_g.plot(xctrl[li:], zctrl[li:],
              color='firebrick', lw=1.2, zorder=4, label='Creeping',
              solid_joinstyle='round', solid_capstyle='round')

    # Mark locking point
    ax_g.plot(x_lock, z_lock, 'ko', ms=2.5, zorder=5,
              label='Locking point')

    ax_g.axhline(0, color='k', lw=0.8, zorder=5)
    ax_g.set_xlim(0, x_plot)
    ax_g.set_ylim(-35, 5)
    ax_g.set_xlabel('Distance')
    ax_g.set_ylabel('Depth')
    ax_g.legend(fontsize=7, frameon=False, loc='lower left')
    ax_g.set_title('Fault geometry')

    plt.savefig("discrete_fault.pdf", bbox_inches="tight")
    plt.show()
    print("Done.")
