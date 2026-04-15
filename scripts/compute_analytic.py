"""
Generate rotating-GIF visualisations and analytic spectra for
3-dimensional domains and closed 2-manifolds:

  Sphere S²         (closed manifold, spherical harmonics)
  Flat torus T²     (closed manifold, Fourier modes)
  Ball B³           (3D Dirichlet, spherical Bessel × spherical harmonics)
  Box [0,a]×[0,b]×[0,c]  (3D Dirichlet, separable)

Output matches the format used by compute_eigenfunctions.py:
  public/files/eigenfunctions/data/<id>.json
  public/files/eigenfunctions/img/<id>_<bc>_<k>.gif

Then scripts/rebuild_index.py can be re-run to pick them up — but here we
also register corresponding FamilySpec entries in the main script, so a
follow-up rebuild_index picks them up automatically.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import scipy.special as sp

ROOT = Path(__file__).resolve().parents[1]
OUT_IMG = ROOT / "public" / "files" / "eigenfunctions" / "img"
OUT_DATA = ROOT / "public" / "files" / "eigenfunctions" / "data"
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_DATA.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# rotating-GIF writer
# --------------------------------------------------------------------------

def _save_rot_gif(draw_frame: Callable[[Axes3D, float], None], out_path: Path,
                  n_frames: int = 12, size: float = 2.6, dpi: int = 95,
                  fps: int = 8, elev: float = 22.0) -> None:
    fig = plt.figure(figsize=(size, size), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def animate(i):
        ax.clear()
        ax.set_axis_off()
        azim = 360.0 * i / n_frames
        draw_frame(ax, azim)
        ax.view_init(elev=elev, azim=azim)

    ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=1000 // fps)
    ani.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)

# --------------------------------------------------------------------------
# Sphere S² (closed manifold)
# --------------------------------------------------------------------------

def sphere_mode_indices(n_eig: int) -> list[tuple[int, int]]:
    """Return list of (l, m) indices ordered by l; take only m >= 0 and real
    spherical harmonics (we'll plot the real part of Y_l^m to cover both
    cos(m φ) and sin(m φ) components by sign of m)."""
    pairs: list[tuple[int, int]] = []
    l = 0
    while len(pairs) < n_eig:
        for m in range(0, l + 1):
            pairs.append((l, m))
            if len(pairs) >= n_eig:
                break
        l += 1
    return pairs


def _sphere_mesh(N: int = 64):
    phi = np.linspace(0, 2 * np.pi, 2 * N)
    theta = np.linspace(0, np.pi, N)
    P, T = np.meshgrid(phi, theta)
    X = np.sin(T) * np.cos(P)
    Y = np.sin(T) * np.sin(P)
    Z = np.cos(T)
    return X, Y, Z, P, T


def plot_sphere_mode(l: int, m: int, R: float, out_gif: Path) -> float:
    """Plot real Y_l^m on sphere of radius R as rotating GIF. Returns λ."""
    X, Y, Z, P, T = _sphere_mesh(64)
    # Real spherical harmonic: use scipy; take real part for m >= 0.
    Ylm = (sp.sph_harm_y(l, m, T, P) if hasattr(sp, "sph_harm_y") else sp.sph_harm(m, l, P, T)).real
    amax = max(1e-12, np.max(np.abs(Ylm)))
    Ylm_n = Ylm / amax
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)
    colors = cm.RdBu_r(norm(Ylm_n))

    def draw(ax, azim):
        ax.plot_surface(R * X, R * Y, R * Z,
                        facecolors=colors, rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=False)
        lim = R * 1.05
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))

    _save_rot_gif(draw, out_gif, n_frames=18, size=3.0, dpi=100, fps=9)
    return l * (l + 1) / (R * R)


# --------------------------------------------------------------------------
# Flat torus T² = ℝ²/(aZ × bZ)    — rendered as embedded 3D torus
# --------------------------------------------------------------------------

def flat_torus_indices(n_eig: int) -> list[tuple[int, int, str]]:
    """Return (m, n, 'cc' | 'cs' | 'sc' | 'ss') modes sorted by m²+n²."""
    triples = []
    limit = int(math.sqrt(n_eig)) + 3
    for m in range(0, limit):
        for n in range(0, limit):
            for kind in ("cc", "cs", "sc", "ss"):
                # filter degenerate: sin(0)=0
                if m == 0 and kind[0] == "s":
                    continue
                if n == 0 and kind[1] == "s":
                    continue
                triples.append((m, n, kind))
    triples.sort(key=lambda t: (t[0] ** 2 + t[1] ** 2, t[0], t[1]))
    return triples[:n_eig]


def plot_flat_torus_mode(m: int, n: int, kind: str, a: float, b: float,
                         out_gif: Path) -> float:
    # Embed torus: major radius R_maj, minor radius r_min
    R_maj, r_min = 1.6, 0.6
    N_u, N_v = 90, 36
    u = np.linspace(0, 2 * np.pi, N_u)
    v = np.linspace(0, 2 * np.pi, N_v)
    U, V = np.meshgrid(u, v)
    X = (R_maj + r_min * np.cos(V)) * np.cos(U)
    Y = (R_maj + r_min * np.cos(V)) * np.sin(U)
    Z = r_min * np.sin(V)

    # Fundamental coordinates on the torus: x_flat in [0,a], y_flat in [0,b]
    x_flat = a * (U / (2 * np.pi))
    y_flat = b * (V / (2 * np.pi))
    fx = np.cos(2 * np.pi * m * x_flat / a) if kind[0] == "c" else np.sin(2 * np.pi * m * x_flat / a)
    fy = np.cos(2 * np.pi * n * y_flat / b) if kind[1] == "c" else np.sin(2 * np.pi * n * y_flat / b)
    W = fx * fy

    amax = max(1e-12, np.max(np.abs(W)))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)
    colors = cm.RdBu_r(norm(W / amax))

    def draw(ax, azim):
        ax.plot_surface(X, Y, Z, facecolors=colors,
                        rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
        lim = 2.3
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-1.3, 1.3)
        ax.set_box_aspect((1, 1, 0.6))

    _save_rot_gif(draw, out_gif, n_frames=18, size=3.0, dpi=100, fps=9, elev=35)
    return 4 * math.pi ** 2 * (m * m / (a * a) + n * n / (b * b))


# --------------------------------------------------------------------------
# Ball B³ (Dirichlet) — half-cutaway showing interior via hemisphere + slice
# --------------------------------------------------------------------------

def _spherical_bessel_zero(l: int, n: int) -> float:
    """Return the n-th positive zero of j_l (spherical Bessel, n >= 1).
    Uses scipy.optimize.brentq to bracket successive zeros of J_{l+1/2}."""
    from scipy.optimize import brentq
    order = l + 0.5
    def f(x):
        return sp.jv(order, x)
    # zeros are interlaced; first zero is near (n + l/2) * π roughly.
    zeros: list[float] = []
    x = 0.5
    step = 0.5
    while len(zeros) < n:
        x_next = x + step
        try:
            fa, fb = f(x), f(x_next)
        except Exception:
            x = x_next; continue
        if np.isnan(fa) or np.isnan(fb):
            x = x_next; continue
        if fa * fb < 0:
            try:
                z = brentq(f, x, x_next, xtol=1e-10)
                zeros.append(z)
            except Exception:
                pass
        x = x_next
        if x > 200:
            break
    return float(zeros[n - 1])


def ball_mode_indices(n_eig: int) -> list[tuple[int, int, int]]:
    """Return (l, n_radial, m) indices sorted by eigenvalue (R=1)."""
    tuples = []
    for l in range(0, 6):
        for nr in range(1, 4):
            lam = _spherical_bessel_zero(l, nr) ** 2
            for m in range(0, l + 1):
                tuples.append((l, nr, m, lam))
    tuples.sort(key=lambda t: (t[3], t[0], t[1], t[2]))
    return [(l, nr, m) for (l, nr, m, _) in tuples[:n_eig]]


def plot_ball_mode(l: int, nr: int, m: int, R: float, out_gif: Path) -> float:
    """Show eigenfunction on the half-open ball (x <= 0 cut away) so interior
    is visible. The hemisphere (x>0) surface is coloured by Y_l^m; the flat
    equatorial/axial cut is coloured by J_{l+1/2}(√λ r)·Y_l^m."""
    k = _spherical_bessel_zero(l, nr)
    lam = (k / R) ** 2

    # Hemispherical surface (x >= 0): theta ∈ [0,π], phi ∈ [-π/2, π/2]
    N = 40
    phi = np.linspace(-np.pi / 2, np.pi / 2, N)
    theta = np.linspace(0, np.pi, N)
    P, T = np.meshgrid(phi, theta)
    X = R * np.sin(T) * np.cos(P)
    Y = R * np.sin(T) * np.sin(P)
    Z = R * np.cos(T)
    # eigenfunction on sphere surface: J_{l+1/2}(kR/R)·Y_l^m = const · Y_l^m
    # (modulo radial value at r=R: j_l(k) at r=R = 0 for Dirichlet, so u=0 on ∂Ω).
    # So the surface itself is the zero set. Instead, colour by Y_l^m alone to
    # show the angular shape (pedagogically useful), normalised.
    S_surf = (sp.sph_harm_y(l, m, T, P) if hasattr(sp, "sph_harm_y") else sp.sph_harm(m, l, P, T)).real
    s_max = max(1e-12, np.max(np.abs(S_surf)))
    col_surf = cm.RdBu_r(TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)(S_surf / s_max))

    # Cut plane at y = 0 (the x-z plane). Parameterise by (x, z):
    M = 80
    xs = np.linspace(-R, R, M)
    zs = np.linspace(-R, R, M)
    Xc, Zc = np.meshgrid(xs, zs)
    Yc = np.zeros_like(Xc)
    Rr = np.sqrt(Xc * Xc + Zc * Zc)
    mask_in = Rr <= R
    # spherical coords on cut: theta_c = arccos(Zc / Rr), phi_c = atan2(Yc, Xc)
    Tc = np.arccos(np.clip(Zc / np.maximum(Rr, 1e-12), -1, 1))
    Pc = np.arctan2(Yc, Xc)
    # radial part: spherical Bessel j_l(kr/R) = √(π/(2kr/R)) J_{l+1/2}(kr/R)
    arg = k * Rr / R
    with np.errstate(divide="ignore", invalid="ignore"):
        j_l_vals = np.where(Rr > 1e-9,
                            np.sqrt(np.pi / (2 * arg)) * sp.jv(l + 0.5, arg),
                            (1.0 if l == 0 else 0.0))
    S_cut = j_l_vals * (sp.sph_harm_y(l, m, Tc, Pc) if hasattr(sp, "sph_harm_y") else sp.sph_harm(m, l, Pc, Tc)).real
    S_cut = np.where(mask_in, S_cut, np.nan)
    c_max = max(1e-12, np.nanmax(np.abs(S_cut)))
    col_cut = cm.RdBu_r(TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)(np.nan_to_num(S_cut) / c_max))

    def draw(ax, azim):
        # translucent hemisphere surface
        ax.plot_surface(X, Y, Z, facecolors=col_surf, rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=False, alpha=0.55)
        # cut plane
        ax.plot_surface(Xc, Yc, Zc, facecolors=col_cut, rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=False)
        lim = R * 1.05
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))

    _save_rot_gif(draw, out_gif, n_frames=18, size=3.0, dpi=100, fps=9)
    return lam


# --------------------------------------------------------------------------
# Box [0,a]×[0,b]×[0,c] Dirichlet — transparent walls + interior isosurface
# --------------------------------------------------------------------------

def box_mode_indices(n_eig: int, a: float, b: float, c: float) -> list[tuple[int, int, int]]:
    triples = []
    for i in range(1, 8):
        for j in range(1, 8):
            for k in range(1, 8):
                triples.append((i, j, k, math.pi ** 2 * ((i / a) ** 2 + (j / b) ** 2 + (k / c) ** 2)))
    triples.sort(key=lambda t: (t[3], t[0], t[1], t[2]))
    return [(i, j, k) for (i, j, k, _) in triples[:n_eig]]


def plot_box_mode(i: int, j: int, k: int, a: float, b: float, c: float, out_gif: Path) -> float:
    """Box with three visible face cuts (mid-planes) coloured by the mode."""
    lam = math.pi ** 2 * ((i / a) ** 2 + (j / b) ** 2 + (k / c) ** 2)

    # mid-plane slices
    N = 60
    def mode(x, y, z):
        return np.sin(i * np.pi * x / a) * np.sin(j * np.pi * y / b) * np.sin(k * np.pi * z / c)

    xs = np.linspace(0, a, N); ys = np.linspace(0, b, N); zs = np.linspace(0, c, N)

    # plane z = c/2
    Xxy, Yxy = np.meshgrid(xs, ys)
    Vxy = mode(Xxy, Yxy, c / 2)
    # plane y = b/2
    Xxz, Zxz = np.meshgrid(xs, zs)
    Vxz = mode(Xxz, b / 2, Zxz)
    # plane x = a/2
    Yyz, Zyz = np.meshgrid(ys, zs)
    Vyz = mode(a / 2, Yyz, Zyz)

    vmax = max(abs(Vxy).max(), abs(Vxz).max(), abs(Vyz).max(), 1e-12)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)

    C_xy = cm.RdBu_r(norm(Vxy / vmax))
    C_xz = cm.RdBu_r(norm(Vxz / vmax))
    C_yz = cm.RdBu_r(norm(Vyz / vmax))

    # light wireframe of the box
    ex = np.array([[0, a, a, 0, 0], [0, a, a, 0, 0]])
    ey = np.array([[0, 0, b, b, 0], [0, 0, b, b, 0]])
    ez = np.array([[0, 0, 0, 0, 0], [c, c, c, c, c]])

    def draw(ax, azim):
        # three mid-plane slices
        ax.plot_surface(Xxy, Yxy, np.full_like(Xxy, c / 2),
                        facecolors=C_xy, rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=False, alpha=0.95)
        ax.plot_surface(Xxz, np.full_like(Xxz, b / 2), Zxz,
                        facecolors=C_xz, rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=False, alpha=0.85)
        ax.plot_surface(np.full_like(Yyz, a / 2), Yyz, Zyz,
                        facecolors=C_yz, rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=False, alpha=0.85)
        # box wireframe
        ax.plot_wireframe(ex, ey, ez, color="#444", linewidth=0.6)
        ax.plot_wireframe(ex, ey, ez - c, color="#444", linewidth=0.0)  # no-op
        for (x0, y0, x1, y1) in [(0, 0, a, 0), (0, 0, 0, b), (a, 0, a, b), (0, b, a, b)]:
            ax.plot([x0, x1], [y0, y1], [0, 0], color="#666", linewidth=0.6)
            ax.plot([x0, x1], [y0, y1], [c, c], color="#666", linewidth=0.6)
        for (xx, yy) in [(0, 0), (a, 0), (a, b), (0, b)]:
            ax.plot([xx, xx], [yy, yy], [0, c], color="#666", linewidth=0.6)
        lim = max(a, b, c) * 1.05
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_zlim(0, lim)
        ax.set_box_aspect((a, b, c))

    _save_rot_gif(draw, out_gif, n_frames=18, size=3.0, dpi=100, fps=9)
    return lam


# --------------------------------------------------------------------------
# Registry of analytic families
# --------------------------------------------------------------------------

# Each entry: produce list of DomainSpec-ish dicts with custom plotting.

@dataclass
class AnalyticMember:
    id: str
    name_en: str
    name_ja: str
    description_en: str
    description_ja: str
    category: str
    family_id: str
    family_param: float
    n_eig: int
    kind: str           # 'sphere' | 'torus' | 'ball' | 'box'
    params: dict        # e.g. {'R': 1.0} or {'a':1,'b':1,'c':1}


def _write_entry(member: AnalyticMember) -> dict:
    """Compute eigenvalues + plot rotating GIFs for all modes; write JSON."""
    modes = []
    for k in range(member.n_eig):
        img_path = OUT_IMG / f"{member.id}_dirichlet_{k + 1}.gif"
        if member.kind == "sphere":
            l, m = sphere_mode_indices(member.n_eig)[k]
            lam = plot_sphere_mode(l, m, member.params["R"], img_path)
            label = f"Y_{{{l}{m}}}"
        elif member.kind == "torus":
            mi, ni, kind = flat_torus_indices(member.n_eig)[k]
            lam = plot_flat_torus_mode(mi, ni, kind, member.params["a"], member.params["b"], img_path)
            label = f"({mi},{ni},{kind})"
        elif member.kind == "ball":
            l, nr, m = ball_mode_indices(member.n_eig)[k]
            lam = plot_ball_mode(l, nr, m, member.params["R"], img_path)
            label = f"(l={l},n={nr},m={m})"
        elif member.kind == "box":
            i, j, kk = box_mode_indices(
                member.n_eig, member.params["a"], member.params["b"], member.params["c"]
            )[k]
            lam = plot_box_mode(
                i, j, kk, member.params["a"], member.params["b"], member.params["c"], img_path
            )
            label = f"({i},{j},{kk})"
        else:
            raise ValueError(member.kind)
        modes.append({
            "k": k + 1, "lambda": float(lam),
            "lambdaExact": float(lam), "relErr": 0.0,
            "image": f"/files/eigenfunctions/img/{member.id}_dirichlet_{k + 1}.gif",
            "label": label,
        })
    d = {
        "id": member.id, "nameEn": member.name_en, "nameJa": member.name_ja,
        "descriptionEn": member.description_en, "descriptionJa": member.description_ja,
        "category": member.category,
        "reference": "",
        "familyId": member.family_id,
        "familyParam": member.family_param,
        "mesh": {"vertices": 0, "triangles": 0, "element": "analytic (closed form)"},
        "analytic": True,
        "dimension": "3d" if member.kind in ("ball", "box") else "manifold-2",
        "boundaries": {"dirichlet": {"modes": modes}},
    }
    (OUT_DATA / f"{member.id}.json").write_text(json.dumps(d, indent=2, ensure_ascii=False))
    print(f"  wrote {member.id}", flush=True)
    return d


def build_members() -> list[AnalyticMember]:
    out: list[AnalyticMember] = []
    # ---- Sphere S² radius r ----
    for r in [round(0.5 + 0.25 * i, 2) for i in range(0, 12)]:
        out.append(AnalyticMember(
            id=f"sphere-{r}",
            name_en=f"Sphere S² (R={r})",
            name_ja=f"球面 S² (R={r})",
            description_en="Unit spherical harmonics Y_l^m on the round 2-sphere of radius R. Spectrum λ_l = l(l+1)/R² with multiplicity 2l+1.",
            description_ja="半径 R の 2-球面上の球面調和関数 Y_l^m。スペクトルは λ_l = l(l+1)/R² で重複度 2l+1。",
            category="manifold",
            family_id="sphere",
            family_param=r,
            n_eig=6,
            kind="sphere",
            params={"R": float(r)},
        ))
    # ---- Flat torus T² — aspect ratio e = a/b, a = 1 ----
    for e in [round(1.0 + 0.4 * i, 2) for i in range(0, 10)]:
        out.append(AnalyticMember(
            id=f"torus-{e}",
            name_en=f"Flat torus T² (a/b={e})",
            name_ja=f"平坦トーラス T² (a/b={e})",
            description_en="Flat torus ℝ²/(aZ × bZ) with a/b = e. Spectrum 4π²(m²/a² + n²/b²); eigenfunctions cos/sin products.",
            description_ja="平坦トーラス ℝ²/(aZ × bZ)（a/b = e）。スペクトル 4π²(m²/a² + n²/b²)、固有関数は cos/sin の積。",
            category="manifold",
            family_id="torus",
            family_param=e,
            n_eig=6,
            kind="torus",
            params={"a": 1.0, "b": 1.0 / float(e)},
        ))
    # ---- Ball B³ radius R ----
    for r in [round(0.5 + 0.25 * i, 2) for i in range(0, 10)]:
        out.append(AnalyticMember(
            id=f"ball-{r}",
            name_en=f"Ball B³ (R={r})",
            name_ja=f"球体 B³ (R={r})",
            description_en="3D Dirichlet Laplacian on the open ball of radius R. Eigenvalues are squares of the zeros of spherical Bessel j_l.",
            description_ja="半径 R の開球上の 3 次元 Dirichlet Laplacian。固有値は球ベッセル j_l の零点の二乗。",
            category="three-d",
            family_id="ball",
            family_param=r,
            n_eig=6,
            kind="ball",
            params={"R": float(r)},
        ))
    # ---- Box [0,a]×[0,1]×[0,1] — aspect ratio ----
    for a in [round(1.0 + 0.4 * i, 2) for i in range(0, 10)]:
        out.append(AnalyticMember(
            id=f"box-{a}",
            name_en=f"Box [0,{a}]×[0,1]×[0,1]",
            name_ja=f"直方体 [0,{a}]×[0,1]×[0,1]",
            description_en="3D Dirichlet Laplacian on an axis-aligned box. Separable spectrum π²((i/a)² + j² + k²).",
            description_ja="軸に沿った直方体上の 3 次元 Dirichlet Laplacian。変数分離スペクトル π²((i/a)² + j² + k²)。",
            category="three-d",
            family_id="box",
            family_param=a,
            n_eig=6,
            kind="box",
            params={"a": float(a), "b": 1.0, "c": 1.0},
        ))
    return out


# Family-level metadata consumed by the atlas page (same shape as
# FamilySpec in compute_eigenfunctions.py).

ANALYTIC_FAMILIES = [
    {
        "id": "sphere",
        "nameEn": "Sphere S² (radius R)",
        "nameJa": "球面 S² (半径 R)",
        "descriptionEn": "Round 2-sphere of radius R with its Laplace–Beltrami operator. Spectrum λ_l = l(l+1)/R² with multiplicity 2l+1.",
        "descriptionJa": "半径 R の 2-球面の Laplace–Beltrami 作用素。スペクトル λ_l = l(l+1)/R² は重複度 2l+1。",
        "category": "manifold",
        "param": "R", "paramJa": "R",
        "paramValues": [round(0.5 + 0.25 * i, 2) for i in range(0, 12)],
        "memberIds": [f"sphere-{round(0.5 + 0.25 * i, 2)}" for i in range(0, 12)],
        "reference": "",
    },
    {
        "id": "torus",
        "nameEn": "Flat torus T² (aspect ratio e=a/b)",
        "nameJa": "平坦トーラス T² (軸比 e=a/b)",
        "descriptionEn": "Flat torus ℝ²/(Z × bZ). Spectrum 4π²(m² + (n/b)²); eigenfunctions are products of sines and cosines.",
        "descriptionJa": "平坦トーラス ℝ²/(Z × bZ)。スペクトル 4π²(m² + (n/b)²)、固有関数は三角関数の積。",
        "category": "manifold",
        "param": "e", "paramJa": "e",
        "paramValues": [round(1.0 + 0.4 * i, 2) for i in range(0, 10)],
        "memberIds": [f"torus-{round(1.0 + 0.4 * i, 2)}" for i in range(0, 10)],
        "reference": "",
    },
    {
        "id": "ball",
        "nameEn": "Ball B³ (radius R)",
        "nameJa": "球体 B³ (半径 R)",
        "descriptionEn": "Open 3-ball of radius R with the Dirichlet Laplacian. Eigenvalues are squared zeros of spherical Bessel j_l; shown as a rotating GIF with a transparent hemisphere and a coloured cut-plane.",
        "descriptionJa": "半径 R の開球の Dirichlet Laplacian。固有値は球ベッセル j_l の零点の二乗。回転 GIF（半透明の半球 + 切断面に色付け）で表示。",
        "category": "three-d",
        "param": "R", "paramJa": "R",
        "paramValues": [round(0.5 + 0.25 * i, 2) for i in range(0, 10)],
        "memberIds": [f"ball-{round(0.5 + 0.25 * i, 2)}" for i in range(0, 10)],
        "reference": "",
    },
    {
        "id": "box",
        "nameEn": "Box [0,a]×[0,1]×[0,1]",
        "nameJa": "直方体 [0,a]×[0,1]×[0,1]",
        "descriptionEn": "Axis-aligned rectangular box with Dirichlet boundary. Separable spectrum. Rotating GIF shows three mid-plane slices coloured by the eigenfunction.",
        "descriptionJa": "軸平行の直方体（Dirichlet）。変数分離可能。回転 GIF で 3 つの中央切断面に固有関数を色付けして表示。",
        "category": "three-d",
        "param": "a", "paramJa": "a",
        "paramValues": [round(1.0 + 0.4 * i, 2) for i in range(0, 10)],
        "memberIds": [f"box-{round(1.0 + 0.4 * i, 2)}" for i in range(0, 10)],
        "reference": "",
    },
]


def main():
    for m in build_members():
        _write_entry(m)
    # Append family metadata via rebuild_index — but since rebuild_index
    # only reads FAMILIES from the main script, we write a small
    # companion file here that the main rebuild_index consumes.
    (OUT_DATA / "_analytic_families.json").write_text(
        json.dumps(ANALYTIC_FAMILIES, indent=2, ensure_ascii=False)
    )
    print("wrote _analytic_families.json")


if __name__ == "__main__":
    main()
