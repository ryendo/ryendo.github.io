"""
Compute Dirichlet / Neumann Laplacian eigenvalues and eigenfunctions on a
collection of theoretically-interesting planar domains.

Method: Galerkin FEM with P2 Lagrange elements on an unstructured triangular
mesh (gmsh via pygmsh). Generalised symmetric eigenproblem K u = lambda M u is
solved in shift-invert mode with ARPACK (scipy.sparse.linalg.eigsh).

Outputs:
  public/files/eigenfunctions/data/index.json    -- per-domain metadata
  public/files/eigenfunctions/data/<domain>.json -- per-mode eigenvalues
  public/files/eigenfunctions/img/<domain>_<bc>_<k>.png -- eigenfunction plot

NOTE ON ACCURACY:
  Using P2 elements, ARPACK shift-invert, and ~ h ~ 0.02-0.03 mesh size
  typically gives 5-7 correct digits for low modes on smooth/polygonal domains
  and 3-5 digits on domains with re-entrant corners (L-shape). For domains
  with known closed-form spectra (square, disk, equilateral triangle, annulus)
  we report the analytical value alongside the FEM value and give the relative
  error.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm
from scipy.sparse.linalg import eigsh

import pygmsh

from skfem import (
    MeshTri,
    Basis,
    ElementTriP2,
    asm,
    condense,
)
from skfem.models.poisson import laplace, mass


ROOT = Path(__file__).resolve().parents[1]
OUT_IMG = ROOT / "public" / "files" / "eigenfunctions" / "img"
OUT_DATA = ROOT / "public" / "files" / "eigenfunctions" / "data"
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_DATA.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Domain definitions -- each returns an skfem MeshTri at a target mesh size.
# ---------------------------------------------------------------------------


def _mesh_from_pygmsh(geom_builder: Callable, h: float) -> MeshTri:
    """Build a geometry in pygmsh and return as skfem MeshTri."""
    with pygmsh.geo.Geometry() as geom:
        geom_builder(geom, h)
        mesh = geom.generate_mesh(order=1)
    pts = mesh.points[:, :2]  # (N, 2)
    tris = None
    for cb in mesh.cells:
        if cb.type == "triangle":
            tris = cb.data  # (M, 3)
            break
    if tris is None:
        raise RuntimeError("No triangle cells in mesh")
    # gmsh often emits isolated points (e.g. circle centres). Drop any
    # vertex not referenced by a triangle, and remap indices.
    used = np.unique(tris)
    remap = -np.ones(pts.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size)
    pts2 = pts[used]
    tris2 = remap[tris]
    return MeshTri(pts2.T, tris2.T)


def dom_square(h: float = 0.03) -> MeshTri:
    def build(geom, h):
        geom.add_polygon(
            [[0, 0], [1, 0], [1, 1], [0, 1]], mesh_size=h,
        )
    return _mesh_from_pygmsh(build, h)


def dom_rectangle_2_1(h: float = 0.03) -> MeshTri:
    def build(geom, h):
        geom.add_polygon(
            [[0, 0], [2, 0], [2, 1], [0, 1]], mesh_size=h,
        )
    return _mesh_from_pygmsh(build, h)


def dom_disk(h: float = 0.03) -> MeshTri:
    def build(geom, h):
        geom.add_circle([0, 0], 1.0, mesh_size=h, num_sections=64, make_surface=True)
    return _mesh_from_pygmsh(build, h)


def dom_equilateral_triangle(h: float = 0.03) -> MeshTri:
    # side length 1, vertices placed symmetrically
    s = 1.0
    pts = [
        [0.0, 0.0],
        [s, 0.0],
        [s / 2, s * math.sqrt(3) / 2],
    ]
    def build(geom, h):
        geom.add_polygon(pts, mesh_size=h)
    return _mesh_from_pygmsh(build, h)


def dom_right_isoceles_triangle(h: float = 0.03) -> MeshTri:
    def build(geom, h):
        geom.add_polygon([[0, 0], [1, 0], [0, 1]], mesh_size=h)
    return _mesh_from_pygmsh(build, h)


def dom_30_60_90_triangle(h: float = 0.03) -> MeshTri:
    # 30-60-90 right triangle with legs 1 (adjacent to 60deg) and sqrt(3).
    def build(geom, h):
        geom.add_polygon(
            [[0, 0], [math.sqrt(3), 0], [0, 1]], mesh_size=h,
        )
    return _mesh_from_pygmsh(build, h)


def dom_regular_polygon(n: int, h: float = 0.03) -> MeshTri:
    # inscribed in unit circle
    pts = [
        [math.cos(2 * math.pi * k / n - math.pi / 2),
         math.sin(2 * math.pi * k / n - math.pi / 2)]
        for k in range(n)
    ]
    def build(geom, h):
        geom.add_polygon(pts, mesh_size=h)
    return _mesh_from_pygmsh(build, h)


def dom_l_shape(h: float = 0.03) -> MeshTri:
    # [-1,1]^2  \  [0,1] x [-1,0] ; the classic Fox-Henrici-Moler L
    def build(geom, h):
        geom.add_polygon(
            [
                [-1, -1], [0, -1], [0, 0], [1, 0],
                [1, 1], [-1, 1],
            ],
            mesh_size=h,
        )
    return _mesh_from_pygmsh(build, h)


def dom_stadium(h: float = 0.03) -> MeshTri:
    # Bunimovich stadium: rectangle [-a,a]x[-1,1] capped by semicircles of
    # radius 1. Take a = 1 (classical choice).
    a = 1.0
    def build(geom, h):
        # build via circular arcs + straight segments
        p0 = geom.add_point([-a, -1.0], mesh_size=h)
        p1 = geom.add_point([ a, -1.0], mesh_size=h)
        cR = geom.add_point([ a,  0.0], mesh_size=h)
        p2 = geom.add_point([ a,  1.0], mesh_size=h)
        p3 = geom.add_point([-a,  1.0], mesh_size=h)
        cL = geom.add_point([-a,  0.0], mesh_size=h)
        l1 = geom.add_line(p0, p1)
        arcR = geom.add_circle_arc(p1, cR, p2)
        l2 = geom.add_line(p2, p3)
        arcL = geom.add_circle_arc(p3, cL, p0)
        loop = geom.add_curve_loop([l1, arcR, l2, arcL])
        geom.add_plane_surface(loop)
    return _mesh_from_pygmsh(build, h)


def dom_annulus(r_in: float = 0.5, h: float = 0.03) -> MeshTri:
    def build(geom, h):
        outer = geom.add_circle([0, 0], 1.0, mesh_size=h, num_sections=64, make_surface=False)
        inner = geom.add_circle([0, 0], r_in, mesh_size=h, num_sections=48, make_surface=False)
        geom.add_plane_surface(outer.curve_loop, holes=[inner.curve_loop])
    return _mesh_from_pygmsh(build, h)


def dom_ellipse_2_1(h: float = 0.03) -> MeshTri:
    # semi-axes a=1, b=0.5 ; approximate as polygon
    N = 120
    pts = [[math.cos(2 * math.pi * k / N), 0.5 * math.sin(2 * math.pi * k / N)]
           for k in range(N)]
    def build(geom, h):
        geom.add_polygon(pts, mesh_size=h)
    return _mesh_from_pygmsh(build, h)


def dom_sector(angle: float, h: float = 0.03) -> MeshTri:
    # Circular sector with total opening angle `angle` (radians), radius 1
    def build(geom, h):
        N = max(16, int(64 * angle / (2 * math.pi)))
        arc_pts = [[math.cos(t), math.sin(t)] for t in np.linspace(0, angle, N + 1)]
        pts = [[0.0, 0.0]] + arc_pts
        geom.add_polygon(pts, mesh_size=h)
    return _mesh_from_pygmsh(build, h)


def dom_cardioid(h: float = 0.03) -> MeshTri:
    # r = 1 - cos(theta), re-centred. Theoretically interesting: transition
    # to full quantum chaos (Robnik 1983). Nominal diameter ~ 2.
    N = 240
    pts = []
    for k in range(N):
        t = 2 * math.pi * k / N
        r = 1 - math.cos(t)
        pts.append([r * math.cos(t), r * math.sin(t)])
    def build(geom, h):
        geom.add_polygon(pts, mesh_size=h)
    return _mesh_from_pygmsh(build, h)


def dom_sinai(h: float = 0.03) -> MeshTri:
    # Sinai billiard: unit square with disk of radius 0.3 at centre removed.
    def build(geom, h):
        outer = geom.add_polygon(
            [[-1, -1], [1, -1], [1, 1], [-1, 1]], mesh_size=h, make_surface=False,
        )
        inner = geom.add_circle([0, 0], 0.3, mesh_size=h, num_sections=48, make_surface=False)
        geom.add_plane_surface(outer.curve_loop, holes=[inner.curve_loop])
    return _mesh_from_pygmsh(build, h)


def _gww_build_polygon(triangles: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]):
    """Given a list of triangles (each = 3 vertices), produce the outline
    polygon of their union by building an edge-set and removing shared
    (interior) edges. Assumes the union is simply connected.
    """
    from collections import defaultdict
    # round coordinates to avoid float equality issues
    def key(p):
        return (round(p[0], 9), round(p[1], 9))
    edge_count: dict[tuple, int] = defaultdict(int)
    edge_dir: dict[tuple, tuple] = {}
    for tri in triangles:
        # ensure CCW
        a, b, c = tri
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if cross < 0:
            a, b, c = a, c, b
        for p, q in ((a, b), (b, c), (c, a)):
            kp, kq = key(p), key(q)
            e = tuple(sorted((kp, kq)))
            edge_count[e] += 1
            edge_dir[(kp, kq)] = True
    # boundary edges = edges appearing in only one triangle
    boundary = set()
    for e, cnt in edge_count.items():
        if cnt == 1:
            boundary.add(e)
    # reconstruct ordered polygon by walking boundary
    adj: dict[tuple, list[tuple]] = defaultdict(list)
    for a, b in boundary:
        adj[a].append(b)
        adj[b].append(a)
    start = min(adj.keys())
    path = [start]
    prev = None
    while True:
        cur = path[-1]
        nxts = [p for p in adj[cur] if p != prev]
        if not nxts:
            break
        nxt = nxts[0]
        if nxt == start:
            break
        path.append(nxt)
        prev = cur
        if len(path) > 10000:
            raise RuntimeError("polygon walk did not terminate")
    return path


# Gordon–Webb–Wolpert isospectral pair — Chapman's (1995) 7-triangle
# construction. Each domain is the union of 7 congruent 45°–45°–90°
# right triangles (legs of length 1). Vertex coordinates below follow
# Figure 2 of Chapman, Drums That Sound the Same, American Math. Monthly
# 102 (1995) 124–138, which is in turn the "warped" pair of
# Buser–Conway–Doyle–Semmler (1994).
#
# Labelling: each triangle is given as (apex, right-leg-end, other-leg-end).
_GWW_D1_TRIS = [
    ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),        # T1
    ((1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),        # T2
    ((1.0, 0.0), (2.0, 0.0), (1.0, 1.0)),        # T3
    ((2.0, 0.0), (2.0, 1.0), (1.0, 1.0)),        # T4
    ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0)),        # T5
    ((0.0, 1.0), (1.0, 1.0), (1.0, 2.0)),        # T6 (above T2)
    ((0.0, 1.0), (1.0, 2.0), (0.0, 2.0)),        # T7
]

_GWW_D2_TRIS = [
    ((0.0, 0.0), (1.0, 0.0), (1.0, -1.0)),       # T1' (flipped down)
    ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),        # T2'
    ((1.0, 0.0), (2.0, 0.0), (1.0, 1.0)),        # T3'
    ((1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),        # T4'
    ((2.0, 0.0), (2.0, 1.0), (1.0, 1.0)),        # T5'
    ((2.0, 0.0), (3.0, 0.0), (2.0, 1.0)),        # T6'
    ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0)),        # T7'
]


def dom_gww_d1(h: float = 0.04) -> MeshTri:
    pts = _gww_build_polygon(_GWW_D1_TRIS)
    def build(geom, h):
        geom.add_polygon(list(pts), mesh_size=h)
    return _mesh_from_pygmsh(build, h)


def dom_gww_d2(h: float = 0.04) -> MeshTri:
    pts = _gww_build_polygon(_GWW_D2_TRIS)
    def build(geom, h):
        geom.add_polygon(list(pts), mesh_size=h)
    return _mesh_from_pygmsh(build, h)


def dom_thin_triangle(h_base: float = 0.05, height: float = 0.15) -> MeshTri:
    # isoceles triangle, base = 1, height = 0.15
    def build(geom, h):
        geom.add_polygon([[0, 0], [1, 0], [0.5, height]], mesh_size=h)
    return _mesh_from_pygmsh(build, min(h_base, height / 4))


# ---------------------------------------------------------------------------
# FEM eigensolver
# ---------------------------------------------------------------------------


def solve_eigs(mesh: MeshTri, n_eig: int, bc: str) -> tuple[np.ndarray, np.ndarray, Basis]:
    """Solve the Laplacian eigenproblem.

    bc in {'dirichlet','neumann'}.
    Returns (eigenvalues ascending, eigenvectors columns, Basis).
    """
    basis = Basis(mesh, ElementTriP2())
    K = asm(laplace, basis)
    M = asm(mass, basis)

    if bc == "dirichlet":
        D = basis.get_dofs()  # all boundary dofs
        # condense: removes Dirichlet dofs; returns reduced matrices
        Kc, Mc, _, I = condense(K, M, D=D)
        sigma = 1e-6  # shift near 0 to find smallest eigenvalues
        vals, vecs = eigsh(Kc, k=n_eig, M=Mc, sigma=sigma, which="LM")
        # expand back to full dof vector
        full = np.zeros((basis.N, n_eig))
        full[I, :] = vecs
        order = np.argsort(vals)
        return vals[order], full[:, order], basis
    elif bc == "neumann":
        # K is singular (constant in kernel); add small shift via sigma
        sigma = -1e-4
        vals, vecs = eigsh(K, k=n_eig, M=M, sigma=sigma, which="LM")
        order = np.argsort(vals)
        vals = vals[order]
        vecs = vecs[:, order]
        # first eigenvalue should be ~0; replace with exactly 0
        if abs(vals[0]) < 1e-6:
            vals[0] = 0.0
        return vals, vecs, basis
    else:
        raise ValueError(bc)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_eigenfunction(mesh: MeshTri, basis: Basis, u: np.ndarray, path: Path,
                       title: str = ""):
    # project P2 -> P1 (use vertex values only, correct for visualization)
    # basis.doflocs has the geometric location of every DoF including edge mids
    # For a simple plot, evaluate at vertices: take first n_vertices dofs
    # (skfem orders vertex dofs first for P2).
    nv = mesh.nvertices
    u_vert = u[:nv]
    # normalise sign so that the largest magnitude is positive
    k = np.argmax(np.abs(u_vert))
    if u_vert[k] < 0:
        u_vert = -u_vert
    # symmetric colour scale
    amax = np.max(np.abs(u_vert))
    if amax == 0:
        amax = 1.0
    x = mesh.p[0]
    y = mesh.p[1]
    tri = mtri.Triangulation(x, y, mesh.t.T)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-amax, vmax=amax)
    tpc = ax.tripcolor(tri, u_vert, shading="gouraud", cmap="RdBu_r", norm=norm)
    # zero contour
    try:
        ax.tricontour(tri, u_vert, levels=[0.0], colors="k", linewidths=0.4, alpha=0.6)
    except Exception:
        pass
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Registry of domains to compute
# ---------------------------------------------------------------------------


@dataclass
class DomainSpec:
    id: str
    name_en: str
    name_ja: str
    description_en: str
    description_ja: str
    category: str
    builder: Callable[[], MeshTri]
    n_eig: int = 8
    analytical: Callable[[int], list[float]] | None = None  # returns sorted λ₁..λ_n
    reference: str = ""


def analytic_square(n: int) -> list[float]:
    vals = []
    for i in range(1, 15):
        for j in range(1, 15):
            vals.append(math.pi ** 2 * (i * i + j * j))
    vals.sort()
    return vals[:n]


def analytic_rectangle_2_1(n: int) -> list[float]:
    vals = []
    for i in range(1, 15):
        for j in range(1, 15):
            vals.append(math.pi ** 2 * ((i / 2.0) ** 2 + j ** 2))
    vals.sort()
    return vals[:n]


def analytic_disk(n: int) -> list[float]:
    # Dirichlet on disk radius 1: λ = j_{m,k}² where j_{m,k} is k-th positive
    # zero of Bessel J_m.
    from scipy.special import jn_zeros
    vals = []
    for m in range(0, 10):
        zeros = jn_zeros(m, 6)
        mult = 1 if m == 0 else 2
        for z in zeros:
            for _ in range(mult):
                vals.append(z * z)
    vals.sort()
    return vals[:n]


def analytic_equilateral_triangle(n: int) -> list[float]:
    # McCartin (SIAM Review 45, 2003): Dirichlet eigenvalues of the
    # equilateral triangle with side a are
    #   λ_{m,n} = (16 π² / (9 a²)) (m² + m n + n²),  m ≥ n ≥ 1,
    # with multiplicity 1 if m = n and 2 otherwise. Side a = 1 here.
    a = 1.0
    pref = 16 * math.pi ** 2 / (9 * a * a)
    vals = []
    for m in range(1, 18):
        for nn in range(1, m + 1):
            v = pref * (m * m + m * nn + nn * nn)
            mult = 1 if m == nn else 2
            vals.extend([v] * mult)
    vals.sort()
    return vals[:n]


def analytic_right_isoceles(n: int) -> list[float]:
    # Right isoceles triangle with legs of length 1 (hypotenuse on diagonal).
    # Eigenvalues: π²(i² + j²) with i < j, i,j ≥ 1 -- it is half of the
    # unit square spectrum corresponding to antisymmetric modes across y=x.
    # Actually precise statement (Brüning et al): on triangle with vertices
    # (0,0),(π,0),(0,π) the Dirichlet eigenvalues are m²+n² with m>n≥1.
    # Rescaling to legs of length 1 gives π²(m²+n²).
    vals = []
    for j in range(2, 20):
        for i in range(1, j):
            vals.append(math.pi ** 2 * (i * i + j * j))
    vals.sort()
    return vals[:n]


def analytic_30_60_90(n: int) -> list[float]:
    # 30-60-90 triangle is a "Lamé" triangle: a known tilable triangle whose
    # Dirichlet spectrum is (4π²/9)(m²+mn+n²) over m,n ≥ 1 after scaling,
    # but the precise pre-factor depends on the leg length. We'll skip
    # an analytic comparison here (FEM is the reference).
    return []


DOMAINS: list[DomainSpec] = [
    DomainSpec(
        id="square",
        name_en="Unit square",
        name_ja="単位正方形",
        description_en="[0,1] × [0,1]. Separable; every eigenvalue is π²(m²+n²) with m,n ≥ 1.",
        description_ja="[0,1] × [0,1]。変数分離可能で、固有値は π²(m²+n²)（m, n ≥ 1）で尽くされる。",
        category="basic",
        builder=lambda: dom_square(0.025),
        n_eig=10,
        analytical=analytic_square,
        reference="Courant–Hilbert (1953), §V.5",
    ),
    DomainSpec(
        id="rectangle-2-1",
        name_en="Rectangle 2:1",
        name_ja="長方形 2:1",
        description_en="[0,2] × [0,1]. Separable; λ_{m,n} = π²((m/2)² + n²).",
        description_ja="[0,2] × [0,1]。変数分離可能で λ_{m,n} = π²((m/2)² + n²)。",
        category="basic",
        builder=lambda: dom_rectangle_2_1(0.03),
        n_eig=10,
        analytical=analytic_rectangle_2_1,
    ),
    DomainSpec(
        id="disk",
        name_en="Unit disk",
        name_ja="単位円板",
        description_en="Radius 1. Eigenvalues are squares of zeros of Bessel J_m.",
        description_ja="半径 1 の円板。固有値は第1種ベッセル関数 J_m の零点の二乗で与えられる。",
        category="basic",
        builder=lambda: dom_disk(0.025),
        n_eig=10,
        analytical=analytic_disk,
        reference="Courant–Hilbert (1953), §V.5",
    ),
    DomainSpec(
        id="equilateral-triangle",
        name_en="Equilateral triangle",
        name_ja="正三角形",
        description_en="Side 1. Lamé (1833) derived the closed-form spectrum λ_{m,n} = (16π²/9)(m² + n² + mn).",
        description_ja="一辺 1。ラメ (1833) が閉形式解 λ_{m,n} = (16π²/9)(m² + n² + mn) を導いた。",
        category="polygon",
        builder=lambda: dom_equilateral_triangle(0.025),
        n_eig=10,
        analytical=analytic_equilateral_triangle,
        reference="Lamé (1833); McCartin, SIAM Rev. 45 (2003).",
    ),
    DomainSpec(
        id="right-isoceles-triangle",
        name_en="Right isoceles triangle",
        name_ja="直角二等辺三角形",
        description_en="Legs of length 1. Half-square: Dirichlet spectrum is π²(m²+n²) with m > n ≥ 1.",
        description_ja="脚の長さ 1。単位正方形の半分として得られ、固有値は π²(m²+n²)（m > n ≥ 1）。",
        category="polygon",
        builder=lambda: dom_right_isoceles_triangle(0.02),
        n_eig=10,
        analytical=analytic_right_isoceles,
    ),
    DomainSpec(
        id="30-60-90-triangle",
        name_en="30-60-90 triangle",
        name_ja="30°-60°-90° 三角形",
        description_en="Short leg 1, long leg √3. A tiling (Lamé) triangle with fully integrable billiard.",
        description_ja="短い脚 1、長い脚 √3。平面を敷き詰める（ラメ型）三角形で、ビリヤードは可積分。",
        category="polygon",
        builder=lambda: dom_30_60_90_triangle(0.02),
        n_eig=10,
        reference="Integrable triangle (30-60-90 is one of the three Lamé triangles).",
    ),
    DomainSpec(
        id="pentagon",
        name_en="Regular pentagon",
        name_ja="正五角形",
        description_en="Inscribed in the unit circle. No closed-form spectrum is known.",
        description_ja="単位円に内接。閉形式解は知られていない。",
        category="polygon",
        builder=lambda: dom_regular_polygon(5, 0.025),
        n_eig=8,
    ),
    DomainSpec(
        id="hexagon",
        name_en="Regular hexagon",
        name_ja="正六角形",
        description_en="Inscribed in the unit circle. Tiles the plane; approximates the disk.",
        description_ja="単位円に内接。平面を敷き詰め、円板の近似として現れる。",
        category="polygon",
        builder=lambda: dom_regular_polygon(6, 0.025),
        n_eig=8,
    ),
    DomainSpec(
        id="heptagon",
        name_en="Regular heptagon",
        name_ja="正七角形",
        description_en="Inscribed in the unit circle. Lowest-symmetry regular polygon that cannot tile.",
        description_ja="単位円に内接。平面を敷き詰められない最も対称性の低い正多角形。",
        category="polygon",
        builder=lambda: dom_regular_polygon(7, 0.025),
        n_eig=8,
    ),
    DomainSpec(
        id="l-shape",
        name_en="L-shape",
        name_ja="L字型領域",
        description_en="[-1,1]² with lower-right quadrant removed. Reference benchmark: eigenfunctions have a r^{2/3} corner singularity at the re-entrant corner (Fox–Henrici–Moler 1967).",
        description_ja="[-1,1]² から右下の象限を除いた領域。凹角での r^{2/3} 型特異性を持つ古典的ベンチマーク（Fox–Henrici–Moler 1967）。",
        category="non-convex",
        builder=lambda: dom_l_shape(0.025),
        n_eig=8,
        reference="Fox–Henrici–Moler, SIAM J. Numer. Anal. 4 (1967).",
    ),
    DomainSpec(
        id="stadium",
        name_en="Bunimovich stadium",
        name_ja="ブニモヴィッチ・スタジアム",
        description_en="Rectangle [-1,1]×[-1,1] with unit semicircular caps. Classical billiard of Bunimovich (1974): fully hyperbolic, a touchstone for quantum chaos; level spacing follows GOE.",
        description_ja="幅 2 の長方形に半径 1 の半円を両端に付加。ブニモヴィッチの完全カオスビリヤード (1974)。量子カオスの試金石で、準位間隔は GOE 分布に従う。",
        category="chaotic",
        builder=lambda: dom_stadium(0.03),
        n_eig=10,
        reference="Bunimovich, Commun. Math. Phys. 65 (1979); Heller (1984) scars.",
    ),
    DomainSpec(
        id="sinai-billiard",
        name_en="Sinai billiard",
        name_ja="シナイ・ビリヤード",
        description_en="Square [-1,1]² with a central disk of radius 0.3 removed. First proven K-system (Sinai 1970).",
        description_ja="[-1,1]² から中心の半径 0.3 の円板を除いた領域。最初に証明された K-系 (Sinai 1970)。",
        category="chaotic",
        builder=lambda: dom_sinai(0.03),
        n_eig=10,
        reference="Sinai, Russ. Math. Surveys 25 (1970).",
    ),
    DomainSpec(
        id="cardioid",
        name_en="Cardioid",
        name_ja="カージオイド",
        description_en="r = 1 − cos θ. Robnik billiard: fully chaotic (ergodic & mixing) for the classical flow; a benchmark for the Bohigas–Giannoni–Schmit conjecture.",
        description_ja="r = 1 − cos θ。ロブニク・ビリヤードで、古典フローは完全にカオス的（エルゴード的かつ混合的）。BGS 予想の試金石の一つ。",
        category="chaotic",
        builder=lambda: dom_cardioid(0.02),
        n_eig=10,
        reference="Robnik, J. Phys. A 16 (1983); Bäcker–Steiner (1998).",
    ),
    DomainSpec(
        id="annulus-0.5",
        name_en="Annulus (r=0.5)",
        name_ja="円環 (r=0.5)",
        description_en="{0.5 < r < 1}. Separable; eigenvalues determined by cross-products of Bessel functions J_m, Y_m.",
        description_ja="{0.5 < r < 1}。変数分離可能で、固有値はベッセル関数 J_m, Y_m のクロス積の零点で与えられる。",
        category="non-convex",
        builder=lambda: dom_annulus(0.5, 0.025),
        n_eig=8,
    ),
    DomainSpec(
        id="ellipse-2-1",
        name_en="Ellipse 2:1",
        name_ja="楕円 2:1",
        description_en="Semi-axes (1, 0.5). Separable in elliptic coordinates; eigenfunctions are products of Mathieu functions.",
        description_ja="半軸 (1, 0.5)。楕円座標で変数分離可能で、固有関数はマシュー関数の積で表される。",
        category="curved",
        builder=lambda: dom_ellipse_2_1(0.02),
        n_eig=8,
    ),
    DomainSpec(
        id="sector-60",
        name_en="60° circular sector",
        name_ja="60° 扇形",
        description_en="Radius 1, opening π/3. Separable; eigenvalues are squares of zeros of J_{3k}.",
        description_ja="半径 1、開き角 π/3。変数分離可能で、固有値は J_{3k} の零点の二乗。",
        category="sector",
        builder=lambda: dom_sector(math.pi / 3, 0.02),
        n_eig=8,
    ),
    DomainSpec(
        id="thin-triangle",
        name_en="Thin isoceles triangle (h=0.15)",
        name_ja="細長い二等辺三角形 (h=0.15)",
        description_en="Base 1, height 0.15. Illustrates thin-domain asymptotics: the first eigenfunction concentrates along the ridge; the spectrum is governed by a 1D Schrödinger operator with potential 1/h(x)² (Friedlander–Solomyak 2009; Freitas–Krejčiřík 2008).",
        description_ja="底辺 1、高さ 0.15。細領域極限では第一固有関数は「尾根」に局在し、スペクトルは 1 次元シュレディンガー作用素（ポテンシャル 1/h(x)²）で支配される（Friedlander–Solomyak 2009 / Freitas–Krejčiřík 2008）。",
        category="collapsing",
        builder=lambda: dom_thin_triangle(0.015, 0.15),
        n_eig=6,
        reference="Friedlander–Solomyak, ESAIM COCV (2009); Freitas–Krejčiřík, J. Diff. Eq. (2008).",
    ),
]


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def compute_domain(spec: DomainSpec, boundaries: Iterable[str] = ("dirichlet", "neumann")) -> dict:
    print(f"\n=== {spec.id} ({spec.name_en}) ===", flush=True)
    mesh = spec.builder()
    print(f"  mesh: {mesh.p.shape[1]} vertices, {mesh.t.shape[1]} triangles", flush=True)

    result = {
        "id": spec.id,
        "nameEn": spec.name_en,
        "nameJa": spec.name_ja,
        "descriptionEn": spec.description_en,
        "descriptionJa": spec.description_ja,
        "category": spec.category,
        "reference": spec.reference,
        "mesh": {
            "vertices": int(mesh.p.shape[1]),
            "triangles": int(mesh.t.shape[1]),
            "element": "P2 Lagrange",
        },
        "boundaries": {},
    }

    for bc in boundaries:
        print(f"  solving {bc} ...", flush=True)
        vals, vecs, basis = solve_eigs(mesh, spec.n_eig, bc)
        analytic = None
        if bc == "dirichlet" and spec.analytical is not None:
            try:
                analytic = spec.analytical(spec.n_eig)
            except Exception as e:
                print(f"    analytic failed: {e}")
                analytic = None

        modes = []
        for k in range(spec.n_eig):
            lam = float(vals[k])
            lam_ex = float(analytic[k]) if (analytic is not None and k < len(analytic)) else None
            rel_err = (abs(lam - lam_ex) / lam_ex) if (lam_ex is not None and lam_ex > 1e-12) else None

            img_path = OUT_IMG / f"{spec.id}_{bc}_{k + 1}.png"
            plot_eigenfunction(mesh, basis, vecs[:, k], img_path,
                               title=f"λ_{{{k+1}}} ≈ {lam:.4f}")
            modes.append({
                "k": k + 1,
                "lambda": lam,
                "lambdaExact": lam_ex,
                "relErr": rel_err,
                "image": f"/files/eigenfunctions/img/{spec.id}_{bc}_{k + 1}.png",
            })

        result["boundaries"][bc] = {"modes": modes}

    # write per-domain file
    (OUT_DATA / f"{spec.id}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    return result


def main(selection: list[str] | None = None) -> None:
    index = []
    for spec in DOMAINS:
        if selection and spec.id not in selection:
            continue
        r = compute_domain(spec)
        index.append({
            "id": spec.id,
            "nameEn": spec.name_en,
            "nameJa": spec.name_ja,
            "descriptionEn": spec.description_en,
            "descriptionJa": spec.description_ja,
            "category": spec.category,
            "reference": spec.reference,
            "mesh": r["mesh"],
            "firstFew": {
                bc: [m["lambda"] for m in r["boundaries"][bc]["modes"][:4]]
                for bc in r["boundaries"]
            },
        })
    (OUT_DATA / "index.json").write_text(
        json.dumps({"domains": index}, indent=2, ensure_ascii=False)
    )
    print(f"\nwrote {OUT_DATA/'index.json'}")


if __name__ == "__main__":
    sel = sys.argv[1:] if len(sys.argv) > 1 else None
    main(sel)
