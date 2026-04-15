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
    FacetBasis,
    ElementTriP2,
    BilinearForm,
    asm,
    condense,
)
from skfem.models.poisson import laplace, mass


@BilinearForm
def boundary_mass(u, v, w):
    # assembled on a FacetBasis -> matrix is zero except on boundary DoFs
    return u * v


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


def _ellipse_mesh(a: float, b: float, h: float) -> MeshTri:
    # semi-axes (a,b), polygonal approximation
    # polygon density chosen so boundary segments are ~h in length
    circumference = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    N = max(60, min(360, int(circumference / h)))
    pts = [[a * math.cos(2 * math.pi * k / N), b * math.sin(2 * math.pi * k / N)]
           for k in range(N)]
    def build(geom, h):
        geom.add_polygon(pts, mesh_size=h)
    return _mesh_from_pygmsh(build, h)


def dom_ellipse_2_1(h: float = 0.03) -> MeshTri:
    return _ellipse_mesh(1.0, 0.5, h)


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


def dom_pacman(open_angle: float, h: float = 0.03) -> MeshTri:
    """Unit disk with a circular sector of opening `open_angle` (rad) removed
    (Pac-Man domain). The re-entrant angle at the origin is 2π − open_angle."""
    # Boundary: origin → (cos α, sin α) arc to (cos(2π−α), sin(−α)) → origin,
    # where α = (2π − open_angle)/2 is the half-angle of the kept sector.
    kept = 2 * math.pi - open_angle  # reflex angle kept as the domain
    # Build as polygon approximation of the arc.
    N = 160
    t_start = open_angle / 2
    t_end = 2 * math.pi - open_angle / 2
    arc = [[math.cos(t), math.sin(t)] for t in np.linspace(t_start, t_end, N + 1)]
    pts = [[0.0, 0.0]] + arc
    def build(geom, h):
        geom.add_polygon(pts, mesh_size=h)
    return _mesh_from_pygmsh(build, h)


def dom_non_concentric_annulus(
    r_out: float = 1.0, r_in: float = 0.25,
    offset: tuple[float, float] = (0.4, 0.0), h: float = 0.025,
) -> MeshTri:
    def build(geom, h):
        outer = geom.add_circle([0, 0], r_out, mesh_size=h, num_sections=96, make_surface=False)
        inner = geom.add_circle(list(offset) + [0.0], r_in, mesh_size=h,
                                num_sections=64, make_surface=False)
        geom.add_plane_surface(outer.curve_loop, holes=[inner.curve_loop])
    return _mesh_from_pygmsh(build, h)


def dom_dumbbell(r: float = 1.0, sep: float = 2.6, neck: float = 0.25,
                 h: float = 0.05) -> MeshTri:
    """Two disks of radius r, centres at (±sep/2, 0), joined by a straight
    rectangular channel of full width 2·neck along the x-axis."""
    N = 60
    ptsL = [[-sep / 2 + r * math.cos(t), r * math.sin(t)] for t in np.linspace(0, 2 * math.pi, N, endpoint=False)]
    ptsR = [[+sep / 2 + r * math.cos(t), r * math.sin(t)] for t in np.linspace(0, 2 * math.pi, N, endpoint=False)]
    # Build left disk, right disk, and rectangle as three separate surfaces
    # then take the union.
    def build(geom, h):
        p_left = geom.add_polygon(ptsL, mesh_size=h, make_surface=False)
        p_right = geom.add_polygon(ptsR, mesh_size=h, make_surface=False)
        rect_pts = [
            [-sep / 2, -neck], [sep / 2, -neck],
            [sep / 2, neck], [-sep / 2, neck],
        ]
        p_rect = geom.add_polygon(rect_pts, mesh_size=h, make_surface=False)
        s_left = geom.add_plane_surface(p_left.curve_loop)
        s_right = geom.add_plane_surface(p_right.curve_loop)
        s_rect = geom.add_plane_surface(p_rect.curve_loop)
        geom.boolean_union([s_left, s_right, s_rect])
    # Boolean union needs OCC; fall back to constructing the outline by hand
    # (union of two disks and rectangle). Build the outline polygon directly:
    # the boundary is: top of left disk → top-left of rectangle → top-right of
    # rectangle → top of right disk → bottom of right → bottom-right of rect →
    # bottom-left of rect → bottom of left → close.
    # Circles stop at x = ±sep/2 (the channel entry points). Angle where disk
    # centred at (-sep/2, 0) intersects x = -sep/2 + 0 ... actually the disk
    # centred at (-sep/2, 0) with radius r passes through the channel at y =
    # ±neck provided r > neck. Find the angle φ with y = neck:
    phi = math.asin(neck / r)  # angle above/below x-axis where y=±neck on disk
    left_cx = -sep / 2
    right_cx = +sep / 2

    # upper-left arc: from angle π (leftmost) going clockwise(?) to angle φ
    # We'll go CCW along the outline:
    # start at (left_cx + r cos(−π + φ), r sin(−π + φ)) = top-left-ish of
    # left disk? Simpler: parameterise each arc explicitly and concatenate.

    M = 80
    # Left disk: arc from angle φ (upper-channel side) going CCW around the
    # back to −φ (lower-channel side): angles from φ through π (back of disk)
    # to 2π − φ.
    left_arc_angles = np.linspace(phi, 2 * math.pi - phi, M)
    left_arc = [[left_cx + r * math.cos(t), r * math.sin(t)] for t in left_arc_angles]

    # Right disk: arc from angle (π − φ) going CCW around to (π + φ).
    right_arc_angles = np.linspace(math.pi - phi, math.pi + phi, M)
    # but we need to traverse in the sense CCW along the combined boundary.
    # The overall outline CCW: starting at upper-right channel corner of
    # left disk (left_cx + r cos φ, neck) moving right along top of channel
    # to (right_cx − r cos φ, neck) (upper-left channel corner of right disk)
    # then CCW around the right disk to (right_cx − r cos φ, −neck) then back
    # along bottom of channel to (left_cx + r cos φ, −neck) then CCW around
    # left disk back to start.

    # Points:
    # top-channel: (left_cx + r cosφ, +neck) → (right_cx − r cosφ, +neck)
    # right-disk arc: angles from π − φ through 0 (rightmost) to −(π − φ)
    right_arc_angles = np.linspace(math.pi - phi, -(math.pi - phi), M)
    right_arc = [[right_cx + r * math.cos(t), r * math.sin(t)] for t in right_arc_angles]
    # bottom-channel: (right_cx − r cosφ, −neck) → (left_cx + r cosφ, −neck)
    # left-disk arc: angles from 2π − (−(π − φ)) ... easier: from −(π − φ) + 2π
    # through π back to π − φ + 2π? Let me just parameterise:
    # angles from (π + φ) going CCW (increasing) to (3π − φ) =
    # same as (π + φ) through 2π to (2π − φ) + 2π etc. Simplify:
    left_arc_angles = np.linspace(2 * math.pi - phi, math.pi + phi, M)  # CCW
    # Wait: at start, left disk point is (left_cx + r cosφ, -neck). That corresponds
    # to angle −φ on the left disk or equivalently 2π − φ. Going CCW we
    # increase the angle, and we should return to +φ (via the back of the
    # disk at angle π). But CCW on disk centred to the left, viewed from
    # outside, is clockwise in the standard angle parameterisation. Let me
    # use the outward normal: for the LEFT part of the dumbbell the outline
    # goes counter-clockwise around the whole shape, which means around the
    # left disk we go CLOCKWISE in the angle parameterisation (increasing y
    # at angle π). So angle goes from 2π − φ DECREASING through π to φ.
    left_arc_angles = np.linspace(2 * math.pi - phi, phi, M)
    left_arc_pts = [[left_cx + r * math.cos(t), r * math.sin(t)] for t in left_arc_angles]

    outline = []
    outline.append([left_cx + r * math.cos(phi), neck])          # start
    outline.append([right_cx - r * math.cos(phi), neck])          # top-right of channel
    outline.extend(right_arc)                                     # around right disk
    outline.append([right_cx - r * math.cos(phi), -neck])         # bottom-right
    outline.append([left_cx + r * math.cos(phi), -neck])          # bottom-left
    outline.extend(left_arc_pts[1:-1])                            # around left disk
    # de-duplicate adjacent points
    dedup = [outline[0]]
    for p in outline[1:]:
        if abs(p[0] - dedup[-1][0]) + abs(p[1] - dedup[-1][1]) > 1e-10:
            dedup.append(p)

    def build2(geom, h):
        geom.add_polygon(dedup, mesh_size=h)
    return _mesh_from_pygmsh(build2, h)


def dom_reuleaux_triangle(h: float = 0.02) -> MeshTri:
    """Reuleaux triangle of side 1: intersection of three unit disks centred
    at the vertices of an equilateral triangle of side 1. Constant width 1."""
    # Equilateral vertices at (0,0), (1,0), (1/2, √3/2).
    V = [(0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(3) / 2)]
    # Boundary: three circular arcs; each arc is on the unit circle centred
    # at the opposite vertex. Go CCW around the triangle:
    # arc from V_A = V0 to V_B = V1, centred at V_C = V2 (angle goes from
    # angle(V0 − V2) to angle(V1 − V2), unit radius).
    def ang(p, c):
        return math.atan2(p[1] - c[1], p[0] - c[0])
    order = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    M = 80
    outline = []
    for a_idx, b_idx, c_idx in order:
        A, B, C = V[a_idx], V[b_idx], V[c_idx]
        aA = ang(A, C); aB = ang(B, C)
        # ensure CCW on the boundary: reflex-free arc is the short one
        if aB < aA:
            aB += 2 * math.pi
        ts = np.linspace(aA, aB, M)
        for t in ts[:-1]:
            outline.append([C[0] + math.cos(t), C[1] + math.sin(t)])
    def build(geom, h):
        geom.add_polygon(outline, mesh_size=h)
    return _mesh_from_pygmsh(build, h)


def dom_robnik(eps: float, h: float = 0.02) -> MeshTri:
    """Robnik billiard: image of the unit disk |w| < 1 under the conformal
    map z = w + eps w². A one-parameter family interpolating between the
    disk (integrable, eps = 0) and increasingly chaotic billiards. For eps ≥
    1/2 the map is no longer injective; we keep eps ∈ [0, 1/2)."""
    if not (0 <= eps < 0.5):
        raise ValueError("Robnik eps must satisfy 0 ≤ eps < 1/2")
    N = 240
    pts = []
    for k in range(N):
        th = 2 * math.pi * k / N
        w = complex(math.cos(th), math.sin(th))
        z = w + eps * w * w
        pts.append([z.real, z.imag])
    def build(geom, h):
        geom.add_polygon(pts, mesh_size=h)
    return _mesh_from_pygmsh(build, h)


def dom_trapezoid(h: float = 0.025) -> MeshTri:
    # Isoceles trapezoid, bases 1 and 2, height 1
    def build(geom, h):
        geom.add_polygon([[-1, 0], [1, 0], [0.5, 1], [-0.5, 1]], mesh_size=h)
    return _mesh_from_pygmsh(build, h)


def dom_rhombus(angle_deg: float, h: float = 0.025) -> MeshTri:
    # Rhombus with a small interior angle `angle_deg` at (0,0). Diagonals of length 2.
    a = math.radians(angle_deg)
    side = 1.0
    pts = [
        [0.0, 0.0],
        [side, 0.0],
        [side + side * math.cos(a), side * math.sin(a)],
        [side * math.cos(a), side * math.sin(a)],
    ]
    def build(geom, h):
        geom.add_polygon(pts, mesh_size=h)
    return _mesh_from_pygmsh(build, h)


# ---------------------------------------------------------------------------
# FEM eigensolver
# ---------------------------------------------------------------------------


ROBIN_ALPHA = 1.0  # default Robin parameter (−∂_n u = α u on ∂Ω)

# Parameter grid for the per-domain Robin α slider.
ROBIN_ALPHAS = [0.1, 1.0, 10.0, 100.0]


def solve_eigs(mesh: MeshTri, n_eig: int, bc: str, *, alpha: float | None = None) -> tuple[np.ndarray, np.ndarray, Basis]:
    """Solve a Laplacian eigenproblem.

    bc in {'dirichlet', 'neumann', 'robin', 'steklov'}.
    Returns (eigenvalues ascending, eigenvector columns shaped (N_dof, k), Basis).

    Weak forms (integration by parts):
        Dirichlet:  ∫∇u·∇v = λ ∫ u v,                 u = 0 on ∂Ω
        Neumann:    ∫∇u·∇v = λ ∫ u v,                 ∂_n u = 0 on ∂Ω
        Robin:      ∫∇u·∇v + α ∫_∂Ω u v = λ ∫ u v,    ∂_n u + α u = 0 on ∂Ω
        Steklov:    ∫∇u·∇v = σ ∫_∂Ω u v,              Δu = 0 in Ω, ∂_n u = σ u on ∂Ω

    Conventions: λ ≥ 0. For Steklov the eigenvalue parameter is written σ
    here for clarity (the gallery UI shows them in the same "λ" column).
    """
    basis = Basis(mesh, ElementTriP2())
    K = asm(laplace, basis)
    M = asm(mass, basis)

    if bc == "dirichlet":
        D = basis.get_dofs()  # all boundary dofs
        Kc, Mc, _, I = condense(K, M, D=D)
        sigma = 1e-6
        vals, vecs = eigsh(Kc, k=n_eig, M=Mc, sigma=sigma, which="LM")
        full = np.zeros((basis.N, n_eig))
        full[I, :] = vecs
        order = np.argsort(vals)
        return vals[order], full[:, order], basis

    if bc == "neumann":
        sigma = -1e-4
        vals, vecs = eigsh(K, k=n_eig, M=M, sigma=sigma, which="LM")
        order = np.argsort(vals)
        vals = vals[order]; vecs = vecs[:, order]
        if abs(vals[0]) < 1e-6:
            vals[0] = 0.0
        return vals, vecs, basis

    if bc == "robin":
        a = ROBIN_ALPHA if alpha is None else float(alpha)
        fbasis = FacetBasis(mesh, ElementTriP2())
        B = asm(boundary_mass, fbasis)
        A = (K + a * B).tocsr()
        # A is SPD when α > 0 (no zero mode), so a positive shift near 0 works.
        sigma = 1e-6
        vals, vecs = eigsh(A, k=n_eig, M=M, sigma=sigma, which="LM")
        order = np.argsort(vals)
        return vals[order], vecs[:, order], basis

    if bc == "steklov":
        fbasis = FacetBasis(mesh, ElementTriP2())
        B = asm(boundary_mass, fbasis)
        # Split DoFs into interior (I) and boundary (D). Reduce K u = σ B u to
        # a problem on the boundary DoFs only, via the Dirichlet-to-Neumann
        # Schur complement S = K_BB − K_BI K_II^{-1} K_IB.
        Dset = basis.get_dofs().flatten()
        allN = basis.N
        mask = np.zeros(allN, dtype=bool); mask[Dset] = True
        Didx = np.where(mask)[0]
        Iidx = np.where(~mask)[0]
        if Didx.size > 1500:
            # dense Schur matrix would be too large; skip Steklov for this mesh
            raise RuntimeError(
                f"Steklov skipped: {Didx.size} boundary DoFs exceed threshold 1500"
            )
        Kcsr = K.tocsr()
        K_II = Kcsr[Iidx][:, Iidx].tocsc()
        K_IB = Kcsr[Iidx][:, Didx].tocsc()
        K_BB = Kcsr[Didx][:, Didx].tocsc()
        Bcsr = B.tocsr()
        M_B = Bcsr[Didx][:, Didx].tocsc()

        from scipy.sparse.linalg import splu, LinearOperator
        lu = splu(K_II.tocsc())

        from scipy.sparse.linalg import aslinearoperator
        K_BB_op = aslinearoperator(K_BB)
        K_IB_op = aslinearoperator(K_IB)

        nB = Didx.size

        def matvec(x):
            # S x = K_BB x − K_BI (K_II^{-1} (K_IB x))
            y1 = K_IB @ x
            y2 = lu.solve(y1)
            y3 = (K_IB.T) @ y2
            return (K_BB @ x) - y3

        S = LinearOperator((nB, nB), matvec=matvec, dtype=float)
        # S has a 1-D kernel (constants) with σ = 0. Shift-invert near 0.
        # eigsh needs the inverse of (S − σ M_B); we can't factor S as a
        # sparse matrix directly because S is an implicit operator. Instead
        # build S densely (nB is small) and solve generalised eigenproblem.
        S_dense = np.zeros((nB, nB))
        I_nB = np.eye(nB)
        for j in range(nB):
            S_dense[:, j] = matvec(I_nB[:, j])
        # Symmetrise (tiny roundoff asymmetry)
        S_dense = 0.5 * (S_dense + S_dense.T)
        from scipy.linalg import eigh
        vals_all, vecs_all = eigh(S_dense, M_B.toarray())
        # take the n_eig smallest
        order = np.argsort(vals_all)
        vals = vals_all[order][:n_eig]
        vecs_bdry = vecs_all[:, order][:, :n_eig]
        if abs(vals[0]) < 1e-8:
            vals[0] = 0.0

        # Reconstruct interior DoFs: u_I = −K_II^{-1} K_IB u_B (harmonic ext.)
        full = np.zeros((allN, n_eig))
        full[Didx, :] = vecs_bdry
        rhs = -(K_IB @ vecs_bdry)
        full[Iidx, :] = lu.solve(rhs)
        return vals, full, basis

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
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=130)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-amax, vmax=amax)
    tpc = ax.tripcolor(tri, u_vert, shading="gouraud", cmap="RdBu_r", norm=norm)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=130, bbox_inches="tight", pad_inches=0.02,
                pil_kwargs={"quality": 86, "method": 4})
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
    analytical: Callable[[int], list[float]] | None = None  # Dirichlet, sorted λ₁..λ_n
    analytical_steklov: Callable[[int], list[float]] | None = None  # Steklov, sorted
    reference: str = ""
    family_id: str | None = None  # None = standalone; otherwise member of a FamilySpec
    family_param: float | int | None = None  # parameter value within the family


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


def analytic_steklov_disk(n: int) -> list[float]:
    # Steklov spectrum of the unit disk: σ_k = k, with multiplicity 1 for
    # k = 0 (constants) and 2 for k ≥ 1 (e.g. r^k cos(k θ), r^k sin(k θ)).
    # See Girouard–Polterovich, J. Spectral Theory 7 (2017).
    vals = [0.0]
    for k in range(1, n + 2):
        vals.append(float(k))
        vals.append(float(k))
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
        builder=lambda: dom_square(0.05),
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
        description_ja="半径 1 の円板。固有値は第 1 種の Bessel 関数 J_m の零点の二乗で与えられる。",
        category="basic",
        builder=lambda: dom_disk(0.05),
        n_eig=10,
        analytical=analytic_disk,
        analytical_steklov=analytic_steklov_disk,
        reference="Courant–Hilbert (1953), §V.5; Girouard–Polterovich, J. Spectral Theory 7 (2017).",
    ),
    DomainSpec(
        id="equilateral-triangle",
        name_en="Equilateral triangle",
        name_ja="正三角形",
        description_en="Side 1. Lamé (1833) derived the closed-form spectrum λ_{m,n} = (16π²/9)(m² + n² + mn).",
        description_ja="一辺 1。Lamé (1833) が閉形式解 λ_{m,n} = (16π²/9)(m² + n² + mn) を導いた。",
        category="polygon",
        builder=lambda: dom_equilateral_triangle(0.05),
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
        builder=lambda: dom_right_isoceles_triangle(0.05),
        n_eig=10,
        analytical=analytic_right_isoceles,
    ),
    DomainSpec(
        id="30-60-90-triangle",
        name_en="30-60-90 triangle",
        name_ja="30°-60°-90° 三角形",
        description_en="Short leg 1, long leg √3. A tiling (Lamé) triangle with fully integrable billiard.",
        description_ja="短い脚 1、長い脚 √3。平面を敷き詰める(Lamé 型)三角形で、ビリヤードは可積分。",
        category="polygon",
        builder=lambda: dom_30_60_90_triangle(0.05),
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
        description_en="[-1,1]² with the lower-right quadrant removed. The eigenfunctions have an r^{2/3} corner singularity at the re-entrant corner (Fox–Henrici–Moler 1967).",
        description_ja="[-1,1]² から右下の象限を除いた領域。凹角 (re-entrant corner) において固有関数は r^{2/3} 型の特異性をもつ（Fox–Henrici–Moler 1967）。",
        category="non-convex",
        builder=lambda: dom_l_shape(0.05),
        n_eig=8,
        reference="Fox–Henrici–Moler, SIAM J. Numer. Anal. 4 (1967).",
    ),
    DomainSpec(
        id="stadium",
        name_en="Bunimovich stadium",
        name_ja="Bunimovich スタジアム",
        description_en="Rectangle [-1,1]² capped by unit semicircles. The classical billiard is ergodic and K-mixing (Bunimovich 1974); the quantum level-spacing statistics agree with GOE random matrix theory (Bohigas–Giannoni–Schmit conjecture).",
        description_ja="長方形 [-1,1]² の両端に半径 1 の半円を付加した領域。古典ビリヤードは ergodic かつ K-mixing (Bunimovich 1974)；量子 spectrum の level-spacingは GOE の予想 (Bohigas–Giannoni–Schmit) に従う。",
        category="chaotic",
        builder=lambda: dom_stadium(0.03),
        n_eig=10,
        reference="Bunimovich, Commun. Math. Phys. 65 (1979); Heller (1984) scars.",
    ),
    DomainSpec(
        id="sinai-billiard",
        name_en="Sinai billiard",
        name_ja="Sinai ビリヤード",
        description_en="Square [-1,1]² with a central disk of radius 0.3 removed. The classical billiard is a K-system (Sinai 1970); a model for the Boltzmann–Gibbs ergodic hypothesis.",
        description_ja="[-1,1]² から中心の半径 0.3 の円板を除いた領域。古典ビリヤードは K 系 (Sinai 1970) で、Boltzmann–Gibbs の ergodic hypothesisに対するモデル。",
        category="chaotic",
        builder=lambda: dom_sinai(0.03),
        n_eig=10,
        reference="Sinai, Russ. Math. Surveys 25 (1970).",
    ),
    DomainSpec(
        id="cardioid",
        name_en="Cardioid",
        name_ja="Cardioid",
        description_en="r = 1 − cos θ. The classical billiard is ergodic, mixing and K (Markarian 1993); extensively studied in quantum chaos (Robnik 1983; Bäcker–Steiner 1998).",
        description_ja="r = 1 − cos θ。古典ビリヤードは ergodic, mixing, かつ K 系 (Markarian 1993)；quantum chaos の文脈で詳しく調べられている (Robnik 1983; Bäcker–Steiner 1998)。",
        category="chaotic",
        builder=lambda: dom_cardioid(0.06),
        n_eig=10,
        reference="Robnik, J. Phys. A 16 (1983); Bäcker–Steiner (1998).",
    ),
    DomainSpec(
        id="annulus-0.5",
        name_en="Annulus (r=0.5)",
        name_ja="円環 (r=0.5)",
        description_en="{0.5 < r < 1}. Separable; eigenvalues determined by cross-products of Bessel functions J_m, Y_m.",
        description_ja="{0.5 < r < 1}。変数分離可能で、固有値は Bessel 関数 J_m, Y_m のクロス積の零点で与えられる。",
        category="non-convex",
        builder=lambda: dom_annulus(0.5, 0.025),
        n_eig=8,
    ),
    DomainSpec(
        id="ellipse-2-1",
        name_en="Ellipse 2:1",
        name_ja="楕円 2:1",
        description_en="Semi-axes (1, 0.5). Separable in elliptic coordinates; eigenfunctions are products of Mathieu functions.",
        description_ja="半軸 (1, 0.5)。楕円座標で変数分離可能で、固有関数は Mathieu 関数の積で表される。",
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
        description_en="Base 1, height 0.15. Illustrates thin-domain asymptotics: as the height h → 0, the eigenvalues are governed by a 1D Schrödinger operator with potential 1/h(x)² (Friedlander–Solomyak 2009; Freitas–Krejčiřík 2008).",
        description_ja="底辺 1、高さ 0.15。細領域極限 h → 0 では、スペクトルは 1 次元 Schrödinger 作用素（ポテンシャル 1/h(x)²）で支配される（Friedlander–Solomyak 2009; Freitas–Krejčiřík 2008）。",
        category="collapsing",
        builder=lambda: dom_thin_triangle(0.015, 0.15),
        n_eig=6,
        reference="Friedlander–Solomyak, ESAIM COCV (2009); Freitas–Krejčiřík, J. Diff. Eq. (2008).",
    ),
    # --- polygon sequence → disk ---
    DomainSpec(
        id="octagon",
        name_en="Regular octagon",
        name_ja="正八角形",
        description_en="Inscribed in the unit circle.",
        description_ja="単位円に内接する正八角形。",
        category="polygon",
        builder=lambda: dom_regular_polygon(8, 0.025),
        n_eig=10,
    ),
    DomainSpec(
        id="decagon",
        name_en="Regular decagon",
        name_ja="正十角形",
        description_en="Inscribed in the unit circle.",
        description_ja="単位円に内接する正十角形。",
        category="polygon",
        builder=lambda: dom_regular_polygon(10, 0.025),
        n_eig=10,
    ),
    DomainSpec(
        id="dodecagon",
        name_en="Regular dodecagon",
        name_ja="正十二角形",
        description_en="Inscribed in the unit circle. Polygonal approximation of the disk.",
        description_ja="単位円に内接する正十二角形。円板の多角形近似。",
        category="polygon",
        builder=lambda: dom_regular_polygon(12, 0.022),
        n_eig=10,
    ),
    # --- sectors ---
    DomainSpec(
        id="sector-30",
        name_en="30° circular sector",
        name_ja="30° 扇形",
        description_en="Unit disk sector of opening π/6. Separable; eigenvalues are squares of zeros of J_{6k}.",
        description_ja="半径 1、開き角 π/6 の扇形。変数分離可能で、固有値は J_{6k} の零点の二乗。",
        category="sector",
        builder=lambda: dom_sector(math.pi / 6, 0.02),
        n_eig=8,
    ),
    DomainSpec(
        id="sector-120",
        name_en="120° circular sector",
        name_ja="120° 扇形",
        description_en="Unit disk sector of opening 2π/3.",
        description_ja="半径 1、開き角 2π/3 の扇形。",
        category="sector",
        builder=lambda: dom_sector(2 * math.pi / 3, 0.02),
        n_eig=10,
    ),
    DomainSpec(
        id="pacman-90",
        name_en="Pac-Man (opening π/2)",
        name_ja="Pac-Man (開口 π/2)",
        description_en="Unit disk with a sector of angle π/2 removed. Re-entrant angle 3π/2 at the origin produces an r^{2/3} corner singularity.",
        description_ja="単位円板から開き角 π/2 の扇形を除いた領域。原点での凹角 3π/2 により r^{2/3} 型特異性が生じる。",
        category="non-convex",
        builder=lambda: dom_pacman(math.pi / 2, 0.02),
        n_eig=10,
    ),
    # --- ellipses ---
    DomainSpec(
        id="ellipse-3-1",
        name_en="Ellipse 3:1",
        name_ja="楕円 3:1",
        description_en="Semi-axes (1, 1/3). Eigenfunctions are products of angular and radial Mathieu functions.",
        description_ja="半軸 (1, 1/3)。固有関数は角度および動径 Mathieu 関数の積で表される。",
        category="curved",
        builder=lambda: _ellipse_mesh(1.0, 1.0 / 3.0, 0.02),
        n_eig=10,
    ),
    DomainSpec(
        id="ellipse-5-1",
        name_en="Ellipse 5:1",
        name_ja="楕円 5:1",
        description_en="Semi-axes (1, 0.2). Anisotropic domain between a disk and a thin strip.",
        description_ja="半軸 (1, 0.2)。円板と細い帯の中間の異方的な領域。",
        category="curved",
        builder=lambda: _ellipse_mesh(1.0, 0.2, 0.015),
        n_eig=10,
    ),
    # --- non-concentric annulus ---
    DomainSpec(
        id="annulus-offcenter",
        name_en="Non-concentric annulus",
        name_ja="非同心円環",
        description_en="Unit disk with an interior disk of radius 0.25 centred at (0.4, 0) removed. Symmetry-broken variant of the annulus.",
        description_ja="単位円板から、中心 (0.4, 0)、半径 0.25 の内側円板を除いた領域。同心円環の対称性を破った変種。",
        category="non-convex",
        builder=lambda: dom_non_concentric_annulus(1.0, 0.25, (0.4, 0.0), 0.04),
        n_eig=10,
    ),
    # --- (individual dumbbell removed; see the `dumbbell` family below) ---
    # --- Reuleaux triangle (constant width) ---
    DomainSpec(
        id="reuleaux-triangle",
        name_en="Reuleaux triangle",
        name_ja="Reuleaux 三角形",
        description_en="Intersection of three unit disks centred at the vertices of an equilateral triangle of side 1; a curve of constant width (Reuleaux 1875).",
        description_ja="一辺 1 の正三角形の各頂点を中心とする単位円板 3 つの共通部分。定幅図形 (Reuleaux 1875)。",
        category="curved",
        builder=lambda: dom_reuleaux_triangle(0.05),
        n_eig=10,
    ),
    # --- Robnik family (integrable → chaotic) ---
    DomainSpec(
        id="robnik-0.15",
        name_en="Robnik billiard (ε=0.15)",
        name_ja="Robnik ビリヤード (ε=0.15)",
        description_en="Image of the unit disk under the conformal map z = w + εw² with ε = 0.15. A smooth one-parameter deformation of the disk introduced by Robnik (1983); classical flow is soft-chaotic with mixed phase space.",
        description_ja="共形写像 z = w + εw² (ε = 0.15) による単位円板の像。Robnik (1983) が導入した円板の滑らかな 1-パラメータ変形族。古典流は mixed phase space をもつ soft-chaotic。",
        category="chaotic",
        builder=lambda: dom_robnik(0.15, 0.02),
        n_eig=10,
        reference="Robnik, J. Phys. A 16 (1983).",
    ),
    DomainSpec(
        id="robnik-0.3",
        name_en="Robnik billiard (ε=0.3)",
        name_ja="Robnik ビリヤード (ε=0.3)",
        description_en="Image of the unit disk under z = w + εw² with ε = 0.3. The classical billiard is chaotic (Markarian) but not yet limiting to the cardioid (ε = 1/2).",
        description_ja="z = w + εw² (ε = 0.3) による単位円板の像。古典流は chaotic (Markarian) であるが、ε = 1/2 で得られる Cardioidにはまだ達していない。",
        category="chaotic",
        builder=lambda: dom_robnik(0.3, 0.02),
        n_eig=10,
        reference="Robnik, J. Phys. A 16 (1983); Markarian, Nonlinearity 6 (1993).",
    ),
    # --- additional thin triangles for the asymptotic family ---
    DomainSpec(
        id="thin-triangle-0.3",
        name_en="Isoceles triangle (h=0.3)",
        name_ja="二等辺三角形 (h=0.3)",
        description_en="Base 1, height 0.3.",
        description_ja="底辺 1、高さ 0.3 の二等辺三角形。",
        category="collapsing",
        builder=lambda: dom_thin_triangle(0.02, 0.3),
        n_eig=8,
    ),
    DomainSpec(
        id="thin-triangle-0.05",
        name_en="Thin isoceles triangle (h=0.05)",
        name_ja="細長い二等辺三角形 (h=0.05)",
        description_en="Base 1, height 0.05. Used as the thinnest member of the asymptotic family h → 0.",
        description_ja="底辺 1、高さ 0.05 の細長い二等辺三角形。漸近族 h → 0 のうち最も細いもの。",
        category="collapsing",
        builder=lambda: dom_thin_triangle(0.008, 0.05),
        n_eig=6,
    ),
    # --- trapezoid, rhombus ---
    DomainSpec(
        id="trapezoid-iso",
        name_en="Isoceles trapezoid",
        name_ja="等脚台形",
        description_en="Isoceles trapezoid with parallel bases 2 and 1, height 1.",
        description_ja="平行な底辺 2 と 1、高さ 1 の等脚台形。",
        category="polygon",
        builder=lambda: dom_trapezoid(0.05),
        n_eig=10,
    ),
    DomainSpec(
        id="rhombus-60",
        name_en="Rhombus (60°)",
        name_ja="菱形 (60°)",
        description_en="Unit-side rhombus with interior angles 60° and 120°. Obtained by gluing two equilateral triangles along a common edge.",
        description_ja="一辺 1 の菱形で、内角は 60° と 120°。正三角形を 2 つ貼り合わせたものに一致。",
        category="polygon",
        builder=lambda: dom_rhombus(60.0, 0.02),
        n_eig=10,
    ),
    DomainSpec(
        id="rhombus-30",
        name_en="Rhombus (30°)",
        name_ja="菱形 (30°)",
        description_en="Unit-side rhombus with acute angle 30°; exhibits strong directional anisotropy.",
        description_ja="一辺 1 の菱形で、鋭角は 30°。強い方向異方性をもつ。",
        category="polygon",
        builder=lambda: dom_rhombus(30.0, 0.018),
        n_eig=10,
    ),
]


# ---------------------------------------------------------------------------
# Parametric families
# ---------------------------------------------------------------------------
#
# A FamilySpec groups a one-parameter family of domains. At FEM time each
# parameter value is just another DomainSpec with the same id-format
# (see member_id_fmt). The index.json lists families alongside individual
# domains; the UI renders a family as a single card with a slider.


@dataclass
class FamilySpec:
    id: str
    name_en: str
    name_ja: str
    description_en: str
    description_ja: str
    category: str
    param: str                # short parameter label, e.g. "n", "ε", "h"
    param_ja: str
    param_values: list       # list of parameter values in the family
    member_id_fmt: str        # python format string, gets `{p}` (or {n}/{idx})
    member_name_en_fmt: str
    member_name_ja_fmt: str
    builder_for: Callable[[object], Callable[[], MeshTri]]  # param -> builder
    n_eig: int = 5
    reference: str = ""


def _fmt_param(p) -> str:
    if isinstance(p, int):
        return str(p)
    if isinstance(p, float):
        if abs(p - round(p)) < 1e-9:
            return str(int(round(p)))
        return f"{p:g}"
    return str(p)


def _family_to_domain_specs(fam: FamilySpec) -> list[DomainSpec]:
    specs: list[DomainSpec] = []
    for p in fam.param_values:
        ps = _fmt_param(p)
        specs.append(DomainSpec(
            id=fam.member_id_fmt.format(p=ps),
            name_en=fam.member_name_en_fmt.format(p=ps),
            name_ja=fam.member_name_ja_fmt.format(p=ps),
            description_en=fam.description_en,
            description_ja=fam.description_ja,
            category=fam.category,
            builder=fam.builder_for(p),
            n_eig=fam.n_eig,
            reference=fam.reference,
            family_id=fam.id,
            family_param=p,
        ))
    return specs


# ---- builders for family members ------------------------------------------

def _dom_rectangle_ar(r: float) -> Callable[[], MeshTri]:
    def build(geom, h):
        geom.add_polygon([[0, 0], [r, 0], [r, 1], [0, 1]], mesh_size=h)
    # keep Steklov boundary DoFs manageable on very long rectangles.
    if r >= 15:   mesh_size = 0.10
    elif r >= 7:  mesh_size = 0.07
    else:         mesh_size = 0.05
    return lambda: _mesh_from_pygmsh(build, mesh_size)


def _dom_isotri(apex_deg: float) -> Callable[[], MeshTri]:
    apex = math.radians(apex_deg)
    height = 1.0 / (2.0 * math.tan(apex / 2.0))
    mesh_size = max(0.025, min(0.06, height / 6))
    def build(geom, h):
        geom.add_polygon([[0, 0], [1, 0], [0.5, height]], mesh_size=h)
    return lambda: _mesh_from_pygmsh(build, mesh_size)


def _dom_rhombus_ang(angle_deg: float) -> Callable[[], MeshTri]:
    mesh_size = 0.04 if angle_deg < 30 else 0.06
    return lambda: dom_rhombus(angle_deg, mesh_size)


def _dom_annulus_rin(r_in: float) -> Callable[[], MeshTri]:
    return lambda: dom_annulus(r_in, 0.06)


def _dom_sector_a(a: float) -> Callable[[], MeshTri]:
    return lambda: dom_sector(a, 0.06)


def _dom_pacman_open(open_angle: float) -> Callable[[], MeshTri]:
    return lambda: dom_pacman(open_angle, 0.06)


def _dom_robnik_eps(eps: float) -> Callable[[], MeshTri]:
    return lambda: dom_robnik(eps, 0.06)


def _dom_thin_tri_h(height: float) -> Callable[[], MeshTri]:
    h_mesh = max(0.015, min(0.06, height / 3))
    return lambda: dom_thin_triangle(h_mesh, height)


def _dom_dumbbell_neck(neck: float) -> Callable[[], MeshTri]:
    return lambda: dom_dumbbell(1.0, 2.6, neck, 0.09)


def _dom_sinai_r(r_in: float) -> Callable[[], MeshTri]:
    def make():
        def build(geom, h):
            outer = geom.add_polygon(
                [[-1, -1], [1, -1], [1, 1], [-1, 1]], mesh_size=h, make_surface=False,
            )
            inner = geom.add_circle([0, 0], r_in, mesh_size=h,
                                    num_sections=32, make_surface=False)
            geom.add_plane_surface(outer.curve_loop, holes=[inner.curve_loop])
        return _mesh_from_pygmsh(build, 0.06)
    return make


def _dom_stadium_a(a: float) -> Callable[[], MeshTri]:
    # Bunimovich-family stadium with half-length a (radius 1 caps).
    def make():
        def build(geom, h):
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
        return _mesh_from_pygmsh(build, 0.06)
    return make


def _dom_ellipse_ar(e: float) -> Callable[[], MeshTri]:
    # semi-axes (1, 1/e)
    return lambda: _ellipse_mesh(1.0, 1.0 / e, max(0.035, 0.07 / math.sqrt(e)))


# ---- family registry -------------------------------------------------------

FAMILIES: list[FamilySpec] = [
    FamilySpec(
        id="regpoly",
        name_en="Regular n-gon",
        name_ja="正 n 角形",
        description_en="Regular n-gon inscribed in the unit circle. As n → ∞, the spectrum converges to that of the unit disk.",
        description_ja="単位円に内接する正 n 角形。n → ∞ で固有値は単位円板の固有値に収束する。",
        category="polygon",
        param="n", param_ja="n",
        param_values=list(range(3, 301)),
        member_id_fmt="regpoly-{p}",
        member_name_en_fmt="Regular {p}-gon",
        member_name_ja_fmt="正 {p} 角形",
        builder_for=lambda n: (lambda nn=n: dom_regular_polygon(int(nn),
                                                                 0.10 if int(nn) > 60 else
                                                                 0.07 if int(nn) > 30 else
                                                                 0.05)),
    ),
    FamilySpec(
        id="rect",
        name_en="Rectangle (aspect ratio r)",
        name_ja="長方形 (辺比 r)",
        description_en="Rectangle [0, r] × [0, 1]. Separable spectrum λ_{m,n} = π²((m/r)² + n²).",
        description_ja="長方形 [0, r] × [0, 1]。変数分離可能で λ_{m,n} = π²((m/r)² + n²)。",
        category="curved",  # kept with ellipse-type curved shapes
        param="r", param_ja="r",
        param_values=[round(1.0 + 0.2 * i, 2) for i in range(0, 16)]
                     + [round(r, 2) for r in [4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 25.0]],
        member_id_fmt="rect-{p}",
        member_name_en_fmt="Rectangle r={p}",
        member_name_ja_fmt="長方形 r={p}",
        builder_for=lambda r: _dom_rectangle_ar(float(r)),
    ),
    FamilySpec(
        id="ellipse",
        name_en="Ellipse (aspect ratio e)",
        name_ja="楕円 (軸比 e)",
        description_en="Ellipse with semi-axes (1, 1/e). Eigenfunctions are products of angular and radial Mathieu functions.",
        description_ja="半軸 (1, 1/e) の楕円。固有関数は角度および動径 Mathieu 関数の積。",
        category="curved",
        param="e", param_ja="e",
        param_values=[round(1.1 + 0.1 * i, 2) for i in range(0, 20)]
                     + [3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0],
        member_id_fmt="ellipse-{p}",
        member_name_en_fmt="Ellipse e={p}",
        member_name_ja_fmt="楕円 e={p}",
        builder_for=lambda e: _dom_ellipse_ar(float(e)),
    ),
    FamilySpec(
        id="isotri",
        name_en="Isoceles triangle (apex angle θ)",
        name_ja="二等辺三角形 (頂角 θ)",
        description_en="Isoceles triangle with base 1 and apex angle θ. At θ = 60° reduces to the equilateral triangle.",
        description_ja="底辺 1、頂角 θ の二等辺三角形。θ = 60° で正三角形に一致。",
        category="polygon",
        param="θ (deg)", param_ja="θ (度)",
        param_values=list(range(10, 175, 3)),
        member_id_fmt="isotri-{p}",
        member_name_en_fmt="Isoceles triangle θ={p}°",
        member_name_ja_fmt="二等辺三角形 θ={p}°",
        builder_for=lambda a: _dom_isotri(float(a)),
    ),
    FamilySpec(
        id="rhombus",
        name_en="Rhombus (acute angle θ)",
        name_ja="菱形 (鋭角 θ)",
        description_en="Unit-side rhombus with acute interior angle θ. At θ = 90° reduces to a unit square.",
        description_ja="一辺 1、鋭角 θ の菱形。θ = 90° で単位正方形に一致。",
        category="polygon",
        param="θ (deg)", param_ja="θ (度)",
        param_values=list(range(5, 91, 2)),
        member_id_fmt="rhombus-{p}",
        member_name_en_fmt="Rhombus θ={p}°",
        member_name_ja_fmt="菱形 θ={p}°",
        builder_for=lambda a: _dom_rhombus_ang(float(a)),
    ),
    FamilySpec(
        id="annulus",
        name_en="Annulus (inner radius r)",
        name_ja="円環 (内径 r)",
        description_en="Concentric annulus { r < |x| < 1 }. Eigenvalues are determined by cross-products of Bessel functions J_m, Y_m.",
        description_ja="同心円環 { r < |x| < 1 }。固有値は Bessel 関数 J_m, Y_m のクロス積の零点で与えられる。",
        category="non-convex",
        param="r", param_ja="r",
        param_values=[round(0.05 + 0.05 * i, 2) for i in range(0, 19)],
        member_id_fmt="annulus-{p}",
        member_name_en_fmt="Annulus r={p}",
        member_name_ja_fmt="円環 r={p}",
        builder_for=lambda r: _dom_annulus_rin(float(r)),
    ),
    FamilySpec(
        id="sector",
        name_en="Circular sector (opening α)",
        name_ja="扇形 (開き角 α)",
        description_en="Unit disk sector of opening α. The eigenfunctions are J_{kπ/α}(√λ r) sin(kπθ/α).",
        description_ja="半径 1、開き角 α の扇形。固有関数は J_{kπ/α}(√λ r) sin(kπθ/α)。",
        category="sector",
        param="α (rad)", param_ja="α (rad)",
        param_values=[round(math.pi * k / 48, 5) for k in range(2, 96, 2)],
        member_id_fmt="sector-{p}",
        member_name_en_fmt="Sector α={p}",
        member_name_ja_fmt="扇形 α={p}",
        builder_for=lambda a: _dom_sector_a(float(a)),
    ),
    FamilySpec(
        id="pacman",
        name_en="Pac-Man (opening θ removed)",
        name_ja="Pac-Man (開口 θ 除去)",
        description_en="Unit disk with a sector of opening angle θ removed; the re-entrant angle at the origin is 2π − θ, generating an r^{π/(2π−θ)} corner singularity.",
        description_ja="単位円板から開き角 θ の扇形を除いた領域。原点での凹角は 2π − θ で、r^{π/(2π−θ)} 型特異性が生じる。",
        category="non-convex",
        param="θ (rad)", param_ja="θ (rad)",
        param_values=[round(math.pi * k / 24, 5) for k in range(1, 23, 1)],
        member_id_fmt="pacman-{p}",
        member_name_en_fmt="Pac-Man θ={p}",
        member_name_ja_fmt="Pac-Man θ={p}",
        builder_for=lambda a: _dom_pacman_open(float(a)),
    ),
    FamilySpec(
        id="robnik",
        name_en="Robnik billiard (ε)",
        name_ja="Robnik ビリヤード (ε)",
        description_en="Image of the unit disk under the conformal map z = w + εw². Interpolates between the integrable disk (ε = 0) and increasingly chaotic billiards; ε = 1/2 yields the cardioid.",
        description_ja="共形写像 z = w + εw² による単位円板の像。ε = 0 の可積分な円板から ε = 1/2 で Cardioid に至る 1 パラメータ族。",
        category="chaotic",
        param="ε", param_ja="ε",
        param_values=[round(0.02 * i, 2) for i in range(0, 25)],
        member_id_fmt="robnik-{p}",
        member_name_en_fmt="Robnik ε={p}",
        member_name_ja_fmt="Robnik ε={p}",
        builder_for=lambda e: _dom_robnik_eps(float(e)),
        reference="Robnik, J. Phys. A 16 (1983).",
    ),
    FamilySpec(
        id="thintri",
        name_en="Isoceles triangle (height h)",
        name_ja="二等辺三角形 (高さ h)",
        description_en="Isoceles triangle with base 1 and height h. The limit h → 0 realises the thin-domain asymptotics λ ~ (const)/h² (Friedlander–Solomyak 2009).",
        description_ja="底辺 1、高さ h の二等辺三角形。h → 0 の細領域極限では λ ~ (定数)/h² (Friedlander–Solomyak 2009)。",
        category="collapsing",
        param="h", param_ja="h",
        param_values=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.28,
                      0.35, 0.45, 0.60, 0.80, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00],
        member_id_fmt="thintri-{p}",
        member_name_en_fmt="Isoceles triangle h={p}",
        member_name_ja_fmt="二等辺三角形 h={p}",
        builder_for=lambda hh: _dom_thin_tri_h(float(hh)),
        reference="Friedlander–Solomyak, ESAIM COCV (2009).",
    ),
    FamilySpec(
        id="dumbbell",
        name_en="Dumbbell (neck half-width n)",
        name_ja="ダンベル (頸部幅 n)",
        description_en="Two unit disks centred at (±1.3, 0) connected by a rectangular channel of half-width n. As n → 0 the first two Dirichlet eigenvalues become asymptotically degenerate (Jimbo 1989; Arrieta 1995).",
        description_ja="中心 (±1.3, 0) の単位円板 2 つを、半幅 n の長方形頸部で連結した領域。n → 0 で第 1, 第 2 Dirichlet 固有値が漸近的に縮退する (Jimbo 1989; Arrieta 1995)。",
        category="collapsing",
        param="n", param_ja="n",
        param_values=[round(0.02 + 0.04 * i, 2) for i in range(0, 25)],
        member_id_fmt="dumbbell-{p}",
        member_name_en_fmt="Dumbbell n={p}",
        member_name_ja_fmt="ダンベル n={p}",
        builder_for=lambda neck: _dom_dumbbell_neck(float(neck)),
        reference="Jimbo, J. Diff. Eq. 77 (1989); Arrieta, J. Diff. Eq. 118 (1995).",
    ),
    FamilySpec(
        id="sinai-family",
        name_en="Sinai billiard (hole radius r)",
        name_ja="Sinai ビリヤード (内半径 r)",
        description_en="Square [-1,1]² with a central disk of radius r removed. The classical billiard is a K-system (Sinai 1970) for 0 < r < 1.",
        description_ja="[-1,1]² から中心の半径 r の円板を除いた領域。0 < r < 1 で古典ビリヤードは K 系 (Sinai 1970)。",
        category="chaotic",
        param="r", param_ja="r",
        param_values=[round(0.05 + 0.05 * i, 2) for i in range(0, 18)],
        member_id_fmt="sinai-{p}",
        member_name_en_fmt="Sinai r={p}",
        member_name_ja_fmt="Sinai r={p}",
        builder_for=lambda r: _dom_sinai_r(float(r)),
        reference="Sinai, Russ. Math. Surveys 25 (1970).",
    ),
    FamilySpec(
        id="stadium-family",
        name_en="Stadium (half-length a)",
        name_ja="スタジアム (半長 a)",
        description_en="Rectangle [-a, a] × [-1, 1] with unit semicircular caps. The classical billiard is ergodic and K-mixing for every a > 0 (Bunimovich 1974); a = 0 is the unit disk.",
        description_ja="長方形 [-a, a] × [-1, 1] の両端に半径 1 の半円を付加した領域。a > 0 で古典ビリヤードは ergodic かつ K-mixing (Bunimovich 1974)；a = 0 は単位円板。",
        category="chaotic",
        param="a", param_ja="a",
        param_values=[round(0.1 + 0.2 * i, 2) for i in range(0, 25)],
        member_id_fmt="stadium-{p}",
        member_name_en_fmt="Stadium a={p}",
        member_name_ja_fmt="スタジアム a={p}",
        builder_for=lambda a: _dom_stadium_a(float(a)),
        reference="Bunimovich, Commun. Math. Phys. 65 (1979).",
    ),
]


# Extend DOMAINS with all family members. Existing individual specs that are
# now part of a family (rectangle-2-1, ellipse-2-1/3-1/5-1, pentagon, hexagon,
# heptagon, octagon, decagon, dodecagon, sector-60/30/120, pacman-90,
# robnik-0.15/0.3, thin-triangle/-0.3/-0.05, sinai-billiard, stadium,
# annulus-0.5, rhombus-30/60) are removed from the DOMAINS list below by
# filtering on id.
_REPLACED_IDS = {
    "rectangle-2-1",
    "ellipse-2-1", "ellipse-3-1", "ellipse-5-1",
    "pentagon", "hexagon", "heptagon",
    "octagon", "decagon", "dodecagon",
    "sector-60", "sector-30", "sector-120",
    "pacman-90",
    "robnik-0.15", "robnik-0.3",
    "thin-triangle", "thin-triangle-0.3", "thin-triangle-0.05",
    "sinai-billiard",
    "stadium",
    "annulus-0.5",
    "rhombus-30", "rhombus-60",
}
DOMAINS = [d for d in DOMAINS if d.id not in _REPLACED_IDS]
for _fam in FAMILIES:
    DOMAINS.extend(_family_to_domain_specs(_fam))


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def compute_domain(
    spec: DomainSpec,
    boundaries: Iterable[str] = ("dirichlet", "neumann", "robin", "steklov"),
) -> dict:
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
        "familyId": spec.family_id,
        "familyParam": spec.family_param,
        "mesh": {
            "vertices": int(mesh.p.shape[1]),
            "triangles": int(mesh.t.shape[1]),
            "element": "P2 Lagrange",
        },
        "boundaries": {},
    }

    def _alpha_tag(a: float) -> str:
        if abs(a - round(a)) < 1e-9:
            return str(int(round(a)))
        return f"{a:g}"

    for bc in boundaries:
        if bc == "robin":
            # parametric α slider: compute Robin for each α in ROBIN_ALPHAS
            entries = []
            for a in ROBIN_ALPHAS:
                print(f"  solving robin (α={a}) ...", flush=True)
                try:
                    vals, vecs, basis = solve_eigs(mesh, spec.n_eig, "robin", alpha=a)
                except Exception as e:
                    print(f"    skipped robin α={a}: {e}", flush=True)
                    continue
                a_tag = _alpha_tag(a)
                modes = []
                for k in range(spec.n_eig):
                    lam = float(vals[k])
                    img_path = OUT_IMG / f"{spec.id}_robin-{a_tag}_{k + 1}.webp"
                    plot_eigenfunction(
                        mesh, basis, vecs[:, k], img_path,
                        title=f"λ_{{{k+1}}} ≈ {lam:.4f} (α={a_tag})",
                    )
                    modes.append({
                        "k": k + 1,
                        "lambda": lam,
                        "lambdaExact": None,
                        "relErr": None,
                        "image": f"/files/eigenfunctions/img/{spec.id}_robin-{a_tag}_{k + 1}.webp",
                    })
                entries.append({"alpha": a, "modes": modes})
            if entries:
                result["boundaries"]["robin"] = {
                    "alphas": [e["alpha"] for e in entries],
                    "entries": entries,
                }
            continue

        print(f"  solving {bc} ...", flush=True)
        try:
            vals, vecs, basis = solve_eigs(mesh, spec.n_eig, bc)
        except Exception as e:
            print(f"    skipped {bc}: {e}", flush=True)
            continue
        analytic = None
        source = None
        if bc == "dirichlet":
            source = spec.analytical
        elif bc == "steklov":
            source = spec.analytical_steklov
        if source is not None:
            try:
                analytic = source(spec.n_eig)
            except Exception as e:
                print(f"    analytic failed: {e}")
                analytic = None

        modes = []
        for k in range(spec.n_eig):
            lam = float(vals[k])
            lam_ex = float(analytic[k]) if (analytic is not None and k < len(analytic)) else None
            rel_err = (abs(lam - lam_ex) / lam_ex) if (lam_ex is not None and lam_ex > 1e-12) else None

            img_path = OUT_IMG / f"{spec.id}_{bc}_{k + 1}.webp"
            plot_eigenfunction(mesh, basis, vecs[:, k], img_path,
                               title=f"λ_{{{k+1}}} ≈ {lam:.4f}")
            modes.append({
                "k": k + 1,
                "lambda": lam,
                "lambdaExact": lam_ex,
                "relErr": rel_err,
                "image": f"/files/eigenfunctions/img/{spec.id}_{bc}_{k + 1}.webp",
            })

        result["boundaries"][bc] = {"modes": modes}

    # write per-domain file
    (OUT_DATA / f"{spec.id}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    return result


def _first_few(bc_info: dict) -> list[float]:
    """Extract first few λ values from a BC info dict (handles Robin's nested structure)."""
    if "modes" in bc_info:
        return [m["lambda"] for m in bc_info["modes"][:4]]
    if "entries" in bc_info:
        # Robin: pick the α=1 entry if present, otherwise the first one.
        entries = bc_info["entries"]
        pick = next((e for e in entries if abs(float(e.get("alpha", 0)) - 1.0) < 1e-9), entries[0] if entries else None)
        if pick is None:
            return []
        return [m["lambda"] for m in pick["modes"][:4]]
    return []


def main(selection: list[str] | None = None, robin_only: bool = False) -> None:
    index = []
    for spec in DOMAINS:
        if selection and spec.id not in selection:
            continue
        if robin_only:
            # Recompute only the Robin family, splicing it into the existing
            # per-domain JSON. Keep Dirichlet/Neumann/Steklov untouched.
            existing_path = OUT_DATA / f"{spec.id}.json"
            if not existing_path.exists():
                r = compute_domain(spec)
            else:
                print(f"\n=== {spec.id} ({spec.name_en}) [robin-only] ===", flush=True)
                mesh = spec.builder()
                print(f"  mesh: {mesh.p.shape[1]} vertices, {mesh.t.shape[1]} triangles", flush=True)
                existing = json.loads(existing_path.read_text())
                # Drop pre-existing robin data (old single-α) and plot images.
                existing.setdefault("boundaries", {}).pop("robin", None)
                # reuse compute_domain but only request robin
                fake = compute_domain(spec, boundaries=("robin",))
                existing["boundaries"]["robin"] = fake["boundaries"].get("robin", {})
                existing["familyId"] = spec.family_id
                existing["familyParam"] = spec.family_param
                existing_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
                r = existing
        else:
            r = compute_domain(spec)
        index.append({
            "id": spec.id,
            "nameEn": spec.name_en,
            "nameJa": spec.name_ja,
            "descriptionEn": spec.description_en,
            "descriptionJa": spec.description_ja,
            "category": spec.category,
            "reference": spec.reference,
            "familyId": spec.family_id,
            "familyParam": spec.family_param,
            "mesh": r["mesh"],
            "firstFew": {
                bc: _first_few(r["boundaries"][bc])
                for bc in r["boundaries"]
            },
        })

    individuals = [d for d in index if d["familyId"] is None]
    family_entries = []
    for fam in FAMILIES:
        members_meta = [d for d in index if d["familyId"] == fam.id]
        if not members_meta:
            continue
        family_entries.append({
            "id": fam.id,
            "nameEn": fam.name_en,
            "nameJa": fam.name_ja,
            "descriptionEn": fam.description_en,
            "descriptionJa": fam.description_ja,
            "category": fam.category,
            "param": fam.param,
            "paramJa": fam.param_ja,
            "paramValues": list(fam.param_values),
            "memberIds": [d["id"] for d in members_meta],
            "reference": fam.reference,
        })

    (OUT_DATA / "index.json").write_text(
        json.dumps({
            "domains": index,  # kept for backwards compatibility
            "individuals": individuals,
            "families": family_entries,
        }, indent=2, ensure_ascii=False)
    )
    print(f"\nwrote {OUT_DATA/'index.json'} ({len(individuals)} individuals, {len(family_entries)} families, {len(index)} total domains)")


if __name__ == "__main__":
    args = sys.argv[1:]
    robin_only = "--robin-only" in args
    sel = [a for a in args if not a.startswith("--")]
    main(sel if sel else None, robin_only=robin_only)
