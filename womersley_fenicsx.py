#------------------------------------------------------------------------
# Author: Erico Lopes de Souza
# Université Paris Saclay / Universidade de Brasilia (UnB)
# FEniCSx simulation of Womersley flow in a hemi-equilateral
# triangular duct (30-60-90 right triangle).
# Validates: mean velocity amplitude w_a and phase angle theta vs alpha.
#------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpi4py import MPI

import gmsh
from dolfinx.io import gmshio
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import (functionspace, Function, Constant,
                          form, assemble_scalar, locate_dofs_topological,
                          dirichletbc)
import ufl
from ufl import dx, grad, inner, TrialFunction, TestFunction, split
from basix.ufl import element as basix_element, mixed_element

# ── 1. Mesh ───────────────────────────────────────────────────────────────
# 30-60-90 hemi-equilateral triangle: (0,0), (1,0), (1, 1/sqrt(3))
# Angles: 30 deg at origin, 90 deg at (1,0), 60 deg at (1, 1/sqrt(3))

gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 0)
gmsh.model.add("hemi_equilateral")
h = 0.025   # mesh element size

pts = [
    gmsh.model.geo.addPoint(0.0, 0.0,            0.0, h),
    gmsh.model.geo.addPoint(1.0, 0.0,            0.0, h),
    gmsh.model.geo.addPoint(1.0, 1.0/np.sqrt(3), 0.0, h),
]
lines = [
    gmsh.model.geo.addLine(pts[0], pts[1]),
    gmsh.model.geo.addLine(pts[1], pts[2]),
    gmsh.model.geo.addLine(pts[2], pts[0]),
]
surf = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(lines)])
gmsh.model.geo.synchronize()

# Physical groups are required by dolfinx's gmshio
gmsh.model.addPhysicalGroup(2, [surf],  tag=1)   # surface (domain)
gmsh.model.addPhysicalGroup(1, lines,   tag=2)   # boundary lines

gmsh.model.mesh.generate(2)

domain, _, _ = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
gmsh.finalize()

# ── 2. Mixed function space W = (wc, ws) ─────────────────────────────────
P2  = basix_element("Lagrange", domain.basix_cell(), 2)
W   = functionspace(domain, mixed_element([P2, P2]))

W0, _ = W.sub(0).collapse()   # collapsed space for wc
W1, _ = W.sub(1).collapse()   # collapsed space for ws

# All boundary facets -> homogeneous Dirichlet (no-slip)
domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
boundary_facets = dmesh.locate_entities_boundary(
    domain, domain.topology.dim - 1,
    lambda x: np.ones(x.shape[1], dtype=bool)
)

dofs_0 = locate_dofs_topological((W.sub(0), W0), domain.topology.dim - 1, boundary_facets)
dofs_1 = locate_dofs_topological((W.sub(1), W1), domain.topology.dim - 1, boundary_facets)

bc0 = dirichletbc(Function(W0), dofs_0, W.sub(0))   # zero by default
bc1 = dirichletbc(Function(W1), dofs_1, W.sub(1))
bcs = [bc0, bc1]

# ── 3. Domain area (sanity check) ─────────────────────────────────────────
A_area = assemble_scalar(form(Constant(domain, np.float64(1.0)) * dx))
print(f"Mesh area:  {A_area:.6f}   (analytical sqrt(3)/6 = {np.sqrt(3)/6:.6f})")

# ── 4. Womersley FEM solver ───────────────────────────────────────────────
# Equations (eq. 11-12, normalized):
#   nabla^2 wc - alpha^2 ws = -1
#   nabla^2 ws + alpha^2 wc =  0
#
# Weak form (integration by parts, zero BCs):
#   int(grad wc . grad vc) + alpha^2 int(ws vc) = int(vc)       [eq 1]
#   int(grad ws . grad vs) - alpha^2 int(wc vs) = 0             [eq 2]

def solve_womersley(alpha):
    uc, us = split(TrialFunction(W))
    vc, vs = split(TestFunction(W))
    a2 = np.float64(alpha**2)

    a = (inner(grad(uc), grad(vc)) + a2 * us * vc +
         inner(grad(us), grad(vs)) - a2 * uc * vs) * dx
    L = Constant(domain, np.float64(1.0)) * vc * dx

    prob = LinearProblem(a, L, bcs=bcs,
                         petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    wh = prob.solve()
    return wh.sub(0).collapse(), wh.sub(1).collapse()

def mean_val(f):
    return assemble_scalar(form(f * dx)) / A_area

# ── 5. Analytical solution (series, eq. 27-29) ───────────────────────────
def analytical_wbar(alpha, N=40):
    A_an  = np.sqrt(3) / 6
    I2    = np.sqrt(3) / 8
    wc = ws = 0.0
    for m in range(1, N + 1):
        for n in range(1, N + 1):
            lmn = 4*np.pi**2/3 * (3*m**2 + 3*m*n + n**2)
            t1  = (1 - (-1)**m)     / (m       * (3*m + 2*n))
            t2  = (1 - (-1)**n)     / (n       * (2*m +   n))
            t3  = (1 - (-1)**(m+n)) / ((m + n) * (3*m +   n))
            Imn = np.sqrt(3) / np.pi**2 * (t1 + t2 - t3)
            if Imn == 0.0:
                continue
            fac = Imn**2 / I2
            den = lmn**2 + alpha**4
            wc += lmn       / den * fac
            ws += alpha**2  / den * fac
    return wc / A_an, ws / A_an

# ── 6. Validation sweep: alpha from 0 to 100 ─────────────────────────────
alphas   = np.linspace(0.0, 100.0, 50)
wa_fem   = np.empty(len(alphas))
th_fem   = np.empty(len(alphas))
wa_ana   = np.empty(len(alphas))
th_ana   = np.empty(len(alphas))

print("\nRunning sweep (FEM + analytical)...")
for i, alpha in enumerate(alphas):
    wc_h, ws_h = solve_womersley(alpha)
    wc_m, ws_m = mean_val(wc_h), mean_val(ws_h)
    wa_fem[i]  = np.hypot(wc_m, ws_m)
    th_fem[i]  = -np.arctan2(ws_m, wc_m)

    wc_a, ws_a = analytical_wbar(alpha)
    wa_ana[i]  = np.hypot(wc_a, ws_a)
    th_ana[i]  = -np.arctan2(ws_a, wc_a)

    print(f"  alpha={alpha:6.1f}  w_a: FEM={wa_fem[i]:.5f}  Ana={wa_ana[i]:.5f}  "
          f"theta: FEM={th_fem[i]:.4f}  Ana={th_ana[i]:.4f}")

# ── 7. Velocity field visualization for a chosen alpha ────────────────────
alpha_show = 5.0
print(f"\nComputing velocity field for alpha = {alpha_show} ...")
wc_f, ws_f = solve_womersley(alpha_show)

# Interpolate to P1 for clean visualization
P1 = basix_element("Lagrange", domain.basix_cell(), 1)
V1 = functionspace(domain, P1)

wc_p1 = Function(V1);  wc_p1.interpolate(wc_f)
ws_p1 = Function(V1);  ws_p1.interpolate(ws_f)

wa_nodes = np.hypot(wc_p1.x.array, ws_p1.x.array)
th_nodes = -np.arctan2(ws_p1.x.array, wc_p1.x.array)

# Triangulation from mesh geometry dofmap
dof_coords = V1.tabulate_dof_coordinates()[:, :2]
triang     = mtri.Triangulation(dof_coords[:, 0], dof_coords[:, 1])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Womersley flow — hemi-equilateral triangle,  alpha = {alpha_show}", fontsize=13)

im0 = axes[0].tripcolor(triang, wa_nodes, cmap="viridis", shading="gouraud")
plt.colorbar(im0, ax=axes[0], label="$w_a$")
axes[0].tricontour(triang, wa_nodes, levels=8, colors="white", linewidths=0.5)
axes[0].set_title("Velocity amplitude $w_a(x,y)$")
axes[0].set_xlabel("x");  axes[0].set_ylabel("y")
axes[0].set_aspect("equal")

im1 = axes[1].tripcolor(triang, th_nodes, cmap="RdBu", shading="gouraud")
plt.colorbar(im1, ax=axes[1], label=r"$\theta$ (rad)")
axes[1].set_title(r"Phase angle $\theta(x,y)$")
axes[1].set_xlabel("x");  axes[1].set_ylabel("y")
axes[1].set_aspect("equal")

plt.tight_layout()
plt.savefig("velocity_field.png", dpi=150)
print("Saved: velocity_field.png")

# ── 8. Validation plots: FEM vs analytical ────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Womersley flow — validation: FEM vs analytical series", fontsize=13)

ax1.plot(alphas, wa_ana, "k-",  lw=2,   label="Analytical (series)")
ax1.plot(alphas, wa_fem, "bo",  ms=5,   label="FEM (FEniCSx)")
ax1.set_xlabel(r"$\alpha$");   ax1.set_ylabel(r"$\bar{w}_a$")
ax1.set_title("Mean velocity amplitude")
ax1.legend();  ax1.grid(True, linestyle="--", alpha=0.6)

ax2.plot(alphas, th_ana, "k-",  lw=2,   label="Analytical (series)")
ax2.plot(alphas, th_fem, "ro",  ms=5,   label="FEM (FEniCSx)")
ax2.axhline(-np.pi/2, color="gray", ls="--", lw=0.8, label=r"$-\pi/2$ limit")
ax2.set_xlabel(r"$\alpha$");   ax2.set_ylabel(r"$\bar{\theta}$ (rad)")
ax2.set_title("Mean phase angle")
ax2.legend();  ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("womersley_validation_fem.png", dpi=150)
print("Saved: womersley_validation_fem.png")

plt.show()

# ── 9. Animation: w_a and theta fields as alpha increases ─────────────────
from matplotlib.animation import FuncAnimation, PillowWriter

alpha_anim = np.linspace(0.0, 50.0, 50)

print("\nPre-computing fields for animation (this may take a few minutes)...")
wa_frames = []
th_frames = []
for k, alpha in enumerate(alpha_anim):
    wc_a, ws_a = solve_womersley(alpha)
    wc_p = Function(V1);  wc_p.interpolate(wc_a)
    ws_p = Function(V1);  ws_p.interpolate(ws_a)
    wa_frames.append(np.hypot(wc_p.x.array, ws_p.x.array).copy())
    th_frames.append(-np.arctan2(ws_p.x.array, wc_p.x.array).copy())
    print(f"  frame {k+1:02d}/{len(alpha_anim)}  alpha={alpha:.1f}")

wa_max = max(f.max() for f in wa_frames)
th_min = min(f.min() for f in th_frames)
th_max = max(f.max() for f in th_frames)

fig_a, axs_a = plt.subplots(1, 2, figsize=(12, 5))

tc0 = axs_a[0].tripcolor(triang, wa_frames[0], cmap="viridis",
                          shading="gouraud", vmin=0, vmax=wa_max)
plt.colorbar(tc0, ax=axs_a[0], label="$w_a$")
axs_a[0].set_aspect("equal");  axs_a[0].set_xlabel("x");  axs_a[0].set_ylabel("y")
axs_a[0].set_title("Velocity amplitude $w_a(x,y)$")

tc1 = axs_a[1].tripcolor(triang, th_frames[0], cmap="RdBu",
                          shading="gouraud", vmin=th_min, vmax=th_max)
plt.colorbar(tc1, ax=axs_a[1], label=r"$\theta$ (rad)")
axs_a[1].set_aspect("equal");  axs_a[1].set_xlabel("x");  axs_a[1].set_ylabel("y")
axs_a[1].set_title(r"Phase angle $\theta(x,y)$")

sup = fig_a.suptitle(f"Womersley flow  —  α = {alpha_anim[0]:.1f}", fontsize=13)
plt.tight_layout()

def update(frame):
    tc0.set_array(wa_frames[frame])
    tc1.set_array(th_frames[frame])
    sup.set_text(f"Womersley flow  —  α = {alpha_anim[frame]:.1f}")
    return tc0, tc1, sup

anim = FuncAnimation(fig_a, update, frames=len(alpha_anim), interval=150, blit=False)

# Save as GIF (requires Pillow: pip install Pillow)
anim.save("womersley_animation.gif", writer=PillowWriter(fps=8), dpi=100)
print("Saved: womersley_animation.gif")

# To save as MP4 instead (requires ffmpeg installed):
# anim.save("womersley_animation.mp4", fps=8, dpi=100)

print("\nDone.")
