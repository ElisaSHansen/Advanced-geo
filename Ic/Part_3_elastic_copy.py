import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# INPUT DATA
# ============================================================

L = 16.0                 # embedded pile length [m]
D = 2.5                  # outer diameter [m]
t = 0.025                # wall thickness [m]
d = D - 2.0 * t          # inner diameter [m]

Ep = 210e9               # Young's modulus of steel [Pa]
Epy = 68.07e6            # foundation stiffness per unit length [N/m^2]

V_head = 1.625e6         # horizontal load at pile head [N]
M_head = 37.86e6         # moment at pile head [Nm]

# ============================================================
# SECTION PROPERTIES
# ============================================================

I = np.pi * (D**4 - d**4) / 64.0
EI = Ep * I
lam = (Epy / (4.0 * EI)) ** 0.25

print("Section properties")
print("------------------------------")
print(f"Outer diameter D = {D:.3f} m")
print(f"Inner diameter d = {d:.3f} m")
print(f"I  = {I:.6e} m^4")
print(f"EI = {EI:.6e} Nm^2")
print(f"lambda = {lam:.6e} 1/m")


# ============================================================
# HELPER: SHEAR FROM MOMENT
# Use one consistent post-processing rule for both methods
# ============================================================

def shear_from_moment(M, x):
    dx = x[1] - x[0]
    V = np.full_like(M, np.nan)

    # Central difference in the interior
    # Minus sign because depth x is positive downward
    V[1:-1] = -(M[2:] - M[:-2]) / (2.0 * dx)

    # One-sided differences at the ends
    V[0] = -(-3.0 * M[0] + 4.0 * M[1] - M[2]) / (2.0 * dx)
    V[-1] = -(3.0 * M[-1] - 4.0 * M[-2] + M[-3]) / (2.0 * dx)

    return V


# ============================================================
# FINITE DIFFERENCE METHOD
# Governing equation:
#     EI * y'''' + Epy * y = 0
#
# Boundary conditions:
#   x = 0 (pile head, free with applied load and moment)
#       EI y''(0)  = M_head
#       EI y'''(0) = V_head
#
#   x = L (pile tip, fixed)
#       y(L)  = 0
#       y'(L) = 0
# ============================================================

def solve_FDM(n=400):
    dx = L / n
    alpha = Epy / EI

    # Unknowns: u[0] ... u[n]
    # Node 0 = head, node n = tip
    N = n + 1
    K = np.zeros((N, N))
    p = np.zeros(N)

    # --------------------------------------------------------
    # Interior nodes: fourth-order central difference
    # --------------------------------------------------------
    for i in range(2, n - 1):
        K[i, i - 2] = 1.0
        K[i, i - 1] = -4.0
        K[i, i]     = 6.0 + alpha * dx**4
        K[i, i + 1] = -4.0
        K[i, i + 2] = 1.0

    # --------------------------------------------------------
    # Head boundary conditions at x = 0
    # EI y''(0) = M_head
    # y''(0) ≈ (2u0 - 5u1 + 4u2 - u3) / dx^2
    # --------------------------------------------------------
    K[0, 0] = 2.0
    K[0, 1] = -5.0
    K[0, 2] = 4.0
    K[0, 3] = -1.0
    p[0] = M_head * dx**2 / EI

    # EI y'''(0) = V_head
    # y'''(0) ≈ (-5u0 + 18u1 - 24u2 + 14u3 - 3u4) / (2 dx^3)
    K[1, 0] = -5.0
    K[1, 1] = 18.0
    K[1, 2] = -24.0
    K[1, 3] = 14.0
    K[1, 4] = -3.0
    p[1] = 2.0 * V_head * dx**3 / EI

    # --------------------------------------------------------
    # Near-head governing equation at i = 1
    # --------------------------------------------------------
    K[2, 0] = 1.0
    K[2, 1] = -4.0
    K[2, 2] = 6.0 + alpha * dx**4
    K[2, 3] = -4.0
    K[2, 4] = 1.0

    # --------------------------------------------------------
    # Near-tip governing equation at i = n-2
    # --------------------------------------------------------
    i = n - 2
    K[i, i - 2] = 1.0
    K[i, i - 1] = -4.0
    K[i, i]     = 6.0 + alpha * dx**4
    K[i, i + 1] = -4.0
    K[i, i + 2] = 1.0

    # --------------------------------------------------------
    # Tip boundary conditions at x = L
    # y'(L) = 0 and y(L) = 0
    # y'(L) ≈ (3u_n - 4u_{n-1} + u_{n-2}) / (2 dx)
    # --------------------------------------------------------
    K[n - 1, n]     = 3.0
    K[n - 1, n - 1] = -4.0
    K[n - 1, n - 2] = 1.0
    p[n - 1] = 0.0

    K[n, n] = 1.0
    p[n] = 0.0

    u = np.linalg.solve(K, p)

    x = np.linspace(0.0, L, N)
    y = u.copy()

    theta = np.full_like(y, np.nan)
    M = np.full_like(y, np.nan)

    # Slope
    theta[1:-1] = (y[2:] - y[:-2]) / (2.0 * dx)
    theta[0] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (2.0 * dx)
    theta[-1] = (3.0 * y[-1] - 4.0 * y[-2] + y[-3]) / (2.0 * dx)

    # Moment
    M[1:-1] = EI * (y[:-2] - 2.0 * y[1:-1] + y[2:]) / dx**2
    M[0] = EI * (2.0 * y[0] - 5.0 * y[1] + 4.0 * y[2] - y[3]) / dx**2
    M[-1] = EI * (2.0 * y[-1] - 5.0 * y[-2] + 4.0 * y[-3] - y[-4]) / dx**2

    return x, y, theta, M


# ============================================================
# ANALYTICAL METHOD
# General solution:
# y = exp(lx)(C1 cos lx + C2 sin lx) + exp(-lx)(C3 cos lx + C4 sin lx)
# ============================================================

def solve_analytical(npts=800):
    l = lam

    A = np.zeros((4, 4))
    b = np.zeros(4)

    # --------------------------------------------------------
    # At x = 0:
    # y''(0)  = 2 l^2 (C2 - C4) = M_head / EI
    # y'''(0) = 2 l^3 (-C1 + C2 + C3 + C4) = V_head / EI
    # --------------------------------------------------------
    A[0, :] = [0.0, 2.0 * l**2, 0.0, -2.0 * l**2]
    b[0] = M_head / EI

    A[1, :] = [-2.0 * l**3, 2.0 * l**3, 2.0 * l**3, 2.0 * l**3]
    b[1] = V_head / EI

    # --------------------------------------------------------
    # At x = L:
    # y(L)  = 0
    # y'(L) = 0
    # --------------------------------------------------------
    epl = np.exp(l * L)
    eml = np.exp(-l * L)
    cL = np.cos(l * L)
    sL = np.sin(l * L)

    A[2, :] = [
        epl * cL,
        epl * sL,
        eml * cL,
        eml * sL
    ]
    b[2] = 0.0

    A[3, :] = [
        epl * l * (cL - sL),
        epl * l * (sL + cL),
        eml * l * (-cL - sL),
        eml * l * (-sL + cL)
    ]
    b[3] = 0.0

    C1, C2, C3, C4 = np.linalg.solve(A, b)

    x = np.linspace(0.0, L, npts)
    ex = np.exp(l * x)
    emx = np.exp(-l * x)
    cx = np.cos(l * x)
    sx = np.sin(l * x)

    y = ex * (C1 * cx + C2 * sx) + emx * (C3 * cx + C4 * sx)

    dy = (
        ex * l * ((C1 + C2) * cx + (C2 - C1) * sx)
        + emx * l * ((-C3 + C4) * cx - (C3 + C4) * sx)
    )

    d2y = (
        ex * 2.0 * l**2 * (-C1 * sx + C2 * cx)
        + emx * 2.0 * l**2 * (C3 * sx - C4 * cx)
    )

    M = EI * d2y
    theta = dy

    return x, y, theta, M


# ============================================================
# SOLVE BOTH METHODS
# ============================================================

x_fdm, y_fdm, th_fdm, M_fdm = solve_FDM(n=400)
x_an,  y_an,  th_an,  M_an  = solve_analytical(npts=800)

# Compute shear consistently from moment for both methods
V_fdm = shear_from_moment(M_fdm, x_fdm)
V_an = shear_from_moment(M_an, x_an)

# ============================================================
# PRINT COMPARISON
# ============================================================

print("\nMaximum values")
print("------------------------------")
print("FDM:")
print(f"Max displacement: {np.nanmax(np.abs(y_fdm)) * 1000:.2f} mm")
print(f"Max rotation:     {np.nanmax(np.abs(th_fdm)) * 1e3:.3f} x10^-3")
print(f"Max moment:       {np.nanmax(np.abs(M_fdm)) / 1e6:.3f} MNm")
print(f"Max shear:        {np.nanmax(np.abs(V_fdm)) / 1e3:.2f} kN")

print("\nAnalytical:")
print(f"Max displacement: {np.nanmax(np.abs(y_an)) * 1000:.2f} mm")
print(f"Max rotation:     {np.nanmax(np.abs(th_an)) * 1e3:.3f} x10^-3")
print(f"Max moment:       {np.nanmax(np.abs(M_an)) / 1e6:.3f} MNm")
print(f"Max shear:        {np.nanmax(np.abs(V_an)) / 1e3:.2f} kN")

print("\nHead values")
print("------------------------------")
print(f"Applied head moment: {M_head / 1e6:.3f} MNm")
print(f"Applied head shear:  {V_head / 1e3:.2f} kN")


# ============================================================
# PLOTS
# ============================================================

fig, axs = plt.subplots(1, 4, figsize=(18, 6))

# Displacement
axs[0].plot(y_fdm * 1000, x_fdm, label="FDM", lw=2)
axs[0].plot(y_an * 1000, x_an, "--", label="Analytical", lw=2)
axs[0].set_title("Displacement [mm]")
axs[0].set_xlabel("y [mm]")
axs[0].set_ylabel("Depth x [m]")
axs[0].invert_yaxis()
axs[0].grid(True)
axs[0].legend()

# Rotation
axs[1].plot(th_fdm * 1e3, x_fdm, lw=2)
axs[1].plot(th_an * 1e3, x_an, "--", lw=2)
axs[1].set_title("Rotation [$10^{-3}$ rad]")
axs[1].set_xlabel(r"$\theta$ [$10^{-3}$]")
axs[1].invert_yaxis()
axs[1].grid(True)

# Moment
axs[2].plot(M_fdm / 1e6, x_fdm, lw=2)
axs[2].plot(M_an / 1e6, x_an, "--", lw=2)
axs[2].set_title("Moment [MNm]")
axs[2].set_xlabel("M [MNm]")
axs[2].invert_yaxis()
axs[2].grid(True)

# Shear
axs[3].plot(V_fdm / 1e3, x_fdm, lw=2, label="FDM")
axs[3].plot(V_an / 1e3, x_an, "--", lw=2, label="Analytical")
axs[3].set_title("Shear [kN]")
axs[3].set_xlabel("V [kN]")
axs[3].invert_yaxis()
axs[3].grid(True)
axs[3].legend()

plt.tight_layout()
plt.show()