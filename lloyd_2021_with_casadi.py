#!/usr/bin/env python

# The unofficial implementation of
# J. E. Lloyd, F. Roewer-Després and I. Stavness, 
# "Muscle Path Wrapping on Arbitrary Surfaces," 
# in IEEE Transactions on Biomedical Engineering, vol. 68, no. 2, pp. 628-638, Feb. 2021, 
# doi: 10.1109/TBME.2020.3009922. 
# 
# The original implementation involves matrix calculation with a tridiagonal matrix. 
# The implementation shown here simply calculates the sum of the potential energy. 
# The wrapping problem is formulated as casadi nonlinear optimization problem.
# The code is written with the help of Google Gemini. 


import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Common Helper Functions ---
def split_with_remainder_at_end(total: int, n_parts: int) -> np.ndarray:
    base = total // n_parts
    remainder = total % n_parts
    parts = np.full(n_parts, base, dtype=int)
    parts[-1] += remainder
    return parts

def divide_segment_with_waypoints(points, divisions):
    if len(divisions) != len(points) - 1:
        raise ValueError("len(divisions) should be equal to len(points) -1.")
    all_points = []
    for i in range(len(divisions)):
        start = np.array(points[i])
        end = np.array(points[i + 1])
        n = divisions[i]
        segment = np.linspace(start, end, n + 1)
        if i != 0:
            segment = segment[1:]
        all_points.append(segment)
    return np.vstack(all_points)


# --- Factory Function for Tendon Wrapping Solver ---
def create_tendon_solver(m, k_line=1.0, k_contact=2.0):
    """
    Accepts an arbitrary SDF formula (symbolic inputs and outputs) and builds 
    a wrapper that manages the inputs and outputs of the NLP solver.
    """
    # 1. Define optimization variables (all internal contact points: 3 * m dimensional vector)
    X_sym = ca.MX.sym('X', 3 * m)
    
    # Define external parameters (fixed endpoints: 3D start point + 3D end point)
    p_start_sym = ca.MX.sym('p_start', 3)
    p_end_sym = ca.MX.sym('p_end', 3)
    
    # 2. Build elastic energy for the line
    E_line = 0.0
    
    # First segment
    p_first = X_sym[0:3]
    E_line += 0.5 * k_line * ca.sumsqr(p_first - p_start_sym)
    
    # Intermediate segments
    for j in range(m - 1):
        p_current = X_sym[j*3 : (j+1)*3]
        p_next    = X_sym[(j+1)*3 : (j+2)*3]
        E_line += 0.5 * k_line * ca.sumsqr(p_next - p_current)
        
    # Last segment
    p_last = X_sym[(m-1)*3 : m*3]
    E_line += 0.5 * k_line * ca.sumsqr(p_end_sym - p_last)
    
    # 3. Build contact energy (SDF)
    # To keep it shape-independent, a "symbolic SDF function" that accepts a 3D input 
    # can be injected from the outside.
    p_single = ca.MX.sym('p_single', 3)
    
    # Create a placeholder function to bind a concrete SDF (MX expression) in the next stage
    def build_solver_with_sdf_expr(sdf_expr_func):
        """
        sdf_expr_func: A Python function that takes ca.MX(3) and returns ca.MX(1) (SDF value).
        """
        E_contact = 0.0
        for j in range(m):
            p_j = X_sym[j*3 : (j+1)*3]
            sdf_j = sdf_expr_func(p_j) # Evaluate the geometry formula provided from outside
            
            penetration = ca.fmin(sdf_j, 0.0)
            E_contact += 0.5 * k_contact * (penetration ** 2)
            
        # Total energy
        objective = E_line + E_contact
        
        # NLP definition
        nlp = {'x': X_sym, 'f': objective, 'p': ca.vcat([p_start_sym, p_end_sym])}
        
        # Solver generation
        opts = {'ipopt.print_level': 0, 'ipopt.max_iter': 200, 'ipopt.tol': 1e-8}
        nlp_solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        return nlp_solver

    return build_solver_with_sdf_expr


# --- Concrete SDF Geometry Definitions (Users can add more freely) ---
def sdf_cylinder(p, radius=0.5, height=2.0):
    d = ca.sqrt(p[0]**2 + p[1]**2) - radius
    return ca.fmax(d, ca.fabs(p[2]) - height / 2.0)

def sdf_sphere(p, radius=0.6, center=None):
    if center is None:
        center = [0.0, 0.2, 0.0] # A sphere with a slightly offset center
    return ca.norm_2(p - ca.MX(center)) - radius

def sdf_rectangle(p, sx=0.5, sy=0.3, height=2.0):
    """
    SDF for a centered rectangular cuboid.
    sx, sy: Half-lengths from the center to each side.
    height: Height along the Z-axis.
    """
    # 1. Calculate the rectangular SDF on the XY plane
    # Absolute the XY coordinates of point p, and compute the difference from the size (outward overflow)
    d_xy = ca.fabs(p[0:2]) - ca.vertcat(sx, sy)
    
    # Distance component when outside (norm of max(d, 0))
    ext_dist_xy = ca.norm_2(ca.fmax(d_xy, 0.0))
    # Distance component when inside (negative distance to the closest edge)
    int_dist_xy = ca.fmin(ca.fmax(d_xy[0], d_xy[1]), 0.0)
    
    sdf_xy = ext_dist_xy + int_dist_xy
    
    # 2. Extrusion processing along the Z-axis (height)
    return ca.fmax(sdf_xy, ca.fabs(p[2]) - height / 2.0)


def sdf_rounded_rectangle(p, sx=0.5, sy=0.3, r=0.1, height=2.0):
    """
    SDF for a rounded rectangular cuboid.
    sx, sy: Half-lengths from the center to the original corners before rounding.
    r: Radius of the corners (*The shape expands outward by this radius).
    height: Height along the Z-axis.
    """
    # The basic idea is the same as the rectangle, but the radius 'r' is subtracted from the size beforehand.
    d_xy = ca.fabs(p[0:2]) - ca.vertcat(sx - r, sy - r)
    
    ext_dist_xy = ca.norm_2(ca.fmax(d_xy, 0.0))
    int_dist_xy = ca.fmin(ca.fmax(d_xy[0], d_xy[1]), 0.0)
    
    # Subtracting the radius 'r' at the end smoothly rounds the corners.
    sdf_xy = ext_dist_xy + int_dist_xy - r
    
    # Extrusion processing along the Z-axis (height)
    return ca.fmax(sdf_xy, ca.fabs(p[2]) - height / 2.0)



def main():

    # 1. Generate initial tendon configuration
    segments = 100
    keysegments = 2
    keypoints = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([-1.0, 0.0, 0.0])
    ]
    tendon_points = divide_segment_with_waypoints(keypoints, split_with_remainder_at_end(segments, keysegments))
    
    all_points = tendon_points.copy().reshape(tendon_points.size)
    iter_points_init = all_points[3:-3]
    p_start_val = all_points[:3]
    p_end_val = all_points[-3:]
    
    m = int(len(iter_points_init)/3)
    
    # 2. Initialize solver factory
    solver_factory = create_tendon_solver(m=m, k_line=1.0, k_contact=2.0)
    
    # 3. Build the solver by passing the geometry as an argument (function object)
    # * Simply switching this line to sdf_sphere instantly changes the wrapping target to a sphere!
    target_sdf = lambda p: sdf_cylinder(p, radius=0.5, height=2.0)
    # target_sdf = lambda p: sdf_sphere(p, radius=0.6) 
    # target_sdf = lambda p: sdf_rectangle(p, sx=0.5, sy=0.3, height=2.0)
    # target_sdf = lambda p: sdf_rounded_rectangle(p, sx=0.5, sy=0.5, r=0.1, height=2.0)
    
    solver = solver_factory(target_sdf)
    
    # 4. Execute solver (concatenate and pass start and end coordinates as parameters)
    p_param = np.concatenate([p_start_val, p_end_val])
    sol = solver(x0=iter_points_init, p=p_param)
    
    # Retrieve optimization results
    x_opt = np.array(sol['x']).flatten()
    tendon_points_opt = np.concatenate([p_start_val, x_opt, p_end_val]).reshape(-1, 3)
    
    print("Optimization finished successfully!")

    # --- Visualization ---
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Optimized tendon
    ax.plot(tendon_points_opt[:, 0], tendon_points_opt[:, 1], tendon_points_opt[:, 2], label='Optimized Tendon', color='k')
    ax.scatter(tendon_points_opt[:, 0], tendon_points_opt[:, 1], tendon_points_opt[:, 2], color='r', s=15)
    
    # Initial configuration (dotted line)
    ax.plot(tendon_points[:, 0], tendon_points[:, 1], tendon_points[:, 2], label='Initial Guess', color='gray', linestyle=':')
    
    # Obstacle guide (draw a simple circle for reference)
    theta = np.linspace(0, 2.0 * np.pi, 100)
    ax.plot(0.5 * np.cos(theta), 0.5 * np.sin(theta), 0, color='blue', linestyle='--', label='Obstacle Bound')
    
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Generalized Tendon Wrapping Solver")
    ax.legend()
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    main()