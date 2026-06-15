import os
import shutil
import unittest
import numpy as np
from numpy import testing
import casadi as ca
import trimesh

import skrobot
from skrobot.data import get_cache_dir, bunny_objpath
from skrobot.sdf import BoxSDF, SphereSDF, CylinderSDF, GridSDF, UnionSDF


# ==============================================================================
# SDF Conversion Function (Skrobot to CasADi)
# ==============================================================================
def convert_to_casadi_sdf(skrobot_sdf):
    # Ensure the skrobot internal coordinate systems are up-to-date
    skrobot_sdf.update()
    
    # 1. Create coordinate transformation (World -> SDF Local)
    coords = skrobot_sdf.copy_worldcoords()
    R_world_to_sdf = coords.rotation.T   # 3x3 rotation matrix
    t_world_to_sdf = coords.translation  # 3D translation vector
    
    p_world = ca.MX.sym('p_world', 3)
    p_local = ca.mtimes(ca.MX(R_world_to_sdf), p_world - ca.MX(t_world_to_sdf))

    # --- Case 1: SphereSDF ---
    if isinstance(skrobot_sdf, SphereSDF):
        r = getattr(skrobot_sdf, '_radius', getattr(skrobot_sdf, 'radius', None))
        sd_val = ca.norm_2(p_local) - r

    # --- Case 2: BoxSDF ---
    elif isinstance(skrobot_sdf, BoxSDF):
        width_val = getattr(skrobot_sdf, '_width', getattr(skrobot_sdf, '_widths', None))
        half_widths = ca.MX(width_val * 0.5)
        d_each_axis = ca.fabs(p_local) - half_widths
        
        positive_dists = ca.norm_2(ca.fmax(d_each_axis, 0))
        negative_dists = ca.fmin(ca.mmax(d_each_axis), 0)
        sd_val = positive_dists + negative_dists

    # --- Case 3: CylinderSDF ---
    elif isinstance(skrobot_sdf, CylinderSDF):
            r = getattr(skrobot_sdf, '_radius', getattr(skrobot_sdf, 'radius', None))
            h = getattr(skrobot_sdf, '_height', getattr(skrobot_sdf, 'height', None))
            half_h = h * 0.5
            
            # Small constant to prevent division by zero
            eps = 1e-8
            
            # Use safe norm calculation instead of ca.norm_2(p_local[0:2])
            r_dist = ca.sqrt(p_local[0]**2 + p_local[1]**2 + eps)
            z_dist = p_local[2]
            
            d_r = ca.fabs(r_dist) - r
            d_z = ca.fabs(z_dist) - half_h
            
            d_each_axis = ca.vertcat(d_r, d_z)
            
            # Also use a smoothed max function or safe norm here
            # Replaced "positive_dists = ca.norm_2(ca.fmax(d_each_axis, 0))" with the following:
            v_max = ca.fmax(d_each_axis, 0)
            positive_dists = ca.sqrt(v_max[0]**2 + v_max[1]**2 + eps)
            
            negative_dists = ca.fmin(ca.mmax(d_each_axis), 0)
            sd_val = positive_dists + negative_dists

    # --- Case 4: UnionSDF ---
    elif isinstance(skrobot_sdf, UnionSDF):
        child_functions = [convert_to_casadi_sdf(child) for child in skrobot_sdf.sdf_list]
        sd_val = child_functions[0](p_world)
        for child_f in child_functions[1:]:
            sd_val = ca.fmin(sd_val, child_f(p_world))

    # --- Case 5: GridSDF (Using fast built-in CasADi interpolant) ---
    elif isinstance(skrobot_sdf, GridSDF):
        data = skrobot_sdf._data        # 3D numpy array
        res = skrobot_sdf._resolution   # float
        origin = skrobot_sdf.origin     # length 3 array
        
        # 1. Generate grid coordinates for each axis (list of 1D arrays)
        # Note: Please adjust this if the data ordering (XYZ) of skrobot's GridSDF requires it.
        nx, ny, nz = data.shape
        x_grid = origin[0] + np.arange(nx) * res
        y_grid = origin[1] + np.arange(ny) * res
        z_grid = origin[2] + np.arange(nz) * res
        
        # 2. Flatten the data (CasADi's interpolant expects 'fortran' (F) order or a specific layout)
        # For typical bspline/linear interpolation, ravel(order='F') is often used.
        # If the signs or values of the result are misaligned, please check the supplementary notes.
        flat_data = data.ravel(order='F')
        
        # 3. Create CasADi interpolant (linear lookup table)
        # This function is optimized on the C++ side, eliminating symbolic index access overhead.
        interp_name = f"sdf_interp_{id(skrobot_sdf)}"  # Unique name to avoid duplication
        sdf_interp = ca.interpolant(interp_name, 'linear', [x_grid, y_grid, z_grid], flat_data)
        
        # 4. Pass local coordinates to the interpolant as a column vector
        # Since the interpolant accepts input as a (3, N) matrix, p_local is formatted as a column vector.
        sd_val = sdf_interp(p_local)

    else:
        raise NotImplementedError(f"SDF type {type(skrobot_sdf)} is not supported.")

    if getattr(skrobot_sdf, 'use_abs', False):
        sd_val = ca.fabs(sd_val)

    return ca.Function('sdf', [p_world], [sd_val])