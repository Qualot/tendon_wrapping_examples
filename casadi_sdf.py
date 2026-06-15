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

    # --- Case 5: GridSDF (Pure Trilinear Interpolation Formula) ---
    elif isinstance(skrobot_sdf, GridSDF):
        data = skrobot_sdf._data        # 3D numpy array
        res = skrobot_sdf._resolution   # float
        origin = skrobot_sdf.origin     # length 3 array
        
        # Convert local coordinates to raw grid indices (float)
        idx_raw = (p_local - ca.MX(origin)) / res
        
        # Clamp indices to stay within grid boundaries
        max_idx = ca.MX([float(data.shape[0]-2), float(data.shape[1]-2), float(data.shape[2]-2)])
        idx_clamped = ca.fmax(0.0, ca.fmin(max_idx, idx_raw))
        
        # Get integer part of the indices using floor function
        idx_f = ca.floor(idx_clamped)
        
        # Calculate interpolation ratios t (0.0 <= t <= 1.0)
        t = idx_clamped - idx_f
        tx, ty, tz = t[0], t[1], t[2]
        
        # Flatten the 3D grid data in 'C' order (matching SciPy's internal layout)
        flat_data_np = data.ravel(order='C')
        ca_data_table = ca.MX(flat_data_np)
        
        # Strides for 3D-to-1D index conversion
        stride_x = data.shape[1] * data.shape[2]
        stride_y = data.shape[2]
        
        def get_val(nx, ny, nz):
            cur_x = idx_f[0] + nx
            cur_y = idx_f[1] + ny
            cur_z = idx_f[2] + nz
            flat_idx = cur_x * stride_x + cur_y * stride_y + cur_z
            return ca_data_table[flat_idx]
            
        # Fetch values from the 8 surrounding grid vertices
        c000 = get_val(0, 0, 0)
        c001 = get_val(0, 0, 1)
        c010 = get_val(0, 1, 0)
        c011 = get_val(0, 1, 1)
        c100 = get_val(1, 0, 0)
        c101 = get_val(1, 0, 1)
        c110 = get_val(1, 1, 0)
        c111 = get_val(1, 1, 1)
        
        # Step-by-step Trilinear Interpolation
        c00 = c000 * (1 - tx) + c100 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c11 = c011 * (1 - tx) + c111 * tx
        
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        
        sd_val = c0 * (1 - tz) + c1 * tz

    else:
        raise NotImplementedError(f"SDF type {type(skrobot_sdf)} is not supported.")

    if getattr(skrobot_sdf, 'use_abs', False):
        sd_val = ca.fabs(sd_val)

    return ca.Function('sdf', [p_world], [sd_val])