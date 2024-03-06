"""
Module by G. Vernassa.
Computes the elasticity constants for a transversely isotropic material according 
to the rules of mixtures.
It assumes that the material is stratified along a Z direction.
Being therefore axisymmetric in the X-Y plane.
"""

import math as m
import numpy as np

def Ex(thicknesses, moduli):
    """
    
    Parameters
    ----------
    thicknesses : (Nx0) np.array
        Sequence of thicknesses for the different material layers
    moduli : (Nx0) np.array
        Sequence of elastic moduli for the respective material layers

    Returns Ex
    -------
    In plane elastic modulus

    """
    fractions = thicknesses/np.sum(thicknesses)
    Eeq = np.dot(fractions,moduli)
    return Eeq 

def Ez(thicknesses, moduli):
    fractions = thicknesses/np.sum(thicknesses)
    Eeq = np.sum(fractions/moduli)**(-1)
    return Eeq

def Gxy(thicknesses, Gi):
    fractions = thicknesses/np.sum(thicknesses)
    Geq = np.dot(fractions,Gi)
    return Geq 

def Gxz(thicknesses, Gi):
    fractions = thicknesses/np.sum(thicknesses)
    Geq = np.sum(fractions/Gi)**(-1)
    return Geq

def nuxz(thicknesses, nui):
    fractions = thicknesses/np.sum(thicknesses)
    nueq = np.dot(fractions,nui)
    return nueq

def nuxy(thicknesses, nui, moduli):
    """
    This formula needs verification
    """
    fractions = thicknesses/np.sum(thicknesses)
    nume = np.sum(nui*fractions*moduli)
    denom = np.dot(moduli,fractions)
    nueq = nume/denom
    return nueq

def C_S(EX, EZ, NUXY, NUXZ, GXY, GXZ):
    """
    Returns elasticity constants' matrix and compliance matrix according to 
    {s_x, s_y, s_z, tau_xy, tau_yz, tau_zx} = [C] {eps_x, eps_y, eps_z, gamma_xy, gamma_yz, gamma_zx}
    """
    S = np.matrix([[1/EX, -NUXY/EX, -NUXZ/EX, 0, 0, 0],
              [-NUXY/EX, 1/EX, -NUXZ/EX, 0, 0, 0],
              [-NUXZ/EX, -NUXZ/EX, 1/EZ, 0, 0, 0],
              [0, 0, 0, 1/GXY, 0, 0],
              [0, 0, 0, 0, 1/GXZ, 0],
              [0, 0, 0, 0, 0, 1/GXZ]]
             )
    C = S.I
    return C, S

def C_S_rthz(ER, ETH, NUTHR, NUTHZ, GTHR, GTHZ):
    """
    Returns elasticity constants' matrix and compliance matrix for a transversely isotropic material in the {r, th, z} csys, stratified along r.
    {s_r, s_th, s_z, tau_rth, tau_thz, tau_rz} = [C] {eps_r, eps_th, eps_z, gamma_rth, gamma_thz, gamma_rz}
    """
    S = np.matrix([[1/ER, -NUTHR/ETH, -NUTHR/ETH, 0, 0, 0],
              [-NUTHR/ETH, 1/ETH, -NUTHZ/ETH, 0, 0, 0],
              [-NUTHR/ETH, -NUTHZ/ETH, 1/ETH, 0, 0, 0],
              [0, 0, 0, 1/GTHR, 0, 0],
              [0, 0, 0, 0, 1/GTHZ, 0],
              [0, 0, 0, 0, 0, 1/GTHR]]
             )
    C = S.I
    return C, S

# Need to add the decomposition of stresses

# For the hypotheses made at the basis of the rules of mixtures we have that:
# - sigmaZ is the same in each layer.
# - Eps_x and Eps_y are the same in each layer

# We can invert the constitutive laws for each layer (l) to have:
# [sigma_x(l)| = [S_XX(l), S_XY(l)| * [eps_X - S_XZ(l)*sigma_z|
# |sigma_y(l)] = |S_XY(l), S_YY(l)] * |eps_Y - S_YZ(l)*sigma_z]

# In case of a cylindrical reference system, with layers piled up along r,
# we have:
# - sigma_r being equal
# - eps_theta and eps_z being equal

# We can invert the constitutive laws for each layer (l) to have:
# [sigma_theta(l)| = [S_thetatheta(l), S_thetaZ(l)| * [eps_theta - S_rtheta(l)*sigma_r|
# |sigma_z(l)    ] = |S_thetaz(l),     S_zz(l)    ] * |eps_z - S_rz(l)*sigma_r        ]

def sigma_th_sigma_z_l(sigma_r, eps_th, eps_z, S_layer):
    """
    This function returns the values of hoop and axial stresses in a layer of material, given the
    components of:
    - sigma_r: Radial stress,
    - eps_th: hoop strain,
    - eps_z: axial strain.
    - Compliance matrix (S) of the layer.
    Assuming the material is stratified along {r}.
    and given the base material compliance matrix S_layer
    """
    dim = np.shape(sigma_r)[0]
    sigma_th = np.zeros_like(sigma_r)
    sigma_z = np.zeros_like(sigma_r)
    S_new = S_layer[1:3, 1:3]

    inv_s = np.linalg.inv(S_new)
    
    for i, sri in enumerate(sigma_r):
        sthi, szi = inv_s.dot(np.array([[eps_th[i] - S_layer[0,1]*sri], 
                                        [eps_z[i] - S_layer[0,2]*sri]]))
        sigma_th[i] = sthi
        sigma_z[i] = szi

    return sigma_th, sigma_z
