#""""
# This file contains the formulas for the Generalized Plane Strain Analysis of 
# superconducting solenoids, according to the publication of Markiewicz.
# The coefficients are however adapted to the pdf I created.
#""""

import numpy as np
import matplotlib.pyplot as plt

def A_ij_energization(obj):
    """
    This function takes in:
    - Obj [ditionary]: contains keys 'C', 'J', 'B0' and 'C0' which are 
    respectively the Matrix of elastic constants of the material, the 
    current density in the section, and the field parameters to describe
    the linear decay of field intensity as B(r) = B0-C0*r

    - radius[float]: In case an orthotropic material is used, with C_theta/C_r = 1,2,3
    a the radius of calculation must be specified.
    """
    # WATCH OUT FOR k == 2, 3
    Crr = obj['C'][0,0]
    Crth = obj['C'][0,1]
    Crz = obj['C'][0,2]
    Cthth =obj['C'][1,1]
    Cthz = obj['C'][1,2]

    JJ = obj['J'] 
    B0 = obj['B0'] 
    C0 = obj['C0'] 
    
    kk = np.sqrt(Cthth/Crr) 
    
    if np.round(kk,5) == 1.0:
        A1 = 0.0
        A2 = -1/3*JJ*B0/Crr 
        A3 = 1/8*JJ*C0/Crr
    else:
        A1 = -1/(1-kk**2)*(Crz-Cthz)/Crr  
        A2 = -1/(4-kk**2)*(JJ*B0)/Crr  
        A3 = 1/(9-kk**2)*(JJ*C0)/Crr 
       
    return A1, A2, A3         
    
    
def coefficients_energization(obj):
    """
    This function generates the alpha, beta, f, F, coefficients to 
    Describe a problem according to Markiewicz formulation.
    
    Inputs:
    - Obj [dictionary]
    
    Returns:
     -     [[alpha1, alpha2, alpha3],
            [beta1, beta2, beta3],
            [f1, f2, f3],
            [Fa, Fb, Ft],
            [A1, A2, A3],
            [aa, bb, kk]]
    """
    Crr = obj['C'][0,0]     # ok
    Crth = obj['C'][0,1]     # ok
    Crz = obj['C'][0,2]     # ok
    Cthth = obj['C'][1,1]     # ok
    Cthz = obj['C'][1,2]     # ok
    Czz = obj['C'][2,2]     # ok
    
    kk = np.sqrt(Cthth/Crr)     # ok
    aa = obj['a']     # ok
    bb = obj['b']     # ok
    
    A1, A2, A3 = A_ij_energization(obj)     # ok

    if np.round(kk,5) == 1.0:
        alpha1 = (Crth + Crr)
        alpha2 = (Crth-Crr)/(aa**2)
        alpha3 = Crz

        beta1 = (Crth + Crr)
        beta2 = (Crth-Crr)/(bb**2)
        beta3 = Crz
    else:
        alpha1 = (Crth + kk*Crr)*aa**(kk-1)     # ok
        alpha2 = (Crth - kk*Crr)*aa**(-kk-1)     # ok
        alpha3 = (Crth+Crr)*A1+Crz     # ok
        
        beta1 = (Crth +kk*Crr)*bb**(kk-1)     # ok
        beta2 = (Crth - kk*Crr)*bb**(-kk-1)     # ok
        beta3 = (Crth+Crr)*A1+Crz     # ok

    if np.round(kk,5) == 1.0:
        f1 = Crz*(bb**2-aa**2)
        f2 = 0.0
        f3 = Czz/2*(bb**2-aa**2)

        Fa = -(2*Crr+Crth)*A2*aa - (3*Crr+Crth)*A3*aa**2
        Fb = -(2*Crr+Crth)*A2*bb - (3*Crr+Crth)*A3*bb**2
        Ft = -Crz*A2*(bb**3-aa**3) - Crz*A3*(bb**4-aa**4)
    else:
        f1 = (Cthz +kk*Crz)/(kk+1)*(bb**(kk+1)-aa**(kk+1))     # ok
        f2 = (Cthz -kk*Crz)/(-kk+1)*(bb**(-kk+1)-aa**(-kk+1))     # ok
        f3 = ((Cthz+Crz)*A1+Czz)/2*(bb**2-aa**2) # careful here if k = 3 there is a problem beacuse A1 should be intergated.
        Fa = -(Crth+2*Crr)*A2*aa - (Crth + 3*Crr)*A3*aa**2     # ok
        Fb = -(Crth+2*Crr)*A2*bb - (Crth + 3*Crr)*A3*bb**2     # ok
        Ft = -((Cthz+2*Crz)*A2)/3*(bb**3-aa**3) - (Cthz+3*Crz)*A3/4*(bb**4-aa**4)      # ok
         
    return [[alpha1, alpha2, alpha3],
            [beta1, beta2, beta3],
            [f1, f2, f3],
            [Fa, Fb, Ft],
            [A1, A2, A3],
            [aa, bb, kk]]


def assemble_problem(layers, Fz_2pi = 0.0, pint=0.0, pext=0.0):
    """
    This function assembles the problem of the Generalized Plane Strain analysis of
    an orthotropic solenoid.

    Parameters
    ----------
    layers : TYPElist
        Classical Python-type list. Its element must be Dictionnaries, who must contain
        the following keys
        'a' : internal radius of the layer;
        'b' : External "   "
        'J' : current density in SI units (A m^-2)
        'B0' : Intercept of B at r=0 from linearization B(r) = B0 - C0*r
        'C0' : Slope of B according to B(r) = B0 - C0*r
        'C' : Elasticity constant's matrix
        'divi': only for plotting purposes. It's the number of subdivisions for line graphs.
        
    Fz_2pi : Float, optional
        Axial force divided by 2 pi. The default is 0.0.
    pint : Float, optional
        Value of the internal pressure at layer 0. The default is 0.0.
    pext : Float, optional
        Value of external pressure of last layer. The default is 0.0.

    Returns
    -------
    matrix : numpy 2D matrix,
        contains the coeffiecients as descibed in Markiewiczl
    values : numpy 1D array
        Vector of known values (essentially boundary conditions).

    """
    n_layers = np.shape(layers)[0]
    # dimensionality of the problem
    dimension = int(2*(n_layers)+1)
    # initialization problem matrix
    matrix = np.zeros([dimension, dimension])
    values = np.zeros([dimension,1])

    if n_layers == 1:
        object = layers[0]
        [alpha, beta, f, F, A, p] = coefficients_energization(object)
        [aa, bb, kk] = p

        matrix[0,0] = alpha[0]
        matrix[0,1] = alpha[1]
        matrix[0,2] = alpha[2]

        matrix[1,0] = beta[0]
        matrix[1,1] = beta[1]
        matrix[1,2] = beta[2]

        matrix[2,0] = f[0]
        matrix[2,1] = f[1]
        matrix[2,2] = f[2]

        values[0,0] = -pint + F[0]
        values[1,0] = -pext + F[1]
        values[2,0] = Fz_2pi + F[2]

        
        
    else:
        
        for ii in range(n_layers):
            object = layers[ii]
        
            [alpha, beta, f, F, A, p] = coefficients_energization(object)
            [aa, bb, kk] = p
        
            if ii == 0:
                # applying the first BC
                matrix[0,0] = alpha[0]   # alpha_11
                matrix[0,1] = alpha[1]   # alpha_12
                matrix[0,-1] = alpha[2]  # alpha_13
        
                values[0,0] = F[0] -pint      # Fa1 with -Pint
            
            elif ii == int(n_layers-1):
                # applying the last bc
                matrix[-2,-3] = beta[0]  # beta_n1
                matrix[-2,-2] = beta[1]  # beta_n2
                matrix[-2,-1] = beta[2]  # beta_n3
        
                # Insert PEXT. The addition of Fb is done just below as for a normal layer
                values[-2,0] = values[-2,0] - pext
            
            index = 2*ii
            
            # this fills last row at correct columns
            matrix[-1,index] = f[0] 
            matrix[-1,index+1] = f[1]
            # this is at very last column
            matrix[-1,-1] = matrix[-1,-1] + f[2]
            # this adds the Fb terms
            values[index+1,0] = values[index+1,0] + F[1]
            values[-1,0] = values[-1,0] + F[2]  # this fills values at last row
            
            if ii != int(n_layers-1):
                
                matrix[index+1, index] = beta[0]
                matrix[index+1, index+1] = beta[1]
                matrix[index+1, -1] = beta[2]
            
                matrix[index+2, index] = bb**kk
                matrix[index+2, index+1] = bb**(-kk)
                matrix[index+2, -1] = A[0]*bb
            
                values[index+2,0] = values[index+2,0] -A[1]*bb**2 - A[2]*bb**3
            
            if ii != 0:
        
                matrix[index-1, index] = -alpha[0]
                matrix[index-1, index+1] = -alpha[1]
                matrix[index-1, -1] = matrix[index-1, -1] -alpha[2]
        
                matrix[index, index] = - aa**kk
                matrix[index, index+1] = - aa**(-kk)
                matrix[index, -1] = matrix[index, -1] - A[0]*aa    
        
                values[index-1,0] = values[index-1,0] - F[0]
                values[index,0] = values[index,0] + A[1]*aa**2+A[2]*aa**3
        
        values[-1,0] = values[-1,0] + Fz_2pi

    # print('matrix assembled')
    
    return matrix, values

def plot_results(layers, solution, figdpi = 150, figsiz = [6,3], disp = 1, stress = 1, strain = 1):
    """
    Plots the results of a Generalized Plane Strain Analysis of a solenoid, according
    to the framework of this module.
    

    Parameters
    ----------
    layers : list of dictionaries.
        Contains the list of layers described as a dictionary.
        Each dictionary must contain:
            
    solution : TYPE
        DESCRIPTION.
    figdpi : TYPE, optional
        DESCRIPTION. The default is 150.
    figsiz : TYPE, optional
        DESCRIPTION. The default is [6,3].
    disp : TYPE, optional
        DESCRIPTION. The default is 1.
    stress : TYPE, optional
        DESCRIPTION. The default is 1.
    strain : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    list
        DESCRIPTION.
    list
        DESCRIPTION.

    """
    
    n_layers = np.shape(layers)[0]

    epsz = solution[-1]
    fig1, ax1 = plt.subplots(figsize = figsiz, dpi = figdpi)
    fig2, ax2 = plt.subplots(figsize = figsiz, dpi = figdpi)
    fig3, ax3 = plt.subplots(figsize = figsiz, dpi = figdpi)

    for ii in range(n_layers):
        index = 2*ii
        D1 = solution[index,0]
        D2 = solution[index+1,0]
        
        object = layers[ii]
        
        [alpha, beta, f, F, A, p] = coefficients_energization(object)
        [aa, bb, kk] = p

        try:
            rr = np.linspace(aa, bb, object['divi'])
        except:
            rr = np.linspace(aa, bb, 5)
                       
        u = D1*rr**kk + D2*rr**(-kk) + A[0]*epsz*rr + A[1]*rr**2 + A[2]*rr**3
        er = p[2]*D1*rr**(p[2]-1) - p[2]*D2*rr**(-p[2]-1) + A[0]*epsz + 2*A[1]*rr + 3*A[2]*rr**2
        eth = D1*rr**(p[2]-1) + D2*rr**(-p[2]-1) + A[0]*epsz + A[1]*rr + A[2]*rr**2
        
        sr = object['C'][0,0]*er + object['C'][0,1]*eth + object['C'][0,2]*epsz 
        sth = object['C'][1,0]*er + object['C'][1,1]*eth + object['C'][1,2]*epsz 
        sz = object['C'][2,0]*er + object['C'][2,1]*eth + object['C'][2,2]*epsz 
        
        if ii ==0:
            ax1.plot(rr,u*1e3,'-ok', markersize = 4, markerfacecolor = 'w', label = r'$u_r$')
            ax2.plot(rr,sr*1e-6,'-^k', markersize = 4, markerfacecolor = 'w', label = r'$\sigma_r$')
            ax2.plot(rr,sth*1e-6,'-sk', markersize = 4, markerfacecolor = 'w', label = r'$\sigma_\theta$')
            ax2.plot(rr,sz*1e-6, '-ok', markersize = 4, markerfacecolor = 'w', label = r'$\sigma_z$')
            ax3.plot(rr,er*1e3,'-^k', markersize = 4, markerfacecolor = 'w', label = r'$\varepsilon_r$')
            ax3.plot(rr,eth*1e3,'-sk', markersize = 4, markerfacecolor = 'w', label = r'$\varepsilon_\theta$')
            ax3.plot(rr,epsz*np.ones_like(rr)*1e3, '-ok', markersize = 4, markerfacecolor = 'w', label = r'$\varepsilon_z$')
        else:
            ax1.plot(rr,u*1e3,'-og', markersize = 4, markerfacecolor = 'w')
            ax2.plot(rr,sr*1e-6,'-^g', markersize = 4, markerfacecolor = 'w')
            ax2.plot(rr,sth*1e-6,'-sg', markersize = 4, markerfacecolor = 'w')
            ax2.plot(rr,sz*1e-6, '-og', markersize = 4, markerfacecolor = 'w')
            ax3.plot(rr,er*1e3,'-^g', markersize = 4, markerfacecolor = 'w')
            ax3.plot(rr,eth*1e3,'-sg', markersize = 4, markerfacecolor = 'w')
            ax3.plot(rr,epsz*np.ones_like(rr)*1e3, '-og', markersize = 4, markerfacecolor = 'w')
    
    
    ax1.set_title('Displacement')
    ax1.set_xlabel('r / m')
    ax1.set_ylabel(r'u$_r$ / mm')
    ax1.grid(visible=True, which='minor', axis='both', alpha = 0.1)
    ax1.grid(visible=True, which='major', axis='both', alpha = 0.5)
    ax1.minorticks_on()
    box1 = ax1.get_position()
    ax1.set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])
    ax1.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax2.set_title('Stresses')
    ax2.set_xlabel('r / m')
    ax2.set_ylabel(r'$\sigma$ / MPa')
    ax2.grid(visible=True, which='minor', axis='both', alpha = 0.1)
    ax2.grid(visible=True, which='major', axis='both', alpha = 0.5)
    ax2.minorticks_on()
    box2 = ax2.get_position()
    ax2.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])
    ax2.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax3.set_title('Strains')
    ax3.set_xlabel('r / m')
    ax3.set_ylabel(r'$\varepsilon$ / mm m$^{-1}$')
    ax3.grid(visible=True, which='minor', axis='both', alpha = 0.1)
    ax3.grid(visible=True, which='major', axis='both', alpha = 0.5)
    ax3.minorticks_on()
    box3 = ax3.get_position()
    ax3.set_position([box3.x0, box3.y0, box3.width * 0.8, box3.height])
    ax3.legend(fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
    
    return [fig1, fig2, fig3], [ax1, ax2, ax3]


def plot_stresses(axes, layers, solution, param = None):
    
    n_layers = np.shape(layers)[0]

    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    
    epsz = solution[-1]
    
    for ii in range(n_layers):
        index = 2*ii
        D1 = solution[index,0]
        D2 = solution[index+1,0]
        
        object = layers[ii]
        
        [alpha, beta, f, F, A, p] = coefficients_energization(object)
        [aa, bb, kk] = p

        try:
            rr = np.linspace(aa, bb, object['divi'])
        except:
            rr = np.linspace(aa, bb, 5)
                       
        er = p[2]*D1*rr**(p[2]-1) - p[2]*D2*rr**(-p[2]-1) + A[0]*epsz + 2*A[1]*rr + 3*A[2]*rr**2
        eth = D1*rr**(p[2]-1) + D2*rr**(-p[2]-1) + A[0]*epsz + A[1]*rr + A[2]*rr**2
        
        sr = object['C'][0,0]*er + object['C'][0,1]*eth + object['C'][0,2]*epsz 
        sth = object['C'][1,0]*er + object['C'][1,1]*eth + object['C'][1,2]*epsz 
        sz = object['C'][2,0]*er + object['C'][2,1]*eth + object['C'][2,2]*epsz 

        if ii ==0:
            ax1.plot(rr,sr*1e-6,'-^', markersize = 4, markerfacecolor = 'w', label = param)
            ax2.plot(rr,sth*1e-6,'-s', markersize = 4, markerfacecolor = 'w', label = param)
            ax3.plot(rr,sz*1e-6, '-o', markersize = 4, markerfacecolor = 'w', label = param)
        else:
            ax1.plot(rr,sr*1e-6,'-^', markersize = 4, markerfacecolor = 'w')
            ax2.plot(rr,sth*1e-6,'-s', markersize = 4, markerfacecolor = 'w')
            ax3.plot(rr,sz*1e-6, '-o', markersize = 4, markerfacecolor = 'w')

    return

def plot_sr(axis, layers, solution, param = None, col = None):
    """ NOT finalized yet added color """
    n_layers = np.shape(layers)[0]

    epsz = solution[-1]
    
    for ii in range(n_layers):
        index = 2*ii
        D1 = solution[index,0]
        D2 = solution[index+1,0]
        
        object = layers[ii]
        
        [alpha, beta, f, F, A, p] = coefficients_energization(object)
        [aa, bb, kk] = p

        try:
            rr = np.linspace(aa, bb, object['divi'])
        except:
            rr = np.linspace(aa, bb, 5)
                       
        er = p[2]*D1*rr**(p[2]-1) - p[2]*D2*rr**(-p[2]-1) + A[0]*epsz + 2*A[1]*rr + 3*A[2]*rr**2
        eth = D1*rr**(p[2]-1) + D2*rr**(-p[2]-1) + A[0]*epsz + A[1]*rr + A[2]*rr**2
        
        sr = object['C'][0,0]*er + object['C'][0,1]*eth + object['C'][0,2]*epsz 
        
        if ii ==0:
            if col:
                axis.plot(rr,sr*1e-6,'-^', color = col, markersize = 4, markerfacecolor = 'w', markeredgecolor = col, label = param)
            else:
                axis.plot(rr,sr*1e-6,'-^', markersize = 4, markerfacecolor = 'w', label = param)
        else:
            if col:
                axis.plot(rr,sr*1e-6,'-^', color = col, markersize = 4, markerfacecolor = 'w', markeredgecolor = col)
            else:
                axis.plot(rr,sr*1e-6,'-^', markersize = 4, markerfacecolor = 'w')

    return

def plot_sth(axis, layers, solution, param = None, col = None):
    """ NOT finalized yet added color """
    n_layers = np.shape(layers)[0]

    epsz = solution[-1]
    
    for ii in range(n_layers):
        index = 2*ii
        D1 = solution[index,0]
        D2 = solution[index+1,0]
        
        object = layers[ii]
        
        [alpha, beta, f, F, A, p] = coefficients_energization(object)
        [aa, bb, kk] = p

        try:
            rr = np.linspace(aa, bb, object['divi'])
        except:
            rr = np.linspace(aa, bb, 5)
                       
        er = p[2]*D1*rr**(p[2]-1) - p[2]*D2*rr**(-p[2]-1) + A[0]*epsz + 2*A[1]*rr + 3*A[2]*rr**2
        eth = D1*rr**(p[2]-1) + D2*rr**(-p[2]-1) + A[0]*epsz + A[1]*rr + A[2]*rr**2
        
        sth = object['C'][1,0]*er + object['C'][1,1]*eth + object['C'][1,2]*epsz 
        
        if ii ==0:
            if col:
                axis.plot(rr,sth*1e-6,'-^', color = col, markersize = 4, markerfacecolor = 'w', markeredgecolor = col, label = param)
            else:
                axis.plot(rr,sth*1e-6,'-^', markersize = 4, markerfacecolor = 'w', label = param)
        else:
            if col:
                axis.plot(rr,sth*1e-6,'-^', color = col, markersize = 4, markerfacecolor = 'w', markeredgecolor = col)
            else:
                axis.plot(rr,sth*1e-6,'-^', markersize = 4, markerfacecolor = 'w')
    return

# def stresses_strains(solution, layers):
#     """
#     made 15.02.2024
#     Takes in the solution (e.g. {D11, D12, D21, D22, D31, D32, Eps_z}) and the sequence of layers,
#     defined as Dictionaries, and restitutes a dictionary containing, for each layer index, a dictionary of arrays
#     containing Sr, Sth, Sz, Er, Eth, Ez
#     """
#     n_layers = np.shape(layers)[0]
#     epsz = solution[-1]
#     SS = {}
#     for ii in range(n_layers):
#         index = 2*ii
#         D1 = solution[index,0]
#         D2 = solution[index+1,0]
        
#         obj = layers[ii]
        
#         [alpha, beta, f, F, A, p] = coefficients_energization(obj)
#         [aa, bb, kk] = p

#         try:
#             rr = np.linspace(aa, bb, obj['divi'])
#         except:
#             rr = np.linspace(aa, bb, 5)
                       
#         er = p[2]*D1*rr**(p[2]-1) - p[2]*D2*rr**(-p[2]-1) + A[0]*epsz + 2*A[1]*rr + 3*A[2]*rr**2
#         eth = D1*rr**(p[2]-1) + D2*rr**(-p[2]-1) + A[0]*epsz + A[1]*rr + A[2]*rr**2
        
#         sr = obj['C'][0,0]*er + obj['C'][0,1]*eth + obj['C'][0,2]*epsz 
#         sth = obj['C'][1,0]*er + obj['C'][1,1]*eth + obj['C'][1,2]*epsz 
#         sz = obj['C'][2,0]*er + obj['C'][2,1]*eth + obj['C'][2,2]*epsz 

#         di = {ii: {'r': rr, 'sr': sr, 'sth': sth, 'sz': sz, 'er': er, 'eth': eth, 'ez': epsz*np.ones_like(rr)}}
#         SS.update(di)
        
#     return SS        

def return_results(layers, solution):
    """
    made 16.02.2024
    Return the global results of a Generalized Plane Strain Analysis of a solenoid, according
    to the framework of this module.
    

    Parameters
    ----------
    layers : list of dictionaries.
        Contains the list of layers described as a dictionary.
        Each dictionary must contain:
            
    solution : TYPE
        DESCRIPTION.
    
    Returns
    -------
    SOLU: Dictionary,
        Contains the global results per layer of an electromechanical analysis.
        It has the following structure:
            - SOLU.keys(): are the layer identification number, '0', '1' and so on
            - SOLU.vals(): are other dictionaries containing:
                    - 'r': radial coordinate
                    - 'sr': ....
                    - 'sth', 'sz', u, 'er'

    """
    
    n_layers = np.shape(layers)[0]
    epsz = solution[-1]
    
    SOLU = {}
    for ii in range(n_layers):
        
        index = 2*ii
        D1 = solution[index,0]
        D2 = solution[index+1,0]
        
        obj = layers[ii]
        
        [alpha, beta, f, F, A, p] = coefficients_energization(obj)
        [aa, bb, kk] = p

        try:
            rr = np.linspace(aa, bb, obj['divi'])
        except:
            rr = np.linspace(aa, bb, 5)
                       
        u = D1*rr**kk + D2*rr**(-kk) + A[0]*epsz*rr + A[1]*rr**2 + A[2]*rr**3
        er = p[2]*D1*rr**(p[2]-1) - p[2]*D2*rr**(-p[2]-1) + A[0]*epsz + 2*A[1]*rr + 3*A[2]*rr**2
        eth = D1*rr**(p[2]-1) + D2*rr**(-p[2]-1) + A[0]*epsz + A[1]*rr + A[2]*rr**2
        
        sr = obj['C'][0,0]*er + obj['C'][0,1]*eth + obj['C'][0,2]*epsz 
        sth = obj['C'][1,0]*er + obj['C'][1,1]*eth + obj['C'][1,2]*epsz 
        sz = obj['C'][2,0]*er + obj['C'][2,1]*eth + obj['C'][2,2]*epsz 
        
        PART = {ii: {'r': rr, 
                     'u': u,
                     'er': er,
                     'eth': eth,
                     'ez': np.ones_like(eth)*epsz,
                     'sr': sr,
                     'sth': sth,
                     'sz': sz}}
        SOLU.update(PART)
        
    return SOLU