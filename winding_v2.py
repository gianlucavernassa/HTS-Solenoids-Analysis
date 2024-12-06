# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 07:57:55 2024

@author: gvernass
"""
import numpy as np
import gps_v3 as gaoc


def create_assembly(mandrel, tape, Nturns, tension):
    layers = [mandrel]
    hh = tape['height']
    tt = tape['thick']
    Ct = tape['C']
    divit = tape['divi']
    
    for i in range(Nturns):
        aa = layers[-1]['b']
        bb = aa + tt
        TT = tension[i]
        
        _newdict = {'a': aa, 'b' : bb, 
                    'height': hh, 'C': Ct, 
                    'tension': TT, 
                    'J':0.0, 'B0':0.0, 'C0': 0.0, 
                    'divi' : divit}
        layers.append(_newdict)
        return layers


def wind(layers):
    """
    

    Parameters
    ----------
    layers : TYPE
        DESCRIPTION.

    Returns
    -------
    solutions : TYPE
        DESCRIPTION.
    additional_epsz : TYPE
        DESCRIPTION.

    """
    
    N = np.shape(layers)[0] 
    solutions = np.zeros([2*N+1, N-1])
    additional_epsz = np.zeros([1, N-1])
    
    solutions[:] = np.nan
    additional_epsz[:] = np.nan
    
    # winding up to N-1 layers because first one is mandrel.
    for i in range(N-1):
        # if you want to include nonlinearities, here you can update the values of inner and outer radii for each object !
        # but be careful with references.
        
        b_w = layers[i]['b']
                           
        T_pl = layers[i+1]['tension'] / layers[i+1]['height']
            
        pres = T_pl/b_w
        
        # solution wound
        if i ==0:
            matrix_wound, values_wound = gaoc.assemble_problem(layers[:i+1], pext = pres)        
        else:
            matrix_wound, values_wound = gaoc.assemble_problem(layers[:i+1], pext = pres)        
        sol_wound = np.linalg.solve(matrix_wound,values_wound)
        # solution single layer
        matrix_last, values_last = gaoc.assemble_problem([layers[i+1]], pint = pres)        
        sol_last = np.linalg.solve(matrix_last,values_last)
        
        solutions[:2*(i+1), i] = sol_wound[:-1,0]
        solutions[2*(i+1):2*(i+2), i] = sol_last[:-1,0]
        
        solutions[-1, i] = sol_wound[-1,0]
        additional_epsz[0, i] = sol_last[-1,0]
        
    return solutions, additional_epsz

def cumulate_effects(layers,solutions, additional_epsz, ax1= None, ax2=None, col = None, param=None):
    """
    

    Parameters
    ----------
    layers : TYPE
        DESCRIPTION.
    solutions : TYPE
        DESCRIPTION.
    additional_epsz : TYPE
        DESCRIPTION.
    ax1 : TYPE, optional
        DESCRIPTION. The default is None.
    ax2 : TYPE, optional
        DESCRIPTION. The default is None.
    col : TYPE, optional
        DESCRIPTION. The default is None.
    param : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    """
    
    # time of the winding process can be chosen by extracting columns up to a certain value
    Nlayers = np.shape(layers)[0]
    
    summed_D = np.nansum(solutions, axis = 1)[None].T
    # print(summed_D)
    
    for ii in range(Nlayers):
        # for each layer it extracts the good D and 
        index = 2*ii
        D1 = summed_D[index,0]
        D2 = summed_D[index+1,0]
        # The epsz is a bit trickier because it shall not account for the epsz coming from the time
        # the layer was added (ii). therefore:
        epsz_tot = np.nansum(solutions[-1,ii:])
        
        # then, on top of this, we add the specific one coming from the layer winding.
        if ii == 0:
            epsz_ii = epsz_tot
        if ii!=0 & ii != Nlayers-1:
            # if it's not the last layer, it sums the global epsz to that generated during winding.
            epsz_ii = epsz_tot + additional_epsz[0,ii-1]
        elif ii == Nlayers-1:
            # Then if it's the last layer it only takes into account the last epsz
            epsz_ii = additional_epsz[ii-1]
        
        obj = layers[ii]
        [alpha, beta, f, F, A, p] = gaoc.coefficients_energization(obj)
        [aa, bb, kk] = p
        
        try:
            rr = np.linspace(aa, bb, obj['divi'])
        except:
            rr = np.linspace(aa, bb, 5)
        
        er = p[2]*D1*rr**(p[2]-1) - p[2]*D2*rr**(-p[2]-1) + A[0]*epsz_ii + 2*A[1]*rr + 3*A[2]*rr**2
        eth = D1*rr**(p[2]-1) + D2*rr**(-p[2]-1) + A[0]*epsz_ii + A[1]*rr + A[2]*rr**2
        
        sr = obj['C'][0,0]*er + obj['C'][0,1]*eth + obj['C'][0,2]*epsz_ii 
        sth = obj['C'][1,0]*er + obj['C'][1,1]*eth + obj['C'][1,2]*epsz_ii 
        
        _newd = {ii:{'r': rr, 'sr': sr, 'sth': sth}}
        if ii ==0:
            data = _newd
        else:
            data.update(_newd)
                
        if ax1:
            if ii ==0:
                if col:
                    ax1.plot(rr*1e3,sr*1e-6,'-', color = col, label = param)
                    ax2.plot(rr*1e3,sth*1e-6,'-', color = col, label = param)
                else:
                    ax1.plot(rr*1e3,sr*1e-6,'-',  label = param)
                    ax2.plot(rr*1e3,sth*1e-6,'-',  label = param)
            else:
                if col:
                    ax1.plot(rr*1e3,sr*1e-6,'-', color = col)
                    ax2.plot(rr*1e3,sth*1e-6,'-', color = col)
                else:
                    ax1.plot(rr*1e3,sr*1e-6,'-')
                    ax2.plot(rr*1e3,sth*1e-6,'-')
                    
    return data
