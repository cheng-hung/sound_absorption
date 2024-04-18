import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


### Fundmental definitions
def shear_m(Young_m, Poisson_r, loss_factor):
    return Young_m*(1+1j*loss_factor)/(2*(1+Poisson_r))


def lame_const(Young_m, Poisson_r, loss_factor):
    return Young_m*(1+1j*loss_factor)*Poisson_r/((1+Poisson_r)*(1-2*Poisson_r))


def longitudinal_m(Yong_m, Poisson_r, loss_factor):
    return Yong_m*(1-Poisson_r)*(1+1j*loss_factor)/((1+Poisson_r)*(1-2*Poisson_r))
    
## eq.(1-13b), in book page 26
def ith_longitudinal_speed(longitudinal_m, density):
    speed = np.sqrt(longitudinal_m/density)
    # damping = np.sqrt(1+1j*loss_factor)
    return speed    

## eq.(1-14), in book page 27
def ith_transverse_speed(shear_m, density):
    speed = np.sqrt(shear_m/density)
    # damping = np.sqrt(1+1j*loss_factor)
    return speed


## Effective radius of cone & horn shape: (25-27) in paper
def effective_radius(p, q, lh, num_segments, shape='cone'):
    lh_n = np.linspace(0, lh, num_segments)
    
    if shape == 'cone':
        # print(f'The input shape is {shape}.')
        pcone, qcone = p, q
        alpha = (qcone-pcone)/lh
        beta = pcone
        r_effective = alpha*lh_n + beta
        
    elif shape == 'horn':
        # print(f'The input shape is {shape}.')
        phorn, qhorn = p, q
        gamma = phorn
        delta = (1/lh)*np.log(qhorn/phorn)
        r_effective = gamma*np.exp(delta*lh_n)
    
    return r_effective, lh_n



### Reference: Influence of hole shape on sound absorption of underwater anechoic layers
#### https://www.sciencedirect.com/science/article/abs/pii/S0022460X1830227X

### Start at eq(15) in the paper
## The effective density: (15)
## ai represents the inner radii of the pipe in the i th layer
def ith_effect_density(ai, cell_radius, rubber_den, air_den):
    return rubber_den*(1-(ai/cell_radius)**2) + air_den*(ai/cell_radius)**2



## The effective impedance of the i th segment: (16)
def ith_effect_impedance(ith_effect_density, omega, wave_number):
    return ith_effect_density*omega/wave_number


## The transfer matrix of the i th segment: (17)
def ith_tran_matrix(wave_number, li, ith_impedance):
    ti = np.zeros((2,2), dtype=complex)
    ti[0,0] = np.cos(wave_number*li)
    ti[0,1] = -1j*np.sin(wave_number*li)*ith_impedance
    ti[1,0] = -1j*np.sin(wave_number*li)/ith_impedance
    ti[1,1] = np.cos(wave_number*li)
    return ti

## The successive multi-plication of the transfer matrix of each segment: (18)
def total_tran_matrix(wave_number, lh, n, impedance):
    li = lh/n
    
    try:
        t0 = ith_tran_matrix(wave_number[0], li, impedance[0])
        tn=t0
        for i in range(1, n):
            tn = np.matmul(tn, ith_tran_matrix(wave_number[i], li, impedance[i]))
        
    except IndexError:
        t0 = ith_tran_matrix(wave_number, li, impedance[0])
        tn=t0
        for i in range(1, n):
            tn = np.matmul(tn, ith_tran_matrix(wave_number, li, impedance[i]))        
    
    return tn
    

## The impedance of the front interface Zf: (22)
def imped_front(tn):
    # zf = np.absolute(tn[0,0]/tn[1,0])
    zf = (tn[0,0]/tn[1,0])
    return zf


## The reflection coefficient of the anechoic layer: (23)
def reflection_coefficient(zf,zw):
    return (zf-zw)/(zf+zw)


## The sound absorption coefficient: (24)
def absorption_coefficient(reflection):
    return 1-np.absolute(reflection)**2