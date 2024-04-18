#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ## Documents of SymPy for solving equations
# 
# #### https://docs.sympy.org/latest/modules/solvers/solvers.html
import sympy
from sympy.solvers import solve
from sympy import Symbol
import time

x = Symbol('x')
ans = solve(x**3 + x + 10, x, numerical=True)



# ## Bessel functions from SciPy
# 
# 
# ##### 1. Bessel function of the first kind of real order and complex argument.
# 
# ##### https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jv.html#scipy.special.jv
# 
# 
# ##### 2. Bessel function of the second kind of real order and complex argument.
# 
# ##### https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.yv.html#scipy.special.yv
import scipy
from scipy.special import jv
from scipy.special import yv

# fig, ax = plt.subplots(figsize = (6, 4), constrained_layout=True)
# x = np.linspace(0., 10., 1000)
# for i in range(4):
#     ax.plot(x, jv(i, x), label=f'$Y_{i!r}$')
# # ax.set_ylim(-3, 1)
# ax.legend()
# plt.show()


# ## Define All twelve elements in the coefficient matrix of Eq. (11) in the paper 
# 
# #### Bessel functions from SymPy
# ##### https://docs.sympy.org/latest/modules/functions/special.html
from sympy import besselj, bessely
from sympy import I
from sympy.abc import a, b, z

mu = Symbol('mu')
lambda_ = Symbol('lambda')
kz_ = Symbol('kz')
omega_ = Symbol('omega')
cl_ = Symbol('cl')
ct_ = Symbol('ct')
kl_ = omega_/cl_
kt_ = omega_/ct_
klr = sympy.sqrt(kl_**2 - kz_**2)
ktr = sympy.sqrt(kt_**2 - kz_**2)

J0klra= besselj(0, klr*a)
J0ktra= besselj(0, ktr*a)
J1klra= besselj(1, klr*a)
J1ktra= besselj(1, ktr*a)
J1klrb= besselj(1, klr*b)
J1ktrb= besselj(1, ktr*b)

Y0klra= bessely(0, klr*a)
Y0ktra= bessely(0, ktr*a)
Y1klra= bessely(1, klr*a)
Y1ktra= bessely(1, ktr*a)
Y1klrb= bessely(1, klr*b)
Y1ktrb= bessely(1, ktr*b)

M11 = -klr*J1klrb
M12 = -klr*Y1klrb
M13 = -kz_*ktr*J1ktrb*I
M14 = -kz_*ktr*Y1ktrb*I

M21 = -(lambda_*(klr**2+kz_**2)+2*mu*klr**2)*J0klra + 2*mu*klr*J1klra/a
M22 = -(lambda_*(klr**2+kz_**2)+2*mu*klr**2)*Y0klra + 2*mu*klr*Y1klra/a
M23 = 2*I*mu*ktr*kz_*(-ktr*J0ktra+J1ktra/a)
M24 = 2*I*mu*ktr*kz_*(-ktr*Y0ktra+Y1ktra/a)

M31 = -2*I*mu*klr*kz_*J1klra
M32 = -2*I*mu*klr*kz_*Y1klra
M33 = mu*ktr*(kz_**2-ktr**2)*J1ktra
M34 = mu*ktr*(kz_**2-ktr**2)*Y1ktra

M41 = -2*I*mu*klr*kz_*J1klrb
M42 = -2*I*mu*klr*kz_*Y1klrb
M43 = mu*ktr*(kz_**2-ktr**2)*J1ktrb
M44 = mu*ktr*(kz_**2-ktr**2)*Y1ktrb

from sympy import Matrix, lambdify
mat = Matrix([[M11, M12, M13, M14], [M21, M22, M23, M24], [M31, M32, M33, M34], [M41, M42, M43, M44]]) 
determinant_00 = mat.det()
determinant_01 = lambdify((kz_, a, b, mu, lambda_, omega_, cl_, ct_), determinant_00)


## Given an arbitrary angular frequency and mechanical properties
frequency = 1500 #np.arange(0.1, 10000, 50)
omega = frequency * 2 * np.pi
Young_molulus = 0.14*(10**9)
Poisson_ratio = 0.49
loss_factor = 0.23
rubber_den=1100
air_den=1.21


# ### Import defined equations for calculation
get_ipython().run_line_magic('run', '-i equations.py')


## Assume cone shape
pcone = 4*0.001
qcone = 4*0.001
lh = 40*0.001
num_segments = 100
cell_radius = 15*0.001 

ai, _ = effective_radius(pcone, qcone, lh, num_segments, shape='cone')
effective_density = ith_effect_density(ai, cell_radius, rubber_den, air_den)


## Calculate shear_modulus & longitudinal_modulus
shear_modulus = shear_m(Young_molulus, Poisson_ratio, loss_factor)
lame_constant = lame_const(Young_molulus, Poisson_ratio, loss_factor)
longitudinal_modulus = longitudinal_m(Young_molulus, Poisson_ratio, loss_factor)
shear_modulus


# ### Defined necessary variables in eq.(f6) for solving eq.(f5) in the reference book Page 70-71 
## Calculate speed of longitudinal & speed of transverse
cl = ith_longitudinal_speed(longitudinal_modulus, effective_density)
ct = ith_transverse_speed(shear_modulus, effective_density)


## kl: longitudinal wavenumber = omega / speed of longitudinal
## kt: transverse wavenumber = omega / speed of transverse
kl = omega/cl
kt = omega/ct



# ## Solve the determinant equation of determinant numerically in Scipy...
from scipy.optimize import newton, fsolve
# root = newton(coefficient_matrix, kl*6, args=(omega, cl, ct, ai[0], cell_radius), tol=80, maxiter=10000)
# kz_root = newton(determinant_01, kl[-1], args=(ai[-1], cell_radius, shear_modulus, lame_constant, omega, cl[-1], ct[-1]), tol=1.48e-12, maxiter=10000)


from cxroots import Circle
root_range = Circle(0, 100)
determinant_02 = lambda kz: determinant_01(kz, ai[-1], cell_radius, shear_modulus, lame_constant, omega, cl[-1], ct[-1])
# roots = root_range.roots(determinant_02)


# C = Circle(0, 3)
# f = lambda z: (z + 1.2) ** 3 * (z - 2.5) ** 2 * (z + 1j)
# C.roots(f, verbose=True)


def axial_wavenumber(ai, cell_radius, shear_modulus, lame_constant, omega, cl, ct):
    kz = []
    failed_kz = []
    kl = omega/cl
    x0 = kl[0]
    for i in range(ai.shape[0]):
        # x0 = kl[i]
        try:
            kz_root = newton(determinant_01, x0, 
                             args=(ai[i], cell_radius, shear_modulus, lame_constant, omega, cl[i], ct[i]), 
                             tol=1.48e-5, maxiter=100)
            kz.append(kz_root)
            x0 = kz_root

        except RuntimeError:
            try:
                time.sleep(1)
                print(f'First try of finding root at frequency = {omega/(2*np.pi):.2f}, {i = } / {ai.shape[0]} failed. Second Try...')
                
                try:
                    x0 = kz_root
                except UnboundLocalError:
                    x0 = 10 + 10j
                    
                kz_root = newton(determinant_01, x0, 
                                args=(ai[i], cell_radius, shear_modulus, lame_constant, omega, cl[i], ct[i]), 
                                tol=1.48e-5, maxiter=1000)
                kz.append(kz_root) 
        
            except RuntimeError:
                print(f'Second try of finding root at {omega = }, {i = } / {ai.shape[0]} failed. Assume root is 0+0j...')
                kz.append(0+0j)
                failed_kz.append([omega, i, ai.shape[0]])
                # break
        
        # time.sleep(1)

    return np.asarray(kz), failed_kz




# frequency = 500
# kz_list, _ = axial_wavenumber(ai, cell_radius, shear_modulus, lame_constant, omega, cl, ct)

frequency_array = np.arange(200, 10001, 200)
# frequency_array = frequency_array[1:]
frequency_array.shape


wavenumer_array = np.zeros((frequency_array.shape[0], num_segments), dtype=complex)
failed_roots = []
i = 0
for frequency in frequency_array:
    omega = frequency * 2 * np.pi
    kz, failed_kz = axial_wavenumber(ai, cell_radius, shear_modulus, lame_constant, omega, cl, ct)
    wavenumer_array[i][:] = kz
    failed_roots.append(failed_kz)
    i += 1
    print(f'Solving determinant at {frequency = } is done.')



def save_wavenumber_csv(frequency_array, wavenumer_array, filename, filepath, 
                        p, q, lh, num_segments, 
                        Young_m, Poisson_r, loss_factor, 
                        cell_radius, shape='cone', rubber_den=1100, air_den=1.21):

    df_const = pd.DataFrame()
    df_const['Variable'] = ['shape', 'p_mm', 'q_mm', 'lh_mm', 'b_mm',  'num_segments', 'Young_GPa', 'Poisson_r', 'loss_factor', 'rubber_kgm-3', 'air_kgm-3']
    df_const['Value'] = [shape, p*1000, q*1000, lh*1000, cell_radius*1000, num_segments, Young_m/(10**9), Poisson_r, loss_factor, rubber_den, air_den]

    df_wave = pd.DataFrame()
    i = 0
    for frequency in frequency_array:
        df_wave[frequency] = wavenumer_array[i][:]
        i += 1

    try:
        fn = os.path.join(filepath, filename)
        writer = pd.ExcelWriter(fn, engine='odsxwriter')   
        df_const.to_excel(writer, sheet_name='Constants')   
        df_wave.to_excel(writer, sheet_name='Wave_numbers')
        writer.close()
        print(f'Save file to {fn}')
    
    except ValueError:
        fn_const = os.path.join(filepath, filename.split('.')[0]+'_const.csv')
        df_const.to_csv(fn_const, sep=' ', index=False, header=False, float_format='{:.8e}'.format)
        fn_wave = os.path.join(filepath, filename.split('.')[0]+'_kz.csv')
        df_wave.to_csv(fn_wave, sep=' ', index=False, header=True, float_format='{:.8e}'.format)
        print(f'Save file to {fn_const}')
        print(f'Save file to {fn_wave}') 
    



def absorption_frequency(frequency_array, wavenumer_array, 
                        p, q, lh, num_segments, 
                        Young_m, Poisson_r, loss_factor, 
                        cell_radius, shape='cone', rubber_den=1100, air_den=1.21):
    
    if frequency_array.shape[0] == wavenumer_array.shape[0]:
        print(f'{frequency_array.shape[0] = } is same as {wavenumer_array.shape[0] = }')
    else:
        raise ValueError(f'{frequency_array.shape = } not same as {wavenumer_array.shape = }')

    frequency_array = np.asarray(frequency_array)
    absorption_list = []
    
    ai, _ = effective_radius(p, q, lh, num_segments, shape=shape)
    effective_density = ith_effect_density(ai, cell_radius, rubber_den, air_den)

    longitudinal_modulus = longitudinal_m(Young_m, Poisson_r, loss_factor)
    shear_modulus = shear_m(Young_molulus, Poisson_ratio, loss_factor)
    lame_constant = lame_const(Young_molulus, Poisson_ratio, loss_factor)

    # effective_speed = ith_longitudinal_speed(longitudinal_modulus, effective_density)
    cl = ith_longitudinal_speed(longitudinal_modulus, effective_density)
    ct = ith_transverse_speed(shear_modulus, effective_density)
    
    for frequency, wave_number in zip(frequency_array, wavenumer_array):
       
        omega = frequency * 2 * np.pi
        
        # wave_number = axial_wavenumber(ai, cell_radius, shear_modulus, lame_constant, omega, cl, ct)
        # print(f'Solving axial_wavenumber kz at {frequency = } is done.')
        
        effective_impedance = ith_effect_impedance(effective_density, omega, wave_number)
        tn = total_tran_matrix(wave_number, lh, num_segments, effective_impedance)

        ## Acoustic impdedance of water: 1.48 MPa.s.m−1
        ## https://www.animations.physics.unsw.edu.au/jw/sound-impedance-intensity.htm

        zw = 998*1483
        zf = imped_front(tn)
        ref = reflection_coefficient(zf, zw)
        alpha = absorption_coefficient(ref)   
        absorption_list.append(alpha)
        
    return np.asarray(absorption_list)




# frequency_array = np.arange(0.1, 10000, 500)
# try: 
#     absorption_array = absorption_frequency(frequency_array, 4*0.001, 8*0.001, 40*0.001, 10,
#                                             0.014*(10**9), 0.49, 0.23, 
#                                             15*0.001, shape='cone')
# except ValueError:
#     print(f'RuntimeError occurred and cannot converge at {frequency = }, {wave_number = }')



# rows = 1
# cols = 1
# f1, ax1 = plt.subplots(rows, cols, figsize = (6, 4), constrained_layout=True)
# ax1.plot(frequency_array, absorption_array)
# plt.show()


# from mpmath import findroot
# from mpmath.calculus.optimization import Secant

# # def F2(x):
# #     global a,b
# #     parameters=list([a,b])
# #     args=tuple(x)+tuple(parameters)
# #     return F1(*args)

# def determinant_02(kz):
#     global ai, cell_radius, shear_modulus, lame_constant, omega, cl, ct
#     parameters=list(ai[0], cell_radius, shear_modulus, lame_constant, omega, cl, ct)
#     args=tuple(kz)+tuple(parameters)
#     return determinant(*args)




