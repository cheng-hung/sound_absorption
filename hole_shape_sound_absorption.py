import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from sympy import Symbol, besselj, bessely, I, Matrix, lambdify, sqrt
from sympy.abc import a, b, z
from scipy.optimize import newton
from tqdm.auto import tqdm


'''
### Reference: Influence of hole shape on sound absorption of underwater anechoic layers
### https://www.sciencedirect.com/science/article/abs/pii/S0022460X1830227X
'''

class anechoic_layers():
    
    ## Section 3.1 and Fig. 7 in the paper
    def __init__(self, material='rubber', shape='cone', 
                 p=4e-3, q=8e-3, lh=40e-3, cell_radius=15e-3, 
                 num_segments=100, layer_density=1100, air_density=1.21):
        
        self.material = material
        self.shape = shape
        self.p_hole = p
        self.q_hole = q
        self.h_hole = lh
        self.cell_r = cell_radius
        self.segments = num_segments
        self.layer_density = layer_density
        self.air_density = air_density
        
    
    ## Eq (25-27) in the paper
    def effective_radius(self):
        lh_n = np.linspace(0, self.h_hole, self.segments)

        if self.shape == 'cone':
            # print(f'The input shape is {shape}.')
            pcone, qcone = self.p_hole, self.q_hole
            alpha = (qcone-pcone)/self.h_hole
            beta = pcone
            r_effective = alpha*lh_n + beta

        elif self.shape == 'horn':
            # print(f'The input shape is {shape}.')
            phorn, qhorn = self.p_hole, self.q_hole
            gamma = phorn
            delta = (1/self.h_hole)*np.log(qhorn/phorn)
            r_effective = gamma*np.exp(delta*lh_n)

        return r_effective, lh_n
    
    
    ## ai represents the inner radii of the pipe in the i th layer
    ## Eq (15) in the paper
    def effective_density(self):
        ai, _ = self.effective_radius()
        return self.layer_density*(1-(ai/self.cell_r)**2) + self.air_density*(ai/self.cell_r)**2
    


class elastic_module(anechoic_layers):
    def __init__(self, Young_modulus=0.14e9, Poisson_ratio=0.49, loss_factor=0.23):
        self.Young = Young_modulus
        self.Poisson = Poisson_ratio
        self.loss_factor = loss_factor
        super().__init__()
        
        
    def shear_m(self):
        return self.Young*(1-1j*self.loss_factor)/(2*(1+self.Poisson))


    def lame_const(self):
        return self.Young*(1-1j*self.loss_factor)*self.Poisson/((1+self.Poisson)*(1-2*self.Poisson))


    def longitudinal_m(self):
        return self.Young*(1-self.Poisson)*(1-1j*self.loss_factor)/((1+self.Poisson)*(1-2*self.Poisson))
        

        

class wavenumber(elastic_module):
    def __init__(self, determinant, frequency_array):
        self.determinant = determinant
        self.frequency_array = np.asarray(frequency_array)
        # self.omega_array = np.asarray(frequency_array)*2*np.pi
        super().__init__()
        
    def omega_array(self):
        return self.frequency_array*2*np.pi
    
    
    
    ## Solve the determinant equation of determinant numerically in Scipy...
    def axial_wavenumber(self, single_omega, guess=0.1+0.1j, leave=False):
        kz = []
        failed_kz = []
        kl = single_omega/self.longitudinal_speed()
        x0 = abs(guess) # kl[0]
        ai, _ = self.effective_radius()
        for i in tqdm(range(ai.shape[0]), position=1, leave=leave, 
                      desc =f'  ... working at frequency = {single_omega/(2*np.pi):.1f} Hz', ):

            if ai[i] == 0:
                kz.append(0+0j)

            else:
                try:
                    kz_root = newton(self.determinant, x0, 
                                        args=(ai[i], self.cell_r, self.shear_m(), self.lame_const(), single_omega, 
                                            self.longitudinal_speed()[i], self.transverse_speed()[i]), 
                                        tol=1.48e-8, maxiter=200)
                    kz.append(kz_root)
                    x0 = kz_root

                except RuntimeError:
                    try:
                        # time.sleep(1)
                        print(f'First try of finding root at frequency = {single_omega/(2*np.pi):.2f}, {i = } / {ai.shape[0]} failed.')
                        print('Try again...')

                        try:
                            x0 = kz_root
                        except UnboundLocalError:
                            x0 = abs(guess)*(i+1)
                        
                        kz_root = newton(self.determinant, x0, 
                                            args=(ai[i], self.cell_r, self.shear_m(), self.lame_const(), single_omega, 
                                                self.longitudinal_speed()[i], self.transverse_speed()[i]), 
                                            tol=1.48e-8, maxiter=1000)
                        kz.append(kz_root)
                        x0 = kz_root

                    except RuntimeError:
                        print(f'Second try of finding root at frequency = {single_omega/(2*np.pi):.2f}, {i = } / {ai.shape[0]} failed.')
                        print('Assume root is 0+0j...')
                        kz.append(0+0j)
                        failed_kz.append([single_omega, i, ai.shape[0]])
                        # break
        
        
        if self.p_hole == 0:
            # print('Since p=0, the root of kz at the last segment cannot be solved.')
            # print('Assign the root next to last to the last (kz[0] = kz[1]).')
            kz[0] = kz[1]
        
        
        if self.q_hole == 0:
            # print('Since q=0, the root of kz at the last segment cannot be solved.')
            # print('Assign the root next to last to the last (kz[-1] = kz[-2]).')
            kz[-1] = kz[-2]


        return np.asarray(kz), failed_kz
    
    
    
    def axial_wavenumber_array(self, guess=0.1+0.1j):    
        wavenumer_array = np.zeros((self.frequency_array.shape[0], self.segments), dtype=complex)
        failed_roots = []
        # i = 0
        print(f"Solving wavenumber in determinant for shape = {self.shape}, p = {self.p_hole}, q = {self.q_hole}, Young's = {self.Young}")
        for i in tqdm(range(self.frequency_array.shape[0]), position=0, maxinterval=1, 
                      desc ='Solving for all frequencies'):
            omega = self.omega_array()[i]
            leave = (i == self.frequency_array.shape[0]-1)
            kz, failed_kz = self.axial_wavenumber(omega, guess=guess, leave=leave)
            wavenumer_array[i][:] = kz
            failed_roots.append(failed_kz)
            guess=kz[0]

        print('\n')
        return np.asarray(wavenumer_array), np.asarray(failed_roots)
    
        

        

class sound_performance(wavenumber):
    ## medium_density: density of water 998 kg/m3
    ## sound_speed_medium: Sound speed of water 1483 m/s
    def __init__(self, determinant, frequency_array, medium_density=998, sound_speed_medium=1483):
        self.zw = medium_density * sound_speed_medium
        self.wavenumer_array = []
        self.failed_root = []
        super().__init__(determinant, frequency_array)
        
    ## eq.(1-13b), in book page 26
    def longitudinal_speed(self):
        return np.sqrt(self.longitudinal_m()/self.effective_density())

    ## eq.(1-14), in book page 27
    def transverse_speed(self):
        return np.sqrt(self.shear_m()/self.effective_density())
    
    
    ## The effective impedance of the total segments under one frequency/omega: (16)
    def effective_impedance(self, omega, wave_number):
        return self.effective_density()*omega/wave_number


    ## The successive multi-plication of the transfer matrix of each segment: (18)
    def total_tran_matrix(self, omega, wave_number):
        
        ## The transfer matrix of the i th segment: (17)
        def ith_tran_matrix(wave_number, li, ith_impedance):
            ti = np.zeros((2,2), dtype=complex)
            ti[0,0] = np.cos(wave_number*li)
            ti[0,1] = -1j*np.sin(wave_number*li)*ith_impedance
            ti[1,0] = -1j*np.sin(wave_number*li)/ith_impedance
            ti[1,1] = np.cos(wave_number*li)
            return ti
    
        li = self.h_hole/self.segments
        impedance = self.effective_impedance(omega, wave_number)

        try:
            t0 = ith_tran_matrix(wave_number[0], li, impedance[0])
            tn=t0
            for i in range(1, self.segments):
                tn = np.matmul(tn, ith_tran_matrix(wave_number[i], li, impedance[i]))

        except IndexError:
            t0 = ith_tran_matrix(wave_number, li, impedance[0])
            tn=t0
            for i in range(1, self.segments):
                tn = np.matmul(tn, ith_tran_matrix(wave_number, li, impedance[i]))        

        return tn


    ## The impedance of the front interface Zf: (22)
    def imped_front(self, omega, wave_number):
        # zf = np.absolute(tn[0,0]/tn[1,0])
        return (self.total_tran_matrix(omega, wave_number)[0,0]/self.total_tran_matrix(omega, wave_number)[1,0])


    ## The reflection coefficient of the anechoic layer: (23)
    def reflection_coefficient(self, omega, wave_number):
        return (self.imped_front(omega, wave_number)-self.zw)/(self.imped_front(omega, wave_number)+self.zw)


    ## The sound absorption coefficient: (24)
    def absorption_coefficient(self, omega, wave_number):
        return 1-np.absolute(self.reflection_coefficient(omega, wave_number))**2
        
               
        
    def absorption_frequency(self):

        if self.frequency_array.shape[0] == self.wavenumer_array.shape[0]:
            print(f'{self.frequency_array.shape[0] = } is same as {self.wavenumer_array.shape[0] = }')
        else:
            raise ValueError(f'{self.frequency_array.shape = } not same as {self.wavenumer_array.shape = }')

        absorption_list = []
        for omega, wave_number in zip(self.omega_array(), self.wavenumer_array):
            # self.effective_impedance(omega, wave_number)
            # self.total_tran_matrix(omega, wave_number)
            # self.imped_front()
            # self.reflection_coefficient()
            # # alpha = absorption_coefficient(ref)   
            absorption_list.append(self.absorption_coefficient(omega, wave_number))

        return np.asarray(absorption_list)
        
        
        
        
    def save_data(self, filepath, filename):

        df_const = pd.DataFrame()
        df_const['Variable'] = ['material', 'shape', 'p_mm', 'q_mm', 'lh_mm', 'b_mm',  'num_segments', 'Young_GPa', 'Poisson_r', 
                                'loss_factor', 'rubber_kgm-3', 'air_kgm-3']
        df_const['Value'] = [self.material, self.shape, self.p_hole*1000, self.q_hole*1000, self.h_hole*1000, self.cell_r*1000, 
                             self.segments, self.Young/(10**9), self.Poisson, self.loss_factor, self.layer_density, self.air_density]

        df_wave = pd.DataFrame()
        df_wave['frequency_Hz'] = np.asarray([f'kz_{i:03d}' for i in range(self.segments)])
        i = 0
        for frequency in self.frequency_array:
            df_wave[frequency] = self.wavenumer_array[i][:]
            i += 1

        df_absorption = pd.DataFrame()
        df_absorption['frequency_Hz'] = self.frequency_array
        df_absorption['absorption'] = self.absorption_frequency()
        
        try:
            fn = os.path.join(filepath, filename)
            # writer = pd.ExcelWriter(fn, engine='xlsxwriter')
            with pd.ExcelWriter(fn, engine='xlsxwriter') as writer:
                df_const.to_excel(writer, sheet_name='Constants', index=False)   
                df_wave.to_excel(writer, sheet_name='Wave_numbers', index=False)
                df_absorption.to_excel(writer, sheet_name='Sound_absorption', index=False)
            print(f'Save file to {fn}')

        except (ValueError, ModuleNotFoundError):
            fn_const = os.path.join(filepath, filename.split('.')[0]+'_const.csv')
            df_const.to_csv(fn_const, sep=' ', index=False, header=False, float_format='{:.8e}'.format)
            print(f'Save file to {fn_const}')
            fn_wave = os.path.join(filepath, filename.split('.')[0]+'_kz.csv')
            df_wave.to_csv(fn_wave, sep=' ', index=False, header=True, float_format='{:.8e}'.format)
            print(f'Save file to {fn_wave}')
            fn_absor = os.path.join(filepath, filename.split('.')[0]+'_absor.csv')
            df_absorption.to_csv(fn_absor, sep=' ', index=False, header=True, float_format='{:.8e}'.format)
            print(f'Save file to {fn_absor}') 
        
        
'''
Define All twelve elements in the coefficient matrix of Eq. (11) in the paper
Bessel functions from SymPy
https://docs.sympy.org/latest/modules/functions/special.html
'''           
def determinant_from_matrix():
    mu = Symbol('mu')
    lambda_ = Symbol('lambda')
    kz_ = Symbol('kz')
    omega_ = Symbol('omega')
    cl_ = Symbol('cl')
    ct_ = Symbol('ct')
    kl_ = omega_/cl_
    kt_ = omega_/ct_
    klr = sqrt(kl_**2 - kz_**2)
    ktr = sqrt(kt_**2 - kz_**2)

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


    mat = Matrix([[M11, M12, M13, M14], [M21, M22, M23, M24], [M31, M32, M33, M34], [M41, M42, M43, M44]]) 
    determinant_00 = mat.det()
    determinant_01 = lambdify((kz_, a, b, mu, lambda_, omega_, cl_, ct_), determinant_00)
    
    return determinant_01, mat