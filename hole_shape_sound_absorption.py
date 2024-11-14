import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from sympy import Symbol, besselj, bessely, I, Matrix, lambdify, sqrt
from sympy.abc import a, b, z
from scipy.optimize import newton
from tqdm.auto import tqdm
# from cxroots import Circle
# from cxroots import Annulus
import mpmath
from scipy.special import jv, yv



'''
### Reference: Influence of hole shape on sound absorption of underwater anechoic layers
### https://www.sciencedirect.com/science/article/abs/pii/S0022460X1830227X
'''

class anechoic_layers():
    
    ## Section 3.1 and Fig. 7 in the paper
    def __init__(self, material='rubber', shape='cone', 
                 p=4e-3, q=8e-3, lh=40e-3, la=0.05, cell_radius=15e-3, 
                 theta=0.203, phi=0.035, length_unit='m', 
                 num_segments=100, layer_density=1100, air_density=1.21, 
                 use_volume=False):
        
        self.material = material
        self.shape = shape
        self.p_hole = p
        self.q_hole = q
        self.h_hole = lh
        self.la = la
        self.cell_r = cell_radius
        self.theta = theta
        self.phi = phi
        self.length_unit = length_unit
        self.segments = num_segments
        self.layer_density = layer_density
        self.air_density = air_density
        self.use_volume = use_volume
        self.density_array = []
        self.radius_array = []
        self.epsilon_array = []
        
    
    ## Eq (25-27) in the paper
    def effective_radius(self, is_segment=True):
        lh_n = np.linspace(0, self.h_hole, self.segments+1)

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

        elif self.shape == 'sin':
            # print(f'The input shape is {shape}.')
            if self.length_unit == 'mm':
                const = 1
                
            elif self.length_unit == 'm':
                const = 1000

            ## To compare theta, phi in the reference, the unit of length in the eq turns to mm
            psin = self.p_hole*const
            qsin = self.q_hole*const
            lhsin = self.h_hole*const
            phi_sin = self.phi*const
            lh_n_sin = lh_n * const
            x = (qsin-psin)/np.sin(self.theta*lhsin)
            alpha = (1/lhsin)*np.log(abs(x))
            r_temp = np.exp(alpha*lh_n_sin)*np.sin(self.theta*lh_n_sin) + phi_sin

            ## Turn the unit of length back to m for SI units
            r_effective = r_temp / const

        if is_segment:
            self.radius_array = r_effective[1:]
            return r_effective[1:], lh_n[1:]
        else:
            return r_effective, lh_n
    
    
    
    ## Try to use actual volume insteas of effecitve radius
    def shape_volume(self):
        lh_n = np.linspace(0, self.h_hole, self.segments+1)

        if self.shape == 'cone':
            pcone, qcone = self.p_hole, self.q_hole
            alpha = (qcone-pcone)/self.h_hole
            beta = pcone
            volume_seg = []
            for i in range(self.segments):
                volume_i = cone_volume_i(lh_n[i], lh_n[i+1], alpha, beta)
                volume_seg.append(volume_i)
            

        elif self.shape == 'horn':
            phorn, qhorn = self.p_hole, self.q_hole
            gamma = phorn
            delta = (1/self.h_hole)*np.log(qhorn/phorn)
            volume_seg = []
            for i in range(self.segments):
                volume_i = horn_volume_i(lh_n[i], lh_n[i+1], gamma, delta)
                volume_seg.append(volume_i)


        elif self.shape == 'sin':
            # print(f'The input shape is {shape}.')
            if self.length_unit == 'mm':
                const = 1
                
            elif self.length_unit == 'm':
                const = 1000

            ## To compare theta, phi in the reference, the unit of length in the eq turns to mm
            psin = self.p_hole*const
            qsin = self.q_hole*const
            lhsin = self.h_hole*const
            phi_sin = self.phi*const
            lh_n_sin = lh_n * const
            x = (qsin-psin)/np.sin(self.theta*lhsin)
            alpha = (1/lhsin)*np.log(abs(x))

            volume_seg = []
            for i in range(self.segments):
                volume_i = sin_volume_i(lh_n[i], lh_n[i+1], alpha, self.theta, phi_sin)
                ## Turn the unit of length back to m for SI units
                volume_i_SI = volume_i / const**3
                volume_seg.append(volume_i_SI)
            
        
        return np.asarray(volume_seg)

    
    
    ## ai represents the inner radii of the pipe in the i th layer
    ## Eq (15) in the paper
    def effective_density(self, use_volume=False):

        if use_volume:
            cell_i_vol = np.pi * (self.cell_r**2) * (self.h_hole/self.segments)
            air_vol = self.shape_volume()
            layer_vol = cell_i_vol - air_vol
            air_vol_ratio = air_vol / cell_i_vol
            layer_vol_ratio = layer_vol / cell_i_vol
            ddd = self.layer_density*layer_vol_ratio + self.air_density*air_vol_ratio

        else:
            ai = self.radius_array
            ddd = self.layer_density*(1-(ai/self.cell_r)**2) + self.air_density*(ai/self.cell_r)**2

        self.density_array = ddd
        return ddd

    
    ## a/b ratio (epsilon)
    def radius_ratio(self):
        ai = self.radius_array
        self.epsilon_array = ai/self.cell_r
        return ai/self.cell_r


    ## Plot the 2D scheme of the hole based on the given shape
    def plot_hole_2D(self, label=True):

        title = f'Hole Shape : {self.shape}'
        if self.shape == 'sin':
            label=f'p= {self.p_hole}, q={self.q_hole}, $\\theta$={self.theta}'
        else:
            label=f'p= {self.p_hole}, q={self.q_hole}'
        
        plt.figure()
        r, z = self.effective_radius(is_segment=False)
        plt.plot(z, r, label=label, color='tab:blue')
        plt.plot(z, -r, 'tab:blue')
        plt.vlines(self.h_hole-self.la, -self.cell_r, self.cell_r)
        plt.vlines(self.h_hole, -self.cell_r, self.cell_r)
        plt.vlines(0, -self.cell_r, self.cell_r, linestyles='--', color='silver')
        plt.hlines(-self.cell_r, self.h_hole-self.la, self.h_hole)
        plt.hlines(self.cell_r, self.h_hole-self.la, self.h_hole)
        plt.hlines(0, self.h_hole-self.la, self.h_hole, linestyles='--', color='silver')
        plt.ylim(-self.cell_r*2, self.cell_r*2)
        plt.vlines(0, -r[0], r[0])
        if label:
            plt.legend()
            plt.title(title, fontsize=15, fontweight='bold')
        # plt.show()


## https://en.wikipedia.org/wiki/Lam%C3%A9_parameters
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

    ## longitudinal_m = lame_const + 2*shear_m
    def longitudinal_m(self):
        return self.Young*(1-self.Poisson)*(1-1j*self.loss_factor)/((1+self.Poisson)*(1-2*self.Poisson))


    ## eq.(1-13b), in book page 26
    def longitudinal_speed(self):
        return np.sqrt(self.longitudinal_m()/self.density_array)


    ## eq.(1-14), in book page 27
    def transverse_speed(self):
        return np.sqrt(self.shear_m()/self.density_array)
        

        

class wavenumber(elastic_module):
    def __init__(self, determinant, frequency_array):
        self.determinant = determinant
        self.frequency_array = np.asarray(frequency_array)
        self.omega_array = np.asarray(frequency_array)*2*np.pi
        super().__init__()
        
    # def omega_array(self):
    #     return self.frequency_array*2*np.pi


    ## Solve the determinant equation of determinant numerically in Scipy...
    def axial_wavenumber_newton(self, single_omega, guess=0.1+0.1j, leave=False):
        kz = []
        failed_kz = []
        # kl = single_omega/self.longitudinal_speed()
        x0 = guess # kl[0]
        # x0 = abs(guess) # kl[0]
        # x0 = guess.imag # kl[0]
        ai = self.radius_array
        for i in tqdm(range(ai.shape[0]), position=1, leave=leave, 
                      desc =f'  ... working at frequency = {single_omega/(2*np.pi):.1f} Hz', ):

            if ai[i] == 0:
                kz.append(0.1+0.1j)

            else:
                try:
                    kz_root = newton(self.determinant, x0, 
                                        args=(ai[i], self.cell_r, self.shear_m(), self.lame_const(), 
                                              single_omega, self.longitudinal_speed()[i], 
                                              self.transverse_speed()[i]), 
                                        tol=1.48e-6, maxiter=1000)
                    kz.append(kz_root)
                    x0 = kz_root

                except RuntimeError:
                    try:
                        # time.sleep(1)
                        print(f'First try of finding root at frequency = {single_omega/(2*np.pi):.2f}, {i = } / {ai.shape[0]} failed.')
                        print('Try again...')

                        # try:
                        #     x0 = kz_root
                        # except UnboundLocalError:
                        x0 = abs(guess)*(i+1)
                        kz_root = newton(self.determinant, x0, 
                                            args=(ai[i], self.cell_r, self.shear_m(), self.lame_const(),
                                                  single_omega, self.longitudinal_speed()[i], 
                                                  self.transverse_speed()[i]), 
                                            tol=1.48e-4, maxiter=2000)
                        kz.append(kz_root)
                        x0 = kz_root

                    except RuntimeError:
                        print(f'Second try of finding root at frequency = {single_omega/(2*np.pi):.2f}, {i = } / {ai.shape[0]} failed.')
                        
                        try:
                            print('Try again...')
                            # x0 = guess*(i+1)
                            x0 = 0.1+0.1j
                            kz_root = newton(self.determinant, x0, 
                                            args=(ai[i], self.cell_r, self.shear_m(), self.lame_const(),
                                                  single_omega, self.longitudinal_speed()[i],
                                                  self.transverse_speed()[i]), 
                                            tol=1.48e-4, maxiter=2000)
                        
                        except RuntimeError:
                            print(f'Third try of finding root at frequency = {single_omega/(2*np.pi):.2f}, {i = } / {ai.shape[0]} failed.')
                        
                        print('Assume root is 0.1+0.1j...')
                        kz.append(0.1+0.1j)
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

        for i in range(len(kz)):
            if kz[i] == 0.1+0.1j:
                kz[i] = kz[3]
                
        return np.asarray(kz), failed_kz
    




    ## Solve the simplified determinant equation for accurate solution numerically in Scipy...
    def axial_wavenumber_newton2(self, single_omega, guess=0.1+0.1j, leave=False):
        kz = []
        failed_kz = []
        # kl = single_omega/self.longitudinal_speed()
        x0 = guess # kl[0]
        # x0 = guess.imag # kl[0]
        ai = self.radius_array
        for i in tqdm(range(ai.shape[0]), position=1, leave=leave, 
                      desc =f'  ... working at frequency = {single_omega/(2*np.pi):.1f} Hz', ):

            if ai[i] == 0:
                kz.append(0+0j)

            else:
                try:
                    kz_root = newton(self.determinant, x0, 
                                        args=(ai[i], self.cell_r, single_omega, 
                                            self.longitudinal_speed()[i], self.transverse_speed()[i]), 
                                        tol=1.48e1, maxiter=200)
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
                                            args=(ai[i], self.cell_r, single_omega, 
                                                self.longitudinal_speed()[i], self.transverse_speed()[i]), 
                                            tol=1.48e1, maxiter=5000)
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
    
    


    ## Solve the determinant equation of determinant numerically in mpmath...
    def axial_wavenumber_mpmath(self, single_omega, guess=0.1+0.1j, leave=False):
        kz = []
        failed_kz = []
        # kl = single_omega/self.longitudinal_speed()
        # x0 = abs(guess) # kl[0]
        ai = self.radius_array
        mpmath.mp.dps = 30
        for i in tqdm(range(ai.shape[0]), position=1, leave=leave, 
                      desc =f'  ... working at frequency = {single_omega/(2*np.pi):.1f} Hz', ):

            if ai[i] == 0:
                kz.append(0+0j)

            else:
                # def f1(kz_):
                #     return self.determinant(kz_, ai[i], self.cell_r, self.shear_m(), self.lame_const(), 
                #                        single_omega, self.longitudinal_speed()[i], 
                #                        self.transverse_speed()[i])
                
                def f1(kz_):
                    return self.determinant(kz_, ai[i], self.cell_r, single_omega, 
                                            self.longitudinal_speed()[i], self.transverse_speed()[i])
                
                kz_root = mpmath.findroot(f1, guess, solver='secant', tol=1.48e-3)
                # kz_root = newton(self.determinant, x0, 
                #                  args=(ai[i], self.cell_r, self.shear_m(), self.lame_const(), 
                #                        single_omega, self.longitudinal_speed()[i], 
                #                        self.transverse_speed()[i]), tol=1.48e-3, maxiter=20)
                kz.append(kz_root)
                x0 = kz_root

        
        
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
            omega = self.omega_array[i]
            leave = (i == self.frequency_array.shape[0]-1)
            if i < 500000:
                kz, failed_kz = self.axial_wavenumber_newton(omega, guess=guess, leave=leave)
            else:
                kz, failed_kz = self.axial_wavenumber_mpmath(omega, guess=guess, leave=leave)
            wavenumer_array[i][:] = kz
            failed_roots.append(failed_kz)
            guess=kz[3]
            # print(f'{guess = }')

        print('\n')
        # return np.asarray(wavenumer_array), np.asarray(failed_roots)
        return np.asarray(wavenumer_array), []
    
        

        

class sound_performance(wavenumber):
    ## medium_density: density of water 998 kg/m3
    ## sound_speed_medium: Sound speed of water 1483 m/s
    def __init__(self, determinant, frequency_array, medium_density=998, sound_speed_medium=1483):
        self.zw = medium_density * sound_speed_medium
        self.absorption_array = []
        self.wavenumer_array = []
        self.failed_root = []
        super().__init__(determinant, frequency_array)
    
    
    ## The effective impedance of the total segments under one frequency/omega: (16)
    def effective_impedance(self, omega, wave_number):
        return self.density_array*omega/wave_number



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

        t0 = ith_tran_matrix(wave_number[0], li, impedance[0])
        tn=t0
        for i in range(1, self.segments):
            tn = np.matmul(tn, ith_tran_matrix(wave_number[i], li, impedance[i]))      

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
        for omega, wave_number in zip(self.omega_array, self.wavenumer_array):
            absorption_list.append(self.absorption_coefficient(omega, wave_number))

        self.absorption_array = absorption_list
        return np.asarray(absorption_list)
        
        
        
        
    def save_data(self, filepath, filename):

        df_const = pd.DataFrame()
        df_const['Variable'] = ['material', 'shape', 'p_mm', 'q_mm', 'lh_mm', 'b_mm', 'theta', 'phi', 
                                'num_segments', 'Young_GPa', 'Poisson_r', 
                                'loss_factor', 'rubber_kgm-3', 'air_kgm-3', 'use_volume', ]
        df_const['Value'] = [self.material, self.shape, self.p_hole*1000, self.q_hole*1000, self.h_hole*1000, self.cell_r*1000, 
                            self.theta, self.phi, self.segments, self.Young/(10**9), self.Poisson, 
                            self.loss_factor, self.layer_density, self.air_density, self.use_volume, ]

        df_wave = pd.DataFrame()
        df_wave['frequency_Hz'] = np.asarray([f'kz_{i:03d}' for i in range(self.segments)])
        i = 0
        for frequency in self.frequency_array:
            df_wave[frequency] = self.wavenumer_array[i][:]
            i += 1

        df_absorption = pd.DataFrame()
        df_absorption['frequency_Hz'] = self.frequency_array
        
        if len(self.absorption_array) == 0:
            self.absorption_frequency()
        else:
            pass
        
        df_absorption['absorption'] = self.absorption_array
        
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
        
        


def anechoic_sound_absorption(determinant, frequency_array,
                              fp = '/Users/chenghunglin/Documents/', 
                              fn = 'cone_6_3_fr50.xlsx', 
                              material='rubber', shape='cone', 
                              p=6e-3, q=6e-3, lh=40e-3, la=0.05, cell_radius=15e-3, 
                              theta=0.203, phi=0.035, length_unit='m', 
                              num_segments=100, layer_density=1100, air_density=1.21, 
                              Young_modulus=0.14e9, Poisson_ratio=0.49, loss_factor=0.23, 
                              medium_density=998, sound_speed_medium=1483, 
                              use_volume=True):
    
    
    par_dict = {'material':material, 'shape':shape, 
                'p_hole': p, 'q_hole': q, 'h_hole': lh, 'la': la, 'cell_r': cell_radius, 
                'theta': theta, 'phi': phi, 'length_unit': length_unit, 
                'segments': num_segments, 'layer_density': layer_density, 'air_density': air_density, 
                'Young': Young_modulus, 'Poisson': Poisson_ratio, 'loss_factor': loss_factor, 
                'zw': medium_density * sound_speed_medium, 'use_volume': use_volume, }
    
    hole_sound = sound_performance(determinant, frequency_array)
    
    for key in par_dict.keys():
        setattr(hole_sound, key, par_dict[key])

    hole_sound.plot_hole_2D()
    plt.show()

    hole_sound.effective_radius()
    hole_sound.effective_density(use_volume=hole_sound.use_volume)
    # hole_sound.radius_ratio()
    hole_sound.wavenumer_array, hole_sound.failed_root = hole_sound.axial_wavenumber_array()
    hole_sound.absorption_frequency()
    
    try:
        hole_sound.save_data(fp, fn)
    except (OSError, FileNotFoundError):
        fp_new = os.getcwd()
        print(f'**** Directory {fp} not found, saving data at {fp_new} ****')
        hole_sound.save_data(fp_new, fn)
    
    return hole_sound
    
    

def exp_sin_shape(z, theta, phi, alpha):
    return np.exp(alpha*z) * np.sin(theta*z) + phi


def cone_volume_i(z0, z1, alpha, beta):
    volume_z0 = np.pi * ((1/3)*(alpha**2)*(z0**3) + (alpha*beta)*(z0**2) + (beta**2)*(z0))
    volume_z1 = np.pi * ((1/3)*(alpha**2)*(z1**3) + (alpha*beta)*(z1**2) + (beta**2)*(z1))
    volume_i = abs(volume_z1-volume_z0)
    return volume_i


def horn_volume_i(z0, z1, gamma, delta):
    volume_z0 = np.pi * (gamma**2) * (1/(2*delta)) * np.exp(2*delta*z0)
    volume_z1 = np.pi * (gamma**2) * (1/(2*delta)) * np.exp(2*delta*z1)
    volume_i = abs(volume_z1-volume_z0)
    return volume_i


def sin_volume_i(z0, z1, alpha, theta, phi):
    def sin_integral_01(z, alpha, theta):
        integral_01 = ((2*alpha)**2-2*alpha*(2*alpha*np.cos(2*theta*z)+2*theta*np.sin(2*theta*z))+4*theta**2)*np.exp(2*alpha*z)/(4*alpha*((2*alpha)**2+4*theta**2))
        return integral_01
    def sin_integral_02(z, alpha, theta, phi):
        integral_02 = 2*phi*(np.exp(alpha*z)*(alpha*np.sin(theta*z)-theta*np.cos(theta*z)))/(alpha**2+theta**2)
        return integral_02
    def sin_integral_03(z, phi):
        integral_03 = z*(phi**2)
        return integral_03

    volume_z0 = sin_integral_01(z0, alpha, theta) + sin_integral_02(z0, alpha, theta, phi) + sin_integral_03(z0, phi)
    volume_z1 = sin_integral_01(z1, alpha, theta) + sin_integral_02(z1, alpha, theta, phi) + sin_integral_03(z1, phi)
    volume_i = abs(volume_z1-volume_z0)
    return volume_i


        
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



'''
Define the accurate solution of eq.2-29 in the reference book, page 70-71
Bessel functions from SymPy
https://docs.sympy.org/latest/modules/functions/special.html
'''           
def matrix_accurate_solution(kz, a, b, omega, cl, ct):
    # kz_ = Symbol('kz')
    # omega_ = Symbol('omega')
    # cl_ = Symbol('cl')
    # ct_ = Symbol('ct')
    
    kl = omega/cl
    kt = omega/ct
    klr = np.sqrt(kl**2 - kz**2)
    ktr = np.sqrt(kt**2 - kz**2)
    # klr = (kl**2 - kz**2)**0.5
    # ktr = (kt**2 - kz**2)**0.5

    # kl_2 = (omega_/cl_)**2
    # kt_2 = (omega_/ct_)**2
    # klr2 = kl_**2 - kz_**2
    # ktr2 = kt_**2 - kz_**2

    beta = kt**2/(2*kl**2)
    y = kz**2/kl**2
    e = b/a

    def theta(a, b, x):
        # J1x = besselj(1, x)
        # Y0ex = bessely(0, e*a)
        # Y1x = bessely(1, x)
        # J0ex = besselJ(0, e*a)
        # Y1ex = bessely(1, e*a)
        # J1ex = besselJ(1, e*a)
        
        e = b/a
        theta_numerator = jv(1, x)*yv(0, e*x) - yv(1, x)*jv(0, e*x)
        theta_denominator = jv(1, x)*yv(1, e*x) - yv(1, x)*jv(1, e*x)

        return e*x*theta_numerator/theta_denominator


    simplified_matrix = ((beta-y)**2)*theta(a, b, klr) + y*(1-y)*theta(a, b, ktr) - (1-y)*beta

    return simplified_matrix
