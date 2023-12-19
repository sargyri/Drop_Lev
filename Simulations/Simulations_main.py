import numpy as np
import levitate
from matplotlib import pyplot as plt
# import src.coordinates
from math import pi
import pandas as pd
import matplotlib
import random
import os
import sys
from importlib import import_module 

path_to_library=str('path_to_the_file_coordinates.py') #For windows
sys.path.insert(0, path_to_library)
source_coordinates = import_module("coordinates")


# ----------------------------------------
#### Simulation Parameters :
lev_list = ["mk3"] #"mk1" or "LangLev"
phase = "opposition"             # ''opposition''  or ''in_phase'' => To have the central node or a central antinode at (0,0,0), respectively

#### Saving parameters:
path_save=str('change_this_to_the_proper_directory_you_want_to_save_the_images_in')
savefigure = False
fig_type='.svg'  

zoom = 0.003                     # Shift (in mm) from the max transducer coordinates (to avoid singularities)                       
linear_nbpoints = 1001
map_nbpoints = 201
max_x = 0.012
max_y = 0.012
max_z = 0.012       

plots = ['x', 'y', 'z', "yz","xz","xy"]
for levitator in lev_list:
    for i in plots:
        #Define mesh for the 2D plots
        meshsize = [5,5,5]          # Init number of points in x, y and z directions for the 3D mesh used in calculation
        if i == "x":
            isx = True
            meshsize[0] = linear_nbpoints
        if i == "y":
            isy = True
            meshsize[1] = linear_nbpoints 
        if i == "z":
            isz = True
            meshsize[2] = linear_nbpoints
        if i =="yz":
            isy = True
            isz = True
            meshsize[1] = map_nbpoints
            meshsize[2] = map_nbpoints
        if i =="xz":
            isx = True
            isz = True
            meshsize[0] = map_nbpoints
            meshsize[2] = map_nbpoints
        if i =="xy":
            isx = True
            isy = True
            meshsize[0] = map_nbpoints
            meshsize[1] = map_nbpoints
    
        #Import coordinates
        if levitator=='LangLev':
            p0=10     # Reference pressure (Pa) used in levitate
            coord= source_coordinates.lev(levitator) / 1000
        else:
            p0=1      # Reference pressure (Pa) used in levitate
            coord = source_coordinates.lev(levitator) / 1000


        # -----------------------------------------
        #### Mesh struture :

        x = np.linspace(-max_x,max_x, meshsize[0])
        y = np.linspace(-max_y,max_y, meshsize[1])
        z = np.linspace(-max_z, max_z, meshsize[2])

        dx = x[1]-x[0]
        dy = y[1]-y[0]
        dz = z[1]-z[0]
        x0 = np.where(x==0)[0][0]
        y0 = np.where(y==0)[0][0]
        z0 = np.where(z==0)[0][0]

        points = np.reshape(np.meshgrid(x, y, z, indexing = 'ij'), (3, -1))

        # ----------------------------------------
        ##### Transducer array :
        
        phases_down = 0.5*np.pi*np.ones((1,len(coord)//2)) 
        if phase == 'quadrature':
            phases_up = -0.5*np.pi*np.ones((1,len(coord)//2))
        elif phase == 'in phase':
            phases_up = 0.5*np.pi*np.ones((1,len(coord)//2)) 
        
        normals = -1*coord / np.linalg.norm(coord, 2, axis=1, keepdims = 1)
        coord = coord.T
        normals = normals.T
        
        
        transducer = levitate.transducers.CircularRing(effective_radius=3.5e-3, p0=p0)
        array = levitate.arrays.TransducerArray(coord, normals, transducer=transducer)
        
        state = 0.5*np.pi*np.ones(array.num_transducers, complex)
        
        top_idx = np.where(coord[2] > 0)
        bottom_idx = np.where(coord[2] < 0)

        if phase=='opposition':
          state[top_idx] = 1j
          state[bottom_idx] = -1j
        elif phase=='in_phase':
          state[top_idx] = 1j
          state[bottom_idx] = 1j
        
        array.state = state
                

        
        # ----------------------------------------
        ##### Levitate calculation :

        pressure_calculation_instance = levitate.fields.Pressure(array) @ points
        pressure = abs(pressure_calculation_instance(state))


        gorkov_calc = levitate.fields.GorkovPotential(array,radius = 1e-3) @ points
        gorkov_grad_calc = levitate.fields.GorkovGradient(array,radius = 1e-3) @ points
        
        gorkov = gorkov_calc(state)
        gorkov = gorkov.reshape(len(x),len(y),len(z))
        gorkov_grad = gorkov_grad_calc(state)
        F = -gorkov_grad.reshape(3,len(x),len(y),len(z))
        pressure = pressure.reshape(len(x),len(y),len(z))

        pressure_z = pressure[x0,y0,:]
        pressure_y = pressure[x0,:,z0]
        pressure_x = pressure[:,y0,z0]

        # ---------------------------------------------------------------
        ##### Plots :

        # --------------------------
        ##### 2d Pressure Map Y, Z :

        if i == "yz":
            fig,ax = plt.subplots(figsize = (500*max_x,402*max_z))

            cs = ax.contourf(y*1000,z*1000,pressure[x0,:,:].T,100, vmin = 0, vmax = np.amax(pressure[x0,:,:].T))
            for a in cs.collections:
                a.set_edgecolor('face')
            plt.xlim((-max_y*1000,max_y*1000))
            plt.ylim((-max_z*1000,max_z*1000))
            plt.xlabel('y (mm)')
            plt.ylabel('z (mm)')
            plt.axis('equal')
            bar = fig.colorbar(cs)
            plt.title("Theoretical pressure field Y, Z")
            if savefigure & isy & isz :
                plt.savefig(path_save+levitator+"_PressureField_yz_"+phase+fig_type, dpi = 300)


        # -------------------------------------
        ##### 2d Pressure Map X, Z :
        if i == "xz":
            fig,ax = plt.subplots(figsize = (500*max_x,402*max_z))
            cs = ax.contourf(x*1000,z*1000,pressure[:,y0,:].T,100, vmin = 0, vmax = np.amax(pressure[:,y0,:].T))
            for a in cs.collections:
                a.set_edgecolor('face')
            plt.xlim((-max_y*1000,max_y*1000))
            plt.ylim((-max_z*1000,max_z*1000))
            plt.xlabel('x (mm)')
            plt.ylabel('z (mm)')
            plt.axis('equal')
            bar = fig.colorbar(cs)
            plt.title("Theoretical pressure field X, Z")
            if savefigure & isx & isz :
                plt.savefig(path_save+levitator+"_PressureField_xz_"+phase+fig_type, dpi = 300)
        # --------------------------------------
        ##### 2d Pressure Map X, Y :
        if i == "xy":
            fig,ax = plt.subplots(figsize = (500*max_x,402*max_y))
            cs = ax.contourf(x*1000,y*1000,pressure[:,:,z0].T,100, vmin = 0, vmax = np.amax(pressure[:,:,z0].T))
            for a in cs.collections:
                a.set_edgecolor('face')
            plt.xlim((-max_y*1000,max_y*1000))
            plt.ylim((-max_z*1000,max_z*1000))
            plt.xlabel('x (mm)')
            plt.ylabel('y (mm)')
            plt.axis('equal')
            bar = fig.colorbar(cs)
            plt.title("Theoretical pressure field X, Y")
            if savefigure & isx & isy:
                plt.savefig(path_save+levitator+"_PressureField_xy_"+phase+fig_type, dpi = 300)

        # ---------------------------------------
        #### 2D Gorkov Potential y,z :
        if i == "yz":
            fig,ax = plt.subplots(figsize = (500*max_y,402*max_z))
            cs = ax.contourf(y*1000,z*1000,gorkov[x0,:,:].T,100, vmin = -2e-8, vmax = np.amax(gorkov[x0,:,:].T))
            for a in cs.collections:
                a.set_edgecolor('face')
            plt.xlabel('y (mm)')
            plt.ylabel('z (mm)')
            plt.axis('equal')
            plt.xlim((-max_y*1000,max_y*1000))
            plt.ylim((-max_z*1000,max_z*1000))
            cbar = fig.colorbar(cs)
            plt.title("Gor'kov potential Y, Z")
            if savefigure & isy & isz:
                plt.savefig(path_save+levitator+"_GorkovField_yz_"+phase+fig_type, dpi = 300)

        # ---------------------------------------
        ##### 2D Gorkov Potential x,z :
        if i == "xz":
            fig,ax = plt.subplots(figsize = (500*max_x,402*max_z))
            cs = ax.contourf(x*1000,z*1000,gorkov[:,y0,:].T,100, vmin = -2e-8, vmax = np.amax(gorkov[:,y0,:].T))
            for a in cs.collections:
                a.set_edgecolor('face')
            plt.xlabel('x (mm)')
            plt.ylabel('z (mm)')
            plt.axis('equal')
            plt.xlim((-max_x*1000,max_x*1000))
            plt.ylim((-max_z*1000,max_z*1000))
            cbar = fig.colorbar(cs)
            plt.title("Gor'kov potential X, Z")
            if savefigure & isx & isz:
                plt.savefig(path_save+levitator+"_GorkovField_xz_"+phase+fig_type, dpi = 300)

        # ---------------------------------------
        ##### 2D Gorkov Potential x,y :
        
        if i == "xy":
            fig,ax = plt.subplots(figsize = (500*max_x,402*max_y))
            cs = ax.contourf(x*1000,y*1000,gorkov[:,:,z0].T,100, vmin = -2e-8, vmax = np.amax(gorkov[:,:,z0].T))
            for a in cs.collections:
                a.set_edgecolor('face')
            plt.xlabel('x (mm)')
            plt.ylabel('y (mm)')
            plt.axis('equal')
            plt.xlim((-max_x*1000,max_x*1000))
            plt.ylim((-max_y*1000,max_y*1000))
            cbar = fig.colorbar(cs)
            plt.title("Gor'kov potential X, Y")
            if savefigure & isx & isy:
                plt.savefig(path_save+levitator+"_GorkovField_xy_"+phase+fig_type, dpi = 300)

        # ------------------------------------------------
        #### Pressure z axis :
        if i == "z":

            plt.figure()
            plt.plot(z*1000,pressure_z)
            plt.title('Calculated pressure along x axis')
            plt.xlabel('x (mm)')
            plt.ylabel('Pressure (Pa)')
            plt.xlim(-max_z*1000,max_z*1000)
            #plt.ylim(0,1700)
            if savefigure & isz:
                plt.savefig(path_save+levitator+"_Pressure_z_"+phase+fig_type, dpi = 300)


        # ------------------------------------------------
        #### Pressure y axis :
        if i == "y":    

            plt.figure()
            plt.plot(y*1000,pressure_y)
            plt.title('Calculated Pressure along y axis')
            plt.xlabel('y (mm)')
            plt.ylabel('Pressure (Pa)')
            plt.xlim(-max_y*1000,max_y*1000)
            if savefigure & isy:
                plt.savefig(path_save+levitator+"_Pressure_y_"+phase+fig_type, dpi = 300)


        # ------------------------------------------------
        #### Pressure x axis :
        if i == "x":

            plt.figure()
            plt.plot(x*1000,pressure_x, label='Less transducers')
            plt.title('Calculated pressure along x axis')
            plt.xlabel('x (mm)')
            plt.ylabel('Pressure (Pa)')
            plt.legend(frameon=False)
            plt.xlim(-max_x*1000,max_x*1000)
            if savefigure & isx:
                plt.savefig(path_save+levitator+"_Pressure_x_"+phase+fig_type, dpi = 300)

