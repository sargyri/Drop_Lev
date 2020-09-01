import numpy as np
import cv2
import os
import re
import glob
import matplotlib.pyplot as plt
from scipy.signal import  gaussian
from scipy.ndimage import filters
from scipy import optimize
from scipy.stats import linregress
from mpl_toolkits.mplot3d import Axes3D

import scipy as sp
import scipy.interpolate
from scipy.integrate import trapz

import time

import pandas as pd

from lmfit import Model, fit_report, Parameters



######### initialize the data ###############
data_input=[]
data_output=[]
l_sample=300
vol_all=[]
vol2_all=[]
data_x=[]
data_y=[]
data_h=[]
data_w=[]
R_sph_all=[]
Ar=[]
#st_all=[]
rho=[]
theta=[]
#List of non-processed images:
discarded=[]
new=[]  #list of images without the discarded ones

#For the simple approach fit
phi_final=[]
theta_az=[]
theta_all=[]
theta_final=[]
r_theta_all=[]
xc_all=[]
yc_all=[]
rho_final=[]
data_Ps=[]
data_dB=[]
data_stderr=[]
data_err=[]

total_num_pictures = len(glob.glob1('.',"*.png"))
frame_number=np.arange(total_num_pictures)

gamma=64 #[mN/m] #example Surface tension value (glycerol)
Cg_air=1/101325 #Pa**(-1)
k_o=2*np.pi*40/340  #wave number in the air. frequency 40kHz (25kH=0.000040sec) and speed of sound in air v=340 m/s from paper (331.5 for 20oC online) 
#Calibration
needle_d=0.83 #mm
needle_pxl=220 #pixels
#calibration=0.00621722846441948**3
cal=needle_d/needle_pxl #[mm/pixels]
calibration=(cal)**3  #[mm**3/pixels**3]


###############################################################################
#### Functions used in the data processing ####################################
###############################################################################

def calc_volume(x,y):
    """
    Calculate the drop volume from a shape matrix.

    Parameters
    ----------
    x, y - cartesian coordinates of contour of the drop [mm]

    Returns
    -------
    Integrated volume using for the given shape matrix.

    """
    # data selection
    idx=x<=0  # r=right
    x_l=x[idx]
    y_l=y[idx]
        
    vol_left=trapz(np.pi*x_l**2, y_l)/2
        
    # data selection
    idx=x>=0  # r=right
    x_r=x[idx]
    y_r=y[idx]
    vol_right=-trapz(np.pi*x_r**2, y_r)/2
   
    return vol_left+vol_right


def calc_R_sph(vol):
    """
    Calculate the spherical radius of a droplet based on the volume.

    Input parameters
    ----------
    vol:         volume   [μL]
    calibration: calibration factor  [mm**3/pixels**3]

    Returns
    -------
    Corresponding spherical radius.  [mm]

    """
#    calibration=0.00621722846441948**3
    R_sph=(3*vol/(4*np.pi))**(1/3)
    return R_sph

def calc_dB(Ps):
    """
    Calculate the deci Bell (dB) value for each acoustic pressure Ps.

    Input parameters
    ----------
    Ps:     aplitude of the acoustic pressure [Pa]

    Returns
    -------
    dB value. [deci Bell]

    """
    Po=2*10**(-5) #[Pa]

    return 20*np.log10(Ps/Po)     


def azimuth(x, y):
    """
    Calculate the azimuthal angle.

    Input parameters
    ----------
    x, y: Cartesian coordinates of contour of the drop [mm]
    Returns
    -------
    Azimuthal angle in [rad]

    """
    a=np.arctan2(x, y)
    return a


def cart2pol(x, y):
    """
    Converts the cartesial coordinates into polar

    Input parameters
    ----------
    x, y: Cartesian coordinates of contour of the drop [mm, mm]
    Returns
    -------
    ρ, φ: Polar coordinates [mm, rad]

    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    """
    Converts the polar coordinates into cartesial

    Input parameters
    ----------
    ρ, φ: Polar coordinates [mm, rad]
    Returns
    -------
    x, y: Cartesian coordinates of contour of the drop [mm, mm]
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def model_fit(th, Ps):
    """
    Express the model that will be used to fit the experimental data (simple approach).

    Input parameters
    ----------
    th:     azimuth angle theta [rad]
    Ps:     Acoustic pressure [Pa]
    ---------------------------------------
    gamma:  Surface tension [mN/m]
    R_sph:  Equivalent spherical radius [mm]
    Cg_air: Air compressibility (constant) [Pa^-1]
    k_o:    Wave number in air [1/mm]
    
    Returns
    -------
    Expression of fitting model.

    """
#    Ps=calc_ampl(dB)
    ct_new=-((3/(64*gamma))*R_sph**2*Ps**2*Cg_air*(1+((7/5)*(k_o*R_sph)**2)))
    return ct_new*(3*(np.cos(th))**2-1)+R_sph
#################################################################

def ST_predict(th, gamma):
    """
    Prediction of ST. 
    After the acoustic pressure is determined. The ST is calculated based on the equation fo our model. 
    
    Input parameters
    ----------
    th:     azimuth angle theta [rad]
    gamma:  Surface tension [mN/m]
    ---------------------------------------
    Ps:     Acoustic pressure [Pa]
    R_sph:  Equivalent spherical radius [mm]
    Cg_air: Air compressibility (constant) [Pa^-1]
    k_o:    Wave number in air [1/mm]
    
    Returns
    -------
    Expression of fitting model.
    """    
#    Ps=data_Ps[i-1]    #[Pa]
    ct_new=-((3/(64*gamma))*R_sph**2*Ps**2*Cg_air*(1+((7/5)*(k_o*R_sph)**2)))
    return ct_new*(3*(np.cos(th))**2-1)+R_sph


# Print iterations progress
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
#################################################################################

def extract_coord(filename):
     """
     This function extracts all the parameters related to the 
     drop via the cv2 library. 
     
     Input parameters:
     --------
     filename:   image (reads in [pixels])
     
     Output parameters:
     ------
     x, y :       Cartesian coordinates [mm, mm]
     rho, phi:    Polar coordinates [mm, rad]
     theta:       azimuth angle  [rad]
     vol:         Volume of drop [μL]
     h_box:       Height of drop [mm] 
     w_box:       Width of drop [mm]
     R_sph:       Equivalent sperical radius [mm]
     
     """
    data_x=[]
    data_y=[]
    data_xvol=[]
    data_yvol=[]
    # determine the contour
    im = cv2.imread(filename)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(imgray, 100, 200) 
    contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(imgray,filename,(40,40), font, .5,(0,0,200),1,cv2.LINE_AA)
    #cv2.imshow('Canny Edges After Contouring', imgray)
    #cv2.waitKey(10) 
    data=contours[0]
     
    threshold_area=1000
    
    for c in contours:
        area = cv2.contourArea(c)
        # print(area, c[0].shape)
    
        if (area > threshold_area): 
            data=c
            (x, y, w, h) = cv2.boundingRect(c)
            #print(area, c[0].shape, h, w, st)
            
    # cv2.drawContours(im, c, -1, (0, 0, 255), 1) 
    # cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
   # cv2.imshow('Contours', im) 
   # cv2.waitKey(10) 
    
    
    for i in data:
        data_xvol.append(i[0,0]-(x+w/2))
        data_yvol.append(i[0,1]-(y+h/2))
        data_x.append(i[0,0])
        data_y.append(i[0,1])
    
    h_box=h*cal
    w_box=w*cal
    
    
    xvol=np.asarray(data_xvol)*cal
    yvol=np.asarray(data_yvol)*cal
    
    x=np.asarray(data_x)*cal
    y=np.asarray(data_y)*cal
    [rho, phi]=cart2pol(xvol, yvol)
    # phi=phi+np.pi
    theta=azimuth(xvol, yvol)
    vol=calc_volume(xvol, yvol)
    
    R_sph=calc_R_sph(vol)
    
    return (x, y, rho, phi, theta, vol, h_box, w_box, R_sph)

def log_interp1d(xx, yy, kind='linear'):            #Used for the ST calculation
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx
