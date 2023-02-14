#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 16:17:49 2022

@author: mirfan
"""

import numpy as np
import healpy as hp
import math
from scipy import stats
import matplotlib.pyplot as plt

def get_im(fullsky, rotarr, widths, resoarcmin):
    
    map1 = hp.visufunc.gnomview(fullsky, coord='G', rot=rotarr, 
                                xsize=widths, ysize=widths, 
                                reso=resoarcmin, flip='astro', 
                                return_projected_map=True)
    plt.close()
    
    return map1[::-1]

def get_arrays(lon1, lon2, lon3, lon4, lat1, lat2, lat3, lat4):
    """ provide arrays for box projection """
    
    tcorn = np.array([math.radians(90. - np.array(lat1)), math.radians(90. - np.array(lat2)), \
            math.radians(90. - np.array(lat3)), math.radians(90. - np.array(lat4))])
    pcorn = np.array([math.radians(np.array(lon1)), math.radians(np.array(lon2)), \
            math.radians(np.array(lon3)), math.radians(np.array(lon4))])
            
    tarray =np.array([np.linspace(tcorn[0], tcorn[1], 100), np.linspace(tcorn[1], tcorn[2], 100), \
            np.linspace(tcorn[2], tcorn[3], 100), np.linspace(tcorn[3], tcorn[0], 100)])
    parray =np.array([np.linspace(pcorn[0], pcorn[1], 100), np.linspace(pcorn[1], pcorn[2], 100), \
            np.linspace(pcorn[2], pcorn[3], 100), np.linspace(pcorn[3], pcorn[0], 100)])
    
    return tarray.flatten(), parray.flatten()

fwhm = 56./60. # alternate between 56 arcmin and 5 degrees, unit = degrees
nside = 512

#haslam
hmap = hp.fitsfunc.read_map('haslam408_ds_Remazeilles2014.fits', field=0, nest=False)
hmap = hmap - 8.9
hmap = hp.sphtfunc.smoothing(hmap, fwhm=np.radians(np.sqrt(fwhm**2 - (56./60.)**2))) 
hmap = hp.pixelfunc.ud_grade(hmap, nside)
fhas = 408.

lmap1 = hp.fitsfunc.read_map('lambda_chipass_healpix_r10.fits', field=0, nest=False)
lmap1 = lmap1/1000. #need to put in K
lmap1[np.where(lmap1 < 0.0)[0]] = hp.UNSEEN
lmap1 = hp.sphtfunc.smoothing(lmap1, fwhm=np.radians(np.sqrt(fwhm**2 - (14.4/60.)**2)))
lmap1 = hp.pixelfunc.ud_grade(lmap1, nside)
frec1 = 1400.

#get zero-level from North Polar Spur
coords = []
coords.append(hp.pixelfunc.ang2vec(math.radians(90. - np.array(15.)), math.radians(np.array(41.)))) #lat then lon
coords.append(hp.pixelfunc.ang2vec(math.radians(90. - np.array(15.)), math.radians(np.array(25.))))
coords.append(hp.pixelfunc.ang2vec(math.radians(90. - np.array(40)), math.radians(np.array(25.))))
coords.append(hp.pixelfunc.ang2vec(math.radians(90. - np.array(40)), math.radians(np.array(41.))))
coords = np.array(coords)
tarea = hp.query_polygon(nside, coords, inclusive=True, nest=False)

result = stats.linregress(hmap[tarea],lmap1[tarea])
slop1 = result.slope
int1 = result.intercept
ierr = result.intercept_stderr

beta1 = np.log((lmap1-int1)/hmap) / np.log(frec1/fhas)

reso_arcmin = hp.nside2resol(nside, arcmin=True)
widt = 64 #64 is 7.3 deg

#lon
firp = np.array([28, 32, 36, 40,\
                       28, 32, 36, 40, \
                           28, 32, 36, 40, 
                               28, 32, 36, 
                                   28, 32, 36, 
                                       32, 
                                           28, 32, 
                                               14, 18, 21, 
                                                   10, 14, 18, 
                                                       10, 14, 18, 
                                                           8, 12, 16, 
                                                               5, 9, 13, 
                                                                   5, 9, 13])
#lat
secp = np.array([18, 18, 18, 18,
                     22, 22, 22, 22, 
                         26, 26, 26, 26, 
                             30, 30, 30, 
                                 34, 34, 34, 
                                     38, 
                                         42, 42, 
                                             46, 46, 46, 
                                                 50, 50, 50, 
                                                     54, 54, 54, 
                                                         58, 58, 58, 
                                                             62, 62, 62, 
                                                                 66, 66, 66])

im1 = get_im(beta1, [20,30,0], widt, reso_arcmin)
xdimen = np.shape(im1)[0]
ydimen = np.shape(im1)[1]
alldat = np.zeros((len(secp), xdimen, ydimen, 1))
for ii in range(len(secp)):
    im = np.zeros((xdimen, ydimen, 1))
    im[:,:,0] = get_im(beta1, [firp[ii], secp[ii]], widt, reso_arcmin).data
    alldat[ii, :, :, :] = im

# alternate between 56 arcmin and 5 degrees
np.save('alldat56arcmin.npy', alldat)






