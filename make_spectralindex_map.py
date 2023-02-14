#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 09:43:32 2022

@author: mirfan
#credit for get_all_faces, put_all_faces, rotate_map functions: Florent Sureau
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from healpy.pixelfunc import pix2vec, vec2ang, get_interp_weights
from healpy.rotator import euler_matrix_new
from healpy.pixelfunc import npix2nside

from model_maker import generate_model

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import math

CST = {"kbolt": 1.3806488e-23, "light": 2.99792458e8, "plancks": 6.626e-34, "cmb_temp": 2.73}

def get_all_faces(Imag,nested=False, newn=128):
   npix = np.shape(Imag)[0]
   numfaces = int(npix / (newn*newn))
   taille_face = np.int32(npix/numfaces)
   CubeFace = np.zeros((numfaces,newn,newn))

   if (nested!=True):
       NewIm = hp.reorder(Imag, r2n = True)
   else :
       NewIm = Imag
   index = np.zeros((newn,newn))
   index = np.array([hp.xyf2pix(newn,x,y,0,True) for x in range(newn) 
                                                         for y in range(newn)])
   for face in range(numfaces):
       CubeFace[face,:,:] = np.resize(NewIm[(index+taille_face*face).astype('int')],(newn,newn))

   return CubeFace

def put_all_faces(CubeFace,nested=False):
   numfaces = np.shape(CubeFace)[0]
   npix = np.shape(CubeFace)[0] * np.shape(CubeFace)[1] * np.shape(CubeFace)[2]
   nside = np.shape(CubeFace)[1]
   taille_face = np.int32(npix/numfaces)
   cote = np.int32(math.sqrt(taille_face))
   Imag = np.zeros((npix))
   index = np.zeros((cote,cote))
   index=np.array([hp.xyf2pix(nside,x,y,0,True) for x in range(nside) 
                                                         for y in range(nside)])
   for face in range(numfaces):
       Imag[index+taille_face*face] = np.resize(CubeFace[face,:,:],(cote*cote))

   if (nested!=True):
       NewIm = hp.reorder(Imag, n2r = True)
   else:
       NewIm = Imag

   return NewIm

def rotate_map(Imag,a1,a2,a3,X=True,Y=False,ZYX=False,deg=False,nested=False):

    npix = np.shape(Imag)[0]

    nside = npix2nside(npix)

    indices = np.arange(0,npix)

    ang_coord = pix2vec(nside, indices,nested)

    ang_coord_array = np.vstack((ang_coord[0],ang_coord[1],ang_coord[2]))

    eul = euler_matrix_new(a1,a2,a3,X=X,Y=Y,ZYX=ZYX,deg=deg)

    new_coord = np.dot(eul,ang_coord_array)

    theta_arr,phi_arr = vec2ang(new_coord.T)

    neigh,weigh = get_interp_weights(nside,theta_arr,phi=phi_arr,nest=nested)

    thr_val = 1e-8

    weigh[np.where(np.abs(weigh)<thr_val)] = 0

    weigh = weigh/np.sum(weigh,axis=0)

    rotIm = np.zeros_like(Imag)

    for k in range(neigh.shape[0]):

        rotIm = rotIm+weigh[k]*Imag[neigh[k]]

    return rotIm


def planckcorr(freq_ghz):
    """ Takes in frequency in GHZ and produces factor to be applied to temp """

    freq = freq_ghz * 10.**9.
    factor = CST["plancks"] * freq / (CST["kbolt"] * CST["cmb_temp"])
    correction = (np.exp(factor)-1.)**2. / (factor**2. * np.exp(factor))

    return correction

def get_im(fullsky, rotarr, widths, resoarcmin):
    
    map1 = hp.visufunc.gnomview(fullsky, coord='G', rot=rotarr, 
                                xsize=widths, ysize=widths, 
                                reso=resoarcmin, flip='astro', 
                                return_projected_map=True)
    plt.close()
    
    return map1[::-1]

#import data 
alldat_old = np.load('alldat5deg.npy')
alldat2_old = np.load('alldat56arcmin.npy')

aa, bb, cc, dd = np.where(np.isnan(alldat2_old))
listi = np.arange(0, np.shape(alldat_old)[0])
fine = np.setdiff1d(listi, aa)

alldat_old = alldat_old[fine,:,:,:]
alldat2_old = alldat2_old[fine,:,:,:]

#going to rotate images so we have 4 times as much data
oldb = np.shape(alldat_old)[0]
xdimen = np.shape(alldat_old)[1]
ydimen = np.shape(alldat_old)[2]
alldat = np.zeros((oldb*4, xdimen, ydimen, 1))
alldat2 = np.zeros((oldb*4, xdimen, ydimen, 1))
bas = 0
for oo in range(oldb):
    alldat[bas, :, :, 0] = np.rot90(alldat_old[oo,:,:,0], k=0)
    alldat[bas+1, :, :, 0] = np.rot90(alldat_old[oo,:,:,0], k=1)
    alldat[bas+2, :, :, 0] = np.rot90(alldat_old[oo,:,:,0], k=2)
    alldat[bas+3, :, :, 0] = np.rot90(alldat_old[oo,:,:,0], k=3)
    
    alldat2[bas, :, :, 0] = np.rot90(alldat2_old[oo,:,:,0], k=0)
    alldat2[bas+1, :, :, 0] = np.rot90(alldat2_old[oo,:,:,0], k=1)
    alldat2[bas+2, :, :, 0] = np.rot90(alldat2_old[oo,:,:,0], k=2)
    alldat2[bas+3, :, :, 0] = np.rot90(alldat2_old[oo,:,:,0], k=3)
    bas += 4
    
# Rescaling code from https://github.com/ai4cmb/ForSE
def rescale_min_max(img, a=-1, b=1, return_min_max=False):
    img_resc = (b-a)*(img-np.min(img))/(np.max(img)-np.min(img))+a
    if return_min_max:
        return img_resc, np.min(img), np.max(img)
    else:
        return img_resc

#We want 75 percent to be for testing 
batchno = int(0.75 * np.shape(alldat)[0])
testno = int(0.25 * np.shape(alldat)[0])

if batchno + testno !=  np.shape(alldat)[0]: 
    print('Nope')
    import pdb
    pdb.set_trace()

X_train = np.zeros((batchno, xdimen, ydimen, 1))
y_train = np.zeros((batchno, xdimen, ydimen, 1))
for bb in range(batchno):
    X_train[bb, :, :, 0] = rescale_min_max(alldat[bb, :, :, 0], \
                            a=-1, b=1, return_min_max=False)
    y_train[bb, :, :, 0] = rescale_min_max(alldat2[bb, :, :, 0], \
                            a=-1, b=1, return_min_max=False)

X_test = np.zeros((testno, xdimen, ydimen, 1))
y_test = np.zeros((testno, xdimen, ydimen, 1))
pbb = 0
for bb in range(batchno, batchno+testno):
    X_test[pbb,:,:,0] = rescale_min_max(alldat[bb, :, :, 0], \
                            a=-1, b=1, return_min_max=False)
    y_test[pbb,:,:,0] = rescale_min_max(alldat2[bb, :, :, 0], \
                            a=-1, b=1, return_min_max=False)
    pbb += 1

# =============================================================================
# Train the U-NET
# =============================================================================
    
model = generate_model([xdimen, ydimen, 1])

print(model.summary())

model.compile(optimizer=keras.optimizers.adam_v2.Adam(learning_rate=0.0001), \
                  loss='mse')

earlyStopping = EarlyStopping(monitor='loss', patience=10, verbose=0, \
                              mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, \
                                monitor='loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=7, \
                                        verbose=1, min_delta=1e-4, mode='min')
    
model.fit(X_train, y_train, batch_size=6, epochs=250, verbose=1, \
          callbacks=[earlyStopping, mcp_save, reduce_lr_loss])
   
# =============================================================================
# =============================================================================
        
pre_gen_imgs = model.predict(X_train)
pre_gen_imgs_test = model.predict(X_test)

gen_imgs = np.zeros_like(pre_gen_imgs)
gen_imgs_test = np.zeros_like(pre_gen_imgs_test)

for bb in range(np.shape(pre_gen_imgs)[0]):
    gen_imgs[bb,:,:,0] = rescale_min_max(pre_gen_imgs[bb, :, :, 0], \
                            a=-1, b=1, return_min_max=False)
for bb in range(np.shape(pre_gen_imgs_test)[0]):
    gen_imgs_test[bb,:,:,0] = rescale_min_max(pre_gen_imgs_test[bb, :, :, 0], \
                            a=-1, b=1, return_min_max=False)


#read MAMD synchrotron spectral index map
s217loc = 'COM_SimMap_synchrotron-ffp10-skyinbands-217_2048_R3.00_full.fits'
s353loc = 'COM_SimMap_synchrotron-ffp10-skyinbands-353_2048_R3.00_full.fits'
sync217 = hp.fitsfunc.read_map(s217loc, field=0, nest=False) / planckcorr(217)
sync353 = hp.fitsfunc.read_map(s353loc, field=0, nest=False) / planckcorr(353)
syncind = np.log(sync353/ sync217) / np.log(353./217.)
nside = 512
npix = 12 * nside * nside
syncind = hp.pixelfunc.ud_grade(syncind, nside)

# Cycle Spinning
#rotations 
Ncycles = 13
rot=np.zeros((Ncycles,3))
rot[0]=[0,0,0]
rot[1]=[math.pi/4.0,0,0]
rot[2]=[0,math.pi/2.0,0]
rot[3]=[-math.pi/4.0,0,0]
rot[4]=[0,-math.pi/2.0,0]
rot[5]=[math.pi/4.,math.pi/2.0,0]
rot[6]=[math.pi/4.,-math.pi/2.0,0]
rot[7]=[-math.pi/4.,math.pi/2.0,0]
rot[8]=[-math.pi/4.,-math.pi/2.0,0]
rot[9]=[math.pi/2.,math.pi/2.0,0]
rot[10]=[math.pi/2.,-math.pi/2.0,0]
rot[11]=[-math.pi/2.,math.pi/2.0,0]
rot[12]=[-math.pi/2.,-math.pi/2.0,0]

themaps = np.zeros((Ncycles, npix))
nested = False

for k in range(Ncycles):
    
    print("Process {0}".format(k))

    large = rotate_map(syncind, rot[k][0], rot[k][1], rot[k][2], \
                       X=True, nested=nested)

    #divide the syncind map into 64 by 64 patches
    thefaces = get_all_faces(large, nested=False, newn=64)
    numfaces = np.shape(thefaces)[0]
    xsides = np.shape(thefaces)[1]
    profaces = thefaces.reshape(numfaces, xsides, xsides, 1)

    #now you need to scale
    scaled_data = np.zeros((numfaces, xsides, xsides, 1))
    for bb in range(numfaces):
        scaled_data[bb, :, :, 0] = rescale_min_max(profaces[bb, :, :, 0], \
                                                   a=-1, b=1, return_min_max=False)

    #make prediction
    pre_small_scales = model.predict(scaled_data)
    small_scales = np.zeros_like(pre_small_scales)
    for bb in range(np.shape(pre_small_scales)[0]):
        small_scales[bb,:,:,0] = rescale_min_max(pre_small_scales[bb, :, :, 0], \
                                a=-1, b=1, return_min_max=False)

    # renormalise and get back full pic
    ss_normed = np.zeros((numfaces, xsides, xsides, 1))
    for bb in range(numfaces):
        ss_normed[bb, :, :, 0] = (small_scales[bb,:,:,0]/np.std(small_scales[bb,:,:,0])*np.std(profaces[bb,:,:,0]))
        ss_normed[bb, :, :, 0] = (ss_normed[bb,:,:,0]-np.mean(ss_normed[bb,:,:,0])+np.mean(profaces[bb,:,:,0]))

    
    backmap = put_all_faces(ss_normed[:,:,:,0], nested=False)
    putback = rotate_map(backmap, -rot[k][2], -rot[k][1], -rot[k][0], \
                         X=True, nested=nested)
        
    themaps[k, :] = putback
    
backmapAll = np.mean(themaps, axis=0)

#smooth from pixel resolution to 56 arcmin
mapres = hp.nside2resol(nside, arcmin=True)
smoobeam = np.sqrt((55./60.0)**2 - (mapres/60.0)**2)
spec_ind_map_56arcmin = hp.sphtfunc.smoothing(backmapAll, fwhm=np.radians(smoobeam))
