#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:44:34 2024

@author: lauracarlton
"""

import numpy as np 
from pysnirf2 import Snirf
from scipy.io import loadmat
import motionArtefactStudy.syntheticHRF_func as synHRF
import imageRecon.ImageRecon as IR
import os
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import pandas as pd 
import random 
from scipy import signal 
from scipy import stats
from sklearn.metrics import mean_squared_error 

from cedalion import units
import cedalion.xrutils as xrutils

import numpy as np
import xarray as xr
import pint
import matplotlib.pyplot as plt
import scipy.signal as signal
import os.path

from tqdm import tqdm 


#%%
def generateHRF(trange, dt, stimDur, paramsBasis = [0.1000,   3.0000,    1.8000,    3.0000], nConc=2, scale = [10, -4]):
    """
    Function to generate HRF basis functions 
    - single gamma function for HbO and HbR
    -> nConc = 2 , number of concentrations, HbO+HbR
    -> paramsBasis = parameters for tau and sigma for the modified gamma function for HbO (1:2) and HbR (3:4) (list of 4 floats)
    -> dt = sampling period
    -> stimDur = duration of stimulus 
    -> trange = duration of HRF, (list of two floats ie, [0, 5])
    -> scale = how much to scale each HRF, [HbO, HbR]
    """

    nPre = np.round(trange[0]/dt)
    nPost_gamma = int(np.round(10/dt))
    
    nPost_stim = int(np.round(trange[1]/dt))
    
    tHRF_gamma = np.arange(nPre*dt, nPost_gamma*dt, dt).T
    stimulus = np.zeros([nPost_stim,nConc])
    
    boxcar = np.zeros(len(tHRF_gamma))
    boxcar[:int(stimDur/dt)] = 1
    
    
    for iConc in range(nConc): 
        tau = paramsBasis[iConc*2]
        sigma = paramsBasis[iConc*2+1]
        
        gamma = (np.exp(1)*(tHRF_gamma-tau)**2/sigma**2) * np.exp( -(tHRF_gamma-tau)**2/sigma**2 );
        lstNeg = np.where(tHRF_gamma<0);
        gamma[lstNeg] = 0;
        
        if tHRF_gamma[0]<tau:
            gamma[0:int(np.round((tau-tHRF_gamma[0])/dt))] = 0;
            
        stimulus[:,iConc] = signal.convolve(boxcar, gamma, mode='full') #[:nPost_stim]
        stimulus[:,iConc] = stimulus[:,iConc] / np.max(abs(stimulus[:,iConc])) * scale[iConc] * 1e-6

    tHRF_stim = np.arange(nPre*dt, nPost_stim*dt, dt).T
    
    tbasis = xr.DataArray(stimulus, dims = ["time", "chromo"], coords={"chromo": ['HbO','HbR'],
                                                                         "time": tHRF_stim }).T
    tbasis = tbasis.assign_coords(samples=('time', np.arange(len(tHRF_stim))))
    tbasis = tbasis.pint.quantify("molar")
    
    return tbasis

#%%
def addHRF_todOD(od, HRF, trange, min_interval = 5, max_interval = 10, numStims = 15):
    """
    add HRF to dOD in channel space
    - added randomly with a random ISI between min_interval and max_interval
    dOD -> OD timeseries data, shape = [length(time), 2] (since there are two wavelengths)
    HRF -> HRF timeseries, shape = [#time points, 2] (again, two wavelengths)
    min_interval = minimum ISI
    max_interval = maximum ISI
    numStims = number of stimuli, ie number of HRFs to be added to dOD 
    """

    current_time = 0
    onset_idxs = []
    onset_times = []
    try:
        nTpts_hrf = len(HRF['time'])
        HRF = HRF.pint.dequantify().values
    except:
        nTpts_hrf = HRF.shape[0]
        HRF = np.reshape(HRF,[HRF.shape[1]//2,2,HRF.shape[0]])
        
    nTpts_data = len(od['time'])
    od_wHRF = od.pint.dequantify().copy()
    time = od_wHRF['time']

    # define the onset times and add HRF at this time
    for stim in range(numStims):
        
        # randomly choose ISI
        interval = random.uniform(min_interval, max_interval)
    
        # if not the first stim, add the duration of the previous stim to current time
        if stim  > 0 :
            current_time += trange[1]
            
        # add the duration of the ISI
        current_time += interval
        
        # get the index correpsonding to the current time
        onset_idx = (np.abs(time-current_time)).argmin()
        
        # if the end of the stimulus goes past the length of the data, don't include it
        if onset_idx + nTpts_hrf > nTpts_data:
            break
        
        # save the onset index and onset_time
        onset_idxs.append(int(onset_idx.values))
        onset_times.append(current_time)        
        
        # add the HRF at this onset index 
        od_wHRF[:, :, int(onset_idx):int(onset_idx+nTpts_hrf)] += HRF

    
    return od_wHRF.pint.quantify(), onset_idxs, onset_times


#%%
def getConnectedVertices(seed, lstV, vertices, faces, dist_thresh=10 ):
    """
    Get connected blob of vertices centered at the seed
    Finds all vertices within dist_thresh=10mm of the seed
    For each of those vertices it finds list of all vertices in adjoining faces since those are also connected 
    Remove duplicates 
    """

    # find all vertices within 10mm of the seed 
    blobV = []
    connectivityDict = {}    


    for v in lstV:
        
        dist = np.linalg.norm(vertices[v,:] - vertices[seed,:])
        if dist < dist_thresh:
            blobV.append(v)
            connectedVertices = []
            
            
            for face in faces:  # LOOP THROUGH ALL THE FACES
                if v in face:   # IF THE VERTEX IS IN THE FACE ADD THOSE VERTICES TO THE LIST OF CONNECTED VERTICES 
                    connectedVertices.extend(face) 
            
            connectedVertices = np.unique(np.array(connectedVertices)) # REMOVE DUPLICATES - GIVES LIST OF ALL VERTICES CONNECTED TO V 
            connectivityDict[str(v)] = connectedVertices
            
            
    blobV = np.unique(np.array(blobV))  # REMOVE DUPLICATES
    
    return blobV, connectivityDict

def diffusionOperator(seed, lstV, vertices, faces, nIterations = 15, nV = 20004):
    """
    calls getConnectedVertices to get blob of activation
    diffusion operator then sets the amplitude of the seed to 1
    then loops over nIterations 
        for each vertex that is not the seed the amplitude is set to the average of all of its connected vertices
    """
    
    blobV, connectivityDict = getConnectedVertices(seed, lstV, vertices, faces, dist_thresh=10 )
    
    diffusionImg = np.zeros(nV)
    
    for n in range(nIterations):
        diffusionImg[seed] = 1
        
        for ii in blobV:
            
            if ii != seed :
                connectedV = connectivityDict[str(ii)]
                
                diffusionImg[ii] = np.mean(diffusionImg[connectedV])

    return blobV, diffusionImg



#%%
def addHRF_toVertices(lstVertices, tbasis, scale = None, nV = 20004):
    """
    Add HbO and HbR to select vertices in image space 
    -> lstVertices = list of vertices to add HbO and HbR to 
    -> tbasis = HRFs for HbO and HbR, must be shape [# time points x 2] where HbO corresponds to [:,0] and HbR corresponds to [:,1]
    -> scale amplitude by the magnitude of scale [nV,1] for each vertex
    -> nV = total number of vertices                                            
    """   
    nTpts = len(tbasis['time'])
    HbO_RealImage = np.zeros([nTpts, nV])
    HbR_RealImage = np.zeros([nTpts, nV])

    # set the image of the vertex to be the HRF
    HbO = tbasis.sel({'chromo' : 'HbO'})
    HbR = tbasis.sel({'chromo' : 'HbR'})
    HbO_RealImage[:,lstVertices] = (HbO.pint.dequantify().values * np.ones([len(lstVertices), nTpts])).T
    HbR_RealImage[:,lstVertices] = (HbR.pint.dequantify().values * np.ones([len(lstVertices), nTpts])).T
    
    if scale is not None:
        HbO_RealImage =  HbO_RealImage*scale
        HbR_RealImage =  HbR_RealImage*scale
    
    HRF_RealImage = np.hstack([HbO_RealImage, HbR_RealImage])
    
    return HRF_RealImage

#%%


def conc2OD(conc, E, ppf=1):
    
    
    OD = xr.dot(E, conc*ppf*units.mm, dims='chromo')
    return OD


def OD2conc(od, E, ppf=1):
    
    Einv = xrutils.pinv(E)
    nChan = int(od.shape[1]/2)
    conc = np.zeros(od.shape)
    # conc =  Einv.values @ od
    # conc = xr.dot(Einv, od / (ppf*1*units.mm), dims=["wavelength"])

    for ii in range(nChan):
        jj = ii+nChan

        conc[:,[ii,jj]] = ( Einv.values @ od[:,[ii, jj]].T).T


    return conc

#%%

def timeseriesChannel2Image(channels, pAinv):
    """
    project channel timeseries into image space 
    channels -> timeseries in channel space 
    Adot -> sensitivity matrix for brain 
    wavelengths -> wavelengths use during acquisition 
    Adot_scalp -> optional; sensitivity matrix for scalp
    """
    nTpts = channels.shape[0]
    nV = pAinv.shape[0]
    # ind = np.hstack([blobV, np.array(blobV)+nV])
    # nV_blob = len(blobV)
    # pAinv_blob = pAinv[ind,:]
    avg_img = np.zeros([nTpts, nV])
    
    for t in range(nTpts):

        img =  pAinv @ channels[t,:].T  # Custo2010NI; BoasDale2005; typical alpha = 0.01

        avg_img[t,:] = img
    
    
    return avg_img


def timeseriesImage2Channel(image, Adot, wavelengths):
    """
    project image timeseries into channel space
    image -> timeseries in image space
    Adot -> sensitivity matrix for brain
    wavelengths -> wavelengths use during acquisition 
    """
    AmatrixDict = IR.generateAmatrix(Adot, wavelengths)
    Amatrix = AmatrixDict['Amatrix']
    nTpts = image.shape[1]
    nChan = Adot.shape[0]*2
    chanSpace = np.zeros([nTpts,nChan])
    
    # make representation in channel space at each time point 
    for t in range(nTpts):
        chanSpace[t,:] = Amatrix@image[:,t]
    
    return chanSpace

#%%

def blockAverage(data, stim):
    """
    block average the data across all the stimuli 
    stim_df -> onsets of all the stimuli
    data -> data being averaged
    
    returns blockaverage -> contains average across both wavelengths/concentrations
    """
    epochs = data.cd.to_epochs(stim,
                                      [1],
                                      before = 2,
                                      after = 18)
    
    baseline = epochs.sel(reltime=(epochs.reltime < 0)).mean("reltime")
    epochs_blcorrected = epochs - baseline
  
    blockaverage = epochs_blcorrected.groupby('trial_type').mean('epoch')
    
    # blockaverage = blockaverage.stack({'measurement' : ['channel', 'chromo']})
    blockaverage = blockaverage.transpose('reltime', 'channel', 'chromo', 'trial_type')
  
    return blockaverage

#%%

def getStats(truth, blockAverage):
    
        
    R_HbO = stats.pearsonr(truth[:,0], blockAverage[:,0])
    R_HbR = stats.pearsonr(truth[:,1], blockAverage[:,1])
    
    
    MSE_HbO = mean_squared_error(truth[:,0], blockAverage[:,0])
    MSE_HbR = mean_squared_error(truth[:,1], blockAverage[:,1])

    statDict = {}
    statDict['R_HbO'] = R_HbO
    statDict['R_HbR'] = R_HbR
    statDict['MSE_HbO'] = MSE_HbO
    statDict['MSE_HbR'] = MSE_HbR
    
    return statDict


