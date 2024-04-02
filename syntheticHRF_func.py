#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:58:17 2023

@author: lauracarlton
"""

#---------------------------------------------------
#%% IMPORT MODULES
#---------------------------------------------------
import numpy as np 
from pysnirf2 import Snirf
from scipy.io import loadmat
import imageRecon.ImageRecon as IR
from mayavi import mlab
import os
import matplotlib.pyplot as plt 
import random 
from tqdm import tqdm 
from scipy import signal 
from scipy import stats
from sklearn.metrics import mean_squared_error 

#---------------------------------------------------
#%% DEFINE USEFUL FUNCTIONS
#---------------------------------------------------

def generateHRF(trange, dt, stimDur, paramsBasis = [0.1000,   3.0000,    1.8000,    3.0000], nConc=2, scale = [10, -4]):
    """
    Function to generate HRF basis functions 
    - single gamma function for HbO and HbR
    -> nConc = 2 , number of concentrations, HbO+HbR
    -> paramsBasis = parameters for tau and sigma for the modified gamma function for HbO (1:2) and HbR (3:4) (list of 4 floats)
    -> dt = sampling period
    -> trange = duration of stimulus, (list of two floats ie, [0, 5])
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
        
        # gamma = gamma*scale[iConc
    
        stimulus[:,iConc] = signal.convolve(boxcar, gamma, mode='full')[:nPost_stim]
        stimulus[:,iConc] = stimulus[:,iConc] / np.max(abs(stimulus[:,iConc])) * scale[iConc] * 1e-6

    tHRF_stim = np.arange(nPre*dt, nPost_stim*dt, dt).T

    return tHRF_stim, stimulus



def getConnectedVertices(seed, lstV, vertices, faces, dist_thresh=10 ):
    
    # GET THE CONNECTED BLOB 
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

def getML(measurementList):
    
    nMeas = len(measurementList) 
    ml = np.zeros([nMeas, 3])
    for ii,m in enumerate(measurementList):
    
        ml[ii,0] = int(m.sourceIndex - 1)
        ml[ii,1] = int(m.detectorIndex - 1)
        ml[ii,2] = int(m.wavelengthIndex - 1)

    return ml 


def OD2Conc(dOD, snirfObj, ppf=1):
    
    probe = snirfObj.nirs[0].probe
    E = IR.GetExtinctions(probe.wavelengths)
    E = E[:, 0:2]/10
    Einv = np.linalg.pinv( E.T @ E ) @ E.T
    nChan = dOD.shape[1]//2
    conc = np.zeros(dOD.shape)
    
    ml = getML(snirfObj.nirs[0].data[0].measurementList)
    
    for ii in range(nChan):
        jj = ii+nChan

        if ppf != 1:
            
            # this code is incorrect right now - does not take into account what happens if you only pass in one channel
            srcInd = int(ml[ii,0])
            detInd = int(ml[ii,1])
            srcPos =probe.sourcePos3D[srcInd,:]
            detPos = probe.detectorPos3D[detInd,:]
        
            rho = np.linalg.norm(srcPos-detPos)
            conc[:,[ii,jj]] = ( Einv @ (dOD[:,[ii, jj]] * rho/ppf).T).T
        else:
            conc[:,[ii,jj]] = ( Einv @ dOD[:,[ii, jj]].T).T


    return conc

def conc2OD(conc, snirfObj,ppf=1):
    
    probe = snirfObj.nirs[0].probe
    
    E = IR.GetExtinctions(probe.wavelengths);
    E = E[:,0:2] / 10; # convert from /cm to /mm
    nChan = conc.shape[1]//2
    dOD = np.zeros(conc.shape)

    ml = getML(snirfObj.nirs[0].data[0].measurementList)
    
    for ii in range(nChan):
            
        jj = ii + nChan
        if ppf != 1:
            
            # this code is incorrect right now - does not take into account what happens if you only pass in one channel
            srcInd = int(ml[ii,0])
            detInd = int(ml[ii,1])
            srcPos =probe.sourcePos3D[srcInd,:]
            detPos = probe.detectorPos3D[detInd,:]
        
            rho = np.linalg.norm(srcPos-detPos)
    
            dOD[:,[ii,jj]] = E @ (conc[:,[ii,jj]] * rho*ppf)
        
        else:
            dOD[:,[ii,jj]] = (E @ conc[:,[ii,jj]].T).T
        
    return dOD


def diffusionOperator(seed, lstV, vertices, faces, nIterations = 15, nV = 20004):
    
    blobV, connectivityDict = getConnectedVertices(seed, lstV, vertices, faces, dist_thresh=10 )
    
    diffusionImg = np.zeros(nV)
    
    # localImg = np.zeros(len(lstV))
    for n in range(nIterations):
        diffusionImg[seed] = 1
        
        for ii in blobV:
            
            if ii != seed :
                connectedV = connectivityDict[str(ii)]
                
                diffusionImg[ii] = np.mean(diffusionImg[connectedV])

    return blobV, diffusionImg




def addHRF_toVertices(lstVertices, tbasis, scale = None, nV = 20004):
    """
    Add HbO and HbR to select vertices in image space 
    -> lstVertices = list of vertices to add HbO and HbR to 
    -> tbasis = HRFs for HbO and HbR, must be shape [# time points x 2] where HbO corresponds to [:,0] and HbR corresponds to [:,1]
    -> nV = total number of vertices                                            
    """   
    nTpts = tbasis.shape[0]
    HbO_RealImage = np.zeros([nTpts, nV])
    HbR_RealImage = np.zeros([nTpts, nV])

    # set the image of the vertex to be the HRF
    HbO = tbasis[:,0]
    HbR = tbasis[:,1]
    HbO_RealImage[:,lstVertices] = (HbO * np.ones([len(lstVertices), nTpts])).T
    HbR_RealImage[:,lstVertices] = (HbR * np.ones([len(lstVertices), nTpts])).T
    
    if scale is not None:
        HbO_RealImage =  HbO_RealImage*scale
        HbR_RealImage =  HbR_RealImage*scale
    
    HRF_RealImage = np.hstack([HbO_RealImage, HbR_RealImage])
    
    return HRF_RealImage




def image2channel(image, Adot, wavelengths):
    """
    project from image space to channel space 
    """
    
    Amatrix = IR.generateAmatrix(Adot, wavelengths)

    chanSpace = Amatrix@image.T
    
    return chanSpace


def timeseriesImage2Channel(image, Adot, wavelengths):
    """
    project image timeseries into channel space
    image -> timeseries in image space
    Adot -> sensitivity matrix for brain
    wavelengths -> wavelengths use during acquisition 
    """
    Amatrix = IR.generateAmatrix(Adot, wavelengths)
    nTpts = image.shape[1]
    nChan = Adot.shape[0]*2
    chanSpace = np.zeros([nTpts,nChan])
    
    # make representation in channel space at each time point 
    for t in range(nTpts):
        chanSpace[t,:] = Amatrix@image[:,t]
    
    return chanSpace



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


def addHRF_todOD(dOD, HRF, trange, time, min_interval = 5, max_interval = 10, numStims = 15):
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
    onsetIdxs = []
    onset_times = []
    nTpts_hrf = HRF.shape[0]
    nTpts_data = dOD.shape[0]
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
        onsetIdx = (np.abs(time-current_time)).argmin()
        
        # if the end of the stimulus goes past the length of the data, don't include it
        if onsetIdx + nTpts_hrf > nTpts_data:
            continue
        
        # save the onset index and onset_time
        onsetIdxs.append(onsetIdx)
        onset_times.append(current_time)        
        
        # add the HRF at this onset index 
        dOD[onsetIdx:onsetIdx+nTpts_hrf,:] += HRF

    
    stimDict = {'dOD': dOD, 'onsets':onsetIdxs, 'onset_times':onset_times}
    return stimDict



def filterData(data, fq, fcut_min=0.01, fcut_max=0.5):
    """
    bandpass filter the data
    """    

    b,a = signal.butter(3,[fcut_min*(2/fq), fcut_max*(2/fq)], btype='bandpass')
    
    data_filt = signal.filtfilt(b,a,data.T).T
    
    return data_filt



def blockAverage(data, onsetIdx, trange, dt):
    """
    block average the data across all the stimuli 
    onsetIdx -> onsets of all the stimuli
    nTpts -> duration of the stimuli
    data -> data being averaged
    
    returns avg -> contains average across both wavelengths/concentrations
    """

    avg = 0
    numStims = len(onsetIdx)
    nTpts = int((trange[1]-trange[0]) / dt)
    
    
    for onset in onsetIdx:
                
        # add the timecourses together 
        start = int(onset + trange[0] / dt)
        stop = start + nTpts
        avg += data[start:stop,:]

    avg = avg/numStims 
    
    return avg


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
    


def plot_blockAvg(trueHRF, avgHRF, tHRF, ax, title=None):
    
    ax.plot(tHRF, avgHRF[:,0], 'r')
    ax.plot(tHRF, avgHRF[:,1], 'b')
    ax.plot(tHRF, trueHRF[:,0], '--r')
    ax.plot(tHRF, trueHRF[:,1], '--b')
    if title is not None:
        ax.set_title(title)
    

def plot_timeseries(time, timeseries, ax, stims = None, title=None):
    
    ax.plot(time, timeseries[:,0], 'r')
    ax.plot(time, timeseries[:,1], 'b')
    
    if stims is not None:
        ax.vlines(stims, color='k', linestyle='dashed',  ymin=min(timeseries[:,0]), ymax=max(timeseries[:,0]))
        
    if title is not None:
        ax.set_title(title)



    
    
