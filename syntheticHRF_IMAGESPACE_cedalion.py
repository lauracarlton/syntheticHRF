#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:00:21 2024

@author: lauracarlton
"""

#---------------------------------------------------
#%% IMPORT MODULES
#---------------------------------------------------
import numpy as np 
import motionArtefactStudy.syntheticHRF_func_cedalion as synHRF_ced
import imageRecon.ImageRecon as IR
import os
import matplotlib.pyplot as plt 
from tqdm import tqdm 

import pandas as pd
import cedalion
import cedalion.io
import cedalion.dataclasses as cdc
# import cedalion.geometry.registration
# import cedalion.geometry.segmentation
import cedalion.plots
import cedalion.xrutils as xrutils
# import cedalion.imagereco.tissue_properties
# import cedalion.imagereco.forward_model as fw
# from cedalion.imagereco.solver import pseudo_inverse_stacked
from cedalion import units
import pickle
import matplotlib.pyplot as p

from cedalion import units
import cedalion.sigproc.quality as quality
from cedalion.sigproc.motion_correct import motion_correct_splineSG, motion_correct_PCA_recurse
import scipy.stats as stats
from sklearn.metrics import mean_squared_error 

import numpy as np
import xarray as xr
import pint
import matplotlib.pyplot as plt
import scipy.signal as signal
import os.path

xr.set_options(display_expand_data=False);
import pyvista as pv
#pv.set_jupyter_backend('html')
pv.set_jupyter_backend('static')
#pv.OFF_SCREEN=True

from scipy.io import loadmat
from tqdm import tqdm 
#---------------------------------------------------
#%% LOAD REQUIRED DATA 
#---------------------------------------------------
'''
need:
    - mesh 
    - A brain
    - A scalp 
    - snirf object
'''
DATADIR = '/Users/lauracarlton/Library/CloudStorage/GoogleDrive-lcarlton@bu.edu/My Drive/fNIRS/Data/motionArtefactStudy/'
rootDir_probe = '/Users/lauracarlton/Library/CloudStorage/GoogleDrive-lcarlton@bu.edu/My Drive/fNIRS/probes/WHHD/WHHD_first_second/'
saveDir = DATADIR + 'export_dod/'

# if not os.path.exists(saveDir):
#     os.makedirs(saveDir)
    
subjpath = 'subj-4/subj-4_task-MA_run-01'

elements = cedalion.io.read_snirf(DATADIR + subjpath)

amp = elements[0].data[0]
amp = amp.pint.dequantify().pint.quantify("V")

geo3d = elements[0].geo3d

dpf = xr.DataArray([1, 1], dims="wavelength", coords={"wavelength" : amp.wavelength})

#---------------------------------------------------
#%% LOAD IMAGE SPACE REQUIREMENTS
#---------------------------------------------------

fwDict = IR.load_fw(rootDir_probe)
Adot = fwDict['Adot']
Adot_scalp = fwDict['Adot_scalp']
#---------------------------------------------------
#%% DEFINE THE HRF
#---------------------------------------------------

time = amp.time
fs =  amp.cd.sampling_rate

trange_HRF = [0, 18]
stimDur = 10
tbasis = synHRF_ced.generateHRF(trange_HRF, 1/fs, stimDur)

dists = cedalion.nirs.channel_distances(amp, geo3d)
dists = dists.pint.to("mm")
geo3d = geo3d.rename({'pos':'digitized'})

E = cedalion.nirs.get_extinction_coefficients('prahl', amp.wavelength)
Einv = xrutils.pinv(E)

tbasis_od = xr.dot(E, tbasis*1*units.mm, dims='chromo')

#---------------------------------------------------
#%% CONFIGURE EVERYTHING FOR THE LOOP
#---------------------------------------------------

# set parameters for adding the HRFs to OD - 16 stims every 5-10 seconds
numStims = 16
min_interval = 5
max_interval = 10
    
# add leading zeros for [-2, 0]seconds 
leadingZeros = int((0 - (-2)) * fs)+1
scale_factor_list = [0.02] #, 0.005]
scale_factor = 0.02
nLoops = 1

run_list = ['noCorrection',
            'pcaRecurse',
            'splineSG'
            ]

subject_list = ['subj-4'] #,'subj-2','subj-3','subj-4']

nRuns = len(run_list)
nSubj = len(subject_list)
nChan = len(amp.channel)
nMeas = nChan * 2

# get list of sensitive vertices
sensitivity_threshold = 0.01
AdotSum = sum(Adot,0);
lstV = np.where(AdotSum[:,1]>sensitivity_threshold*max(AdotSum[:,1]))[0]

nV = Adot.shape[1]
nV_scalp = Adot_scalp.shape[1]

# get cluster of vertices by defining a blob within 50mm of each other - just for now to save compuational time
r = 100
seed = lstV[r]
lstV_temp = synHRF_ced.getConnectedVertices(seed, lstV, fwDict['vertices_brain'], fwDict['faces_brain'], dist_thresh = 50 )[0]

# set image reconstruction parameters 
lambda2_scalp = 0.1
lambda2_brain = 0.1
lambda1 = 0.01

trange_blockAvg = [-2, 18]

# parameters for channel pruning 
snr_thresh = 5 # the SNR (std/mean) of a channel. 
sd_threshs = [0, 4.5]*units.cm # defines the lower and upper bounds for the source-detector separation that we would like to keep
amp_threshs = [1e-3, 1e7]*units.volt # define whether a channel's amplitude is within a certain range

#---------------------------------------------------
#%% RUN THE LOOP FOR ALL VERTICES
#---------------------------------------------------

for ss,subj in enumerate(subject_list):
    #%%
    subjpath = subj +'/' + subj + '_task-MA_run-01'
    elements = cedalion.io.read_snirf(DATADIR + subjpath)
#%%
    amp = elements[0].data[0]
    amp = amp.pint.dequantify().pint.quantify("V")
    
    # PRUNE CHANNELS AND CONVERT TO OD 

    # then we calculate the masks for each metric: SNR, SD distance and mean amplitude
    _, snr_mask = quality.snr(amp, snr_thresh)
    _, sd_mask = quality.sd_dist(amp, geo3d, sd_threshs)
    _, amp_mask = quality.mean_amp(amp, amp_threshs)

    # put all masks in a list
    masks = [snr_mask, sd_mask, amp_mask]
    amp_pruned, drop_list = quality.prune_ch(amp, masks, "all")
    
    idx = [np.where(amp.channel==ii) for ii in drop_list]
    chan_mask = np.ones(len(amp.channel), dtype=bool)
    chan_mask[idx] = False
    nChan_subj = sum(chan_mask)
    
    # convert to OD
    od = cedalion.nirs.int2od(amp_pruned)
    od = od.pint.dequantify()
    od = xr.where(np.isinf(od), 0, od)
    od = xr.where(np.isnan(od), 0, od)
    #%%
    # apply channel mask to Adot 
    Adot_subj = Adot[chan_mask,:,:]
    Adot_scalp_subj = Adot_scalp[chan_mask,:,:]
    fwDict_subj = {'Adot':Adot_subj, 'Adot_scalp':Adot_scalp_subj}
    AmatrixDict = IR.generateAmatrix(Adot_subj, od['wavelength'], Adot_scalp=None)
    Amatrix = AmatrixDict['Amatrix']
    pAinv = IR.getPseudoInverseA_v2(fwDict_subj, od['wavelength'], svr = 1, lambda1=lambda1, lambda2_brain=lambda2_brain, lambda2_scalp=lambda2_scalp, spatial_bases = 0)
#%%
    for scale_factor in scale_factor_list:
        
          
         R_V_hbo = np.zeros([nRuns, nV, nLoops]); R_V_hbr = np.zeros([nRuns, nV, nLoops])
         R_C_hbo = np.zeros([nRuns, nV, nLoops]); R_C_hbr = np.zeros([nRuns, nV, nLoops])
         MSE_V_hbo = np.zeros([nRuns, nV, nLoops]); MSE_V_hbr = np.zeros([nRuns, nV, nLoops])
         MSE_C_hbo = np.zeros([nRuns, nV, nLoops]); MSE_C_hbr = np.zeros([nRuns, nV, nLoops])
         
         numVcalc = 0
         V_visited = []
         #%%
         for seed in tqdm(lstV_temp):
             
            #if vertex already has stats then skip it 
            if seed in V_visited:
                continue
                
            #%%
            # SETUP BLOB OF ACTIVATION AND HRF IMAGE SPACE TIMESERIES 
            # get the list of vertices in blob and the image of the activation of blob 
            blobV, blobImg = synHRF_ced.diffusionOperator(seed, lstV, fwDict['vertices_brain'], fwDict['faces_brain'])
            activeV = np.where(blobImg > 0.5)[0] # all vertices with activation strong enough so will be assigned values in this loop
            V_visited.extend(activeV) # keep track of vertices visited so don't repeat 
            
            # find the channel that is most sensitive to the blob
            vertexSensitivity = Adot_subj[:, seed, 1]
            channel = np.argmax(vertexSensitivity)
            
            # to find vertex most sensitive after image recon
            # project blob into channel space
            blobChan = Amatrix @ np.hstack([blobImg, -0.4*blobImg])
            #%%
            # project back into image space 
            recon_blobV = pAinv @ blobChan
            recon_blobV_brain = recon_blobV[:nV]
            
            # get the vertex with the maximum amplitude 
            maxV = np.argmax(recon_blobV_brain)
            # just select the rows of pAinv corresponding to this vertex to speed up downstream computation
            pAinv4projection = pAinv[[maxV, maxV+nV+nV_scalp], :]

            # add the HRF to the vertices in the blob and scale based on the diffusion
            HRF_image = synHRF_ced.addHRF_toVertices(blobV, tbasis, scale=blobImg)
        
            # transform image to channel space 
            chanSpace_HRFs_OD = synHRF_ced.timeseriesImage2Channel(HRF_image.T, Adot_subj, od['wavelength'])
        
            # scale based on the max of 850nm (WL2)
            scale = (scale_factor/np.max(chanSpace_HRFs_OD[:,nChan_subj:]))
            chanSpace_HRFs_OD_scaled = chanSpace_HRFs_OD * scale

            # DEFINE THE GROUND TRUTH
            # project the chanSpace HRFs back to image space to the ground truth for this scale + regularization
            groundtruth_ImageSpace = synHRF_ced.timeseriesChannel2Image(chanSpace_HRFs_OD_scaled, pAinv4projection)
            groundtruth_ImageSpace = np.concatenate([np.zeros([leadingZeros,2]), groundtruth_ImageSpace], axis=0) # add leading zeros to account for [-2,0]
        
            # convert OD to concentration for the channel to get ground truth for channel space
            groundtruth_ChannelSpace = synHRF_ced.OD2conc(chanSpace_HRFs_OD_scaled[:,[channel, channel+nChan_subj]], E)  
            groundtruth_ChannelSpace = np.concatenate([np.zeros([leadingZeros,2]), groundtruth_ChannelSpace], axis=0) # add leading zeros to account for [-2,0]
           
#%%
            for loop in np.arange(nLoops):
                #%%
                #% ADD OD HRFS TO OD DATA
                od_wHRF, onset_idx, onset_times = synHRF_ced.addHRF_todOD(od, chanSpace_HRFs_OD_scaled, trange_HRF, min_interval=min_interval, max_interval=max_interval, numStims=numStims)
    
                # POPULATE STIM CLASS                
                numStims = len(onset_times)
                stimDict = {"onset": onset_times,
                            "duration": np.ones(len(onset_times))*trange_HRF[1],
                            "value": np.ones(len(onset_times)),
                            "trial_type": np.ones(len(onset_times))
                    }
                stim_df = pd.DataFrame(stimDict)
                
                
                # FILTER AND CLEAN THE DATA
                od_wHRF = od_wHRF.transpose('channel', 'wavelength', 'time')
    #%%
                for rr, run in enumerate(run_list): 
    
            
                    if run == 'splineSG':
                        od_cleaned = motion_correct_splineSG(od_wHRF)
                    if run == 'pcaRecurse':
                        od_cleaned, svs, nSV, tInc = motion_correct_PCA_recurse(od_wHRF)
                    if run == 'noCorrection':
                        od_cleaned = od_wHRF.copy()
                    
                    # convert that channel to HbO/HbR
                    concChan = synHRF_ced.OD2conc(od_wHRF[channel,:,:], E) # FIXME
                    
                    # block average 
                    avgChanSpace = synHRF_ced.blockAverage(concChan, stim_df) #*np.max(abs(tbasis), axis=0)
                    tHRF_avg = np.linspace(-2, 18, len(avgChanSpace))
                
                    # get statistics for channel
                    statDictC = synHRF_ced.getStats(groundtruth_ChannelSpace, avgChanSpace)
                    R_C_hbo[activeV[0]] = statDictC['R_HbO'][0]
                    R_C_hbr[activeV[0]] = statDictC['R_HbR'][0]
                    MSE_C_hbo[activeV[0]] = statDictC['MSE_HbO']
                    MSE_C_hbr[activeV[0]] = statDictC['MSE_HbR']
                    # project back to image space    
                    channels2project = stimDict['dOD']
                    avg_img = synHRF_ced.timeseriesChannel2Image(channels2project, pAinv4projection) # FIXME
                    avg_img[np.isnan(avg_img)] = 0
                    avg_img[np.isinf(avg_img)] = 0
                
                    # block average
                    avgImgSpace = synHRF_ced.blockAverage(avg_img, stim_df)

                    statDictV = synHRF_ced.getStats(groundtruth_ImageSpace, avgImgSpace)
                    R_V_hbo[rr,activeV,loop] = statDictV['R_HbO'][0]
                    R_V_hbr[rr,activeV,loop] = statDictV['R_HbR'][0]
                    MSE_V_hbo[rr,activeV,loop] = statDictV['MSE_HbO']
                    MSE_V_hbr[rr,activeV,loop] = statDictV['MSE_HbR']
                    
                    numVcalc +=1
                    if numVcalc > 300:
                        break
                
            resultDict = {'R_V_hbo':R_V_hbo,'R_V_hbr':R_V_hbr, 'R_C_hbo':R_C_hbo, 'R_C_hbr':R_C_hbr, 
                          'MSE_V_hbo':MSE_V_hbo, 'MSE_V_hbr':MSE_V_hbr, 'MSE_C_hbo':MSE_C_hbo, 'MSE_C_hbr':MSE_C_hbr 
                          }
            
            
            # FIXME need to get results for each run separately 
            # generate save path and folder
            # saveDir = rootDir_data + 'export_' + run + '/'
        
            # if not os.path.exists(saveDir):
            #     os.makedirs(saveDir)
                    
            df = pd.DataFrame(resultDict)
            df.to_csv(saveDir+'statsResults_scalefactor-' + str(scale_factor) +'_'+ run + '.csv')
                
    
    
    
    
    
