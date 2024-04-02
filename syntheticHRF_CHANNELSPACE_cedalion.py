#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthetic HRF pipeline in channel space only using cedalion

steps:
    1. generate HRFs in concentration 
    2. scale for each channel so that when you convert to OD it results in 0.02 change in amplitude
    3. convert OD data to concentration
    4. add synthetic HRFs to each channel at random timings
    5. block avergae 
    6. calculate correlation + MSE with ground truth 
    7. loop over many times 


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
import random 
import cedalion
import cedalion.nirs
import cedalion.xrutils as xrutils

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
rootDir_probe = '/Users/lauracarlton/Library/CloudStorage/GoogleDrive-lcarlton@bu.edu/My Drive/fNIRS/probes/WHHD/WHHD_first_second/fw/'
saveDir = DATADIR + 'export_dod/'

# if not os.path.exists(saveDir):
#     os.makedirs(saveDir)
    
subjpath = 'subj-4/subj-4_task-MA_run-01'

elements = cedalion.io.read_snirf(DATADIR + subjpath)

amp = elements[0].data[0]
amp = amp.pint.dequantify().pint.quantify("V")

geo3d = elements[0].geo3d
# geo3d = geo3d.rename({'pos':'digitized'})

dpf = xr.DataArray([1, 1], dims="wavelength", coords={"wavelength" : amp.wavelength})

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
nLoops = 10

#---------------------------------------------------
#%% LOOP FOR CHANNELS
#---------------------------------------------------

run_list = ['noCorrection',
            'pcaRecurse',
            'splineSG'
            ]

subject_list = ['subj-1','subj-2','subj-3','subj-4']

nRuns = len(run_list)
nSubj = len(subject_list)
nChan = len(amp.channel)
nMeas = nChan * 2

R_HbO_loopAvg = np.zeros([nRuns, nSubj])
R_HbR_loopAvg = np.zeros([nRuns, nSubj])
MSE_HbO_loopAvg = np.zeros([nRuns, nSubj])
MSE_HbR_loopAvg = np.zeros([nRuns, nSubj])

R_HbO_loopSEM = np.zeros([nRuns, nSubj])
R_HbR_loopSEM = np.zeros([nRuns, nSubj])
MSE_HbO_loopSEM = np.zeros([nRuns, nSubj])
MSE_HbR_loopSEM = np.zeros([nRuns, nSubj])

geo3d = geo3d.rename({'pos':'digitized'})

#%%
snr_thresh = 5 # the SNR (std/mean) of a channel. 
sd_threshs = [0, 4.5]*units.cm # defines the lower and upper bounds for the source-detector separation that we would like to keep
amp_threshs = [1e-3, 1e7]*units.volt # define whether a channel's amplitude is within a certain range


for s,subj in enumerate(subject_list): 
    
    subjpath = subj +'/' + subj + '_task-MA_run-01'
    elements = cedalion.io.read_snirf(DATADIR + subjpath)

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
    
    # convert to OD
    od = cedalion.nirs.int2od(amp_pruned)
    od = od.pint.dequantify()
    od = xr.where(np.isinf(od), 0, od)
    od = xr.where(np.isnan(od), 0, od)
   
    for scale_factor in scale_factor_list:
    
        scale = (scale_factor/np.max(tbasis_od[1,:]))
        tbasis_od_scaled = tbasis_od * scale
       
        tbasis_conc = xr.dot(Einv, tbasis_od_scaled / (dpf*1*units.mm), dims=["wavelength"])
        tbasis_conc = tbasis_conc.rename("concentrations")

        groundTruth = np.concatenate([np.zeros([2,leadingZeros]), tbasis_conc.values], axis=1)

        R_HbO = np.zeros([nRuns, nChan, nLoops])
        R_HbR = np.zeros([nRuns, nChan, nLoops])
        MSE_HbO = np.zeros([nRuns, nChan, nLoops])
        MSE_HbR = np.zeros([nRuns, nChan, nLoops])
            
        for loop in np.arange(nLoops):
            
            #% ADD OD HRFS TO OD DATA
            od_wHRF, onset_idx, onset_times = synHRF_ced.addHRF_todOD(od, tbasis_od_scaled, trange_HRF, min_interval=min_interval, max_interval=max_interval, numStims=numStims)

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
                
                
            for rr, run in enumerate(run_list): 
    

                if run == 'splineSG':
                    od_cleaned = motion_correct_splineSG(od_wHRF)
                if run == 'pcaRecurse':
                    od_cleaned, svs, nSV, tInc = motion_correct_PCA_recurse(od_wHRF)
                if run == 'noCorrection':
                    od_cleaned = od_wHRF.copy()
                    
                
                od_filt = od_cleaned.cd.freq_filter(0.01, 0.5, 4)
                
                # CONVERT TO CONCENTRATION 
                conc_wHRF = xr.dot(Einv, od_filt / (dpf*1*units.mm), dims=["wavelength"])
      
                # BLOCK AVERAGE
                trange = [-2, trange_HRF[1]]
                avg = 0
                nTpts = int((trange[1]-trange[0]) * fs)
                
                epochs = conc_wHRF.cd.to_epochs(stim_df,
                                                [1],
                                                before = 2,
                                                after = 18)

                baseline = epochs.sel(reltime=(epochs.reltime < 0)).mean("reltime")
                epochs_blcorrected = epochs - baseline

                blockaverage = epochs_blcorrected.groupby('trial_type').mean('epoch')
                
                # blockaverage = blockaverage.stack({'measurement' : ['channel', 'chromo']})
                blockaverage = blockaverage.transpose('reltime', 'channel', 'chromo', 'trial_type')
 
                testing = 0
                if testing :
                    idx = 20
                    
                    ## PLOT BLOCK AVERAGE IN CHANNEL SPACE ##
                    fig, ax = plt.subplots(1,1)
                    tHRF = blockaverage.reltime
                    ax.plot(tHRF, groundTruth[0,:], 'r')
                    ax.plot(tHRF, groundTruth[1,:], 'b')
                    ax.plot(tHRF, blockaverage[:,idx,0,:], '--r')
                    ax.plot(tHRF, blockaverage[:,idx,1,:], '--b')                   
                    # plt.savefig(saveDir + 'channelSpace_blockAvg_V' + str(seed) + '_lambda2_scalp-' + str(lambda2_scalp) + '.jpeg', dpi = 500)
                
                    blockaverageAVG = blockaverage.mean('channel')
                    blockaverageSTD = blockaverage.std('channel')
                    
                # get statistics for channel
                i = 0
                for c in range(nChan):
                    if chan_mask[c]:
                        
                        R_HbO[rr,c,loop] = stats.pearsonr(groundTruth[0,:], np.squeeze(blockaverage[:,i,0,:].values))[0]
                        R_HbR[rr,c,loop] = stats.pearsonr(groundTruth[1,:], np.squeeze(blockaverage[:,i,1,:].values))[0]

                        MSE_HbO[rr,c,loop] = mean_squared_error(groundTruth[0,:],np.squeeze(blockaverage[:,i,0,:].values)) 
                        MSE_HbR[rr,c,loop] = mean_squared_error(groundTruth[1,:], np.squeeze(blockaverage[:,i,1,:].values)) 
                        i = i+1
                
            R_HbO[:,~chan_mask,:] = np.nan
            R_HbR[:,~chan_mask,:] = np.nan
            MSE_HbO[:,~chan_mask,:] = np.nan
            MSE_HbR[:,~chan_mask,:] = np.nan
    
            R_HbO_loopAvg[:,s] = np.nanmean(R_HbO[:,chan_mask,:], axis=(1,2))
            R_HbR_loopAvg[:,s] = np.nanmean(R_HbR[:,chan_mask,:], axis=(1,2))
            MSE_HbO_loopAvg[:,s] = np.nanmean(MSE_HbO[:,chan_mask,:], axis=(1,2))
            MSE_HbR_loopAvg[:,s] = np.nanmean(MSE_HbR[:,chan_mask,:], axis=(1,2))

 

#%%  plot differences

plt.rcParams.update({'font.size':16})
run_list_plotting = run_list

# for scale_factor in scale_factor_list:
        
#     R_HbO_subjAvg = np.mean(R_HbO_loopAvg, axis = 1)
#     R_HbR_subjAvg = np.mean(R_HbR_loopAvg, axis = 1)
#     MSE_HbO_subjAvg = np.mean(MSE_HbO_loopAvg, axis = 1)
#     MSE_HbR_subjAvg = np.mean(MSE_HbR_loopAvg, axis = 1)
    
# R_HbO_subjSEM = stats.sem(R_HbO_Avg, axis = 1)
# R_HbR_subjSEM = stats.sem(R_HbR_loopAvg, axis = 1)
# MSE_HbO_subjSEM = stats.sem(MSE_HbO_loopAvg, axis = 1)
# MSE_HbR_subjSEM = stats.sem(MSE_HbR_loopAvg, axis = 1)
 
if 1 : 
    fig, ax = plt.subplots(1,2, figsize = [12,20], sharey=True, sharex = True)
    fig.suptitle('R: scalefactor-' + str(scale_factor))
    ax[0].bar(run_list_plotting, np.squeeze(R_HbO_loopAvg), color='r', alpha=0.7) #, yerr=R_HbO_subjSEM,)
    ax[0].set_xticklabels(run_list_plotting, rotation='vertical')
    # ax[0].set_ylim([0.5, 1.2])
    ax[0].set_title('Channel Space; HbO')
    
    ax[1].bar(run_list_plotting, np.squeeze(R_HbR_loopAvg),  color='b', alpha=0.7) #, yerr=R_HbR_subjSEM)
    ax[1].set_xticklabels(run_list_plotting, rotation='vertical')
    # ax[1].set_ylim([0.5, 1])
    ax[1].set_title('Channel Space; HbR')

    plt.tight_layout()
    # plt.savefig(rootDir_data + 'export_compareMethods/CS_barCompareR_scalefactor-' + str(scale_factor) + '.jpeg', dpi=500)


if 1 : 
    fig, ax = plt.subplots(1,2, figsize = [12,20], sharey=False, sharex = True)
    fig.suptitle('MSE: scalefactor-' + str(scale_factor))
    ax[0].bar(run_list_plotting, np.squeeze(MSE_HbO_loopAvg),  color='r', alpha=0.7) #, yerr=MSE_HbO_subjSEM,)
    ax[0].set_xticklabels(run_list_plotting, rotation='vertical')
    # ax[0].set_ylim([0.5, 1.2])
    ax[0].set_title('Channel Space; HbO')
    
    ax[1].bar(run_list_plotting, np.squeeze(MSE_HbR_loopAvg), color='b', alpha=0.7) #, yerr=MSE_HbR_subjSEM, )
    ax[1].set_xticklabels(run_list_plotting, rotation='vertical')
    # ax[1].set_ylim([0.5, 1])
    ax[1].set_title('Channel Space; HbR')
    
    plt.tight_layout()
    # plt.savefig(rootDir_data + 'export_compareMethods/CS_barCompareMSE_scalefactor-' + str(scale_factor) + '.jpeg', dpi=500)



        