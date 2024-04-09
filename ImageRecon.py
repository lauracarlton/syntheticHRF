#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run image reconstruction 

sections:
    1. get image parameters 
        want to do brain and scalp 
        need trange
        rhoSD_ssThresh - short channel threshold (defines what is considered a short channel)
        
    2. get the A matrix for brain and scalp 
            
    3. get the data
        - get the SD file 
        - get the data 
            - need a yavgimg
        - need alpha -> apparently usually around 0.01 for the first method
        
    4. call hmrPyImageReconConc

questions:
    do we need spatial regularization ?
        yes do image recon with brain+scalp and spatial regularization 
        
        
    do I need to build a GUI? or is there a way to just visualize the final image?
    

plan for now:
    generate function that takes A matrix, yavgimg,  runs the recon
    
    need to figure out where alpha and beta come from 
    
@author: lauracarlton
"""
import numpy as np 
from mayavi import mlab
from scipy.io import loadmat 

#%% 

# LOAD ALL REQUIREMENTS FROM AV FW OUTPUT
def load_fw(av_path):
    
    ## load the A matrices ##
    Adot_obj = loadmat(av_path+'Adot.mat')
    Adot = Adot_obj['Adot']

    Adot_scalp_obj = loadmat(av_path+'Adot_scalp.mat')
    Adot_scalp = Adot_scalp_obj['Adot_scalp']

    try :
        ## load the G matrices if they are there ##
        G_obj = loadmat(av_path + 'G.mat')
        G_brain = G_obj['G_brain']
        G_scalp = G_obj['G_scalp']
    except:
        G_brain = []
        G_scalp = []

    ## load the brain mesh 
    mesh_obj = loadmat(av_path +'mesh_brain.mat')
    mesh = mesh_obj['mesh']
    faces = mesh['faces'][0][0]-1
    vertices = mesh['vertices'][0][0]
    
    ## load the brain mesh 
    mesh_obj = loadmat(av_path +'mesh_scalp.mat')
    mesh_scalp = mesh_obj['mesh_scalp']
    faces_scalp = mesh_scalp['faces'][0][0]-1
    vertices_scalp = mesh_scalp['vertices'][0][0]

    fwDict = {'Adot': Adot,
              'Adot_scalp': Adot_scalp,
              'faces_brain': faces,
              'vertices_brain': vertices,
              'G_brain': G_brain,
              'G_scalp': G_scalp,
              'faces_scalp': faces_scalp,
              'vertices_scalp': vertices_scalp
        }
    
    return fwDict



#%%

def generateAmatrix(Adot, wavelengths, Adot_scalp=None):
    
    E = GetExtinctions(wavelengths);
    E = E/10 # convert from /cm to /mm  E rows: wavelength, columns 1:HbO, 2:HbR
    AmatrixDict = {}
    
    AmatrixDict['Abrain_wl1_hbo'] = np.squeeze(Adot[:,:,0])*E[0,0]
    AmatrixDict['Abrain_wl1_hbr'] = np.squeeze(Adot[:,:,0])*E[0,1]
    
    AmatrixDict['Abrain_wl2_hbo'] = np.squeeze(Adot[:,:,1])*E[1,0]
    AmatrixDict['Abrain_wl2_hbr'] = np.squeeze(Adot[:,:,1])*E[1,1]
        
    if Adot_scalp is None:
 
        Acat = np.vstack([np.hstack([AmatrixDict['Abrain_wl1_hbo'], AmatrixDict['Abrain_wl1_hbr']]),
                  np.hstack([AmatrixDict['Abrain_wl2_hbo'], AmatrixDict['Abrain_wl2_hbr']])]) 
        
     
    else :
        
        AmatrixDict['Ascalp_wl1_hbo'] = np.squeeze(Adot_scalp[:,:,0])*E[0,0]
        AmatrixDict['Ascalp_wl1_hbr'] = np.squeeze(Adot_scalp[:,:,0])*E[0,1]
    
        AmatrixDict['Ascalp_wl2_hbo'] = np.squeeze(Adot_scalp[:,:,1])*E[1,0]
        AmatrixDict['Ascalp_wl2_hbr'] = np.squeeze(Adot_scalp[:,:,1])*E[1,1]
        
        Acat1 = np.hstack([AmatrixDict['Abrain_wl1_hbo'], AmatrixDict['Ascalp_wl1_hbo'], AmatrixDict['Abrain_wl1_hbr'], AmatrixDict['Ascalp_wl1_hbr']])
        Acat2 = np.hstack([AmatrixDict['Abrain_wl2_hbo'], AmatrixDict['Ascalp_wl2_hbo'], AmatrixDict['Abrain_wl2_hbr'], AmatrixDict['Ascalp_wl2_hbr']])
        Acat = np.vstack([Acat1, Acat2])
   
    
    AmatrixDict['Amatrix'] = Acat
    
    return AmatrixDict

#%%

def generateHmatrix(AmatrixDict, fwDict):
    
    G_brain = fwDict['G_brain']
    G_scalp = fwDict['G_scalp']
    
    
    H_brain_wl1_hbo = AmatrixDict['Abrain_wl1_hbo'] @ G_brain.T
    H_brain_wl1_hbr = AmatrixDict['Abrain_wl1_hbr'] @ G_brain.T
    H_brain_wl2_hbo = AmatrixDict['Abrain_wl2_hbo'] @ G_brain.T
    H_brain_wl2_hbr = AmatrixDict['Abrain_wl2_hbr'] @ G_brain.T
    
    H_scalp_wl1_hbo = AmatrixDict['Ascalp_wl1_hbo'] @ G_scalp.T
    H_scalp_wl1_hbr = AmatrixDict['Ascalp_wl1_hbr'] @ G_scalp.T
    H_scalp_wl2_hbo = AmatrixDict['Ascalp_wl2_hbo'] @ G_scalp.T
    H_scalp_wl2_hbr = AmatrixDict['Ascalp_wl2_hbr'] @ G_scalp.T
    
    # A_brain_hbo = np.vstack([AmatrixDict['Abrain_wl1_hbo'], AmatrixDict['Abrain_wl2_hbo']])
    # H_brain_hbo = A_brain_hbo @ G_brain.T

    # A_scalp_hbo = np.vstack([AmatrixDict['Ascalp_wl1_hbo'], AmatrixDict['Ascalp_wl2_hbo']])
    # H_scalp_hbo = A_scalp_hbo @ G_scalp.T

    # A_brain_hbr = np.vstack([AmatrixDict['Abrain_wl1_hbr'], AmatrixDict['Abrain_wl2_hbr']])
    # H_brain_hbr = A_brain_hbr @ G_brain.T

    # A_scalp_hbr = np.vstack([AmatrixDict['Ascalp_wl1_hbr'], AmatrixDict['Ascalp_wl2_hbr']])
    # H_scalp_hbr = A_scalp_hbr @ G_scalp.T
    
    Hcat1 = np.hstack([H_brain_wl1_hbo, H_scalp_wl1_hbo, H_brain_wl1_hbr, H_scalp_wl1_hbr])
    Hcat2 = np.hstack([H_brain_wl2_hbo, H_scalp_wl2_hbo, H_brain_wl2_hbr, H_scalp_wl2_hbr])
    Hmatrix = np.vstack([Hcat1, Hcat2])
    
    # Hmatrix = np.hstack([H_brain_hbo, H_scalp_hbo, H_brain_hbr, H_scalp_hbr])
    
    return Hmatrix

#%%
def getPseudoInverseA(Adot, wavelengths, alpha=1e-2, Adot_scalp=None):
    
    Amatrix = generateAmatrix(Adot, wavelengths, Adot_scalp=Adot_scalp)
    
    Amatrix = Amatrix.astype('double')
    B = Amatrix@Amatrix.T

    ###  Tikhonov regularization ###
    # solution to arg min ||Y-Ax||^2 + lamda * ||x||^2 (prior only on state estimate)
    pAinv = Amatrix.T @ np.linalg.pinv(B + (alpha * np.linalg.eig(B)[0][0] * np.eye(np.shape(B)[0])))
    
    return pAinv



def getPseudoInverseA_v2(fwDict, wavelengths, svr = 1, lambda1=0.01, lambda2_brain = 0.1, lambda2_scalp=0.1, spatial_bases = 1):
    
    
    AmatrixDict = generateAmatrix(fwDict['Adot'], wavelengths, Adot_scalp=fwDict['Adot_scalp'])
    
    if spatial_bases:
        H = generateHmatrix(AmatrixDict, fwDict)
        nBrain = fwDict['G_brain'].shape[0]
        nScalp = fwDict['G_scalp'].shape[0]
    else:
        H = AmatrixDict['Amatrix']
        nBrain = fwDict['Adot'].shape[1]
        nScalp = fwDict['Adot_scalp'].shape[1]
        
    [nM,nV] = H.shape

    
    if svr:
       ll_0 = np.sum(H**2, axis=0)
       # L = np.sqrt(ll_0 + lambda2*max(ll_0))
       ll_0 = np.sum(H**2, axis=0)
       
       ll_brain = np.hstack([ll_0[:nBrain], ll_0[nBrain+nScalp: nScalp+nBrain*2]])
       ll_scalp = np.hstack([ll_0[nBrain:nBrain+nScalp], ll_0[nBrain*2+nScalp:]])
       
       L_brain = np.sqrt(ll_brain + lambda2_brain*max(ll_brain))
       L_scalp = np.sqrt(ll_scalp + lambda2_scalp*max(ll_scalp))
       L = np.squeeze(np.hstack([L_brain[:nBrain], L_scalp[:nScalp], L_brain[nBrain:], L_scalp[nScalp:]]))
       
       H = H/L    
    #%
    if nV < nM:
       Htt = (H.T @ H).astype('single')
       
       ss = np.linalg.norm(Htt, 2)
       penalty = np.sqrt(ss) * lambda1
       pHinv = np.linalg.pinv(Htt + penalty**2 * np.eye(nV)) @ H.T
      
    else:
       Htt = (H @ H.T).astype('single')
       
       ss = np.linalg.norm(Htt, 2)
       penalty = np.sqrt(ss) * lambda1
       
       pHinv = H.T @ np.linalg.pinv(Htt + penalty**2 * np.eye(nM))
          
      #%
    if svr:
       L = L.reshape([len(L), 1]) 
       pHinv = pHinv / L

       
    return pHinv

def getPseudoInverseA_v3(Adot, lambda1,  wavelengths, svr = 1, lambda2_brain = 0.1, lambda2_scalp=0.1, Adot_scalp=None):
        
    A = generateAmatrix(Adot, wavelengths, Adot_scalp=Adot_scalp)
    [nM,nV] = A.shape
    nV_brain = Adot.shape[1]
    nV_scalp = Adot_scalp.shape[1]
    
    if svr:
       ll_0 = np.sum(A**2, axis=0)
       # L = np.sqrt(ll_0 + lambda2*max(ll_0))
       ll_0 = np.sum(A**2, axis=0)
       
       ll_brain = np.hstack([ll_0[:nV_brain], ll_0[nV_brain+nV_scalp: nV_scalp+nV_brain*2]])
       ll_scalp = np.hstack([ll_0[nV_brain:nV_brain+nV_scalp], ll_0[nV_brain*2+nV_scalp:]])
       
       L_brain = np.sqrt(ll_brain + lambda2_brain*max(ll_brain))
       L_scalp = np.sqrt(ll_scalp + lambda2_scalp*max(ll_scalp))
       L = np.squeeze(np.hstack([L_brain[:nV_brain], L_scalp[:nV_scalp], L_brain[nV_brain:], L_scalp[nV_scalp:]]))
       
       A = A/L    
    #%
    if nV < nM:
       Att = (A.T @ A).astype('single')
       
       ss = np.linalg.norm(Att, 2)
       penalty = np.sqrt(ss) * lambda1
       pAinv = np.linalg.pinv(Att + penalty * np.eye(nV)) @ A.T
      
    else:
       Att = (A @ A.T).astype('single')
       
       ss = np.linalg.norm(Att, 2)
       penalty = np.sqrt(ss) * lambda1
       
       pAinv = A.T @ np.linalg.pinv(Att + penalty * np.eye(nM))
          
      #%
    if svr:
       L = L.reshape([len(L), 1]) 
       pAinv = pAinv / L

       
    return pAinv

#%%

def ImageReconOD(dodavgimg, fwDict, wavelengths, lambda1=0.01, lambda2_brain = 0.1, lambda2_scalp=0.01, scalp=1, svr = 1, v = 2, spatial_bases=1):
    
    # Adot must be double (rather than single) to calculate eigs(B,1)
    if not fwDict['Adot'].dtype == 'float': 
        fwDict['Adot'] = fwDict['Adot'].astype(float)
    
    if v == 2 : 
        pAinv = getPseudoInverseA_v2(fwDict, wavelengths, svr = svr, lambda1=lambda1, lambda2_brain =lambda2_brain, lambda2_scalp=lambda2_scalp, spatial_bases = spatial_bases)
        
    elif v == 3:
        # for v3 lambda1 = list of variance for each channel 
        pAinv = getPseudoInverseA_v3(fwDict['Adot'], lambda1, wavelengths, svr = svr, lambda2_brain=lambda2_brain, lambda2_scalp=lambda2_scalp, Adot_scalp=fwDict['Adot_scalp'])
    
    else :
        pAinv = getPseudoInverseA(fwDict['Adot'], wavelengths, alpha = lambda1, Adot_scalp=fwDict['Adot_scalp'])
        
    img =  pAinv @ dodavgimg  # Custo2010NI; BoasDale2005; typical alpha = 0.01

    # get HbO and HbR
    split = np.shape(img)[0]//2
    HbO = img[:split]
    HbR = img[split:]
        
    return  HbO, HbR

#%%

def plot_image(faces, vertices, yimg, vmin=0, vmax=1, savePath=None):
    
    # Create a triangular mesh using Mayavi
    fig = mlab.figure(size=(700,700))
    mesh = mlab.triangular_mesh(vertices[:, 1], vertices[:, 0], vertices[:, 2], faces, figure=fig, colormap='jet', scalars=yimg, resolution=16, vmax=vmax, vmin=vmin)
    mesh.scene.background = (1, 1, 1)
    
    # Create a custom colorbar
    # lut = mesh.module_manager.scalar_lut_manager.lut.table.to_array()
    # lut[0:10,0:10,0:10,0:10] = [192,192,192,1]
    # mesh.module_manager.scalar_lut_manager.lut.table = lut
    
    # Add a colorbar to the plot
    mlab.draw()
    if savePath != None:
        
        cb = mlab.colorbar(orientation='vertical')
        cb.label_text_property.color = (0,0,0)
        
        # mlab.view(azimuth= -90, elevation = 270)
        # mlab.savefig(savePath+'left.jpeg')
        
        mlab.view(azimuth = 270, elevation = -270)
        mlab.savefig(savePath+'right.jpeg')
        
        # mlab.view(azimuth= 180, elevation= 90)
        # mlab.savefig(savePath+'superior.jpeg')
        
    else :
        cb = mlab.colorbar(orientation='horizontal')
        cb.label_text_property.color = (0,0,0) 
        
        
    # mlab.figure(bgcolor=(1,1,1))
    
    mlab.show()
    return

#%% 

def  GetExtinctions(lam,WhichSpectrum=1):
    # %
    # % GetExtinctions( lam )
    # %
    # %       Returns the specific absorption coefficients for
    # %         [HbO Hb H2O lipid aa3]
    # %       for the specified wavelengths. Note that the specific
    # %       absorption coefficient (defined base e) is equal to the 
    # %       specific extinction coefficient (defined base 10) times 2.303.
    # %
    # %	These values for the molar extinction coefficient e
    # % 	in [cm-1/(moles/liter)] were compiled by Scott Prahl
    # %	(prahl@ece.ogi.edu) using data from
    # %	W. B. Gratzer, Med. Res. Council Labs, Holly Hill, London
    # %	N. Kollias, Wellman Laboratories, Harvard Medical School, Boston
    # %	To convert this data to absorbance A, multiply by the
    # %	molar concentration and the pathlength. For example, if x is the
    # %	number of grams per liter and a 1 cm cuvette is being used,
    # %	then the absorbance is given by
    # %
    # %        (e) [(1/cm)/(moles/liter)] (x) [g/liter] (1) [cm]
    # %  A =  ---------------------------------------------------
    # %                          66,500 [g/mole]
    # %
    # %	using 66,500 as the gram molecular weight of hemoglobin.
    # %	To convert this data to absorption coefficient in (cm-1), multiply
    # %	by the molar concentration and 2.303,
    # %
    # %	µa = (2.303) e (x g/liter)/(66,500 g Hb/mole)
    # %	where x is the number of grams per liter. A typical value of x
    # %	for whole blood is x=150 g Hb/liter.


    lam = np.array(lam)
    lam = np.reshape(lam, [lam.size,1])
    num_lam = lam.size
    
    if WhichSpectrum == 1:

        # Citation=print('W. B. Gratzer, Med. Res. Council Labs, Holly Hill,London \nN. Kollias, Wellman Laboratories, Harvard Medical School, Boston')

    #         These values for the molar extinction coefficient e in [cm-1/(moles/liter)] were compiled by Scott Prahl (prahl@ece.ogi.edu) using data from
    #         
    #         W. B. Gratzer, Med. Res. Council Labs, Holly Hill, London
    #         N. Kollias, Wellman Laboratories, Harvard Medical School, Boston
    #         To convert this data to absorbance A, multiply by the molar concentration and the pathlength. For example, if x is the number of grams per liter and a 1 cm cuvette is being used, then the absorbance is given by
    #         
    #                 (e) [(1/cm)/(moles/liter)] (x) [g/liter] (1) [cm]
    #           A =  ---------------------------------------------------
    #                                   66,500 [g/mole]
    #         
    #         using 66,500 as the gram molecular weight of hemoglobin.
    #         To convert this data to absorption coefficient in (cm-1), multiply by the molar concentration and 2.303,
    #         
    #         µa = (2.303) e (x g/liter)/(66,500 g Hb/mole)
    #         where x is the number of grams per liter. A typical value of x for whole blood is x=150 g Hb/liter.
    
 
        vlamHbOHb = np.array([
            [250.0, 106112.0, 112736.0],
            [252.0, 105552.0, 112736.0],
            [254.0, 107660.0, 112736.0],
            [256.0, 109788.0, 113824.0],
            [258.0, 112944.0, 115040.0],
            [260.0, 116376.0, 116296.0],
            [262.0, 120188.0, 117564.0],
            [264.0, 124412.0, 118876.0],
            [266.0, 128696.0, 120208.0],
            [268.0, 133064.0, 121544.0],
            [270.0, 136068.0, 122880.0],
            [272.0, 137232.0, 123096.0],
            [274.0, 138408.0, 121952.0],
            [276.0, 137424.0, 120808.0],
            [278.0, 135820.0, 119840.0],
            [280.0, 131936.0, 118872.0],
            [282.0, 127720.0, 117628.0],
            [284.0, 122280.0, 114820.0],
            [286.0, 116508.0, 112008.0],
            [288.0, 108484.0, 107140.0],
            [290.0, 104752.0, 98364.0],
            [292.0, 98936.0, 91636.0],
            [294.0, 88136.0, 85820.0],
            [296.0, 79316.0, 77100.0],
            [298.0, 70884.0, 69444.0],
            [300.0, 65972.0, 64440.0],
            [302.0, 63208.0, 61300.0],
            [304.0, 61952.0, 58828.0],
            [306.0, 62352.0, 56908.0],
            [308.0, 62856.0, 57620.0],
            [310.0, 63352.0, 59156.0],
            [312.0, 65972.0, 62248.0],
            [314.0, 69016.0, 65344.0],
            [316.0, 72404.0, 68312.0],
            [318.0, 75536.0, 71208.0],
            [320.0, 78752.0, 74508.0],
            [322.0, 82256.0, 78284.0],
            [324.0, 85972.0, 82060.0],
            [326.0, 89796.0, 85592.0],
            [328.0, 93768.0, 88516.0],
            [330.0, 97512.0, 90856.0],
            [332.0, 100964.0, 93192.0],
            [334.0, 103504.0, 95532.0],
            [336.0, 104968.0, 99792.0],
            [338.0, 106452.0, 104476.0],
            [340.0, 107884.0, 108472.0],
            [342.0, 109060.0, 110996.0],
            [344.0, 110092.0, 113524.0],
            [346.0, 109032.0, 116052.0],
            [348.0, 107984.0, 118752.0],
            [350.0, 106576.0, 122092.0],
            [352.0, 105040.0, 125436.0],
            [354.0, 103696.0, 128776.0],
            [356.0, 101568.0, 132120.0],
            [358.0, 97828.0, 133632.0],
            [360.0, 94744.0, 134940.0],
            [362.0, 92248.0, 136044.0],
            [364.0, 89836.0, 136972.0],
            [366.0, 88484.0, 137900.0],
            [368.0, 87512.0, 138856.0],
            [370.0, 88176.0, 139968.0],
            [372.0, 91592.0, 141084.0],
            [374.0, 95140.0, 142196.0],
            [376.0, 98936.0, 143312.0],
            [378.0, 103432.0, 144424.0],
            [380.0, 109564.0, 145232.0],
            [382.0, 116968.0, 145232.0],
            [384.0, 125420.0, 148668.0],
            [386.0, 135132.0, 153908.0],
            [388.0, 148100.0, 159544.0],
            [390.0, 167748.0, 167780.0],
            [392.0, 189740.0, 180004.0],
            [394.0, 212060.0, 191540.0],
            [396.0, 231612.0, 202124.0],
            [398.0, 248404.0, 212712.0],
            [400.0, 266232.0, 223296.0],
            [402.0, 284224.0, 236188.0],
            [404.0, 308716.0, 253368.0],
            [406.0, 354208.0, 270548.0],
            [408.0, 422320.0, 287356.0],
            [410.0, 466840.0, 303956.0],
            [412.0, 500200.0, 321344.0],
            [414.0, 524280.0, 342596.0],
            [416.0, 521880.0, 363848.0],
            [418.0, 515520.0, 385680.0],
            [420.0, 480360.0, 407560.0],
            [422.0, 431880.0, 429880.0],
            [424.0, 376236.0, 461200.0],
            [426.0, 326032.0, 481840.0],
            [428.0, 283112.0, 500840.0],
            [430.0, 246072.0, 528600.0],
            [432.0, 214120.0, 552160.0],
            [434.0, 165332.0, 552160.0],
            [436.0, 132820.0, 547040.0],
            [438.0, 119140.0, 501560.0],
            [440.0, 102580.0, 413280.0],
            [442.0, 92780.0, 363240.0],
            [444.0, 81444.0, 282724.0],
            [446.0, 76324.0, 237224.0],
            [448.0, 67044.0, 173320.0],
            [450.0, 62816.0, 103292.0],
            [452.0, 58864.0, 62640.0],
            [454.0, 53552.0, 36170.0],
            [456.0, 49496.0, 30698.8],
            [458.0, 47496.0, 25886.4],
            [460, 44480, 23388.8],
            [462, 41320, 20891.2],
            [464, 39807.2, 19260.8],
            [466, 37073.2, 18142.4],
            [468, 34870.8, 17025.6],
            [470, 33209.2, 16156.4],
            [472, 31620, 15310],
            [474, 30113.6, 15048.4],
            [476, 28850.8, 14792.8],
            [478, 27718, 14657.2],
            [480, 26629.2, 14550],
            [482, 25701.6, 14881.2],
            [484, 25180.4, 15212.4],
            [486, 24669.6, 15543.6],
            [488, 24174.8, 15898],
            [490, 23684.4, 16684],
            [492, 23086.8, 17469.6],
            [494, 22457.6, 18255.6],
            [496, 21850.4, 19041.2],
            [498, 21260, 19891.2],
            [500, 20932.8, 20862],
            [502, 20596.4, 21832.8],
            [504, 20418, 22803.6],
            [506, 19946, 23774.4],
            [508, 19996, 24745.2],
            [510, 20035.2, 25773.6],
            [512, 20150.4, 26936.8],
            [514, 20429.2, 28100],
            [516, 21001.6, 29263.2],
            [518, 22509.6, 30426.4],
            [520, 24202.4, 31589.6],
            [522, 26450.4, 32851.2],
            [524, 29269.2, 34397.6],
            [526, 32496.4, 35944],
            [528, 35990, 37490],
            [530, 39956.8, 39036.4],
            [532, 43876, 40584],
            [534, 46924, 42088],
            [536, 49752, 43592],
            [538, 51712, 45092],
            [540, 53236, 46592],
            [542, 53292, 48148],
            [544, 52096, 49708],
            [546, 49868, 51268],
            [548, 46660, 52496],
            [550, 43016, 53412],
            [552, 39675.2, 54080],
            [554, 36815.2, 54520],
            [556, 34476.8, 54540],
            [558, 33456, 54164],
            [560, 32613.2, 53788],
            [562, 32620, 52276],
            [564, 33915.6, 50572],
            [566, 36495.2, 48828],
            [568, 40172, 46948],
            [570, 44496, 45072],
            [572, 49172, 43340],
            [574, 53308, 41716],
            [576, 55540, 40092],
            [578, 54728, 38467.6],
            [580, 50104, 37020],
            [582, 43304, 35676.4],
            [584, 34639.6, 34332.8],
            [586, 26600.4, 32851.6],
            [588, 19763.2, 31075.2],
            [590, 14400.8, 28324.4],
            [592, 10468.4, 25470],
            [594, 7678.8, 22574.8],
            [596, 5683.6, 19800],
            [598, 4504.4, 17058.4],
            [600, 3200, 14677.2],
            [602, 2664, 13622.4],
            [604, 2128, 12567.6],
            [606, 1789.2, 11513.2],
            [608, 1647.6, 10477.6],
            [610, 1506, 9443.6],
            [612, 1364.4, 8591.2],
            [614, 1222.8, 7762],
            [616, 1110, 7344.8],
            [618, 1026, 6927.2],
            [620, 942, 6509.6],
            [622, 858, 6193.2],
            [624, 774, 5906.8],
            [626, 707.6, 5620],
            [628, 658.8, 5366.8],
            [630, 610, 5148.8],
            [632, 561.2, 4930.8],
            [634, 512.4, 4730.8],
            [636, 478.8, 4602.4],
            [638, 460.4, 4473.6],
            [640, 442, 4345.2],
            [642, 423.6, 4216.8],
            [644, 405.2, 4088.4],
            [646, 390.4, 3965.08],
            [648, 379.2, 3857.6],
            [650, 506.0, 3743.0],
            [652, 488.0, 3677.0],
            [654, 474.0, 3612.0],
            [656, 464.0, 3548.0],
            [658, 454.3, 3491.3],
            [660, 445.0, 3442.0],
            [662, 438.3, 3364.7],
            [664, 433.8, 3292.8],
            [666, 431.3, 3226.3],
            [668, 429.0, 3133.0],
            [670, 427.0, 3013.0],
            [672, 426.5, 2946.0],
            [674, 426.0, 2879.0],
            [676, 424.0, 2821.7],
            [678, 423.0, 2732.3],
            [680, 423.0, 2610.8],
            [682, 422.0, 2497.3],
            [684, 420.0, 2392.0],
            [686, 418.0, 2292.7],
            [688, 416.5, 2209.3],
            [690, 415.5, 2141.8],
            [692, 415.0, 2068.7],
            [694, 415.0, 1990.0],
            [696, 415.5, 1938.5],
            [698, 416.0, 1887.0],
            [700, 419.3, 1827.7],
            [702, 422.5, 1778.5],
            [704, 425.5, 1739.5],
            [706, 429.7, 1695.7],
            [708, 435.0, 1647.0],
            [710, 441.0, 1601.7],
            [712, 446.5, 1562.5],
            [714, 451.5, 1529.5],
            [716, 458.0, 1492.0],
            [718, 466.0, 1450.0],
            [720, 472.7, 1411.3],
            [722, 479.5, 1380.0],
            [724, 486.5, 1356.0],
            [726, 494.3, 1331.7],
            [728, 503.0, 1307.0],
            [730, 510.0, 1296.5],
            [732, 517.0, 1286.0],
            [734, 521.0, 1286.0],
            [736, 530.7, 1293.0],
            [738, 546.0, 1307.0],
            [740, 553.5, 1328.0],
            [742, 561.0, 1349.0],
            [744, 571.0, 1384.3],
            [746, 581.3, 1431.3],
            [748, 592.0, 1490.0],
            [750, 600.0, 1532.0],
            [752, 608.0, 1574.0],
            [754, 618.7, 1620.7],
            [756, 629.7, 1655.3],
            [758, 641.0, 1678.0],
            [760, 645.5, 1669.0],
            [762, 650.0, 1660.0],
            [764, 666.7, 1613.3],
            [766, 681.0, 1555.0],
            [768, 693.0, 1485.0],
            [770, 701.5, 1425.0],
            [772, 710.0, 1365.0],
            [774, 722.0, 1288.3],
            [776, 733.7, 1216.3],
            [778, 745.0, 1149.0],
            [780, 754.0, 1107.5],
            [782, 763.0, 1066.0],
            [784, 775.0, 1021.3],
            [786, 787.0, 972.0],
            [788, 799.0, 918.0],
            [790, 808.0, 913.0],
            [792, 817.0, 908.0],
            [794, 829.0, 887.3],
            [796, 840.7, 868.7],
            [798, 852.0, 852.0],
            [800, 863.3, 838.7],
            [802, 873.3, 828.0],
            [804, 881.8, 820.0],
            [806, 891.7, 812.0],
            [808, 903.0, 804.0],
            [810, 914.3, 798.7],
            [812, 924.7, 793.7],
            [814, 934.0, 789.0],
            [816, 943.0, 787.0],
            [818, 952.0, 785.0],
            [820, 962.7, 783.0],
            [822, 973.0, 781.0],
            [824, 983.0, 779.0],
            [826, 990.5, 778.5],
            [828, 998.0, 778.0],
            [830, 1008.0, 778.0],
            [832, 1018.0, 777.7],
            [834, 1028.0, 777.0],
            [836, 1038.0, 777.0],
            [838, 1047.7, 777.0],
            [840, 1057.0, 777.0],
            [842, 1063.5, 777.5],
            [844, 1070.0, 778.0],
            [846, 1079.3, 779.3],
            [848, 1088.3, 780.3],
            [850, 1097.0, 781.0],
            [852, 1105.7, 783.0],
            [854, 1113.0, 785.0],
            [856, 1119.0, 787.0],
            [858, 1126.0, 789.3],
            [860, 1134.0, 792.0],
            [862, 1142.0, 795.3],
            [864, 1149.7, 799.0],
            [866, 1157.0, 803.0],
            [868, 1163.7, 807.7],
            [870, 1170.3, 812.3],
            [872, 1177.0, 817.0],
            [874, 1182.0, 820.5],
            [876, 1187.0, 824.0],
            [878, 1193.0, 830.0],
            [880, 1198.7, 835.7],
            [882, 1204.0, 841.0],
            [884, 1209.3, 847.0],
            [886, 1214.3, 852.7],
            [888, 1219.0, 858.0],
            [890, 1223.7, 863.3],
            [892, 1227.5, 867.8],
            [894, 1230.5, 871.3],
            [896, 1234.0, 875.3],
            [898, 1238.0, 880.0],
            [900, 1241.3, 883.3],
            [902, 1202, 765.04],
            [904, 1206, 767.44],
            [906, 1209.2, 769.8],
            [908, 1211.6, 772.16],
            [910, 1214, 774.56],
            [912, 1216.4, 776.92],
            [914, 1218.8, 778.4],
            [916, 1220.8, 778.04],
            [918, 1222.4, 777.72],
            [920, 1224, 777.36],
            [922, 1225.6, 777.04],
            [924, 1227.2, 776.64],
            [926, 1226.8, 772.36],
            [928, 1224.4, 768.08],
            [930, 1222, 763.84],
            [932, 1219.6, 752.28],
            [934, 1217.2, 737.56],
            [936, 1215.6, 722.88],
            [938, 1214.8, 708.16],
            [940, 1214, 693.44],
            [942, 1213.2, 678.72],
            [944, 1212.4, 660.52],
            [946, 1210.4, 641.08],
            [948, 1207.2, 621.64],
            [950, 1204, 602.24],
            [952, 1200.8, 583.4],
            [954, 1197.6, 568.92],
            [956, 1194, 554.48],
            [958, 1190, 540.04],
            [960, 1186, 525.56],
            [962, 1182, 511.12],
            [964, 1178, 495.36],
            [966, 1173.2, 473.32],
            [968, 1167.6, 451.32],
            [970, 1162, 429.32],
            [972, 1156.4, 415.28],
            [974, 1150.8, 402.28],
            [976, 1144, 389.288],
            [978, 1136, 374.944],
            [980, 1128, 359.656],
            [982, 1120, 344.372],
            [984, 1112, 329.084],
            [986, 1102.4, 313.796],
            [988, 1091.2, 298.508],
            [990, 1080, 283.22],
            [992, 1068.8, 267.932],
            [994, 1057.6, 252.648],
            [996, 1046.4, 237.36],
            [998, 1035.2, 222.072],
            [1000, 1024, 206.784]])

        vlamHbOHb[:,1] = vlamHbOHb[:,1] * 2.303
        vlamHbOHb[:,2] = vlamHbOHb[:,2] * 2.303
        
    #  **************************************************************************
    #  **************************************************************************
    #  **************************************************************************
    #  **************************************************************************
    
    elif WhichSpectrum == 2:    
        
            Citation=print('J.M. Schmitt, "Optical Measurement of Blood Oxygenation by Implantable Telemetry," Technical Report G558-15, Stanford." \nM.K. Moaveni, "A Multiple Scattering Field Theory Applied to Whole Blood," Ph.D. dissertation, Dept. of Electrical Engineering, University of Washington, 1970');
    
            # These values for the molar extinction coefficient e in [cm-1/(moles/liter)] were compiled by Scott Prahl (prahl@ece.ogi.edu) using data from
            # 
            #  J.M. Schmitt, "Optical Measurement of Blood Oxygenation by Implantable Telemetry," Technical Report G558-15, Stanford."
            #  M.K. Moaveni, "A Multiple Scattering Field Theory Applied to Whole Blood," Ph.D. dissertation, Dept. of Electrical Engineering, University of Washington, 1970.
            #  To convert this data to absorbance A, multiply by the molar concentration and the pathlength. For example, if x is the number of grams per liter and a 1 cm cuvette is being used, then the absorbance is given by
            # 
            #          (e) [(1/cm)/(moles/liter)] (x) [g/liter] (1) [cm]
            #    A =  ---------------------------------------------------
            #                            66,500 [g/mole]
            # 
            #  using 66,500 as the gram molecular weight of hemoglobin.
            #  To convert this data to absorption coefficient in (cm-1), multiply by the molar concentration and 2.303,
            # 
            #  µa = (2.303) e (x g/liter)/(66,500 g Hb/mole)
            
            vlamHbOHb = np.array([
                [630, 680, 4280],
                [640, 440, 3640],
                [650, 380, 3420],
                [660, 320, 3200],
                [670, 320, 3080],
                [680, 320, 2960],
                [690, 280, 2560],
                [700, 320, 2160],
                [710, 340, 1840],
                [720, 360, 1520],
                [730, 400, 1500],
                [740, 440, 1520],
                [750, 520, 1620],
                [760, 600, 1720],
                [770, 660, 1420],
                [780, 720, 1120],
                [790, 760, 1020],
                [800, 800, 920],
                [810, 860, 880],
                [820, 920, 840],
                [830, 980, 840],
                [840, 1040, 840],
                [850, 1060, 800],
                [860, 1080, 840],
                [870, 1120, 840],
                [880, 1160, 840],
                [890, 1180, 860],
                [900, 1200, 880],
                [910, 1220, 920],
                [920, 1240, 880],
                [930, 1240, 800],
                [940, 1200, 800],
                [950, 1200, 720]])
   
    
    
            vlamHbOHb[:,1] = vlamHbOHb[:,1] * 2.303;
            vlamHbOHb[:,2] = vlamHbOHb[:,2] * 2.303;
            
    #         %**************************************************************************
    #         %**************************************************************************
    #         %**************************************************************************
    #         %******************************************************************
            
    
    elif WhichSpectrum == 3:
    
        Citation=print('S. Takatani and M. D. Graham, "Theoretical analysis of diffuse reflectance from a two-layer tissue model," IEEE Trans. Biomed. Eng., BME-26, 656--664, (1987). ')
    
    #         These values for the molar extinction coefficient e in [cm-1/(moles/liter)] were compiled by Scott Prahl (prahl@ece.ogi.edu) using data from
    #         
    #          S. Takatani and M. D. Graham, "Theoretical analysis of diffuse reflectance from a two-layer tissue model," IEEE Trans. Biomed. Eng., BME-26, 656--664, (1987).
    #          To convert this data to absorbance A, multiply by the molar concentration and the pathlength. For example, if x is the number of grams per liter and a 1 cm cuvette is being used, then the absorbance is given by
    #         
    #                  (e) [(1/cm)/(moles/liter)] (x) [g/liter] (1) [cm]
    #            A =  ---------------------------------------------------
    #                                    66,500 [g/mole]
    #         
    #          using 66,500 as the gram molecular weight of hemoglobin.
    #          To convert this data to absorption coefficient in (cm-1), multiply by the molar concentration and 2.303,
    #        
    #          µa = (2.303) e (x g/liter)/(66,500 g Hb/mole)
    #         where x is the number of grams per liter. A typical value of x for whole blood is x=150 g Hb/liter.
    
        vlamHbOHb = np.array([
            [450, 68000, 58000],
            [460, 45040, 20600],
            [480, 27360, 13360],
            [500, 20200, 16360],
            [507, 19240, 19240],
            [510, 19040, 20000],
            [520, 23520, 25080],
            [522, 25680, 25680],
            [540, 57080, 41120],
            [542, 57480, 44000],
            [549, 49840, 49840],
            [555, 36000, 52160],
            [560, 33880, 50160],
            [569, 45080, 45080],
            [577, 61480, 36800],
            [579, 54920, 35440],
            [586, 28920, 28920],
            [600, 3200, 14600],
            [605, 1860, 9496],
            [615, 1152, 5776],
            [625, 732, 4400],
            [635, 488, 3796],
            [645, 396, 3436],
            [655, 340, 3244],
            [665, 292, 3156],
            [675, 288, 3028],
            [685, 272, 2796],
            [695, 280, 2424],
            [705, 300, 1988],
            [715, 328, 1628],
            [725, 368, 1464],
            [735, 412, 1464],
            [745, 480, 1616],
            [755, 556, 1756],
            [765, 616, 1640],
            [775, 684, 1340],
            [785, 736, 1040],
            [795, 776, 964],
            [805, 880, 896],
            [815, 880, 880],
            [825, 952, 832],
            [835, 996, 820],
            [845, 1048, 820],
            [855, 1068, 820],
            [865, 1116, 820],
            [875, 1140, 848],
            [885, 1168, 832],
            [895, 1188, 884],
            [905, 1208, 896],
            [915, 1220, 924],
            [925, 1228, 860],
            [935, 1216, 848],
            [945, 1212, 756],
            [955, 1196, 704],
            [965, 1176, 616],
            [975, 1148, 552],
            [985, 1108, 424],
            [995, 1052, 372]])

        vlamHbOHb[:,1] = vlamHbOHb[:,1] * 2.303;
        vlamHbOHb[:,2] = vlamHbOHb[:,2] * 2.303;
        
   ######################################
    # % ABSORPTION SPECTRUMOF H20
    # % FROM G. M. Hale and M. R. Querry, "Optical constants of water in the 200nm to
    # % 200µm wavelength region," Appl. Opt., 12, 555--563, (1973).
    # %
    # % ON THE WEB AT
    # % http://omlc.ogi.edu/spectra/water/abs/index.html
    # %

    vlamH2O = np.array([
        [200.00, 0.069000],
        [225.00, 0.027400],
        [250.00, 0.016800],
        [275.00, 0.010700],
        [300.00, 0.0067000],
        [325.00, 0.0041800],
        [350.00, 0.0023300],
        [375.00, 0.0011700],
        [400.00, 0.00058000],
        [425.00, 0.00038000],
        [450.00, 0.00028000],
        [475.00, 0.00024700],
        [500.00, 0.00025000],
        [525.00, 0.00032000],
        [550.00, 0.00045000],
        [575.00, 0.00079000],
        [600.00, 0.0023000],
        [625.00, 0.0028000],
        [650.00, 0.0032000],
        [675.00, 0.0041500],
        [700.00, 0.0060000],
        [725.00, 0.015900],
        [750.00, 0.026000],
        [775.00, 0.024000],
        [800.00, 0.020000],
        [810.00, 0.019858],
        [820.00, 0.023907],
        [825.00, 0.028000],
        [830.00, 0.029069],
        [840.00, 0.034707],
        [850.00, 0.043000],
        [860.00, 0.046759],
        [870.00, 0.051999],
        [875.00, 0.056000],
        [880.00, 0.055978],
        [890.00, 0.060432],
        [900.00, 0.068000],
        [910.00, 0.072913],
        [920.00, 0.10927],
        [925.00, 0.14400],
        [930.00, 0.17296],
        [940.00, 0.26737],
        [950.00, 0.39000],
        [960.00, 0.42000],
        [970.00, 0.45000],
        [975.00, 0.45000],
        [980.00, 0.43000],
        [990.00, 0.41000],
        [1000.0, 0.36000]])

    ### Extinction coefficient for lipid. ###
    #  I got this from Brian Pogue who got this from Matcher and Cope (DAB)
    #  In units of per mm and convert to per cm.

    vlamLipid = np.array([
        [650, 0.000080],
        [652, 0.000080],
        [654, 0.000080],
        [656, 0.000080],
        [658, 0.000080],
        [660, 0.000080],
        [662, 0.000080],
        [664, 0.000080],
        [666, 0.000080],
        [668, 0.000080],
        [670, 0.000080],
        [672, 0.000080],
        [674, 0.000080],
        [676, 0.000080],
        [678, 0.000080],
        [680, 0.000080],
        [682, 0.000080],
        [684, 0.000080],
        [686, 0.000080],
        [688, 0.000080],
        [690, 0.000080],
        [692, 0.000080],
        [694, 0.000080],
        [696, 0.000080],
        [698, 0.000080],
        [700, 0.000080],
        [702, 0.000080],
        [704, 0.000080],
        [706, 0.000080],
        [708, 0.000080],
        [710, 0.000080],
        [712, 0.000080],
        [714, 0.000080],
        [716, 0.000080],
        [718, 0.000080],
        [720, 0.000096],
        [722, 0.000101],
        [724, 0.000096],
        [726, 0.000100],
        [728, 0.000090],
        [730, 0.000089],
        [732, 0.000093],
        [734, 0.000105],
        [736, 0.000123],
        [738, 0.000148],
        [740, 0.000179],
        [742, 0.000214],
        [744, 0.000254],
        [746, 0.000296],
        [748, 0.000341],
        [750, 0.000385],
        [752, 0.000426],
        [754, 0.000462],
        [756, 0.000491],
        [758, 0.000510],
        [760, 0.000515],
        [762, 0.000498],
        [764, 0.000458],
        [766, 0.000399],
        [768, 0.000333],
        [770, 0.000267],
        [772, 0.000206],
        [774, 0.000155],
        [776, 0.000106],
        [778, 0.000063],
        [780, 0.000033],
        [782, 0.000021],
        [784, 0.000023],
        [786, 0.000029],
        [788, 0.000027],
        [790, 0.000017],
        [792, 0.000006],
        [794, 0.000000],
        [796, 0.000002],
        [798, 0.000009],
        [800, 0.000021],
        [802, 0.000037],
        [804, 0.000055],
        [806, 0.000073],
        [808, 0.000091],
        [810, 0.000109],
        [812, 0.000128],
        [814, 0.000146],
        [816, 0.000163],
        [818, 0.000178],
        [820, 0.000193],
        [822, 0.000205],
        [824, 0.000214],
        [826, 0.000219],
        [828, 0.000221],
        [830, 0.000219],
        [832, 0.000213],
        [834, 0.000205],
        [836, 0.000193],
        [838, 0.000180],
        [840, 0.000167],
        [842, 0.000155],
        [844, 0.000145],
        [846, 0.000138],
        [848, 0.000135],
        [852, 0.000130],
        [854, 0.000130],
        [856, 0.000137],
        [858, 0.000153],
        [860, 0.000179],
        [862, 0.000216],
        [864, 0.000265],
        [866, 0.000329],
        [868, 0.000408],
        [870, 0.000505],
        [872, 0.000618],
        [874, 0.000758],
        [876, 0.000919],
        [878, 0.001103],
        [880, 0.001314],
        [882, 0.001552],
        [884, 0.001809],
        [886, 0.002082],
        [888, 0.002365],
        [890, 0.002653],
        [892, 0.002939],
        [894, 0.003224],
        [896, 0.003521],
        [898, 0.003833],
        [900, 0.004168],
        [902, 0.004545],
        [904, 0.004976],
        [906, 0.005451],
        [908, 0.005969],
        [910, 0.006522],
        [912, 0.007106],
        [914, 0.007686],
        [916, 0.008235],
        [918, 0.008744],
        [920, 0.009179],
        [922, 0.009484],
        [924, 0.009644],
        [926, 0.009602],
        [928, 0.009363],
        [930, 0.008895],
        [932, 0.008275],
        [934, 0.007497],
        [936, 0.006648],
        [938, 0.005792],
        [940, 0.004947],
        [942, 0.004166],
        [944, 0.003467],
        [946, 0.002859],
        [948, 0.002337],
        [950, 0.001898],
        [952, 0.001544],
        [954, 0.001258],
        [956, 0.001022],
        [958, 0.000830],
        [960, 0.000678],
        [962, 0.000553],
        [964, 0.000451],
        [966, 0.000365],
        [968, 0.000298],
        [970, 0.000244],
        [972, 0.000204],
        [974, 0.000176],
        [976, 0.000163],
        [978, 0.000165],
        [980, 0.000185],
        [982, 0.000221],
        [984, 0.000278],
        [986, 0.000351],
        [988, 0.000448],
        [990, 0.000564],
        [992, 0.000698],
        [994, 0.000853],
        [996, 0.001027],
        [998, 0.001215],
        [1000, 0.001420],
        [1002, 0.001643],
        [1004, 0.001882],
        [1006, 0.002127],
        [1008, 0.002371],
        [1010, 0.002619],
        [1012, 0.002858],
        [1014, 0.003077],
        [1016, 0.003279],
        [1018, 0.003463],
        [1020, 0.003635],
        [1022, 0.003793],
        [1024, 0.003941],
        [1026, 0.004084],
        [1028, 0.004220],
        [1030, 0.004338],
        [1032, 0.004438],
        [1034, 0.004505],
        [1036, 0.004537],
        [1038, 0.004524],
        [1040, 0.004468],
        [1042, 0.004365],
        [1044, 0.004229],
        [1046, 0.004065],
        [1048, 0.003880],
        [1050, 0.003677],
        [1052, 0.003469],
        [1054, 0.003259],
        [1056, 0.003051],
        [1058, 0.002843]]) * 10


    vlamAA3 = np.array([
        [6.5000000e+002,  1.1361272e+000],
        [6.5050000e+002,  1.1318779e+000],
        [6.5100000e+002,  1.1276285e+000],
        [6.5150000e+002,  1.1233792e+000],
        [6.5200000e+002,  1.1191298e+000],
        [6.5250000e+002,  1.1135125e+000],
        [6.5300000e+002,  1.1078952e+000],
        [6.5350000e+002,  1.1022779e+000],
        [6.5400000e+002,  1.0966605e+000],
        [6.5450000e+002,  1.0903251e+000],
        [6.5500000e+002,  1.0839896e+000],
        [6.5550000e+002,  1.0776542e+000],
        [6.5600000e+002,  1.0713187e+000],
        [6.5650000e+002,  1.0643820e+000],
        [6.5700000e+002,  1.0574453e+000],
        [6.5750000e+002,  1.0505086e+000],
        [6.5800000e+002,  1.0435719e+000],
        [6.5850000e+002,  1.0362062e+000],
        [6.5900000e+002,  1.0288404e+000],
        [6.5950000e+002,  1.0214747e+000],
        [6.6000000e+002,  1.0141090e+000],
        [6.6050000e+002,  1.0062588e+000],
        [6.6100000e+002,  9.9840857e-001],
        [6.6150000e+002,  9.9055837e-001],
        [6.6200000e+002,  9.8270816e-001],
        [6.6250000e+002,  9.7439221e-001],
        [6.6300000e+002,  9.6607626e-001],
        [6.6350000e+002,  9.5776031e-001],
        [6.6400000e+002,  9.4944435e-001],
        [6.6450000e+002,  9.4080158e-001],
        [6.6500000e+002,  9.3215880e-001],
        [6.6550000e+002,  9.2351603e-001],
        [6.6600000e+002,  9.1487325e-001],
        [6.6650000e+002,  9.0655165e-001],
        [6.6700000e+002,  8.9823004e-001],
        [6.6750000e+002,  8.8990843e-001],
        [6.6800000e+002,  8.8158682e-001],
        [6.6850000e+002,  8.7315553e-001],
        [6.6900000e+002,  8.6472424e-001],
        [6.6950000e+002,  8.5629295e-001],
        [6.7000000e+002,  8.4786166e-001],
        [6.7050000e+002,  8.3951725e-001],
        [6.7100000e+002,  8.3117284e-001],
        [6.7150000e+002,  8.2282843e-001],
        [6.7200000e+002,  8.1448402e-001],
        [6.7250000e+002,  8.0625998e-001],
        [6.7300000e+002,  7.9803594e-001],
        [6.7350000e+002,  7.8981191e-001],
        [6.7400000e+002,  7.8158787e-001],
        [6.7450000e+002,  7.7355802e-001],
        [6.7500000e+002,  7.6552817e-001],
        [6.7550000e+002,  7.5749832e-001],
        [6.7600000e+002,  7.4946847e-001],
        [6.7650000e+002,  7.4157127e-001],
        [6.7700000e+002,  7.3367407e-001],
        [6.7750000e+002,  7.2577687e-001],
        [6.7800000e+002,  7.1787968e-001],
        [6.7850000e+002,  7.1103726e-001],
        [6.7900000e+002,  7.0419484e-001],
        [6.7950000e+002,  6.9735242e-001],
        [6.8000000e+002,  6.9051001e-001],
        [6.8050000e+002,  6.8506908e-001],
        [6.8100000e+002,  6.7962815e-001],
        [6.8150000e+002,  6.7418722e-001],
        [6.8200000e+002,  6.6874629e-001],
        [6.8250000e+002,  6.6413673e-001],
        [6.8300000e+002,  6.5952716e-001],
        [6.8350000e+002,  6.5491759e-001],
        [6.8400000e+002,  6.5030802e-001],
        [6.8450000e+002,  6.4610502e-001],
        [6.8500000e+002,  6.4190202e-001],
        [6.8550000e+002,  6.3769903e-001],
        [6.8600000e+002,  6.3349603e-001],
        [6.8650000e+002,  6.2987997e-001],
        [6.8700000e+002,  6.2626392e-001],
        [6.8750000e+002,  6.2264786e-001],
        [6.8800000e+002,  6.1903181e-001],
        [6.8850000e+002,  6.1593488e-001],
        [6.8900000e+002,  6.1283796e-001],
        [6.8950000e+002,  6.0974103e-001],
        [6.9000000e+002,  6.0664411e-001],
        [6.9050000e+002,  6.0393052e-001],
        [6.9100000e+002,  6.0121692e-001],
        [6.9150000e+002,  5.9850333e-001],
        [6.9200000e+002,  5.9578974e-001],
        [6.9250000e+002,  5.9354479e-001],
        [6.9300000e+002,  5.9129984e-001],
        [6.9350000e+002,  5.8905489e-001],
        [6.9400000e+002,  5.8680994e-001],
        [6.9450000e+002,  5.8419281e-001],
        [6.9500000e+002,  5.8157569e-001],
        [6.9550000e+002,  5.7895857e-001],
        [6.9600000e+002,  5.7634144e-001],
        [6.9650000e+002,  5.7395588e-001],
        [6.9700000e+002,  5.7157032e-001],
        [6.9750000e+002,  5.6918475e-001],
        [6.9800000e+002,  5.6679919e-001],
        [6.9850000e+002,  5.6423313e-001],
        [6.9900000e+002,  5.6166706e-001],
        [6.9950000e+002,  5.5910100e-001],
        [7.0000000e+002,  5.5653494e-001],
        [7.0050000e+002,  5.5386633e-001],
        [7.0100000e+002,  5.5119773e-001],
        [7.0150000e+002,  5.4852912e-001],
        [7.0200000e+002,  5.4586052e-001],
        [7.0250000e+002,  5.4295036e-001],
        [7.0300000e+002,  5.4004020e-001],
        [7.0350000e+002,  5.3713004e-001],
        [7.0400000e+002,  5.3421989e-001],
        [7.0450000e+002,  5.3157177e-001],
        [7.0500000e+002,  5.2892366e-001],
        [7.0550000e+002,  5.2627555e-001],
        [7.0600000e+002,  5.2362744e-001],
        [7.0650000e+002,  5.2040945e-001],
        [7.0700000e+002,  5.1719147e-001],
        [7.0750000e+002,  5.1397349e-001],
        [7.0800000e+002,  5.1075551e-001],
        [7.0850000e+002,  5.0755727e-001],
        [7.0900000e+002,  5.0435903e-001],
        [7.0950000e+002,  5.0116079e-001],
        [7.1000000e+002,  4.9796255e-001],
        [7.1050000e+002,  4.9508362e-001],
        [7.1100000e+002,  4.9220470e-001],
        [7.1150000e+002,  4.8932577e-001],
        [7.1200000e+002,  4.8644684e-001],
        [7.1250000e+002,  4.8309329e-001],
        [7.1300000e+002,  4.7973974e-001],
        [7.1350000e+002,  4.7638620e-001],
        [7.1400000e+002,  4.7303265e-001],
        [7.1450000e+002,  4.6986524e-001],
        [7.1500000e+002,  4.6669783e-001],
        [7.1550000e+002,  4.6353042e-001],
        [7.1600000e+002,  4.6036302e-001],
        [7.1650000e+002,  4.5755399e-001],
        [7.1700000e+002,  4.5474496e-001],
        [7.1750000e+002,  4.5193593e-001],
        [7.1800000e+002,  4.4912691e-001],
        [7.1850000e+002,  4.4661044e-001],
        [7.1900000e+002,  4.4409397e-001],
        [7.1950000e+002,  4.4157750e-001],
        [7.2000000e+002,  4.3906103e-001],
        [7.2050000e+002,  4.3703350e-001],
        [7.2100000e+002,  4.3500597e-001],
        [7.2150000e+002,  4.3297844e-001],
        [7.2200000e+002,  4.3095091e-001],
        [7.2250000e+002,  4.2950659e-001],
        [7.2300000e+002,  4.2806227e-001],
        [7.2350000e+002,  4.2661795e-001],
        [7.2400000e+002,  4.2517362e-001],
        [7.2450000e+002,  4.2406847e-001],
        [7.2500000e+002,  4.2296331e-001],
        [7.2550000e+002,  4.2185816e-001],
        [7.2600000e+002,  4.2075300e-001],
        [7.2650000e+002,  4.1981891e-001],
        [7.2700000e+002,  4.1888482e-001],
        [7.2750000e+002,  4.1795073e-001],
        [7.2800000e+002,  4.1701664e-001],
        [7.2850000e+002,  4.1654428e-001],
        [7.2900000e+002,  4.1607191e-001],
        [7.2950000e+002,  4.1559955e-001],
        [7.3000000e+002,  4.1512718e-001],
        [7.3050000e+002,  4.1505277e-001],
        [7.3100000e+002,  4.1497836e-001],
        [7.3150000e+002,  4.1490395e-001],
        [7.3200000e+002,  4.1482954e-001],
        [7.3250000e+002,  4.1477623e-001],
        [7.3300000e+002,  4.1472291e-001],
        [7.3350000e+002,  4.1466960e-001],
        [7.3400000e+002,  4.1461628e-001],
        [7.3450000e+002,  4.1479630e-001],
        [7.3500000e+002,  4.1497632e-001],
        [7.3550000e+002,  4.1515633e-001],
        [7.3600000e+002,  4.1533635e-001],
        [7.3650000e+002,  4.1555474e-001],
        [7.3700000e+002,  4.1577314e-001],
        [7.3750000e+002,  4.1599153e-001],
        [7.3800000e+002,  4.1620993e-001],
        [7.3850000e+002,  4.1670986e-001],
        [7.3900000e+002,  4.1720980e-001],
        [7.3950000e+002,  4.1770974e-001],
        [7.4000000e+002,  4.1820967e-001],
        [7.4050000e+002,  4.1892740e-001],
        [7.4100000e+002,  4.1964513e-001],
        [7.4150000e+002,  4.2036286e-001],
        [7.4200000e+002,  4.2108058e-001],
        [7.4250000e+002,  4.2179731e-001],
        [7.4300000e+002,  4.2251404e-001],
        [7.4350000e+002,  4.2323076e-001],
        [7.4400000e+002,  4.2394749e-001],
        [7.4450000e+002,  4.2453814e-001],
        [7.4500000e+002,  4.2512878e-001],
        [7.4550000e+002,  4.2571943e-001],
        [7.4600000e+002,  4.2631007e-001],
        [7.4650000e+002,  4.2678948e-001],
        [7.4700000e+002,  4.2726889e-001],
        [7.4750000e+002,  4.2774830e-001],
        [7.4800000e+002,  4.2822770e-001],
        [7.4850000e+002,  4.2880242e-001],
        [7.4900000e+002,  4.2937714e-001],
        [7.4950000e+002,  4.2995185e-001],
        [7.5000000e+002,  4.3052657e-001],
        [7.5050000e+002,  4.3148348e-001],
        [7.5100000e+002,  4.3244039e-001],
        [7.5150000e+002,  4.3339730e-001],
        [7.5200000e+002,  4.3435422e-001],
        [7.5250000e+002,  4.3511742e-001],
        [7.5300000e+002,  4.3588062e-001],
        [7.5350000e+002,  4.3664382e-001],
        [7.5400000e+002,  4.3740702e-001],
        [7.5450000e+002,  4.3781744e-001],
        [7.5500000e+002,  4.3822787e-001],
        [7.5550000e+002,  4.3863830e-001],
        [7.5600000e+002,  4.3904872e-001],
        [7.5650000e+002,  4.3940898e-001],
        [7.5700000e+002,  4.3976923e-001],
        [7.5750000e+002,  4.4012949e-001],
        [7.5800000e+002,  4.4048975e-001],
        [7.5850000e+002,  4.4084169e-001],
        [7.5900000e+002,  4.4119364e-001],
        [7.5950000e+002,  4.4154558e-001],
        [7.6000000e+002,  4.4189753e-001],
        [7.6050000e+002,  4.4222139e-001],
        [7.6100000e+002,  4.4254525e-001],
        [7.6150000e+002,  4.4286911e-001],
        [7.6200000e+002,  4.4319298e-001],
        [7.6250000e+002,  4.4377194e-001],
        [7.6300000e+002,  4.4435090e-001],
        [7.6350000e+002,  4.4492986e-001],
        [7.6400000e+002,  4.4550882e-001],
        [7.6450000e+002,  4.4615044e-001],
        [7.6500000e+002,  4.4679205e-001],
        [7.6550000e+002,  4.4743367e-001],
        [7.6600000e+002,  4.4807529e-001],
        [7.6650000e+002,  4.4866327e-001],
        [7.6700000e+002,  4.4925126e-001],
        [7.6750000e+002,  4.4983925e-001],
        [7.6800000e+002,  4.5042724e-001],
        [7.6850000e+002,  4.5104584e-001],
        [7.6900000e+002,  4.5166445e-001],
        [7.6950000e+002,  4.5228305e-001],
        [7.7000000e+002,  4.5290166e-001],
        [7.7050000e+002,  4.5346397e-001],
        [7.7100000e+002,  4.5402629e-001],
        [7.7150000e+002,  4.5458860e-001],
        [7.7200000e+002,  4.5515091e-001],
        [7.7250000e+002,  4.5580540e-001],
        [7.7300000e+002,  4.5645989e-001],
        [7.7350000e+002,  4.5711438e-001],
        [7.7400000e+002,  4.5776888e-001],
        [7.7450000e+002,  4.5874658e-001],
        [7.7500000e+002,  4.5972429e-001],
        [7.7550000e+002,  4.6070200e-001],
        [7.7600000e+002,  4.6167971e-001],
        [7.7650000e+002,  4.6304014e-001],
        [7.7700000e+002,  4.6440057e-001],
        [7.7750000e+002,  4.6576100e-001],
        [7.7800000e+002,  4.6712143e-001],
        [7.7850000e+002,  4.6842946e-001],
        [7.7900000e+002,  4.6973749e-001],
        [7.7950000e+002,  4.7104552e-001],
        [7.8000000e+002,  4.7235355e-001],
        [7.8050000e+002,  4.7364174e-001],
        [7.8100000e+002,  4.7492993e-001],
        [7.8150000e+002,  4.7621812e-001],
        [7.8200000e+002,  4.7750631e-001],
        [7.8250000e+002,  4.7897505e-001],
        [7.8300000e+002,  4.8044380e-001],
        [7.8350000e+002,  4.8191254e-001],
        [7.8400000e+002,  4.8338128e-001],
        [7.8450000e+002,  4.8455794e-001],
        [7.8500000e+002,  4.8573459e-001],
        [7.8550000e+002,  4.8691125e-001],
        [7.8600000e+002,  4.8808791e-001],
        [7.8650000e+002,  4.8910580e-001],
        [7.8700000e+002,  4.9012369e-001],
        [7.8750000e+002,  4.9114158e-001],
        [7.8800000e+002,  4.9215948e-001],
        [7.8850000e+002,  4.9300490e-001],
        [7.8900000e+002,  4.9385033e-001],
        [7.8950000e+002,  4.9469576e-001],
        [7.9000000e+002,  4.9554119e-001],
        [7.9050000e+002,  4.9630072e-001],
        [7.9100000e+002,  4.9706025e-001],
        [7.9150000e+002,  4.9781978e-001],
        [7.9200000e+002,  4.9857931e-001],
        [7.9250000e+002,  5.0001031e-001],
        [7.9300000e+002,  5.0144130e-001],
        [7.9350000e+002,  5.0287230e-001],
        [7.9400000e+002,  5.0430330e-001],
        [7.9450000e+002,  5.0577552e-001],
        [7.9500000e+002,  5.0724773e-001],
        [7.9550000e+002,  5.0871995e-001],
        [7.9600000e+002,  5.1019217e-001],
        [7.9650000e+002,  5.1177102e-001],
        [7.9700000e+002,  5.1334987e-001],
        [7.9750000e+002,  5.1492872e-001],
        [7.9800000e+002,  5.1650758e-001],
        [7.9850000e+002,  5.1767295e-001],
        [7.9900000e+002,  5.1883832e-001],
        [7.9950000e+002,  5.2000370e-001],
        [8.0000000e+002,  5.2116907e-001],
        [8.0050000e+002,  5.2170964e-001],
        [8.0100000e+002,  5.2225020e-001],
        [8.0150000e+002,  5.2279077e-001],
        [8.0200000e+002,  5.2333133e-001],
        [8.0250000e+002,  5.2390635e-001],
        [8.0300000e+002,  5.2448137e-001],
        [8.0350000e+002,  5.2505639e-001],
        [8.0400000e+002,  5.2563141e-001],
        [8.0450000e+002,  5.2629161e-001],
        [8.0500000e+002,  5.2695181e-001],
        [8.0550000e+002,  5.2761202e-001],
        [8.0600000e+002,  5.2827222e-001],
        [8.0650000e+002,  5.2901213e-001],
        [8.0700000e+002,  5.2975205e-001],
        [8.0750000e+002,  5.3049196e-001],
        [8.0800000e+002,  5.3123187e-001],
        [8.0850000e+002,  5.3190100e-001],
        [8.0900000e+002,  5.3257012e-001],
        [8.0950000e+002,  5.3323925e-001],
        [8.1000000e+002,  5.3390837e-001],
        [8.1050000e+002,  5.3437521e-001],
        [8.1100000e+002,  5.3484206e-001],
        [8.1150000e+002,  5.3530890e-001],
        [8.1200000e+002,  5.3577574e-001],
        [8.1250000e+002,  5.3610282e-001],
        [8.1300000e+002,  5.3642990e-001],
        [8.1350000e+002,  5.3675698e-001],
        [8.1400000e+002,  5.3708406e-001],
        [8.1450000e+002,  5.3738179e-001],
        [8.1500000e+002,  5.3767952e-001],
        [8.1550000e+002,  5.3797726e-001],
        [8.1600000e+002,  5.3827499e-001],
        [8.1650000e+002,  5.3858777e-001],
        [8.1700000e+002,  5.3890055e-001],
        [8.1750000e+002,  5.3921333e-001],
        [8.1800000e+002,  5.3952611e-001],
        [8.1850000e+002,  5.3983715e-001],
        [8.1900000e+002,  5.4014818e-001],
        [8.1950000e+002,  5.4045922e-001],
        [8.2000000e+002,  5.4077026e-001],
        [8.2050000e+002,  5.4086522e-001],
        [8.2100000e+002,  5.4096019e-001],
        [8.2150000e+002,  5.4105515e-001],
        [8.2200000e+002,  5.4115011e-001],
        [8.2250000e+002,  5.4097464e-001],
        [8.2300000e+002,  5.4079916e-001],
        [8.2350000e+002,  5.4062369e-001],
        [8.2400000e+002,  5.4044822e-001],
        [8.2450000e+002,  5.4014545e-001],
        [8.2500000e+002,  5.3984269e-001],
        [8.2550000e+002,  5.3953992e-001],
        [8.2600000e+002,  5.3923716e-001],
        [8.2650000e+002,  5.3884421e-001],
        [8.2700000e+002,  5.3845126e-001],
        [8.2750000e+002,  5.3805832e-001],
        [8.2800000e+002,  5.3766537e-001],
        [8.2850000e+002,  5.3726601e-001],
        [8.2900000e+002,  5.3686664e-001],
        [8.2950000e+002,  5.3646728e-001],
        [8.3000000e+002,  5.3606792e-001],
        [8.3050000e+002,  5.3569046e-001],
        [8.3100000e+002,  5.3531300e-001],
        [8.3150000e+002,  5.3493554e-001],
        [8.3200000e+002,  5.3455808e-001],
        [8.3250000e+002,  5.3425944e-001],
        [8.3300000e+002,  5.3396080e-001],
        [8.3350000e+002,  5.3366216e-001],
        [8.3400000e+002,  5.3336352e-001],
        [8.3450000e+002,  5.3316411e-001],
        [8.3500000e+002,  5.3296470e-001],
        [8.3550000e+002,  5.3276529e-001],
        [8.3600000e+002,  5.3256588e-001],
        [8.3650000e+002,  5.3247892e-001],
        [8.3700000e+002,  5.3239197e-001],
        [8.3750000e+002,  5.3230501e-001],
        [8.3800000e+002,  5.3221806e-001],
        [8.3850000e+002,  5.3211789e-001],
        [8.3900000e+002,  5.3201773e-001],
        [8.3950000e+002,  5.3191756e-001],
        [8.4000000e+002,  5.3181739e-001],
        [8.4050000e+002,  5.3159895e-001],
        [8.4100000e+002,  5.3138050e-001],
        [8.4150000e+002,  5.3116205e-001],
        [8.4200000e+002,  5.3094361e-001],
        [8.4250000e+002,  5.3066563e-001],
        [8.4300000e+002,  5.3038766e-001],
        [8.4350000e+002,  5.3010968e-001],
        [8.4400000e+002,  5.2983171e-001],
        [8.4450000e+002,  5.2956491e-001],
        [8.4500000e+002,  5.2929812e-001],
        [8.4550000e+002,  5.2903132e-001],
        [8.4600000e+002,  5.2876453e-001],
        [8.4650000e+002,  5.2859686e-001],
        [8.4700000e+002,  5.2842920e-001],
        [8.4750000e+002,  5.2826153e-001],
        [8.4800000e+002,  5.2809386e-001],
        [8.4850000e+002,  5.2801195e-001],
        [8.4900000e+002,  5.2793003e-001],
        [8.4950000e+002,  5.2784811e-001],
        [8.5000000e+002,  5.2776619e-001],
        [8.5050000e+002,  5.2746834e-001],
        [8.5100000e+002,  5.2717049e-001],
        [8.5150000e+002,  5.2687263e-001],
        [8.5200000e+002,  5.2657478e-001],
        [8.5250000e+002,  5.2601940e-001],
        [8.5300000e+002,  5.2546403e-001],
        [8.5350000e+002,  5.2490865e-001],
        [8.5400000e+002,  5.2435328e-001],
        [8.5450000e+002,  5.2352321e-001],
        [8.5500000e+002,  5.2269315e-001],
        [8.5550000e+002,  5.2186309e-001],
        [8.5600000e+002,  5.2103302e-001],
        [8.5650000e+002,  5.2004711e-001],
        [8.5700000e+002,  5.1906120e-001],
        [8.5750000e+002,  5.1807528e-001],
        [8.5800000e+002,  5.1708937e-001],
        [8.5850000e+002,  5.1629274e-001],
        [8.5900000e+002,  5.1549610e-001],
        [8.5950000e+002,  5.1469947e-001],
        [8.6000000e+002,  5.1390283e-001],
        [8.6050000e+002,  5.1338676e-001],
        [8.6100000e+002,  5.1287070e-001],
        [8.6150000e+002,  5.1235463e-001],
        [8.6200000e+002,  5.1183857e-001],
        [8.6250000e+002,  5.1133847e-001],
        [8.6300000e+002,  5.1083837e-001],
        [8.6350000e+002,  5.1033827e-001],
        [8.6400000e+002,  5.0983817e-001],
        [8.6450000e+002,  5.0916645e-001],
        [8.6500000e+002,  5.0849474e-001],
        [8.6550000e+002,  5.0782302e-001],
        [8.6600000e+002,  5.0715131e-001],
        [8.6650000e+002,  5.0635557e-001],
        [8.6700000e+002,  5.0555984e-001],
        [8.6750000e+002,  5.0476411e-001],
        [8.6800000e+002,  5.0396838e-001],
        [8.6850000e+002,  5.0282658e-001],
        [8.6900000e+002,  5.0168478e-001],
        [8.6950000e+002,  5.0054299e-001],
        [8.7000000e+002,  4.9940119e-001],
        [8.7050000e+002,  4.9772793e-001],
        [8.7100000e+002,  4.9605467e-001],
        [8.7150000e+002,  4.9438141e-001],
        [8.7200000e+002,  4.9270815e-001],
        [8.7250000e+002,  4.9123397e-001],
        [8.7300000e+002,  4.8975980e-001],
        [8.7350000e+002,  4.8828562e-001],
        [8.7400000e+002,  4.8681145e-001],
        [8.7450000e+002,  4.8567142e-001],
        [8.7500000e+002,  4.8453139e-001],
        [8.7550000e+002,  4.8339137e-001],
        [8.7600000e+002,  4.8225134e-001],
        [8.7650000e+002,  4.8129036e-001],
        [8.7700000e+002,  4.8032938e-001],
        [8.7750000e+002,  4.7936841e-001],
        [8.7800000e+002,  4.7840743e-001],
        [8.7850000e+002,  4.7750085e-001],
        [8.7900000e+002,  4.7659428e-001],
        [8.7950000e+002,  4.7568770e-001],
        [8.8000000e+002,  4.7478112e-001],
        [8.8050000e+002,  4.7390575e-001],
        [8.8100000e+002,  4.7303039e-001],
        [8.8150000e+002,  4.7215502e-001],
        [8.8200000e+002,  4.7127965e-001],
        [8.8250000e+002,  4.7019206e-001],
        [8.8300000e+002,  4.6910447e-001],
        [8.8350000e+002,  4.6801688e-001],
        [8.8400000e+002,  4.6692929e-001],
        [8.8450000e+002,  4.6576740e-001],
        [8.8500000e+002,  4.6460552e-001],
        [8.8550000e+002,  4.6344363e-001],
        [8.8600000e+002,  4.6228175e-001],
        [8.8650000e+002,  4.6108543e-001],
        [8.8700000e+002,  4.5988910e-001],
        [8.8750000e+002,  4.5869278e-001],
        [8.8800000e+002,  4.5749646e-001],
        [8.8850000e+002,  4.5599046e-001],
        [8.8900000e+002,  4.5448446e-001],
        [8.8950000e+002,  4.5297846e-001],
        [8.9000000e+002,  4.5147246e-001],
        [8.9050000e+002,  4.5001098e-001],
        [8.9100000e+002,  4.4854950e-001],
        [8.9150000e+002,  4.4708802e-001],
        [8.9200000e+002,  4.4562653e-001],
        [8.9250000e+002,  4.4404932e-001],
        [8.9300000e+002,  4.4247210e-001],
        [8.9350000e+002,  4.4089488e-001],
        [8.9400000e+002,  4.3931766e-001],
        [8.9450000e+002,  4.3759003e-001],
        [8.9500000e+002,  4.3586239e-001],
        [8.9550000e+002,  4.3413476e-001],
        [8.9600000e+002,  4.3240713e-001],
        [8.9650000e+002,  4.3071228e-001],
        [8.9700000e+002,  4.2901743e-001],
        [8.9750000e+002,  4.2732258e-001],
        [8.9800000e+002,  4.2562773e-001],
        [8.9850000e+002,  4.2413598e-001],
        [8.9900000e+002,  4.2264422e-001],
        [8.9950000e+002,  4.2115247e-001],
        [9.0000000e+002,  4.1966072e-001],
        [9.0050000e+002,  4.1850323e-001],
        [9.0100000e+002,  4.1734574e-001],
        [9.0150000e+002,  4.1618825e-001],
        [9.0200000e+002,  4.1503076e-001],
        [9.0250000e+002,  4.1411717e-001],
        [9.0300000e+002,  4.1320358e-001],
        [9.0350000e+002,  4.1228999e-001],
        [9.0400000e+002,  4.1137640e-001],
        [9.0450000e+002,  4.1056765e-001],
        [9.0500000e+002,  4.0975889e-001],
        [9.0550000e+002,  4.0895014e-001],
        [9.0600000e+002,  4.0814138e-001],
        [9.0650000e+002,  4.0740348e-001],
        [9.0700000e+002,  4.0666558e-001],
        [9.0750000e+002,  4.0592768e-001],
        [9.0800000e+002,  4.0518978e-001],
        [9.0850000e+002,  4.0446823e-001],
        [9.0900000e+002,  4.0374669e-001],
        [9.0950000e+002,  4.0302514e-001],
        [9.1000000e+002,  4.0230360e-001],
        [9.1050000e+002,  4.0118053e-001],
        [9.1100000e+002,  4.0005746e-001],
        [9.1150000e+002,  3.9893439e-001],
        [9.1200000e+002,  3.9781132e-001],
        [9.1250000e+002,  3.9667644e-001],
        [9.1300000e+002,  3.9554156e-001],
        [9.1350000e+002,  3.9440668e-001],
        [9.1400000e+002,  3.9327180e-001],
        [9.1450000e+002,  3.9214808e-001],
        [9.1500000e+002,  3.9102437e-001],
        [9.1550000e+002,  3.8990065e-001],
        [9.1600000e+002,  3.8877694e-001],
        [9.1650000e+002,  3.8718648e-001],
        [9.1700000e+002,  3.8559602e-001],
        [9.1750000e+002,  3.8400556e-001],
        [9.1800000e+002,  3.8241511e-001],
        [9.1850000e+002,  3.8079458e-001],
        [9.1900000e+002,  3.7917406e-001],
        [9.1950000e+002,  3.7755354e-001],
        [9.2000000e+002,  3.7593302e-001],
        [9.2050000e+002,  3.7428388e-001],
        [9.2100000e+002,  3.7263474e-001],
        [9.2150000e+002,  3.7098560e-001],
        [9.2200000e+002,  3.6933647e-001],
        [9.2250000e+002,  3.6742515e-001],
        [9.2300000e+002,  3.6551383e-001],
        [9.2350000e+002,  3.6360251e-001],
        [9.2400000e+002,  3.6169119e-001],
        [9.2450000e+002,  3.5935763e-001],
        [9.2500000e+002,  3.5702408e-001],
        [9.2550000e+002,  3.5469052e-001],
        [9.2600000e+002,  3.5235696e-001],
        [9.2650000e+002,  3.5020949e-001],
        [9.2700000e+002,  3.4806201e-001],
        [9.2750000e+002,  3.4591454e-001],
        [9.2800000e+002,  3.4376706e-001],
        [9.2850000e+002,  3.4207196e-001],
        [9.2900000e+002,  3.4037685e-001],
        [9.2950000e+002,  3.3868174e-001],
        [9.3000000e+002,  3.3698664e-001],
        [9.3050000e+002,  3.3531468e-001],
        [9.3100000e+002,  3.3364273e-001],
        [9.3150000e+002,  3.3197077e-001],
        [9.3200000e+002,  3.3029882e-001],
        [9.3250000e+002,  3.2829975e-001],
        [9.3300000e+002,  3.2630069e-001],
        [9.3350000e+002,  3.2430162e-001],
        [9.3400000e+002,  3.2230256e-001],
        [9.3450000e+002,  3.2014129e-001],
        [9.3500000e+002,  3.1798002e-001],
        [9.3550000e+002,  3.1581875e-001],
        [9.3600000e+002,  3.1365747e-001],
        [9.3650000e+002,  3.1151386e-001],
        [9.3700000e+002,  3.0937026e-001],
        [9.3750000e+002,  3.0722665e-001],
        [9.3800000e+002,  3.0508304e-001],
        [9.3850000e+002,  3.0292138e-001],
        [9.3900000e+002,  3.0075971e-001],
        [9.3950000e+002,  2.9859805e-001],
        [9.4000000e+002,  2.9643639e-001],
        [9.4050000e+002,  2.9481181e-001],
        [9.4100000e+002,  2.9318724e-001],
        [9.4150000e+002,  2.9156266e-001],
        [9.4200000e+002,  2.8993809e-001],
        [9.4250000e+002,  2.8864886e-001],
        [9.4300000e+002,  2.8735962e-001],
        [9.4350000e+002,  2.8607039e-001],
        [9.4400000e+002,  2.8478116e-001],
        [9.4450000e+002,  2.8361146e-001],
        [9.4500000e+002,  2.8244176e-001],
        [9.4550000e+002,  2.8127206e-001],
        [9.4600000e+002,  2.8010236e-001],
        [9.4650000e+002,  2.7845904e-001],
        [9.4700000e+002,  2.7681573e-001],
        [9.4750000e+002,  2.7517241e-001],
        [9.4800000e+002,  2.7352909e-001],
        [9.4850000e+002,  2.7153475e-001],
        [9.4900000e+002,  2.6954041e-001],
        [9.4950000e+002,  2.6754607e-001],
        [9.5000000e+002,  2.6555173e-001]])
            

    exs = np.zeros([num_lam, 5])
    
    ### HbO, Hb ###
    ind=np.where(np.logical_and(lam>=250 , lam<=1000))[0]
    exs[ind,0]=np.interp(lam[ind,0], vlamHbOHb[:,0],vlamHbOHb[:,1])
    exs[ind,1]=np.interp(lam[ind,0], vlamHbOHb[:,0],vlamHbOHb[:,2])
    
    ### H2O ###
    ind=np.where(np.logical_and(lam>=200, lam<=1000))[0];
    exs[ind,2]=np.interp(lam[ind,0], vlamH2O[:,0], vlamH2O[:,1]);
    
    ### lipid ###
    ind=np.where(np.logical_and(lam>=650, lam<=1058));
    exs[ind,3]=np.interp(lam[ind,0], vlamLipid[:,0],vlamLipid[:,1]);
    
    ### AA3 ###
    ind=np.where(np.logical_and(lam>=650, lam<=950));
    exs[ind,4]=np.interp(lam[ind,0], vlamAA3[:,0],vlamAA3[:,1]);

    return  exs








