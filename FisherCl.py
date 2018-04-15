#! /usr/bin/env python
"""
  Name:
    FisherCl (branch quickCl)
  Purpose:
    Calculate Fisher matrices for angular power spectra C_l as observables
  Uses:
    crosspower.py (for the angular power spectra)
    pycamb (aka camb)
  Modification History:
    Written by Z Knight, 2017.08.27
    Added galaxy bias and lensing amplitude as one parameter per galaxy bin;
      ZK, 2017.09.01
    Modified weighting on A_i averaging to use WinK rather than DNDZ;
      ZK, 2017.09.06
    Added crossClBins, dClVecs, Fisher matrix calculations; ZK, 2017.09.08
    Created some testing+plotting functions; ZK, 2017.09.10
    Reworked power spectra to include 'A' as amplitude of matter distribution,
      rather than of CMB lensing; ZK, 2017.09.19
    Removed redundant elements in covariance matrix, etc.  This reduced the
      number of observables from nMaps**2 to nMaps*(nMaps+1)/2;
      ZK, 2017.09.20
    Fixed indexing error in covar creation; ZK, 2017.09.21
    Fixed indenting error in crossCls creation; ZK, 2017.09.22
    Modiefied so all d/dA are 0 for testing; ZK, 2017.09.29
    Added noAs condition to plotting function for testing; ZK, 2017.10.01
    Added noAs parameter to FisherMatrix init and plotSigmasByNBins;
      ZK, 2017.10.02
    Fixed big problem in dClVecs creation involving missing delta 
      functions; ZK, 2017.10.04
    Fixed indexing problem (bin vs map) in creation of crossClbins;
      ZK, 2017.10.04
    Note: Lloyd thought I should get this working in 2 weeks. 
      It took 6 weeks for me to do it.  ugh.  ZK, 2017.10.05
    Added overlapping redshift bin functionality (dndzMode1); ZK, 2017.10.06
    Added bin smoothing with Gaussian (dndzMode2); ZK, 2017.10.10
    Expanded cosmological parameter set for nuLambdaCDM; ZK, 2017.10.15
    Reparameterized from H0 to cosmomc_theta; ZK, 2017.10.16

    FisherCl split into two versions. 
      The other version (FisherCl_Ab) uses parameters A b
      This version (FisherCl) uses m nu LCDM b; ZK, 2017.10.16
    Added H0,Hs,zs to FisherMatrix object; ZK, 2017.10.20

    Branched off of master.  This version reverts to function getCl doing
      rough approximation to integration; ZK, 2017.12.13
    Removed CLtools; ZK, 2017.12.13
    Renamed cp.matterPower as cp.MatterPower; ZK, 2017.12.14
    Modified to use cp.Window objects and new version of getCl; moved bin 
      biasing to cp.Window; added w, AccuracyBoost to matrix params; 
      ZK, 2017.12.18
    Added fieldNames and obsNames to FisherMatrix; ZK, 2017.12.19
    Added descriptive data to class MatterPower; ZK,2017.12.24
    Modified FisherMatrix to use lmin; ZK, 2018.01.02
    Pulled nonlinear parameter up to FisherMatrix.__init__; ZK, 2018.01.28
    Adjusted deltaP dw from 0.3 to 0.05; ZK,
      Added neutrino_hierarchy field to FisherMatrix; 
      Fixed sign error in lmin correction in makeFisher; ZK, 2018.02.16
    Modified so MatterPower objects previously stored in myPksUpper and 
      myPksLower are used immediatly then discarded; ZK, 2018.02.17 
    Added primary CMB Fisher functions to class MatterPower; 
      Removed redundant Cl calculation since Sigma_i W_i = W is enforced
      in crosspower.py, and A_i is fixed; ZK, 2018.02.19
    Fixed missing self. on self.crossClsP and other fixes 
      for primary CMB; ZK, 2018.02.23
    Added noise to kappa, galxies, and primary CMB; control via useNoise param; 
      Consolidated makeCovar codes to a function; 
      Raised lminP, lmaxP to init parameters; ZK, 2018.03.10
    Hard coded lmax values into FisherMatrix.makeFisher method; 
      ZK, 2018.03.26
    Set TE noise to zero, instead of having similar form to TT,EE noise;
      ZK, 2018.04.10
    Added Cl^BB and BB noise to primary CMB Fisher Matrix calculation;
      Fixed major dark energy parameterization error: Re-organized code to 
      accomodate the use of global variables; added wa param; ZK, 2018.04.15
     

"""

#import sys, platform, os
import numpy as np
import matplotlib.pyplot as plt
#import scipy.integrate as sint
from scipy.interpolate import interp1d
import camb
#from camb import model, initialpower
#from scipy import polyfit,poly1d
import crosspower as cp
import noiseCl as ncl

# Dan Coe's CLtools (confidence limit; c2009), for Fisher Matrix, etc.
#import CLtools as cl 

################################################################################
# some functions




################################################################################
# the Fisher Matrix class

class FisherMatrix:
  """ 
    Name:
      FisherMatrix
    Purpose:
      create objects to calculate and store the Fisher matrix for a set of 
        fields.  Intended for use with CMB lensing and galaxy maps

  """


  def __init__(self,nz=10000,lmin=2,lmax=3000,zmin=0.0,zmax=16.0,dndzMode=2, 
               nBins=10,z0=0.5,doNorm=True,useWk=False,binSmooth=0,BPZ=True, 
               usePrimaryCMB=False,biasByBin=True,AccuracyBoost=3,
               nonlinear=False,useNoise=True,lminP=2,lmaxP=5000,**cos_kwargs):
    """
    
      Inputs:
        nz: the number of z points to use between here and last scattering surface
          Important usage is as the number of points to use in approximation of
            C_l integrals
        lmin,lmax: minimum,maximum ell value to use in k,g summations
          note: lmax used in creating Fij is now hard-coded to 
            2000 for kk,kg,gg; 3000 for TT,TE; 5000 for EE
        lminP,lmaxP: minimum,maximum ell value to use in T,E summations
          note: see above note.
        zmin,zmax: minimum,maximum z to use in binning for A_i, b_i parameters
        doNorm: set to True to normalize dN/dz.
          Default: True
        BPZ: set to true to use BPZ dNdz curves in dndzMode 1, False for TPZ
          Default: True
        biasByBin: set to True to use one bias b_i per bin rather than b(z)
          Default: True
        Parameters only used in dndzMode = 2:
          nBins: number of bins to create
            Default: 10
          z0: controls width of full dNdz distribution
            Defalut: 0.5, for use with cp.modelDNDZ3 for LSST distribution
          useWk: set to True to use W^kappa as dN/dz
            Defalut: False
          binSmooth: parameter that controls the amount of smoothing of bin edges
            Default: 0 (no smoothing)
        usePrimaryCMB = True: set to True to include TT,TE,EE observables.
          In this implementation, we assume that the T and E fields are not
            correlated at all with the kappa and galaxies fields.
            Thus, the correlation matrix can be split into two disjoint sections.
          Default: False
        AccuracyBoost: to pass to set_accuracy to set accuracy
          Note that this sets accuracy globally, not just for this object
          Default: 3
        nonlinear: set to True to use Halofit for nonlinear evolution.
          Default: False
        useNoise: set to True to add noise to primary CMB
          Default: True
        Parameters for camb's set_params and set_cosmology:
          **cos_kwargs

    """
################################################################################
    # preliminaries

    # set cosmological parameters
    self.cosParams = {
        'H0'    : None, #67.51, #setting H0=None allows cosmomc_theta to be used instead
        'cosmomc_theta'           : 1.04087e-2,
        'ombh2' : 0.02226,
        'omch2' : 0.1193,
        'omk'   : 0,
        'tau'   : 0.063,

        'As'    : 2.130e-9,
        'ns'    : 0.9653,
        'r'     : 0,
        'kPivot': 0.05,

        'w0'    : -1.0, # DARK ENERGY!!!
        'wa'    : 0.0,

        # if fiducial mnu is changed, need to adjust omch2 as well
        'mnu'   : 0.06, # (eV)
        #'mnu'   : 0.058, # Lloyd suggested this value for fiducial; adjust omch2 if I do use it
        'nnu'   : 3.046,
        'standard_neutrino_neff'  : 3.046,
        'num_massive_neutrinos'   : 1,
        'neutrino_hierarchy'      : 'normal'}
    self.cosParams.update(cos_kwargs)
    if self.cosParams['mnu'] != 0.06:
      print '------!!! Warning! Wrong Mnu detected! Adjustment not yet implemented! !!!------' 

    # modify for dndzMode = 1
    if dndzMode == 1:
      nBins = 5
      zmin = 0
      zmax = 1.5 # to match dndz files

    # set other parameters
    self.nz = nz
    self.dndzMode = dndzMode
    self.BPZ = BPZ
    self.zmin = zmin
    self.zmax = zmax
    self.nBins = nBins
    self.z0 = z0
    self.lmin = lmin
    self.lmax = lmax
    self.lminP = lminP
    self.lmaxP = lmaxP
    self.AccuracyBoost=AccuracyBoost
    self.doNorm = doNorm
    self.useWk = useWk
    self.binSmooth = binSmooth
    self.biasByBin = biasByBin
    self.usePrimaryCMB = usePrimaryCMB
    self.nonlinear = nonlinear
    self.neutrino_hierarchy = cos_kwargs['neutrino_hierarchy']

    if binSmooth == 0 and dndzMode == 2:
      tophatBins = True # true if bins do not overlap, false if they do
    else:
      tophatBins = False
    
    # observables lists
    # self.obsList, obsListP created along with self.covar, covarP
    nMaps = nBins+1 # +1 for kappa map
    nCls = nMaps*(nMaps+1)/2 # This way removes redundancies, eg C_l^kg = C_l^gk
    self.fieldNames = ['k']
    for binNum in range(1,nBins+1):
        self.fieldNames.append('g'+str(binNum))
    self.obsNames = []
    for map1 in range(nMaps):
      for map2 in range(map1, nMaps):
        self.obsNames.append(self.fieldNames[map1]+','+self.fieldNames[map2])
    if usePrimaryCMB:
        # select 2 for T,E, 3 for T,E,B
        nMapsP = 3 # T,E,B
        nClsP = nMapsP*(nMapsP+1)/2
        self.fieldNamesP = ['T','E','B']
        self.obsNamesP = ['TT','TE','EE','BB'] # is this enough?    
    
    # parameters list:
    # step sizes for discrete derivatives: must correspond to paramList entries!
    #   from Allison et. al. (2015) Table III.
    nCosParams = 9 # 6 LCDM + Mnu + w0 + wa
    paramList = ['ombh2','omch2','cosmomc_theta',  'As', 'ns','tau','mnu', 'w', 'wa']
    deltaP =    [ 0.0008, 0.0030,      0.0050e-2,0.1e-9,0.010,0.020,0.020,0.05,0.025] #mnu one in eV
    for bin in range(nBins):
      paramList.append('bin'+str(bin+1))
    self.nParams   = nParams
    self.nCosParams = nCosParams
    self.paramList = paramList

    # get MatterPower object
    print 'creating MatterPower object . . . '
    myPk = cp.MatterPower(nz=nz,AccuracyBoost=AccuracyBoost,nonlinear=nonlinear,
                          **self.cosParams)
    PK,chistar,chis,dchis,zs,dzs,pars = myPk.getPKinterp()
    #self.H0 = myPk.H0
    self.H0 = pars.H0
    self.Hs = myPk.Hs
    self.zs = myPk.zs

    # get Window object
    print 'creating Window object . . . '
    myWin = cp.Window(myPk,zmin=zmin,zmax=zmax,nBins=nBins,biasK=cp.ones,
                      biasG=cp.byeBias,dndzMode=dndzMode,z0=z0,
                      doNorm=doNorm,useWk=useWk,BPZ=BPZ,binSmooth=binSmooth,
                      biasByBin=biasByBin)

    # get modified parameter lists for numeric differentiation
    print 'creating modified parameter lists . . . '
    myParams = self.cosParams
    myParamsUpper = []
    myParamsLower = []
    for cParamNum in range(nCosParams):
      # add parameter dictionary to lists; HAVE TO BE COPIES!!!
      myParamsUpper.append(myParams.copy())
      myParamsLower.append(myParams.copy())
      # modify parameter number cParamNum in ditionaries
      myParamsUpper[cParamNum][paramList[cParamNum]] += deltaP[cParamNum]
      myParamsLower[cParamNum][paramList[cParamNum]] -= deltaP[cParamNum]

      # after discussion we prefer not to use this for simplicity
      """
      # check for mnu modification and adjust omch2 if necessary
      if paramList[cParamNum] == 'mnu':
        omch2Index = np.where(np.array(paramList) == 'omch2')[0][0]
        deltaOmnh2 = deltaP[cParamNum]/94 #eq.n 12 from Wu et. al.
        #deltaOmnh2 = pars.omegan*(pars.H0/100)**2 # probably a better measure of omega_nu
        # note the -=,+= signs get reversed in next 2 lines compared to above
        myParamsUpper[cParamNum][paramList[omch2Index]] -= deltaOmnh2
        myParamsLower[cParamNum][paramList[omch2Index]] += deltaOmnh2
      """

      #print 'cParamNum: ',cParamNum,', param name: ',paramList[cParamNum]
      #print 'myParamsUpper[cParamNum][paramList[cParamNum]]: ',myParamsUpper[cParamNum][paramList[cParamNum]]
      #print 'myParamsLower[cParamNum][paramList[cParamNum]]: ',myParamsLower[cParamNum][paramList[cParamNum]]
      #print 'deltaP[cParamNum]: ',deltaP[cParamNum]

    # save some of this
    self.myPk = myPk
    self.myWin = myWin
    self.myParamsUpper = myParamsUpper
    self.myParamsLower = myParamsLower
    self.binAs = myWin.binBKs[1:]                   # kappa has one power of A
    self.binBs = myWin.binBGs[1:]/myWin.binBKs[1:]  # galaxies have A*b



################################################################################
    # get all cross power spectra

    # If I use AOfZ not all 1, this needs to be changed 
    #   to include summation over bins for kk
    # If tophatBins, only the diagonal and 0th row and column will be filled
    
    print 'starting kappa, galaxy cross power (with entire kappa) . . . '

    #self.crossCls      = np.zeros((nMaps,nMaps,           lmax-lmin+1))
    #self.crossClsPlus  = np.zeros((nMaps,nMaps,nCosParams,lmax-lmin+1))
    #self.crossClsMinus = np.zeros((nMaps,nMaps,nCosParams,lmax-lmin+1))

    # get main set of power spectra    
    self.crossCls = self.getCrossCls(['(All Fiducial)'],[myParams],myPk,
                                     lmin,lmax,nMaps,tophatBins=tophatBins)
    # get the perturbed versions
    print 'starting upper spectra for numeric derivatives . . . '
    self.crossClsPlus = self.getCrossCls(paramList,myParamsUpper,myPk,
                                         lmin,lmax,nMaps,tophatBins=tophatBins)    
    print 'starting lower spectra for numeric derivatives . . . '
    self.crossClsMinus = self.getCrossCls(paramList,myParamsLower,myPk,
                                          lmin,lmax,nMaps,tophatBins=tophatBins)
    
    # reset global (dark energy and AccuracyBoost) settings
    pars = myPk.getPars(lmax=lmax,lpa=lpa,AccuracyBoost=AccuracyBoost)

    self.ells = np.arange(lmin,lmax+1)
    
    
    # this section needed for dCl/dAi or overlapping bins
    """
    # divide K,G into bins and get crossClbins
    self.crossClBinsKK = np.zeros((nBins,nBins,lmax-lmin+1))
    self.crossClBinsKG = np.zeros((nBins,nBins,lmax-lmin+1))
    self.crossClBinsGG = np.zeros((nBins,nBins,lmax-lmin+1))

    # if tophatBins, only the diagonals will be filled
    # note: cp.getCl has a +1 offset to bin numbers, 
    #   since bin 0 indicates sum of all bins
    print 'starting cross power with binned kappa . . . '
    for bin1 in range(nBins):
      for bin2 in range(bin1,nBins):
        if bin1==bin2 or not tophatBins:
          print '  starting angular cross power spectrum',bin1+1,', ',bin2+1,'. . . '
          # kk
          ells,Cls = cp.getCl(myPk,myWin,binNum1=bin1+1,binNum2=bin2+1,
                           cor1=cp.Window.kappa,cor2=cp.Window.kappa,
                           lmin=lmin,lmax=lmax)
          self.crossClBinsKK[bin1,bin2] = Cls
          self.crossClBinsKK[bin2,bin1] = Cls #symmetric
          # kg
          ells,Cls = cp.getCl(myPk,myWin,binNum1=bin1+1,binNum2=bin2+1,
                           cor1=cp.Window.kappa,cor2=cp.Window.galaxies,
                           lmin=lmin,lmax=lmax)
          self.crossClBinsKG[bin1,bin2] = Cls
          self.crossClBinsKG[bin2,bin1] = Cls #symmetric
          # gg
          ells,Cls = cp.getCl(myPk,myWin,binNum1=bin1+1,binNum2=bin2+1,
                           cor1=cp.Window.galaxies,cor2=cp.Window.galaxies,
                           lmin=lmin,lmax=lmax)
          self.crossClBinsGG[bin1,bin2] = Cls
          self.crossClBinsGG[bin2,bin1] = Cls #symmetric
    """

    # do the primary CMB component
    if usePrimaryCMB:
        print 'starting primary CMB cross-power . . . '
        lpa = 5.0 # set_for_lmax also sets lensing 
        pars = myPk.getPars(lmax=lmax,lpa=lpa,AccuracyBoost=AccuracyBoost)
        
        #get the total lensed CMB power spectra versus unlensed
        #myClName = 'total'
        myClName = 'unlensed_scalar'
        
        #self.crossClsP      = np.zeros((nMapsP,nMapsP,           lmaxP-lminP+1))
        #self.crossClsPPlus  = np.zeros((nMapsP,nMapsP,nCosParams,lmaxP-lminP+1))
        #self.crossClsPMinus = np.zeros((nMapsP,nMapsP,nCosParams,lmaxP-lminP+1))

        # get main set of power spectra
        crossClsP = self.getCrossClsP(['(All Fiducial)'],[myParams],myPk,
                                      nMaps=nMapsP,lmax=lmaxP,lpa=lpa,
                                      myClName=myClName,
                                      AccuracyBoost=AccuracyBoost)
        # get the perturbed versions
        print 'starting upper spectra for numeric derivatives . . . '
        crossClsPPlus  = self.getCrossClsP(paramList,myParamsUpper,myPk,
                                           nMaps=nMapsP,lmax=lmaxP,lpa=lpa,
                                           myClName=myClName,
                                           AccuracyBoost=AccuracyBoost)
        print 'starting lower spectra for numeric derivatives . . . '
        crossClsPMinus = self.getCrossClsP(paramList,myParamsLower,myPk,
                                           nMaps=nMapsP,lmax=lmaxP,lpa=lpa,
                                           myClName=myClName,
                                           AccuracyBoost=AccuracyBoost)
        
        self.crossClsP = crossClsP[:,:,lminP:]
        self.crossClsPPlus = crossClsPPlus[:,:,:,lminP:]
        self.crossClsPMinus = crossClsPMinus[:,:,:,lminP:]
        
        # reset global (dark energy and AccuracyBoost) settings
        pars = myPk.getPars(lmax=lmax,lpa=lpa,AccuracyBoost=AccuracyBoost)
      
        self.ellsP = np.arange(lminP,lmaxP+1)


################################################################################
    # create noise arrays
    if useNoise:
        print 'creating noise arrays... '

        print 'getting (EB) lensing reconstruction noise... '
        # noise levels from a possible CMB-S4 design:
        nlev_t     = 1.   # temperature noise level, in uK.arcmin.
        nlev_p     = 1.414   # polarization noise level, in uK.arcmin.
        beam_fwhm  = 1.   # Gaussian beam full-width-at-half-maximum.

        ells,EB_noise = ncl.getRecNoise(self.lmax,nlev_t,nlev_p,beam_fwhm)

        # convert Nl^dd to Nl^kk
        # use Cl^kk = 1/4*[l*(l+1)]^2 * Cl^phiphi?
        # but d is in between k and phi, so...?
        # Scratch that.
        # Assume output of quicklens is Cl^phiphi
        Nlkk = EB_noise * (ells*(ells+1)/2)**2


        print 'getting galaxy shot noise... '
        # From Schaan et. al.: LSST n_source = 26/arcmin^2 for full survey
        #nbar = 26 # arcmin^-2
        nbar = 66 # 66 arcmin^-2 to match Bye's value

        # the selection of beesBins must be consistent with that which is selected in cp.tophat
        beesBins = True
        if beesBins:
            binEdges = [0.0,0.5,1.0,2.0,3.0,4.0,7.0]
            nBins = 6
        else:
            binEdges = np.linspace(self.zmin,self.zmax,self.nBins+1)
            nBins = self.nBins

        # the selection of dndz function must be consistent with that which is selected in cp.getDNDZinterp
        # myDNDZ must be a function only of z
        #myDNDZ = lambda z: cp.modelDNDZ(z,z0)
        myDNDZ = lambda z: cp.modelDNDZ3(z,z0)

        N_gg = ncl.shotNoise(nbar,binEdges,myDNDZ=myDNDZ)



        # create noiseCls array to accompany crossCls: 
        #  Nl^kk will be reconstruction noise
        #  Nl^gigi will be shot noise
        #  Nl^kg, Nl^gigj are zero because noise is uncorrelated
        self.noiseCls = np.zeros(self.crossCls.shape)
        self.noiseCls[0,0] = Nlkk[self.lmin:self.lmax+1]
        for binNum in range(nBins):
            self.noiseCls[binNum+1,binNum+1] = N_gg[binNum]*np.ones(self.lmax-self.lmin+1)

        if usePrimaryCMB:
            print 'getting (primary CMB) detector noise...'
            # CMBS4 v1
            fwhm = 1; ST = 1; SP = ST*1.414
            # CMBS4 v2
            #fwhm = 2; ST = 1; SP = ST*1.414
            noiseCMBS4_TT1 = ncl.noisePower(ST,ST,fwhm,self.ellsP)
            #noiseCMBS4_TP1 = ncl.noisePower(ST,SP,fwhm,self.ellsP)
            noiseCMBS4_PP1 = ncl.noisePower(SP,SP,fwhm,self.ellsP)

            # assume zero TE noise
            zerosList = np.zeros(lmax+1)
            noiseCMBS4_TP1 = zerosList

            # shape like crossCls
            if nMaps == 2:
                noiseClsP = np.array([[noiseCMBS4_TT1,noiseCMBS4_TP1],[noiseCMBS4_TP1,noiseCMBS4_PP1]])
            elif nMaps == 3:
                noiseClsP = np.array([[noiseCMBS4_TT1, noiseCMBS4_TP1,      zerosList],
                                      [noiseCMBS4_TP1, noiseCMBS4_PP1,      zerosList],
                                      [     zerosList,      zerosList, noiseCMBS4_PP1]])
            else:
                'bad nMaps for noise maker'

    
    else: # no noise
        self.noiseCls = np.zeros(self.crossCls.shape)
        if usePrimaryCMB:
            self.noiseClsP = np.zeros(self.crossClsP.shape)
   
    
    
    
################################################################################
    # create covariance matrix
    print 'building covariance matrix... '

    # need noiseCls before running this
    self.covar,self.invCov,self.ells,self.obsList = \
        self.makeCovar(self.crossCls,self.noiseCls,self.lmin,self.lmax)
    if usePrimaryCMB:
        self.covarP,self.invCovP,self.ellsP,self.obsListP = \
            self.makeCovar(self.crossClsP,self.noiseClsP,0,self.lmaxP)
    


################################################################################
    # get derivatives wrt parameters
    print 'starting creation of C_l derivatives... '

    # get dC_l^munu/da_i (one vector of derivatives of C_ls for each param a_i)
    # store as matrix with additional dimension for a_i)
    # uses same (shortened) nCls as self.covar and self.obsList
    self.dClVecs = np.empty((nCls, nParams, lmax-lmin+1))
    Clzeros = np.zeros(lmax-lmin+1) # for putting into dClVecs when needed
    for map1 in range(nMaps):
      print 'starting derivative set ',map1+1,' of ',nMaps,'... '
      for map2 in range(map1,nMaps):
        mapIdx  = map1*nMaps+map2 -map1*(map1+1)/2  
                                   # mapIdx = map index
        for pIdx in range(nBins):  # pIdx = parameter index
          bi   = nCosParams+pIdx   # Bs to be after other params

          if map1 == 0: #kappa
            if map2 == 0:          #kk
              # this section assumes Sum_i^{nBins+1} W^{k_i} = W^k (completeness)
              self.dClVecs[    mapIdx, bi] = Clzeros 

            else:                  #kg,gk
              # this section assumes no bin overlap (update later)
              if pIdx+1 == map2: # +1 since 1 more map than bin
                #self.dClVecs[  mapIdx, bi] = 1/self.binBs[pIdx] * self.crossClBinsKG[pIdx,pIdx]
                self.dClVecs[  mapIdx, bi] = 1/self.binBs[pIdx] * self.crossCls[0,pIdx+1]
              else: # parameter index does not match bin index
                self.dClVecs[  mapIdx, bi] = Clzeros

          else: #galaxies          #gg
            if pIdx+1 == map2: # +1 since 1 more map than bin
              if map1 == map2:
                #self.dClVecs[  mapIdx, bi] = 2/self.binBs[pIdx] * self.crossClBinsGG[pIdx,pIdx]
                self.dClVecs[  mapIdx, bi] = 2/self.binBs[pIdx] * self.crossCls[pIdx+1,pIdx+1]
              else:
                # this section assumes no bin overlap (update later)
                self.dClVecs[  mapIdx ,bi] = Clzeros 
            else: # parameter index does not match bin index
              self.dClVecs[    mapIdx, bi] = Clzeros
    
        # next do numerical derivs wrt nuLCDM params
        for pIdx in range(nCosParams):
          dClPlus  = self.crossClsPlus[map1,map2,pIdx]
          dClMinus = self.crossClsMinus[map1,map2,pIdx]
          self.dClVecs[mapIdx, pIdx] = (dClPlus-dClMinus)/(2*deltaP[pIdx])

    # do the primary CMB component
    # note: most of this is copied from above.  
    #   Later I should consolidate the two versions into a function.
    if usePrimaryCMB:
        dClVecsP = np.empty((nClsP, nCosParams, lmaxP-lminP+1))

        for map1 in range(nMapsP):
            print 'starting primary CMB derivative set ',map1+1,' of ',nMapsP,'... '
            for map2 in range(map1,nMapsP):
                mapIdx  = map1*nMapsP+map2 -map1*(map1+1)/2  # mapIdx = map index
                for pIdx in range(nCosParams):
                    dClPlus  = self.crossClsPPlus[map1,map2,pIdx]
                    dClMinus = self.crossClsPMinus[map1,map2,pIdx]
                    dClVecsP[mapIdx, pIdx] = (dClPlus-dClMinus)/(2*deltaP[pIdx])
        self.dClVecsP = dClVecsP



################################################################################
    #Build Fisher matrix
    self.Fij = self.makeFisher(self.lmin)
    if usePrimaryCMB:
        self.FijTE = self.makeFisher(self.lminP,TE=True)
    print 'creation of Fisher Matrix complete!\n'
    # end of init function


################################################################################
# other methods

  def getCrossCls(self,paramList,myParams,myPk,lmin,lmax,nMaps,tophatBins=True):
    """
        Purpose:
            get the crossCls for kappa, galaxies
        Inputs:
            paramList: like FisherMatrix.paramList
                Note: a list of length 1 will cause a differently shaped array 
                to be returned
            myParams: a list of lists of parameters like FisherMatrix.cosParams
                This list must have the same length as paramList
            myPk: a MatterPower object

            nMaps: number of fields being used (should be 1+nBins)
            lmin,lmax: min,max ell value to use
            tophatBins: set to True to use tophat-shaped, non-overlapping bins
        Returns:
            crossCls: numpy array of shape (nMaps,nMaps,nCosParams,lmax)
              unless the lenth of paramList is 1, then shape will be 
              (nMaps,nMaps,lmax)
    """
    nCosParams = paramList.__len__()
    
    crossCls = np.zeros((nMaps,nMaps,nCosParams,lmax+1))
    for cParamNum in range(nCosParams):
        print 'calculating MatterPower and Window objects for ',\
              paramList[cParamNum], ' derivative . . . '

        myPks = cp.MatterPower(nz=self.nz,AccuracyBoost=self.AccuracyBoost,
                               nonlinear=self.nonlinear,**myParams[cParamNum])
        myWins = cp.Window(myPks,zmin=self.zmin,zmax=self.zmax,
                           nBins=self.nBins,biasK=cp.ones,biasG=cp.byeBias,
                           dndzMode=self.dndzMode,z0=self.z0,doNorm=self.doNorm,
                           useWk=self.useWk,BPZ=self.BPZ,
                           binSmooth=self.binSmooth,biasByBin=self.biasByBin)
        
        # save pars for use in primary CMB
        # it would be nice to restore this functionality 
        #  but global DE settings make this hard
        #myParsUpper.append(myPksUpper.pars)
        #myParsLower.append(myPksLower.pars)

        for map1 in range(nMaps):
          if map1==0:
            cor1 = cp.Window.kappa
          else:
            cor1 = cp.Window.galaxies
          for map2 in range(map1,nMaps):
            print '  starting angular cross power spectrum ',map1,', ',map2,'... '
            if map2==0:
              cor2 = cp.Window.kappa
            else:
              cor2 = cp.Window.galaxies
            # since nonoverlapping bins have zero correlation use this condition:
            if map1==0 or map1==map2 or not tophatBins:
              ells,Cls = cp.getCl(myPks,myWins,binNum1=map1,binNum2=map2,
                                  cor1=cor1,cor2=cor2,lmin=lmin,lmax=lmax)
              crossCls[map1,map2,cParamNum] = Cls
              crossCls[map2,map1,cParamNum] = Cls #symmetric
        
        # ditch the MatterPower and Window objects - wait, isn't this automatic?
        del myPks
        del myWins

    # reshape for unperterbed version
    if nCosParams == 1:
        crossCls = np.reshape(crossCls,(nMaps,nMaps,lmax+1))
    return crossCls

    

  def getCrossClsP(self,paramList,myParams,myPk, nMaps=3,lmax=5000,lpa=5.0,
                   AccuracyBoost=3,myClName='unlensed_scalar'):
    """
    Purpose:
        get the crossCls for primary CMB: TT,TE,EE,BB
    Inputs:
        paramList: like FisherMatrix.paramList
            Note: a list of length 1 will cause a differently shaped array 
            to be returned
        myParams: a list of lists of parameters like FisherMatrix.cosParams
            This list must have the same length as paramList
        myPk: a MatterPower object
        
        nMaps: number of maps.  Set to 3 for T,E,B
        lmax: lmax for returned array.  lmin will be zero
        lpa: lensing_power_accuracy setting
        AccuracyBoost: accuracy setting
        myClName: which type of power spectrum to get from CAMB
    Returns:
        crossCls: numpy array of shape (nMaps,nMaps,nCosParams,lmax)
          unless the lenth of paramList is 1, then shape will be 
            (nMaps,nMaps,lmax)
          Note Cl arrays are all zero based (starting at L=0).
          Note L=0,1 entries will be zero by default.
    """
    nCosParams = paramList.__len__()
    
    crossCls = np.zeros((nMaps,nMaps,nCosParams,lmax+1))
    for paramNum in range(nCosParams):
        print 'getting Cl power spectra for perturbed parameter ', \
              paramList[paramNum]
        # get pars
        pars = cp.MatterPower.getPars(myPk,lmax=lmax+1,lpa=lpa,
                                      AccuracyBoost=AccuracyBoost,
                                      **myParams[paramNum])
        #get the selected power spectra
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars)
        myCl = powers[myClName]
        
        #store them
        #The different CL are always in the order TT, EE, BB, TE 
        #  (with BB=0 for unlensed scalar results).
        crossCls[0,0,paramNum]  = myCl[:lmax+1,0] # TT
        crossCls[0,1,paramNum]  = myCl[:lmax+1,3] # TE
        crossCls[1,0,paramNum]  = myCl[:lmax+1,3] # ET
        crossCls[1,1,paramNum]  = myCl[:lmax+1,1] # EE
        if nMaps == 3:
            crossCls[2,2,paramNum]  = myCl[:lmax+1,2] # BB
        
        # reshape for unperterbed version
        if nCosParams == 1:
            crossCls = np.reshape(crossCls,(nMaps,nMaps,lmax+1))
    return crossCls
    

  def makeCovar(self,crossCls,noiseCls,lmin,lmax):
    """
    Purpose: make a covariance matrix
    Inputs:
        crossCls:
        noiseCls:
        lmin,lmax: The lowest,highest ell in crossCls and noiseCls
    Returns: 
        covar,invCov,ells,obsList
    """
    # start by finding sizes
    nMaps = crossCls.shape[0]
    nCls = nMaps*(nMaps+1)/2
    
    # create return objects
    covar = np.zeros((nCls,nCls,lmax-lmin+1))
    ells = np.arange(lmin,lmax+1)

    # create obsList to contain base nMaps representation of data label
    obsList = np.zeros(nCls)
    
    # no noise or noise?
    # disable this, since python doesn't know how to compare an np.array to None
    """
    if noiseCls == None:
        print 'no noise'
        myCls = crossCls
    else:
    """
    myCls = crossCls + noiseCls
        
    # fill out covar matrix
    for map1 in range(nMaps):
        print 'starting covariance set ',map1+1,' of ',nMaps,'... '
        for map2 in range(map1, nMaps):
            covIndex1 = map1*nMaps+map2-map1*(map1+1)/2     # shortens the array
            obsList[covIndex1] = map1*nMaps+map2            # base nMaps representation
            for map3 in range(nMaps):
              for map4 in range(map3, nMaps):
                covIndex2 = map3*nMaps+map4-map3*(map3+1)/2 # shortens the array
                if covIndex1 <= covIndex2:
                  covar[covIndex1,covIndex2] = ( \
                    myCls[map1,map3]*myCls[map2,map4] + \
                    myCls[map1,map4]*myCls[map2,map3] )/ \
                    (2.*ells+1)
                else:   # avoid double calculation
                  covar[covIndex1,covIndex2] = covar[covIndex2,covIndex1]
    
    invCov = np.transpose(np.linalg.inv(np.transpose(covar)))

    return covar,invCov,ells,obsList


  #def makeFisher(self,lmin,lmax,TE=False,verbose=False):
  def makeFisher(self,lmin,TE=False,verbose=False):
    """
      Purpose:
        multply vectorT,invcov,vector and add up
      Inputs:
        self: a FisherMatrix object
        lmin: the lowest ell to include in the sum
          must be GE self.lmin
        #Note: this version no longer has an lmax parameter.
        #lmax: the highest ell to inlude in the sum
        #  must be LE self.lmax
        TE: set to True to compute Fij for T,E instead of k,g
          Default: False
        verbose: set to True to have extra output
          Default: False
      Returns:
        a Fisher Matrix, dimensions self.nParams x self.nParams
    """
    if TE:
      lmax = 5000 # lmax for E, not T
      if lmin < self.lminP or lmax > self.lmaxP:
        print 'makeFisher: bad lminP or lmaxP!'
        return 0
      selfLmin = self.lminP
      selfLmax = self.lmaxP
      nParams = self.nCosParams
      invCov = self.invCovP
      dClVecs = self.dClVecsP.copy()

    else:
      lmax = 2000 # lmax for k,g
      if lmin < self.lmin or lmax > self.lmax:
        print 'makeFisher: bad lmin or lmax!'
        return 0
      selfLmin = self.lmin
      selfLmax = self.lmax
      nParams = self.nParams
      invCov = self.invCov
      dClVecs = self.dClVecs.copy()
    
    if verbose:
      print 'building Fisher matrix from components...'
      print 'invCov.shape: ',invCov.shape,', dClVecs.shape: ',dClVecs.shape

    Fij = np.zeros((nParams,nParams)) # indices match those in paramList
    for i in range(nParams):
      if verbose:
        print 'starting bin set ',i+1,' of ',nParams
      dClVec_i = dClVecs[:,i,:] # shape (nCls,nElls)
      for j in range(nParams):
        dClVec_j = dClVecs[:,j,:] # shape (nCls,nElls)

        # adjust lmax for TT case (not TE,EE)
        if TE: 
          print 'adjusting lmax for TT... '
          lmaxTT = 3000
          dClVec_i[0,lmaxTT+1:] = np.zeros(5000-lmaxTT)
          dClVec_j[0,lmaxTT+1:] = np.zeros(5000-lmaxTT)
        
        # ugh.  don't like nested loops in Python... but easier to program...
        for ell in range(lmax-lmin+1):
          ellInd = ell+lmin-selfLmin # adjust ell to match indices in arrays
          myCov = invCov[:,:,ellInd]
          fij = np.dot(dClVec_i[:,ellInd],np.dot(myCov,dClVec_j[:,ellInd]))
          Fij[i,j] += fij

        # now the TT part above ell=3000
        #if TE:
        #  print 'removing TT part above ell=3000...'
        #  lmin = 3001
        #  lmax = 5000
        #  for ell in range(lmax-lmin+1):
        #    ellInd = ell+lmin-selfLmin # adjust ell to match indices in arrays
        #    fij = dClVec_i[0,ellInd]*invCov[0,0,ellInd]*dClVec_j[0,ellInd]
        #    #fij = dClVec_i[1,ellInd]*invCov[1,1,ellInd]*dClVec_j[1,ellInd]
        #    Fij[i,j] -= fij


    return Fij

    
  def getBinCenters(self):
    """
      return array of centers of bins
    """
    if self.dndzMode == 1:
      return (0.3,0.5,0.7,0.9,1.1)
    elif self.dndzMode == 2:
      halfBinWidth = (self.zmax-self.zmin)/(2*self.nBins)
      nHalfBins = (2*np.arange(self.nBins)+1)
      return halfBinWidth*nHalfBins+self.zmin
    else:
      print 'die screaming'
      return 0

  def getSigmas(self):
    """
      get the sigmas from the Fisher Matrix
      Returns:
        #sigmasA,sigmasB
        sigmas
    """
    Finv = np.linalg.inv(self.Fij)
    sigmas = np.sqrt(np.diag(Finv))
    #sigmasA = sigmas[:self.nBins]
    #sigmasB = sigmas[self.nBins:]
    #return sigmasA,sigmasB
    return sigmas

  def showCovar(self,ell,doLog=False):
    """
      ell: which ell value to show covar for
      doLog: set to true to take logarithm of covar first
        Note: this will give divide by zero warning
    """
    print 'C_l^ij codes (index in covar array): ',self.obsList
    nMaps = self.nBins+1
    map1List = np.floor(self.obsList/nMaps)
    map2List = self.obsList%nMaps
    print 'map i numbers: ',map1List
    print 'map j numbers: ',map2List

    if doLog:
      plt.imshow(np.log(self.covar[:,:,ell]),interpolation='nearest')
    else:
      plt.imshow(self.covar[:,:,ell],interpolation='nearest')

    plt.show()

  def saveFish(self,filename='saveFish.npz'):
    """
      calls the saveFish function, defined below
    """
    saveFish(self,filename=filename)

  def getFish(self,filename='saveFish.npz'):
    """
      saves the fisher matrix data, loads it into a Fisher object
      Inputs:
        filename:
      Returns:
        a Fisher object
    """
    saveFish(self,fileame=filename)
    return Fisher(filename=filename)




# end of class FisherMatrix


################################################################################
# saving and loading Fisher Matrix results

def saveFish(Fobj,filename='saveFish.npz'):
  """
    Save the Fij matrix, fiducial parameter names, and their values
    to a npz archive.
    These can be loaded again using loadFish
    Inputs: 
      Fobj: the Fisher Matrix object to get data from
      filename: the name of the file to save to
  """
  Fij = Fobj.Fij
  cosParams = Fobj.cosParams
  bParams = Fobj.binBs
  paramList = Fobj.paramList

  paramVals = []
  for paramNum in range(7):
    paramVals.append(cosParams[paramList[paramNum]])
  paramVals = np.hstack((paramVals,bParams))

  np.savez(filename,paramList=paramList,paramVals=paramVals,Fij=Fij)
    
def loadFish(filename):
  """
    Load fisher matrix data in format used by saveFish
    Inputs:
      filename: the name of the file to load data from
    Returns:
      paramList,paramVals,Fij (all numpy arrays)
  """
  npzFile = np.load(filename)
  return npzFile['paramList'],npzFile['paramVals'],npzFile['Fij']


################################################################################
# object for using Fisher matrix once it's created

class Fisher:
  """
    This class uses the results of the FisherMatrix calculation
  """
  def __init__(self,filename='saveFish.npz'):
    """
      Inputs:
        filename: a fisher matrix save file (made by saveFish)
    """
    self.paramList,self.paramVals,self.Fij = loadFish(filename)

  def dxdyp(self, aIndex, bIndex):
    """
      Return uncertainty in two parameters and their correlation coefficient
      Function modified from method in Dan Coe's Fisher object
      Inputs: 
        aIndex: the index of the first parameter
        bIndex: the index of the second parameter
    """
    C = self.Fij
    C = C.take((aIndex,bIndex),0)
    C = C.take((aIndex,bIndex),1)
    dx = np.sqrt(C[0,0])
    dy = np.sqrt(C[1,1])
    dxy = C[0,1]
    p = dxy / (dx * dy)
    #self.C = C
    return dx, dy, p

  def twoParamConf(self,aIndex,bIndex):
    """
      plot 2d confidence limit ellipses for 2 parameters
      Inputs: 
        aIndex: the index of the first parameter
        bIndex: the index of the second parameter
    """
    """
    # fiducial values
    xFid = self.paramVals[aIndex]
    yFid = self.paramVals[bIndex]
    # labels
    xLabel = self.paramList[aIndex]
    yLabel = self.paramList[bIndex]

    dx,dy,p = self.dxdyp(aIndex,bIndex)
    alpha=0.9
    limFac = 3
    cl.plotellsp(xFid,yFid, dx,dy,p, colors=cl.reds,alpha=alpha)
    plt.xlim([xFid-limFac*dx,xFid+limFac*dx])
    plt.ylim([yFid-limFac*dy,yFid+limFac*dy])
    #cl.finishup(xFid,yFid,xLabel,yLabel,c='k',dc='w',sh=0)
    plt.show()
    """
    pass

################################################################################
# plotting functions

def plotSigmas(FInv):
  """
    plot sigma_A_i
    Input:
      FInv: Inverse of a Fisher Matrix
      binCenters: array of centers of redshift bins

  """
  # get diagonal of Fmatrix
  Fdiag = np.diag(FInv)
  sigmaAs = Fdiag[0:nBins]
  sigmaBs = Fdiag[nBins:2*nBins]

  binCenters = 0
  # hold off on finishing this one...


def plotSigmasByNBins(nz=1000,lmax=2000,zmax=16,z0=1.5,
                      doNorm=True,useWk=False,**cos_kwargs):
  """
    plot several sigmas for various values of nBins at one zmax
    Inputs:
      nz:
      lmax:
      zmax:
      z0:
      doNorm: normalize dndz
      useWk: use Wk as dndz
      **cos_kwargs: for set_cosmology

  """
  # NOTE: when running this with nBinsVals = (4,8,12,16,20), when inverting the final 40x40 matrix, the computer (which had 8Gb RAM) became memory-bound, with around 7.5Gb for python and crawling on all processes. I aborted it.
  nBinsVals = (2,4,8,16)
  #nBinsVals = (4,8,12,16)#,20)
  #nBinsVals = (3,6,9,12)#,15)
  labels = ('2 bins','4 bins','8 bins','16 bins')
  #labels = ('4 bins','8 bins','12 bins','16 bins')
  #labels = ('3 bins','6 bins','9 bins','12 bins')
  #nBinsVals = (5,10,15)
  nnBins = 4#5 #number of nBins values (just a label, really)

  # to collect eigenvalues, As, bs, sigmas
  eigs    = []
  fidAs   = []
  fidBs   = []
  sigmaAs = []
  sigmaBs = []
  # get Fisher matrix objects and do plots
  fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,6))
  for nBinsIndex,nBinsNum in enumerate(nBinsVals):
    print '\n starting Fisher Matrix ',nBinsIndex+1,' of ',nnBins, ', with nBins=',nBinsNum
    Fobj = FisherMatrix(nz=nz,lmax=lmax,zmax=zmax,nBins=nBinsNum,z0=z0,
                        doNorm=doNorm,useWk=useWk,**cos_kwargs)
    fidAs = np.append(fidAs,Fobj.binAs)
    fidBs = np.append(fidBs,Fobj.binBs)

    print 'inverting Fij ',nBinsIndex+1,' of ',nnBins
    FInv = np.linalg.inv(Fobj.Fij)

    # check eigenvalues
    w,v = np.linalg.eigh(FInv)
    print 'eigenvalues: ',w
    eigs = np.append(eigs,w)

    binCenters = Fobj.getBinCenters()
    #print 'binCenters: ',binCenters
    diags = np.diag(FInv)
    sigmas = np.sqrt(diags)
    print 'sigmas: ',sigmas
    nBins = nBinsVals[nBinsIndex]
    As = sigmas[:nBins]
    Bs = sigmas[nBins:]
    sigmaAs = np.append(sigmaAs,As)
    sigmaBs = np.append(sigmaBs,Bs)
    #print 'As: ',As,', Bs: ',Bs,'\n'
    ax1.plot(binCenters,As,label=labels[nBinsIndex])
    ax2.plot(binCenters,Bs,label=labels[nBinsIndex])

  print 'eigenvalues of all inverses of Fisher matrices: ',eigs

  ax1.set_title(r'matter amplitudes $A_i$')
  ax2.set_title(r'galaxy biases $b_i$')
  ax1.set_xlabel('redshift',fontsize=15)
  ax2.set_xlabel('redshift',fontsize=15)
  ax1.set_ylabel(r'$\sigma_A$',fontsize=20)
  ax2.set_ylabel(r'$\sigma_b$',fontsize=20)
  ax1.set_xlim([0,zmax])
  ax2.set_xlim([0,zmax])
  #ax1.set_ylim([0,1e-3])
  #ax2.set_ylim([0,1e-3])
  ax1.legend(loc='upper left')
  plt.show()

  return eigs, fidAs, fidBs, sigmaAs, sigmaBs
  

def plotSigmasByZmax(nz=1000,lmax=2000,nBins=16,z0=1.5,**cos_kwargs):
  """
    plot several sigmas for various values of nBins at one zmax
    Inputs:
      nz:
      lmax:
      nBins: total number of bins to use
      z0:
      **cos_kwargs: for set_cosmology

  """

  zMaxVals = (4,8,16,20)
  labels = ('zMax = 4','zMax = 8','zMax = 16','zMax = 20')
  nnZmax = 4  #number of nBins values (just a label, really)

  # to collect eigenvalues, As, bs, sigmas
  eigs    = []
  fidAs   = []
  fidBs   = []
  sigmaAs = []
  sigmaBs = []
  # get Fisher matrix objects and do plots
  fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,6))
  for zMaxIndex,zMaxNum in enumerate(zMaxVals):
    print '\n starting Fisher Matrix ',zMaxIndex+1,' of ',nnZmax, ', with zMax=',zMaxNum
    Fobj = FisherMatrix(nz=nz,lmax=lmax,zmax=zMaxNum,nBins=nBins,z0=z0,**cos_kwargs)
    fidAs = np.append(fidAs,Fobj.binAs)
    fidBs = np.append(fidBs,Fobj.binBs)

    print 'inverting Fij ',zMaxIndex+1,' of ',nnZmax
    FInv = np.linalg.inv(Fobj.Fij)

    # check eigenvalues
    w,v = np.linalg.eigh(FInv)
    print 'eigenvalues: ',w
    eigs = np.append(eigs,w)

    binCenters = Fobj.getBinCenters()
    #print 'binCenters: ',binCenters
    diags = np.diag(FInv)
    sigmas = np.sqrt(diags)
    print 'sigmas: ',sigmas
    #nBins = nBinsVals[nBinsIndex]
    As = sigmas[:nBins]
    Bs = sigmas[nBins:]
    sigmaAs = np.append(sigmaAs,As)
    sigmaBs = np.append(sigmaBs,Bs)
    #print 'As: ',As,', Bs: ',Bs,'\n'
    ax1.plot(binCenters,As,label=labels[zMaxIndex])
    ax2.plot(binCenters,Bs,label=labels[zMaxIndex])

  print 'eigenvalues of all inverses of Fisher matrices: ',eigs

  ax1.set_title(r'matter amplitudes $A_i$')
  ax2.set_title(r'galaxy biases $b_i$')
  ax1.set_xlabel('redshift',fontsize=15)
  ax2.set_xlabel('redshift',fontsize=15)
  ax1.set_ylabel(r'$\sigma_A$',fontsize=20)
  ax2.set_ylabel(r'$\sigma_b$',fontsize=20)
  ax1.set_xlim([0,20])
  ax2.set_xlim([0,20])
  #ax1.set_ylim([0,1e-3])
  #ax2.set_ylim([0,1e-3])
  ax2.legend(loc='upper left')
  plt.show()
    
  return eigs, fidAs, fidBs, sigmaAs, sigmaBs



def plotSigmasByBinSmooth(nz=1000,lmax=2000,nBins=8,z0=1.5,zmax=4.0,**cos_kwargs):
  """
    plot several sigmas for various values of binSmooth
      with fixed zmax, nBins
    Inputs:
      nz:
      lmax:
      nBins: total number of bins to use
      z0:
      zmax:
      **cos_kwargs: for set_cosmology

  """

  #binSmoothVals = (0,0.05,0.1,0.2,0.4)
  #labels = ('smooth = 0','smooth = 0.05','smooth = 0.1','smooth = 0.2','smooth = 0.4')
  binSmoothVals = (0,0.001,0.005,0.01,0.05)
  labels = ('smooth = 0','smooth = 0.001','smooth = 0.005','smooth = 0.01','smooth = 0.05')
  nBinSmooth = 5  #number of binSmooth values (just a label, really)

  # to collect eigenvalues, As, bs, sigmas
  eigs    = []
  fidAs   = []
  fidBs   = []
  sigmaAs = []
  sigmaBs = []
  # get Fisher matrix objects and do plots
  fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,6))
  for binSmoothIndex,binSmooth in enumerate(binSmoothVals):
    print '\n starting Fisher Matrix ',binSmoothIndex+1,' of ',nBinSmooth, ', with binSmooth=',binSmooth
    Fobj = FisherMatrix(nz=nz,lmax=lmax,zmax=zmax,nBins=nBins,z0=z0,
                        binSmooth=binSmooth,**cos_kwargs)
    fidAs = np.append(fidAs,Fobj.binAs)
    fidBs = np.append(fidBs,Fobj.binBs)

    print 'inverting Fij ',binSmoothIndex+1,' of ',nBinSmooth
    FInv = np.linalg.inv(Fobj.Fij)

    # check eigenvalues
    w,v = np.linalg.eigh(FInv)
    print 'eigenvalues: ',w
    eigs = np.append(eigs,w)

    binCenters = Fobj.getBinCenters()
    #print 'binCenters: ',binCenters
    diags = np.diag(FInv)
    sigmas = np.sqrt(diags)
    print 'sigmas: ',sigmas
    #nBins = nBinsVals[nBinsIndex]
    As = sigmas[:nBins]
    Bs = sigmas[nBins:]
    sigmaAs = np.append(sigmaAs,As)
    sigmaBs = np.append(sigmaBs,Bs)
    #print 'As: ',As,', Bs: ',Bs,'\n'
    ax1.plot(binCenters,As,label=labels[binSmoothIndex])
    ax2.plot(binCenters,Bs,label=labels[binSmoothIndex])

  print 'eigenvalues of all inverses of Fisher matrices: ',eigs

  ax1.set_title(r'matter amplitudes $A_i$')
  ax2.set_title(r'galaxy biases $b_i$')
  ax1.set_xlabel('redshift',fontsize=15)
  ax2.set_xlabel('redshift',fontsize=15)
  ax1.set_ylabel(r'$\sigma_A$',fontsize=20)
  ax2.set_ylabel(r'$\sigma_b$',fontsize=20)
  ax1.set_xlim([0,4])
  ax2.set_xlim([0,4])
  #ax1.set_ylim([0,1e-3])
  #ax2.set_ylim([0,1e-3])
  ax2.legend(loc='upper left')
  plt.show()
    
  return eigs, fidAs, fidBs, sigmaAs, sigmaBs



################################################################################
# testing code

def simpleCovar(crossCls,ells):
  """
    for testing the covariance matrix code
    Copied here from FisherMatrix class and simplified
    Inputs:
      #crossCls: an nMaps*nMaps*nElls numpy array of cross power
      crossCls: an nMaps*nMaps numpy array of cross power
      #ells: array of length nElls
      ells: the single ell value to use
    Returns:
      covar: the covariance matrix
      obsList: base nMaps representation of data label
  """
  nMaps = crossCls.shape[0] # or [1]
  nElls = 1 #crossCls.shape[2]

  # create covariance matrix
  print 'building covariance matrix... '
  nCls = nMaps*(nMaps+1)/2 # This way removes redundancies, eg C_l^kg = C_l^gk
  covar = np.zeros((nCls,nCls)) #,nElls)) #lmax-1))

  # create obsList to contain base nMaps representation of data label
  #   where kappa:0, g1:1, g2:2, etc.
  #   eg, C_l^{kappa,g1} -> 0*nMaps+1 = 01 = 1
  obsList = np.zeros(nCls)

  for map1 in range(nMaps):
    print 'starting covariance set ',map1+1,' of ',nMaps,'... '
    for map2 in range(map1, nMaps):
      covIndex1 = map1*nMaps+map2-map1*(map1+1)/2     # shortens the array
      obsList[covIndex1] = map1*nMaps+map2       # base nMaps representation
      for map3 in range(nMaps):
        for map4 in range(map3, nMaps):
          covIndex2 = map3*nMaps+map4-map3*(map3+1)/2 # shortens the array

          # output for debugging
          print 'map1,2,3,4: ',map1,map2,map3,map4
          print 'covIndex1,2: ',covIndex1,covIndex2

          if covIndex1 <= covIndex2:
            #covar[covIndex1,covIndex2] = (crossCls[map1,map2]*crossCls[map3,map4] + crossCls[map1,map4]*crossCls[map3,map2] )/(2.*ells+1)
            covar[covIndex1,covIndex2] = (crossCls[map1,map3]*crossCls[map2,map4] + crossCls[map1,map4]*crossCls[map2,map3] )/(2.*ells+1)
          else:                                       # avoid double calculation
            covar[covIndex1,covIndex2] = covar[covIndex2,covIndex1]

  return covar,obsList




def test(nz=1000,lmax=2000,zmax=16,z0=1.5):
  """
    function for testing the FisherMatrix object
  """

  # test __file__
  print 'file: ',__file__,'\n'

  # 10 min to make Fmatrix, Finv with nz=10000
  # create and initialize object
  #Fmatrix = FisherMatrix(nz=nz,lmax=lmax)

  # invert it
  #Finv = np.linalg.inv(Fmatrix.Fij)

  # plot something
  # note this took 15 minutes wtih lmax=2000, zmax=4, 
  #   nBinsVals = (4,8,12,16)
  plotSigmasByNBins(nz=nz,lmax=lmax,zmax=zmax,z0=z0)


if __name__=='__main__':
  test()





