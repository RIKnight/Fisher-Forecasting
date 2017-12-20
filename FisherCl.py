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

"""

#import sys, platform, os
import numpy as np
import matplotlib.pyplot as plt
#import scipy.integrate as sint
from scipy.interpolate import interp1d
#import camb
#from camb import model, initialpower
#from scipy import polyfit,poly1d
import crosspower as cp

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


  def __init__(self,nz=10000,lmin=2,lmax=2000,zmin=0.0,zmax=16.0,dndzMode=2, 
               nBins=10,z0=1.5,doNorm=True,useWk=False,binSmooth=0,BPZ=True, 
               biasByBin=True,AccuracyBoost=3,**cos_kwargs):
    """
    
      Inputs:
        nz: the number of z points to use between here and last scattering surface
          Important usage is as the number of points to use in approximation of
            C_l integrals
        lmin,lmax: minimum,maximum ell value to use in summations
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
          useWk: set to True to use W^kappa as dN/dz
            Defalut: False
          binSmooth: parameter that controls the amount of smoothing of bin edges
            Default: 0 (no smoothing)
        AccuracyBoost: to pass to set_accuracy to set accuracy
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

        'w'     : -1.0, # DARK ENERGY!!!

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
    self.dndzMode = dndzMode
    self.BPZ = BPZ
    self.zmin = zmin
    self.zmax = zmax
    self.nBins = nBins
    self.z0 = z0
    self.lmin = lmin
    self.lmax = lmax
    self.AccuracyBoost=AccuracyBoost
    if binSmooth == 0 and dndzMode == 2:
      tophatBins = True # true if bins do not overlap, false if they do
    else:
      tophatBins = False
    nMaps = nBins+1 # +1 for kappa map

    # observables list: defined as self.obsList; created along with self.covar
    nCls = nMaps*(nMaps+1)/2 # This way removes redundancies, eg C_l^kg = C_l^gk

    # parameters list:
    nCosParams = 8 # 6 LCDM + Mnu + w
    nParams = nCosParams+nBins
    paramList = ['ombh2','omch2','cosmomc_theta','As','ns','tau','mnu','w']
    for bin in range(nBins):
      paramList.append('bin'+str(bin+1))
    self.nParams   = nParams
    self.paramList = paramList

    # step sizes for discrete derivatives: must correspond to paramList entries!
    #   from Allison et. al. (2015) Table III.
    deltaP = [0.0008,0.0030,0.0050e-2,0.1e-9,0.010,0.020,0.020,0.3] #mnu one in eV

    # get MatterPower object
    print 'creating MatterPower object...'
    myPk = cp.MatterPower(nz=nz,AccuracyBoost=AccuracyBoost,**self.cosParams)
    PK,chistar,chis,dchis,zs,dzs,pars = myPk.getPKinterp()
    #self.H0 = myPk.H0
    self.H0 = pars.H0
    self.Hs = myPk.Hs
    self.zs = myPk.zs

    # get Window object
    print 'creating Window object...'
    myWin = cp.Window(myPk,zmin=zmin,zmax=zmax,nBins=nBins,biasK=cp.ones,
                      biasG=cp.byeBias,dndzMode=dndzMode,z0=z0,
                      doNorm=doNorm,useWk=useWk,BPZ=BPZ,binSmooth=binSmooth,
                      biasByBin=biasByBin)

    # get bucketloads more of them for numeric differentiation
    print 'creating more matter power objects...'
    myParams = self.cosParams
    myParamsUpper = []
    myParamsLower = []
    myPksUpper = []
    myPksLower = []
    myWinsUpper = []
    myWinsLower = []
    for cParamNum in range(nCosParams):
      print 'creating matter power spectra and window functions for ',\
            paramList[cParamNum],' derivative...'
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

      #print 'cPramNum: ',cParamNum,', param name: ',paramList[cParamNum]
      #print 'myParamsUpper[cParamNum][paramList[cParamNum]]: ',myParamsUpper[cParamNum][paramList[cParamNum]]
      #print 'myParamsLower[cParamNum][paramList[cParamNum]]: ',myParamsLower[cParamNum][paramList[cParamNum]]
      #print 'deltaP[cParamNum]: ',deltaP[cParamNum]

      # create MatterPower objects and add to lists
      myPksUpper.append(cp.MatterPower(nz=nz,AccuracyBoost=AccuracyBoost,
                        **myParamsUpper[cParamNum]))
      myPksLower.append(cp.MatterPower(nz=nz,AccuracyBoost=AccuracyBoost,
                        **myParamsLower[cParamNum]))
      # create Window objects and add to lists
      myWinsUpper.append(cp.Window(myPksUpper[cParamNum],zmin=zmin,zmax=zmax,
                         nBins=nBins,biasK=cp.ones,biasG=cp.byeBias,
                         dndzMode=dndzMode,z0=z0,doNorm=doNorm,useWk=useWk,
                         BPZ=BPZ,binSmooth=binSmooth,biasByBin=biasByBin))
      myWinsLower.append(cp.Window(myPksLower[cParamNum],zmin=zmin,zmax=zmax,
                         nBins=nBins,biasK=cp.ones,biasG=cp.byeBias,
                         dndzMode=dndzMode,z0=z0,doNorm=doNorm,useWk=useWk,
                         BPZ=BPZ,binSmooth=binSmooth,biasByBin=biasByBin))

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

    self.crossCls      = np.zeros((nMaps,nMaps,           lmax-1)) #-1 to omit ell=1
    self.crossClsPlus  = np.zeros((nMaps,nMaps,nCosParams,lmax-1))
    self.crossClsMinus = np.zeros((nMaps,nMaps,nCosParams,lmax-1))

    # if tophatBins, only the diagonal and 0th row and column will be filled
    print 'starting cross power with entire kappa... '
    for map1 in range(nMaps):
      if map1==0:
        cor1 = cp.Window.kappa
      else:
        cor1 = cp.Window.galaxies
      for map2 in range(map1,nMaps):
        print 'starting angular cross power spectrum ',map1,', ',map2,'... '
        if map2==0:
          cor2 = cp.Window.kappa
        else:
          cor2 = cp.Window.galaxies
        # since nonoverlapping bins have zero correlation use this condition:
        if map1==0 or map1==map2 or not tophatBins:
          ells,Cls = cp.getCl(myPk,myWin,binNum1=map1,binNum2=map2,
                           cor1=cor1,cor2=cor2,lmin=lmin,lmax=lmax)
          self.crossCls[map1,map2] = Cls
          self.crossCls[map2,map1] = Cls #symmetric

          # now the adjustments for numeric derivatives
          for cParamNum in range(nCosParams):
            ells,Cls = cp.getCl(myPksUpper[cParamNum],myWinsUpper[cParamNum],
                                binNum1=map1,binNum2=map2,
                                cor1=cor1,cor2=cor2,lmin=lmin,lmax=lmax)
            self.crossClsPlus[map1,map2,cParamNum] = Cls
            self.crossClsPlus[map2,map1,cParamNum] = Cls #symmetric
            ells,Cls = cp.getCl(myPksLower[cParamNum],myWinsLower[cParamNum],
                                binNum1=map1,binNum2=map2,
                                cor1=cor1,cor2=cor2,lmin=lmin,lmax=lmax)
            self.crossClsMinus[map1,map2,cParamNum] = Cls
            self.crossClsMinus[map2,map1,cParamNum] = Cls #symmetric
            
    self.ells = ells

    # divide K,G into bins and get crossClbins
    self.crossClBinsKK = np.zeros((nBins,nBins,lmax-1))
    self.crossClBinsKG = np.zeros((nBins,nBins,lmax-1))
    self.crossClBinsGG = np.zeros((nBins,nBins,lmax-1))

    # if tophatBins, only the diagonals will be filled
    # note: cp.getCl has a +1 offset to bin numbers, 
    #   since bin 0 indicates sum of all bins
    print 'starting cross power with binned kappa... '
    for bin1 in range(nBins):
      for bin2 in range(bin1,nBins):
        if bin1==bin2 or not tophatBins:
          print 'starting angular cross power spectrum ',bin1,', ',bin2,'... '
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


################################################################################
    # create covariance matrix
    print 'building covariance matrix... '

    #nCls = nMaps*(nMaps+1)/2 # This way removes redundancies, eg C_l^kg = C_l^gk
    # nCls defined above
    self.covar = np.zeros((nCls,nCls,lmax-1))

    # create obsList to contain base nMaps representation of data label
    #   where kappa:0, g1:1, g2:2, etc.
    #   eg, C_l^{kappa,g1} -> 0*nMaps+1 = 01 = 1
    self.fieldNames = ['k']
    for binNum in range(1,nBins+1):
        self.fieldNames.append('g'+str(binNum))
    self.obsList = np.zeros(nCls)
    self.obsNames = []

    for map1 in range(nMaps):
      print 'starting covariance set ',map1+1,' of ',nMaps,'... '
      for map2 in range(map1, nMaps):
        covIndex1 = map1*nMaps+map2-map1*(map1+1)/2     # shortens the array
        self.obsList[covIndex1] = map1*nMaps+map2       # base nMaps representation
        self.obsNames.append(self.fieldNames[map1]+','+self.fieldNames[map2])
        for map3 in range(nMaps):
          for map4 in range(map3, nMaps):
            covIndex2 = map3*nMaps+map4-map3*(map3+1)/2 # shortens the array
            if covIndex1 <= covIndex2:
              #self.covar[covIndex1,covIndex2] = (self.crossCls[map1,map2]*self.crossCls[map3,map4] + self.crossCls[map1,map4]*self.crossCls[map3,map2] )/(2.*self.ells+1)
              self.covar[covIndex1,covIndex2] = (self.crossCls[map1,map3]*self.crossCls[map2,map4] + self.crossCls[map1,map4]*self.crossCls[map2,map3] )/(2.*self.ells+1)
            else:                                       # avoid double calculation
              self.covar[covIndex1,covIndex2] = self.covar[covIndex2,covIndex1]

    # invert covariance matrix
    print 'inverting covariance matrix... '
    # transpose of inverse of transpose is inverse of original
    # need to do this to get indices in order that linalg.inv wants them
    self.invCov = np.transpose(np.linalg.inv(np.transpose(self.covar)))


################################################################################
    # get derivatives wrt parameters
    print 'starting creation of C_l derivatives... '

    # get dC_l^munu/da_i (one vector of derivatives of C_ls for each param a_i)
    # store as matrix with additional dimension for a_i)
    # uses same (shortened) nCls as self.covar and self.obsList
    self.dClVecs = np.empty((nCls, nParams, lmax-1))
    Clzeros = np.zeros(lmax-1) # for putting into dClVecs when needed
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
                self.dClVecs[  mapIdx, bi] = 1/self.binBs[pIdx] * self.crossClBinsKG[pIdx,pIdx]
              else: # parameter index does not match bin index
                self.dClVecs[  mapIdx, bi] = Clzeros

          else: #galaxies          #gg
            if pIdx+1 == map2: # +1 since 1 more map than bin
              if map1 == map2:
                self.dClVecs[  mapIdx, bi] = 2/self.binBs[pIdx] * self.crossClBinsGG[pIdx,pIdx]
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
    


################################################################################
    #Build Fisher matrix
    self.Fij = self.makeFisher(self.lmin,self.lmax)
    print 'creation of Fisher Matrix complete!\n'
    # end of init function



################################################################################
# other methods


  def makeFisher(self,lmin,lmax):
    #multply vectorT,invcov,vector and add up
    print 'building Fisher matrix from components...'
    print 'invCov.shape: ',self.invCov.shape,', dClVecs.shape: ',self.dClVecs.shape
    nParams = self.nParams
    Fij = np.zeros((nParams,nParams)) # indices match those in paramList
    for i in range(nParams):
      print 'starting bin set ',i+1,' of ',nParams
      dClVec_i = self.dClVecs[:,i,:] # shape (nCls,nElls)
      for j in range(nParams):
        dClVec_j = self.dClVecs[:,j,:] # shape (nCls,nElls)
        # ugh.  don't like nested loops in Python... but easier to program...
        for ell in range(lmax-1):
          myCov = self.invCov[:,:,ell]
          fij = np.dot(dClVec_i[:,ell],np.dot(myCov,dClVec_j[:,ell]))
          Fij[i,j] += fij
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





