#! /usr/bin/env python
"""
  Name:
    FisherCl
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


  def __init__(self,nz=1000,lmax=2000,zmax=16,nBins=10,z0=1.5,
               noAs=False, **cos_kwargs):
    """
    
      Inputs:
        nz: the number of z points to use between here and last scattering surface
          Important usage is as the number of points to use in approximation of
            C_l integrals
        lmax: maximum ell value to use in summations
        zmax: maximum z to use in binning for A_i, b_i parameters
        nBins: number of bins to use for A_i, b_i parameters
        z0: controls the width of the model galaxy distribution
        noAs: set to True if dCl/dA are all zero (As are fixed)
          This will just set the derivatives to zero, but rows and columns
          in Fij will still be there, but all zero.
        **cos_kwargs: the cosmological parameters for camb's set_cosmology

    """
################################################################################
    # preliminaries

    # set cosmological parameters
    self.cosParams = {
        'H0'    : 67.51,
        'ombh2' : 0.022,
        'omch2' : 0.119,
        'mnu'   : 0.06,
        'omk'   : 0,
        'tau'   : 0.06  }
    self.cosParams.update(cos_kwargs)

    # set other parameters
    self.lmax = lmax

    # set dndzMode and related params for use in winGalaxies and getNormalizedDNDZbin functions
    dndzMode = 2
    #zmax = 4
    #nBins = 10
    #z0 = 0.3
    self.zmax=zmax
    self.zmin=0.0
    self.nBins = nBins
    self.z0 = z0
    tophatBins = True # true if bins do not overlap, false if they do
    nMaps = nBins+1 # +1 for kappa map

    # observables list: defined as self.obsList; created along with self.covar
    # parameters list: not saved in data structure; described later
    
    # get matter power object
    print 'creating matter power spectrum object...'
    myPk = cp.matterPower(nz=nz,**self.cosParams)  #don't need to save as self.myPk...?
    PK,chistar,chis,dchis,zs,dzs,pars = myPk.getPKinterp()


################################################################################
    # create fiducial galxy bias and lensing amplitude as one parameter per bin
    # to match zs, dzs exactly to what normalization routine does, 
    #   use normPoints = 0 (no added points)
    zmin = self.zmin
    normPoints = 0
    verbose = False

    # redshift points for entire range zmin to zmax:
    myNormPoints = nBins*1000
    zArray = np.linspace(zmin,zmax,myNormPoints+1)
    deltaZ = (zmax-zmin)/(myNormPoints)

    bOfZfit = cp.byeBiasFit()  #Byeonghee's bias function
    bOfZ = bOfZfit(zArray)
    AOfZ = np.ones(myNormPoints+1) #fiducially all ones
    binBs = np.empty(nBins)
    binAs = np.empty(nBins)
    for binNum in range(nBins):
      # get weighted average over dNdz as integral of product with normalized dndz
      normalizedDNDZ = cp.getNormalizedDNDZbin(binNum+1,zArray,z0,zmax,nBins,
                          dndzMode=dndzMode,zmin=zmin,normPoints=normPoints,verbose=verbose)
      binBs[binNum] = np.sum(normalizedDNDZ*bOfZ)*deltaZ 
      # get weighted average over dWdz (lensing kernel)
      normalizedWinK = cp.getNormalizedWinKbin(myPk,binNum+1,zArray,zmin=zmin,
                          zmax=zmax,nBins=nBins,normPoints=normPoints,verbose=verbose)
      binAs[binNum] = np.sum(normalizedWinK*AOfZ)*deltaZ
    self.binBs = binBs
    self.binAs = binAs
    print 'fiducial bs: ',binBs
    print 'fiducial As: ',binAs


################################################################################
    # get all cross power spectra

    # transfer binBs,binAs to biases1,biases2 arrays
    # If I use AOfZ not all 1, this needs to be changed to include summation over bins for kk

    self.crossCls  = np.zeros((nMaps,nMaps,lmax-1)) #-1 to omit ell=1; this one for kappa

    # if tophatBins, only the diagonal and 0th row and column will be filled
    print 'starting cross power with entire kappa... '
    for map1 in range(nMaps):
      if map1==0:
        winfunc1 = cp.winKappa
        biases1=np.ones(zs.size)*binAs[map1-1]
      else:
        winfunc1 = cp.winGalaxies
        biases1=np.ones(zs.size)*binBs[map1-1]*binAs[map1-1]  # -1 since nMaps=nBins+1
      for map2 in range(map1,nMaps):
        print 'starting angular cross power spectrum ',map1,', ',map2,'... '
        if map2==0:
          winfunc2 = cp.winKappa
          biases2=np.ones(zs.size)*binAs[map2-1]
        else:
          winfunc2 = cp.winGalaxies
          biases2=np.ones(zs.size)*binBs[map2-1]*binAs[map2-1]  # -1 since nMaps=nBins+1
        # since nonoverlapping bins have zero correlation use this condition:
        if map1==0 or map1==map2 or not tophatBins:
          ells,Cls = cp.getCl(myPk,biases1=biases1,biases2=biases2,
              winfunc1=winfunc1,winfunc2=winfunc2,
              dndzMode1=dndzMode,dndzMode2=dndzMode,binNum1=map1,binNum2=map2,
              lmax=lmax,zmax=zmax,nBins=nBins,z0=z0)
          self.crossCls[map1,map2] = Cls
          self.crossCls[map2,map1] = Cls #symmetric
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
          biasesKi = np.ones(zs.size)*binAs[bin1]
          biasesGi = np.ones(zs.size)*binBs[bin2]*binAs[bin2]
          # kk
          ells,Cls = cp.getCl(myPk,biases1=biasesKi,biases2=biasesKi,
              winfunc1=cp.winKappa,winfunc2=cp.winKappa,
              binNum1=bin1+1,binNum2=bin2+1,
              lmax=lmax,zmax=zmax,nBins=nBins,z0=z0)
          self.crossClBinsKK[bin1,bin2] = Cls
          self.crossClBinsKK[bin2,bin1] = Cls #symmetric
          # kg
          ells,Cls = cp.getCl(myPk,biases1=biasesKi,biases2=biasesGi,
              winfunc1=cp.winKappa,winfunc2=cp.winGalaxies,
              dndzMode2=dndzMode,binNum1=bin1+1,binNum2=bin2+1,
              lmax=lmax,zmax=zmax,nBins=nBins,z0=z0)
          self.crossClBinsKG[bin1,bin2] = Cls
          self.crossClBinsKG[bin2,bin1] = Cls #symmetric
          # gg
          ells,Cls = cp.getCl(myPk,biases1=biasesGi,biases2=biasesGi,
              winfunc1=cp.winGalaxies,winfunc2=cp.winGalaxies,
              dndzMode1=dndzMode,dndzMode2=dndzMode,binNum1=bin1+1,binNum2=bin2+1,
              lmax=lmax,zmax=zmax,nBins=nBins,z0=z0)
          self.crossClBinsGG[bin1,bin2] = Cls
          self.crossClBinsGG[bin2,bin1] = Cls #symmetric




################################################################################
    # create covariance matrix
    print 'building covariance matrix... '
    nCls = nMaps*(nMaps+1)/2 # This way removes redundancies, eg C_l^kg = C_l^gk
    self.covar = np.zeros((nCls,nCls,lmax-1))

    # create obsList to contain base nMaps representation of data label
    #   where kappa:0, g1:1, g2:2, etc.
    #   eg, C_l^{kappa,g1} -> 0*nMaps+1 = 01 = 1
    self.obsList = np.zeros(nCls)

    for map1 in range(nMaps):
      print 'starting covariance set ',map1+1,' of ',nMaps,'... '
      for map2 in range(map1, nMaps):
        covIndex1 = map1*nMaps+map2-map1*(map1+1)/2     # shortens the array
        self.obsList[covIndex1] = map1*nMaps+map2       # base nMaps representation
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

    # check this
    #myL = 423
    #print 'check inverse at ell = ',myL,': '
    #print np.dot(self.invCov[:,:,myL],self.covar[:,:,myL])
    #print np.dot(self.covar[:,:,myL],self.invCov[:,:,myL])




################################################################################
    # get derivatives wrt parameters

    # parameters list: a_i in {A_1, A_2, ..., A_nBins, b_1, b_2, ..., b_nBins}
    #nParams = 2*nBins
    
    # use self.crossCls for relevant power spectra
    # get dC_l^munu/da_i (one vector of derivatives of C_ls for each a_i)
    # store as matrix with additional dimension for a_i)
    print 'starting creation of C_l derivatives... '
    # uses same (shortened) nCls as self.covar and self.obsList
    self.dClVecs = np.empty((nCls, 2*nBins, lmax-1))
    Clzeros = np.zeros(lmax-1) # for putting into dClVecs when needed
    for map1 in range(nMaps):
      print 'starting derivative set ',map1+1,' of ',nMaps,'... '
      for map2 in range(map1,nMaps):
        mapIdx  = map1*nMaps+map2 -map1*(map1+1)/2  
                                   # mapIdx = map index
        for pIdx in range(nBins):  # pIdx = parameter index
          Ai   = pIdx
          bi   = nBins+pIdx        # Bs to be after the As

          if map1 == 0: #kappa
            if map2 == 0:          #kk
              # this section assumes Sum_i^{nBins+1} W^{k_i} = W^k (completeness)
              if noAs:
                self.dClVecs[  mapIdx, Ai] = Clzeros
              else:
                self.dClVecs[  mapIdx, Ai] = 2/binAs[pIdx] * self.crossClBinsKK[pIdx,pIdx]
              self.dClVecs[    mapIdx, bi] = Clzeros 

            else:                  #kg,gk
              # this section assumes no bin overlap (update later)
              if pIdx+1 == map2: # +1 since 1 more map than bin
                if noAs:
                  self.dClVecs[mapIdx, Ai] = Clzeros
                else:
                  self.dClVecs[mapIdx, Ai] = 2/binAs[pIdx] * self.crossClBinsKG[pIdx,pIdx]
                self.dClVecs[  mapIdx, bi] = 1/binBs[pIdx] * self.crossClBinsKG[pIdx,pIdx]
              else: # parameter index does not match bin index
                self.dClVecs[  mapIdx, Ai] = Clzeros
                self.dClVecs[  mapIdx, bi] = Clzeros

          else: #galaxies          #gg
            if pIdx+1 == map2: # +1 since 1 more map than bin
              if map1 == map2:
                if noAs:
                  self.dClVecs[mapIdx, Ai] = Clzeros
                else:
                  self.dClVecs[mapIdx, Ai] = 2/binAs[pIdx] * self.crossClBinsGG[pIdx,pIdx]
                self.dClVecs[  mapIdx, bi] = 2/binBs[pIdx] * self.crossClBinsGG[pIdx,pIdx]
              else:
                # this section assumes no bin overlap (update later)
                self.dClVecs[  mapIdx ,Ai] = Clzeros 
                self.dClVecs[  mapIdx ,bi] = Clzeros 
            else: # parameter index does not match bin index
              self.dClVecs[    mapIdx, Ai] = Clzeros
              self.dClVecs[    mapIdx, bi] = Clzeros
            

       



    # this bit was from when A was A_lens, not A_matter
    """
    print 'starting creation of C_l derivatives... '
    self.dClVecs = np.empty((nCls, 2*nBins, lmax-1))
    # uses same (shortened) nCls as self.covar and self.obsList
    Clzeros = np.zeros(lmax-1) # for putting into dClVecs when needed
    for map1 in range(nMaps):
      print 'starting derivative set ',map1+1,' of ',nMaps,'... '
      for map2 in range(map1,nMaps):
        #mapIdx  = map1*nMaps+map2  # mapIdx = map index
        #mapIdxT = map2*nMaps+map1  # index for transpose C_l^kg <-> C_l^gk
        mapIdx  = map1*nMaps+map2 -map1*(map1+1)/2  # mapIdx = map index
        #mapIdxT = map2*nMaps+map1 -map2*(map2+1)/2  # index for transpose C_l^kg <-> C_l^gk
        for pIdx in range(nBins):  # pIdx = parameter index
          Ai = pIdx
          bi = nBins+pIdx          # Bs to be after the As

          if map1 == 0: #kappa
            if map2 == 0:          #kk
              # this section assumes Sum_i^{nBins+1} W^{k_i} = W^k (completeness)
              self.dClVecs[mapIdx,Ai] = 2/binAs[pIdx] * self.crossClBinsKK[pIdx,pIdx]
              self.dClVecs[mapIdx,bi] = Clzeros 
            else:                  #kg,gk
              # this section assumes no bin overlap (update later)
              self.dClVecs[mapIdx,Ai] = 1/binAs[pIdx] * self.crossClBinsKG[pIdx,pIdx]
              self.dClVecs[mapIdx,bi] = 1/binBs[pIdx] * self.crossClBinsKG[pIdx,pIdx]
              # fill in transpose via symmetry
              #self.dClVecs[mapIdxT,Ai] = self.dClVecs[mapIdx,Ai]
              #self.dClVecs[mapIdxT,bi] = self.dClVecs[mapIdx,bi]
          else: #galaxies          #gg
            if map1 == map2:
              self.dClVecs[mapIdx,Ai] = Clzeros
              self.dClVecs[mapIdx,bi] = 2/binBs[pIdx] * self.crossClBinsGG[pIdx,pIdx]
            else:
              # this section assumes no bin overlap (update later)
              self.dClVecs[mapIdx ,Ai] = Clzeros 
              self.dClVecs[mapIdx ,bi] = Clzeros 
              #self.dClVecs[mapIdxT,Ai] = Clzeros 
              #self.dClVecs[mapIdxT,bi] = Clzeros 
    """

              

################################################################################
    #Build Fisher matrix
    #multply vectorT,invcov,vector and add up
    print 'building Fisher matrix from components...'
    print 'invCov.shape: ',self.invCov.shape,', dClVecs.shape: ',self.dClVecs.shape
    self.Fij = np.zeros((2*nBins,2*nBins)) # indexed by parameters A_i, b_i
    for i in range(2*nBins):
      print 'starting bin set ',i+1,' of ',2*nBins
      dClVec_i = self.dClVecs[:,i,:] # shape (nCls,nElls)
      for j in range(2*nBins):
        dClVec_j = self.dClVecs[:,j,:] # shape (nCls,nElls)
        # ugh.  don't like nested loops in Python... but easier to program...
        for ell in range(lmax-1):
          myCov = self.invCov[:,:,ell]
          #print
          fij = np.dot(dClVec_i[:,ell],np.dot(myCov,dClVec_j[:,ell]))
          

          #test = np.where(fij>1e14)
          #print 'fij>1e14 at ',test
          self.Fij[i,j] += fij
    
    print 'creation of Fisher Matrix complete!\n'
    # end of init function


################################################################################
  def getBinCenters(self):
    """
      return array of centers of bins
    """
    halfBinWidth = (self.zmax-self.zmin)/(2*self.nBins)
    nHalfBins = (2*np.arange(self.nBins)+1)
    
    return halfBinWidth*nHalfBins





  # end of class FisherMatrix

################################################################################


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


def plotSigmasByNBins(nz=1000,lmax=2000,zmax=16,z0=1.5,noAs=False):
  """
    plot several sigmas for various values of nBins at one zmax
    Inputs:
      nz:
      lmax:
      zmax:
      z0:
      noAs: set to True if dCl/dA are all zero (As are fixed)

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

  # to collect eigenvalues
  eigs = []
  # get Fisher matrix objects and do plots
  fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,6))
  for nBinsIndex,nBinsNum in enumerate(nBinsVals):
    print '\n starting Fisher Matrix ',nBinsIndex+1,' of ',nnBins, ', with nBins=',nBinsNum
    Fobj = FisherMatrix(nz=nz,lmax=lmax,zmax=zmax,nBins=nBinsNum,z0=z0,noAs=noAs)

    print 'inverting Fij ',nBinsIndex+1,' of ',nnBins
    if noAs:
      upperQIndices = np.arange(Fobj.nBins)
      myFij = Fobj.Fij
      myFij[upperQIndices,upperQIndices] = 1
      FInv = np.linalg.inv(myFij)
    else:
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
    
  
def plotSigmasByZmax(nz=1000,lmax=2000,nBins=16,z0=1.5,noAs=False):
  """
    plot several sigmas for various values of nBins at one zmax
    Inputs:
      nz:
      lmax:
      nBins: total number of bins to use
      z0:
      noAs: set to True if dCl/dA are all zero (As are fixed)

  """

  zMaxVals = (4,8,16,20)
  labels = ('zMax = 4','zMax = 8','zMax = 16','zMax = 20')
  nnZmax = 4  #number of nBins values (just a label, really)

  # to collect eigenvalues
  eigs = []
  # get Fisher matrix objects and do plots
  fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,6))
  for zMaxIndex,zMaxNum in enumerate(zMaxVals):
    print '\n starting Fisher Matrix ',zMaxIndex+1,' of ',nnZmax, ', with zMax=',zMaxNum
    Fobj = FisherMatrix(nz=nz,lmax=lmax,zmax=zMaxNum,nBins=nBins,z0=z0,noAs=noAs)

    print 'inverting Fij ',zMaxIndex+1,' of ',nnZmax
    if noAs:
      upperQIndices = np.arange(Fobj.nBins)
      myFij = Fobj.Fij
      myFij[upperQIndices,upperQIndices] = 1
      FInv = np.linalg.inv(myFij)
    else:
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




