#! /usr/bin/env python
"""
  Name:
    crosspower.py (branch quickCl)
  Purpose:
    calculate various theoretical angular power spectra for gravitational lensing 
      and galaxies
  Uses:
    pycamb (aka camb)
  Modification History:
    Written by Z Knight, 2017.07.31
    Added dNdz interpolation; ZK, 2017.08.04
    Added sum of dNdz curves in dNdz interp; ZK, 2017.08.08
    Added gBias from Gian16 data; ZK, 2017.08.15
    Fixed error of missing chi factors in winKappa and getCl; ZK, 2017.08.16
    Modified functions so that cosmological parameters are **kwargs; 
      Consolodated some code into matterPower object;
      Added nz as parameter to matterPower init; 
      Modified getCl, winKappa, winGalaxies for bias/amp params; ZK, 2017.08.27
    Added plotRedshiftAppx; ZK, 2017.08.28
    Added modelDNDZ; ZK, 2017.08.29
    Modified winGalaxies and getCl to take modelDNDZ; ZK, 2017.08.30
    Added byeBiasFit, getNormalizedDNDZbin; ZK, 2017.09.01
    Added getWinKinterp, getNormalizedWinKbin to match DNDZ versions;
      Reduced winKappa and winGalaxies parameter lists; ZK, 2017.09.06
    Added bin option to winKappa; ZK, 2017.09.07
    Added note on biases to winGalaxies; ZK, 2017.09.19
    Modified winGalaxies to omit normalization if needed; ZK, 2017.09.27
    Added z0 *= 1.0 to modelDNDZ to fix dividing by integer problem; 
      ZK, 2017.09.28
    Added switch to use W^k as dNdz in winGalaxies; ZK, 2017.09.29
    Modified getNormalizedDNDZbin and getNormalizedWinKbin to always
      use the zs points given to them in normalization, and to call same
      normBin routine; ZK, 2017.10.01
    Modified plotModelDNDZbins and plotWinKbins to have normalized or
      non-normalized plotting ability; ZK, 2017.10.02
    Modified normBin to interpret redshift points as bin centers rather than
      edges; ZK, 2017.10.04
    Added overlapping redshift bin functionality (dndzMode1); ZK, 2017.10.06
    Added bin smoothing with Gaussian (dndzMode2); ZK, 2017.10.10
    Expanded cosmological parameter set for nuLambdaCDM; ZK, 2017.10.15
    Reparameterized from H0 to cosmomc_theta; ZK, 2017.10.16
    Added H0, Hs fields to matterPower; ZK, 2017.10.20
    Added omega_n to omega_m sum in winKappa; ZK, 2017.10.23
    Changed matter power interpolator zmax to zstar; ZK, 2017.10.24

    Branched off of master.  This version reverts to function getCl doing
      rough approximation to integration; ZK, 2017.12.13
    Fixed dz/dchi problem in winGalaxies; made matterPower default nz=10000;
      renamed matterPower as MatterPower; ZK, 2017.12.14

"""

#import sys, platform, os
import numpy as np
import matplotlib.pyplot as plt
#import scipy.integrate as sint
from scipy.interpolate import interp1d
import camb
from camb import model, initialpower
from scipy import polyfit,poly1d


################################################################################
# the matterPower class

class MatterPower:
  """
    Purpose: 
      create and manipulate matter power spectrum objects
    Description:

    Data:
      cosParams: the cosmological parameters for camb's set_cosmology
        Default: 
          cosParams = {
            'H0'    : None, #67.51,
            'cosmomc_theta'           : 1.04087e-2,
            'ombh2' : 0.02226,
            'omch2' : 0.1193,
            'omk'   : 0,
            'tau'   : 0.063,

            'mnu'   : 0.06, #(eV)
            'nnu'   : 3.046,
            'standard_neutrino_neff'  : 3.046,
            'num_massive_neutrinos'   : 1,
            'neutrino_hierarchy'      : 'normal'}
      ns: primordial running
      r: something else primordial (tensor to scalar, probably)
      nz: number of z points to collect z, chi, dz, dchi at between here and last scattering
        Default: 1000
      kmax: maximum k value to use in Pk

      PK:
      chistar:
      chis:
      dchi:
      zs:
      dzs:
      pars:

      H0: hubble constant in Mpc^-1 units
      Hs: hubble parameter H(z) for all zs points

    Methods:
      __init__
      getPars: gets camb parameters object
      makePKinterp: make the Pk interpolator
      getPKinterp: return the Pk interpolator and other related info

  """

  def __init__(self,nz=10000,As=2.130e-9,ns=0.9653,r=0,kPivot=0.05,
               nonlinear=True,**cos_kwargs):
    """
    
      Inputs:
        nz: the number of z points to use between here and last scattering surface
          Important usage is as the number of points to use in approximation of
            C_l integrals (endpoints will be dropped for that calculation)
        As: "comoving curvature power at k=piveo_scalar"(sic)
        ns: "scalar spectral index"
        r: "tensor to scalar ratio at pivot"
        kPivot: "pivot scale for power spectrum"
        nonlinear: set to True to use CAMB's non-linear correction from halo model
        **cos_kwargs: the cosmological parameters for camb's set_cosmology

    """
    # set cosmological parameters
    self.cosParams = {
        #'H0'    : 67.51,
        'H0'    : None,
        'cosmomc_theta'           : 1.04087e-2,
        'ombh2' : 0.02226,
        'omch2' : 0.1193,
        'omk'   : 0,
        'tau'   : 0.063,

        'mnu'   : 0.06, # (eV)
        'nnu'   : 3.046,
        'standard_neutrino_neff'  : 3.046,
        'num_massive_neutrinos'   : 1,
        'neutrino_hierarchy'      : 'normal'}
    self.updateParams(**cos_kwargs)
    self.As = As
    self.ns = ns
    self.r  = r
    self.kPivot = kPivot
    self.nonlinear=nonlinear

    # more parameters
    self.nz = nz # this will actually be 2 more than the number of z points
    self.kmax = 10
    k_per_logint = None #100 # I really don't know what this will do

    # make the PK interpolator (via camb)
    self.makePKinterp(newPk=True,nz=nz,kmax=self.kmax,As=As,ns=ns,r=r,kPivot=kPivot,
                      k_per_logint=k_per_logint,nonlinear=nonlinear)



  def updateParams(self,**cos_kwargs):
    """
      modify object's cosParams variable
      Inputs:
        **cos_kwargs: dictionary of modifications to be passed to set_cosmology
    """
    self.cosParams.update(cos_kwargs)


  def getPars(self,As=2.130e-9,ns=0.9653,r=0,kPivot=0.05,**cos_kwargs):
    """
      Purpose:
        quickly get camb parameters object
        follows example code from http://camb.readthedocs.io/en/latest/CAMBdemo.html
          but with slightly different parameters
      Notes:
        This method uses object's parameters by default, but these can be overriden.
        This method does not overwrite object's stored parameters
      Inputs:
        To pass on to set_params:
          As: "comoving curvature power at k=piveo_scalar"(sic)
          ns: "scalar spectral index"
          r: "tensor to scalar ratio at pivot"
          kPivot: "pivot scale for power spectrum"
        **cos_kwargs: keyword args to pass to set_cosmology 
          if not included, object defaults will be used
      Returns:
        the pars object
    """
    # get cosmological parameters
    cosParams = self.cosParams
    cosParams.update(cos_kwargs)

    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    #pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06) #why 0.122?
    #pars.set_cosmology(H0=67.51, ombh2=0.022, omch2=0.119, mnu=0.06, omk=0, tau=0.06)
    pars.set_cosmology(**cosParams)
    pars.set_dark_energy() #re-set defaults
    #pars.set_matter_power() # get_matter_power_interpolater does this
    pars.InitPower.set_params(As=As,ns=ns,r=r,pivot_scalar=kPivot)

    #pars.set_accuracy(AccuracyBoost=2)

    return pars


  def makePKinterp(self,newPk=True,nz=1000,kmax=10,As=2.130e-9,ns=0.9653,r=0,
                   kPivot=0.05,myVar=model.Transfer_tot,k_per_logint=None,
                   nonlinear=True,**cos_kwargs):
    """
      example code from http://camb.readthedocs.io/en/latest/CAMBdemo.html
        (modified to have nz,kmax,myVar as inputs)
        (and to have dzs as additional output)
      Purpose:
        For calculating large-scale structure and lensing results yourself, get a power spectrum
        interpolation object. In this example we calculate the CMB lensing potential power
        spectrum using the Limber approximation, using PK=camb.get_matter_power_interpolator() function.
        calling PK(z, k) will then get power spectrum at any k and redshift z in range.
      Inputs:
        newPk: set to True to calculate new pars, Pk, etc. for object variables
        nz:   number of steps to use for the radial/redshift integration
          (endpoints will be dropped in actual calculation)
        kmax: kmax to use
        
        As: "comoving curvature power at k=piveo_scalar"(sic)
        ns: "scalar spectral index"
        r: "tensor to scalar ratio at pivot"
        kPivot: "pivot scale for power spectrum"
        myVar: the variable to get autopower spectrum of
          Default: model.Transfer_tot for delta_tot
        k_per_logint=None: to pass to get_matter_power_interpolater 
          which passes it to set_matter_power
        nonlinear: set to True to use CAMB's non-linear correction from halo model
          Default: True
        **cos_kwargs: keyword args to pass to getPars and set_cosmology
          if not used, getPars will use object defaults
      Outputs:
        sets various object variables

    """
    if newPk:
      self.updateParams(**cos_kwargs)
      self.pars = self.getPars(As=As,ns=ns,r=r,kPivot=kPivot)

    #For Limber result, want integration over \chi (comoving radial distance), from 0 to chi_*.
    #so get background results to find chistar, set up arrage in chi, and calculate corresponding redshifts
    results= camb.get_background(self.pars)
    self.chistar = results.conformal_time(0)- model.tau_maxvis.value
    self.chis = np.linspace(0,self.chistar,nz)
    self.zs=results.redshift_at_comoving_radial_distance(self.chis)
    self.zstar = self.zs[-1]
    #Calculate array of delta_chi, and drop first and last points where things go singular
    self.dchis = (self.chis[2:]-self.chis[:-2])/2 #overkill since chis are evenly spaced
    self.dzs   = (  self.zs[2:]-  self.zs[:-2])/2 #not as nice as with chi since zs not evenly spaced
    self.chis = self.chis[1:-1]
    self.zs = self.zs[1:-1]
    print 'zs.size: ',self.zs.size

    #Get the matter power spectrum interpolation object (based on RectBivariateSpline). 
    #Here for lensing we want the power spectrum of the Weyl potential.
    self.PK = camb.get_matter_power_interpolator(self.pars, nonlinear=nonlinear, 
        hubble_units=False, k_hunit=False, kmax=kmax,k_per_logint=k_per_logint,
        var1=myVar,var2=myVar, zmax=self.zstar) #self.zs[-1])

    #Get H(z) values (in Mpc^-1 units)
    #print 'calculating H(z) at each z...'
    self.Hs = np.empty(nz-2)
    self.H0 = results.h_of_z(0)
    for zIndex, z in enumerate(self.zs):
      self.Hs[zIndex] = results.h_of_z(z)


  def getPKinterp(self):
    """
      Purpose:
        Just returns some values from object
      Returns:
        the PK(z,k) interpolator
        chistar (chi of last scattering surface)
        chis (array of chi values) (actually appears to be conformal time)
        dchis (delta chi array)
        zs (redshift array)
        dzs (delta z array)
        pars (CAMB parameters)
    """

    return self.PK,self.chistar,self.chis,self.dchis,self.zs,self.dzs,self.pars




################################################################################
# window functions

def getDNDZ(binNum=1,BPZ=True):
  """
    function to load dN/dz(z) data points from file
    Note:
      data was digitized from figure 3 of Crocce et al, 2016
      data files should be in directory dNdz
    Inputs:
      binNum: integer in {1,2,3,4,5}.
        each corresponds to a redshift bin of width Delta.z=0.2:
        0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0, 1.0-1.2, 1.2-1.4
      BPZ: selects which photo-z result to use
        True: use BPZ result
        False: use TPZ result
    Returns:
      z, dNdz(z)
  """
  myDir = 'dNdz/'
  if binNum == 1:
    if BPZ:
      filename = 'BPZ 1 - two to four.csv'
    else:
      filename = 'TPZ 1 - two to four.csv'
  elif binNum == 2:
    if BPZ:
      filename = 'BPZ 2 - four to six.csv'
    else:
      filename = 'TPZ 2 - four to six.csv'
  elif binNum == 3:
    if BPZ:
      filename = 'BPZ 3 - six to eight.csv'
    else:
      filename = 'TPZ 3 - six to eight.csv'
  elif binNum == 4:
    if BPZ:
      filename = 'BPZ 4 - eight to ten.csv'
    else:
      filename = 'TPZ 4 - eight to ten.csv'
  elif binNum == 5:
    if BPZ:
      filename = 'BPZ 5 - ten to twelve.csv'
    else:
      filename = 'TPZ 5 - ten to twelve.csv'
  else:
    print 'wtf. c\'mon.'
    return 0,0
  myFileName = myDir+filename
  #print myFileName

  z, dNdz = np.loadtxt(myFileName,unpack=True,delimiter=', ')
  return z, dNdz


def extendZrange(zmin,zmax,nBins,binSmooth):
  """
    Function to extend range of zs, nBins
    Inputs:
      zmin,zmax:
      nBins:
      binSmooth:
    Returns:
      extraZ, extraBins
  """
  if binSmooth !=0:
    extraZ = 4*binSmooth
    binSize = (zmax-zmin)/(nBins*1.0)
    extraBins = np.int_(np.ceil(extraZ/binSize))
    extraZ = extraBins*binSize
  else:
    extraBins = 0
    extraZ = 0
  return extraZ, extraBins


def getDNDZinterp(binNum=1,BPZ=True,zmin=0.0,zmax=4.0,nZvals=100,dndzMode=2,
                  z0=1.5,nBins=10,binSmooth=0):
  """
    Purpose:
      get interpolator for dNdz(z) data points
    Inputs:
      binNum: integer in {1,2,3,4,5}. (in dndzMode 1)
        each corresponds to a redshift bin of width Delta.z=0.2:
        0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0, 1.0-1.2, 1.2-1.4;
        except for binNum=0, which indicates a sum of all other bins
        In dndzMode 2: any integer in {1,2,...,nBins}
      BPZ: selects which photo-z result to use in dndzMode 1
        True: use BPZ result
        False: use TPZ result
      zmin,zmax: the lowest,highest redshift points in the interpolation
        if dndzMode == 1: zmin will be 0, zmax will be 1.5, to match dndz files,
          and these points will be added to zs with dndz=0
      nZvals: number of z values to use for interpolating sum of all curves 
        when binNum=0 (for sum of all) is selected (dndzMode 1)
        and number of z values per bin when evaluating model DNDZ (dndzMode 2)
      dndzMode: indicate which method to use to select dNdz curves
        1: use observational DES-SV data from Crocce et al
        2: use LSST model distribution, divided into rectagular bins (Default)
      Parameters used only in dndzMode = 2:
        z0: controls width of distribution
        zmax: maximum redshift in range of bins
        nBins: number of bins to divide range into
        binSmooth: controls smoothing of tophat-stamped bins (unless binNum == 0)
          Default: 0 (no smoothing)
    Returns:
      interpolation function dNdz(z)
      should cover slightly larger region than bin of interest for full interpolation
  """
  if dndzMode == 1:
    zs = np.linspace(zmin,zmax,nZvals)
    zmin = 0.0
    zmax = 1.5 # match dndz files
    if binNum==0:
      myDNDZvals = np.zeros(nZvals)
      for bN in range(1,6): #can not include 0 in this range
        myFunc = getDNDZinterp(binNum=bN,BPZ=BPZ,zmin=zmin,zmax=zmax,dndzMode=1)
        myDNDZvals += myFunc(zs)
      myDNDZ = myDNDZvals
    else:
      zs,myDNDZ = getDNDZ(binNum=binNum,BPZ=BPZ)
      zs     = np.concatenate([[zmin],zs,[zmax]])
      myDNDZ = np.concatenate([[0.],myDNDZ,[0.] ])
  elif dndzMode == 2:
    # do nZvals*nBins+1 to get bin ends in set
    zs = np.linspace(zmin,zmax,nZvals*nBins+1)
    #zs = np.linspace(zmin,zmax,nZvals)
    myDNDZ = modelDNDZ(zs,z0)
    myDNDZ = tophat(myDNDZ,zs,zmin,zmax,nBins,binNum)

    #binEdges =np.linspace(zmin,zmax,nBins+1)
    #print 'edges: ',binEdges[binNum-1],binEdges[binNum]
    #print 'dndz nozero at z = : ',zs[np.where(myDNDZ != 0)]
  else: 
    print 'covfefe!'
    return 0

  # do smoothing
  if binSmooth != 0 and dndzMode == 2:
    myDNDZ = gSmooth(zs,myDNDZ,binSmooth)

  return interp1d(zs,myDNDZ,assume_sorted=True,kind='slinear')#'quadratic')



def getDNDZratio(binNum=1,BPZ=True,zmin=0.0,zmax=1.5,nZvals=100):
  """
    Purpose:
      get interpolator for ratio of (dNdz)_i/(dNdz)_tot data points
    Inputs:
      binNum: integer in {1,2,3,4,5}.
        each corresponds to a redshift bin of width Delta.z=0.2:
        0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0, 1.0-1.2, 1.2-1.4;
      BPZ: selects which photo-z result to use
        True: use BPZ result
        False: use TPZ result
      zmin,zmax: the lowest,highest redshift points in the interpolation
      nZvals: number of z values to use for interpolating sum of all curves 
        when binNum=0 (for sum of all) is selected
    Returns:
      interpolation function dNdz(z)
  """
  numeratorInterp = getDNDZinterp(binNum=binNum,BPZ=BPZ,zmin=zmin,zmax=zmax,
                                  nZvals=nZvals,dndzMode=1)
  denominatorInterp = getDNDZinterp(binNum=0,BPZ=BPZ,zmin=zmin,zmax=zmax,
                                    nZvals=nZvals,dndzMode=1)

  myZvals = np.linspace(zmin,zmax,nZvals)
  numVals = numeratorInterp(myZvals)
  denVals = denominatorInterp(myZvals)

  # what if denom goes to zero?  does it ever?  prob. only at endpoints zmin, zmax
  #print 'denVals == 0 at: ',np.where(denVals == 0)
  # kludge endpoints
  denVals[0] = denVals[1]
  denVals[-1] = denVals[-2]
  ratioVals = numVals/denVals

  return interp1d(myZvals,ratioVals,assume_sorted=True,kind='slinear')#'quadratic')


def modelDNDZ(z,z0):
  """
    Purpose:
      Implement an analytic function that can be used to approximate galaxy 
        distributions of future galaxy surveys
    Inputs:
      z:  the redshift(s) at which dN/dz is evaluated
      z0: the control of the width and extent of the distribution
        note: LSST science book uses z0 = 0.3
    Returns:
      magnitude of dN/dz at redshift z
  """
  # cast z0 as float
  z0 *= 1.0
  zRatio = z/z0
  return 1/(2*z0) * (zRatio)**2 * np.exp(-1*zRatio)


def tophat(FofZ,zs,zmin,zmax,nBins,binNum,includeEdges=False):
  """
  Purpose:
    multiply function by tophat to get a slice of the function
  Inputs:
    FofZ: array of a function evaluated at a set of redshifts
    zs: redshift(s) at which FofZ was evaluated
    zmin,zmax: minimum,maximum z to use when slicing FofZ into sections
    nBins: number of evenly spaced sections to divide (0,zmax) into
    binNum: which bin to return.  bin starting at z=zmin is number 1,
      bin ending at z=zmax is bin number nBins
      Must have 0 < binNum <= nBins, 
    includeEdges: set to True to add one more myPk.zs point outside of bin range
      on each end of bin in set of nonzero values returned; 
      useful for interpolation over entire bin if bin edges are not included in zs
      Default: False
  Returns:
    Tophat slice of FofZ function

  """
  myFofZ = FofZ
  binEdges = np.linspace(zmin,zmax,nBins+1)
  if binNum != 0:
    #myDNDZ[np.where(z< binEdges[binNum-1])] = 0
    #myDNDZ[np.where(z>=binEdges[binNum  ])] = 0
    indicesBelowBin = np.where(zs< binEdges[binNum-1])
    indicesAboveBin = np.where(zs> binEdges[binNum  ])
    #print '\n before attempting overlap:'
    #print 'bin edges: ',binEdges
    #print 'below bin: ',zs[indicesBelowBin]
    #print 'above bin: ',zs[indicesAboveBin]

    if includeEdges: # this is here to allow interp over entire bin
      indicesBelowBin = indicesBelowBin[0][:-1]
      indicesAboveBin = indicesAboveBin[0][1:]
    myFofZ[indicesBelowBin] = 0
    myFofZ[indicesAboveBin] = 0

    #print '\n after attempting overlap:'
    #print 'bin edges: ',binEdges
    #print 'below bin: ',zs[indicesBelowBin]
    #print 'above bin: ',zs[indicesAboveBin]

  return myFofZ


def normBin(FofZ,binZmin,binZmax,zs,normPoints,verbose=False):
  """
    Purpose:
      find normalization factor for a function with a redshift bin
    Note:
      Left bin edges are used for bin height when normalizing
    Inputs:
      FofZ: a function of redshift, z, to be normalized
      binZmin, binZmax: redshift min, max for normalization range
      zs: the points that redshift should eventually be evaluated at
      normPoints: number of points between each point in zs to be used in calculation
      verbose=False: set to true to have more output
    Returns:
      normalization factor

  """
  # select redshift points to work with
  myZs = zs[np.where(np.logical_and( zs>binZmin, zs<binZmax ))]
  myZs = np.append([binZmin],myZs)
  myZs = np.append(myZs,[binZmax])

  if verbose: print '\nmyZs: ',myZs

  # add in normPoints number of points in between each point in myZs
  if normPoints == 0:
    manyZs = myZs
  else:
    stepSize = 1./(normPoints+1) # +1 for endpoint
    xp = np.arange(myZs.size) # x coords for interp
    if verbose: print myZs.size,stepSize,myZs.size-stepSize
    targets = np.arange(0,myZs.size-1+stepSize,stepSize) # y coords for interp
    if verbose: print 'targets: ',targets
    manyZs = np.interp(targets,xp,myZs)

  if verbose: print 'manyZs: ',manyZs

  # get interval widths: spacing between 
  deltaZs = manyZs-np.roll(manyZs,1)
  
  
  # ditch endpoint residuals
  if binZmin in zs:
    deltaZs = deltaZs[1:]
  else:
    deltaZs = deltaZs[2:]
  if binZmax not in zs:
    deltaZs = deltaZs[:-1]

  if verbose: print 'deltaZs: ',deltaZs

  # evaluate FofZ
  myF = FofZ(manyZs)
  
  # interpret redshifts as left bin edges; ditch final F(z)
  myF = myF[:-1]
  if binZmin not in zs:
    myF = myF[1:]
  if binZmax not in zs:
    myF = myF[:-1]



  # interpret redshifts as bin centers; ditch first and last z
  #myF = myF[1:-1]

  if verbose: print 'myF: ',myF

  # area under curve
  area = np.dot(myF,deltaZs)

  return 1./area


def getNormalizedDNDZbin(binNum,zs,z0,zmax,nBins,BPZ=True,dndzMode=2,
                      zmin=0.0,normPoints=1000,binSmooth=0,verbose=False):
  """
    Purpose:
      return normalized dndz array
    Note:
      Left bin edges are used for bin height when normalizing
    Inputs:
      binNum: index indicating which redshift distribution to use
        {1,2,...,nBins}
      zs: the redshifts to calculate DNDZ at
        redshift values outside range (zmin,zmax) will return zero
      dndzMode: indicate which method to use to select dNdz curves
        1: use observational DES-SV data from Crocce et al
        2: use LSST model distribution, divided into rectagular bins (Default)
      Parameters used only in dndzMode = 2:
        z0: controls width of distribution
        zmax: maximum redshift in range of bins
        nBins: number of bins to divide range into
        binSmooth: controls smoothing of tophat-stamped bins (unless binNum == 0)
          Default: 0 (no smoothing)
      BPZ=True: controls which dndz curves to use in dndzMode = 1
      zmin=0.0: minimum redshift in range of bins
      normPoints=100: number of points per zs interval to use when normalizing
      verbose: control ammount of output from normBin
    Returns:
      Normalized array of dNdz values within the bin, corresponding to redshifts zs
      
  """
  # check binNum
  if binNum < 1 or binNum > nBins:
    print 'die screaming'
    return 0

  # get dNdz
  if dndzMode == 1:
    zmax = 1.5 # hard coded to match domain in dndz files
    rawDNDZ = getDNDZinterp(binNum=binNum,BPZ=BPZ,zmin=zmin,zmax=zmax,dndzMode=1)
    binZmin = zmin
    binZmax = zmax
  elif dndzMode == 2:
    # extend Z range for smoothing
    extraZ,extraBins = extendZrange(zmin,zmax,nBins,binSmooth)
    zmax += extraZ
    nBins += extraBins

    rawDNDZ = getDNDZinterp(binNum=binNum,BPZ=BPZ,zmin=zmin,zmax=zmax,dndzMode=2,
                            binSmooth=binSmooth,z0=z0,nBins=nBins)

    # divide redshift range into bins and select desired bin
    binEdges = np.linspace(zmin,zmax,nBins+1)
    binZmin = binEdges[binNum-1] # first bin starting at zmin has binNum=1
    binZmax = binEdges[binNum]

    if binSmooth != 0:
      # adjust binZmin, binZmax to appropriate cutoff for smoothed bin
      numSigmas = 4 # should match what is in gSmooth function
      binZmin -= numSigmas*binSmooth
      binZmax += numSigmas*binSmooth
      if binZmin < 0:
        binZmin = 0

  else: # really, it should be dndzMode = 1 or 2
    print 'covfefe!'
    return 0

  # select bin indices
  binIndices = np.where(np.logical_and( zs>=binZmin, zs<=binZmax ))

  # get normalization factor
  normFac = normBin(rawDNDZ,binZmin,binZmax,zs[binIndices],normPoints,verbose=verbose)

  # get non-normalized DNDZ
  binDNDZ = rawDNDZ(zs[binIndices])
  myDNDZ = np.zeros(zs.size)
  myDNDZ[binIndices] = binDNDZ
  
  if verbose:
    myZs = zs[np.where(np.logical_and( zs>=binZmin, zs<binZmax ))]
    print 'zs in getNormalizedDNDZbin: ',myZs

  # normalize
  normDNDZ = normFac*myDNDZ

  return normDNDZ



def gBias():
  """
    Purpose:
      function that returns data points with errors
    Data:
      galaxy biases from Giannantonio et al 2016, table 2, gal-gal, real space
    Returns:
      three triples: redshifts,biases,bias errors
  """
  zs = (0.3, 0.5, 0.7, 0.9, 1.1)  #redshifts (just bin centers; could be improved)
  bs = (1.03,1.28,1.32,1.57,1.95) #galaxy biases
  es = (0.06,0.04,0.03,0.03,0.04) #bias errors

  return zs,bs,es


def getBiasFit(deg=1):
  """
    Purpose:
      polynomial fit to bias data
    Inputs:
      deg: degree of polynomial to fit to bias data
        This really should be 1. 
    Returns:
      function to evaluate bias b(z)
  """
  zs,bs,es = gBias()
  coefs = polyfit(zs,bs,deg=deg,w=1/np.array(es))

  return poly1d(coefs)


def byeBiasFit():
  """
    Purpose:
      polynomial fit to bias from simulations
    Note:
      Byeonghee Yu appears to have taken this formula from digitizing a plot
        in Weinberg ea 2004 (fig. 9?)
    Returns:
      a function that can evaluate bias at redshift z, or redshifts z if an array is passed
  """
  return lambda z: 1+0.84*z


def gaussian(x,mu,sig):
  """
    Purpose:
      evaluate Gaussian function at input points
    Inputs:
      x: numpy array of points to evaluate gaussian at
      mu: the mean
      sig: the standard deviation
    Returns:
      the value of the Gaussian at each point in x
  """
  return 1./(np.sqrt(2.*np.pi)*sig) * np.exp(-np.power((x-mu)/sig, 2)/2.)


def gSmooth(zs,FofZ,binSmooth,numSigmas=3):
  """
    Purpose:
      smooth a tophat-stamped bin by convolution with a Gaussian
    Inputs:
      zs: the redshift points
      FofZ: a function evaluated at these points
      binSmooth: the standard deviation of the smoothing Gaussian
      numSigmas: how many std devs from edge of old bin new bin goes
        Default: 3
    Returns:
      A smoothed FofZ array
  """
  # exted array below zero to reflect power back into positive
  reflectIndices = np.where(zs < binSmooth*(numSigmas+1))[0]
  reflectZs = np.flip(zs[reflectIndices],0)*-1
  reflectFs = np.zeros(np.size(reflectIndices))
  szzs = np.hstack((reflectZs,zs))
  ZfoFFofZ = np.hstack((reflectFs,FofZ))
  #szzs = zs
  #ZfoFFofZ = FofZ

  # find deltaZs
  # for gaussian function, use redshift points as pixel centers
  deltaZs = (np.roll(szzs,-1)-np.roll(szzs,1))/2

  # find points within bin
  # this is where the tophatness of the bin is used
  binIndices = np.where(ZfoFFofZ != 0)[0]

  # new array to put results into
  smoothed = np.zeros(ZfoFFofZ.size)

  for binI in binIndices:
    lowZ  = szzs[binI]-numSigmas*binSmooth
    midZ  = szzs[binI]
    highZ = szzs[binI]+numSigmas*binSmooth
    myIndices = np.where(np.logical_and(szzs>=lowZ,szzs<=highZ))[0] 
    myZs = szzs[myIndices]
    myGs = gaussian(myZs,midZ,binSmooth)
    smoothed[myIndices] += myGs*ZfoFFofZ[binI]*deltaZs[binI]
    
  # do the reflection
  refl = np.flip(smoothed[reflectIndices],0)
  smoothed = smoothed[reflectIndices.size:]
  smoothed[reflectIndices] += refl

  return smoothed


def winGalaxies(myPk,biases=None,BPZ=True,dndzMode=2,
                binNum=0,zmin=0.0,zmax=4.0,nBins=10,z0=0.3,
                doNorm=True,useWk=False,binSmooth=0):
  """
    window function for galaxy distribution
    Inputs:
      myPk: MatterPower object
      biases: array of galaxy biases * matter biases (b*A)
        corresponding to zs values
        default: all equal to 1
      BPZ: flag to indicate BPZ or TPZ when dndzMode=1; True for BPZ
      binNum: index indicating which redshift distribution to use
        {1,2,...,nBins}
        if binNum == 0, sum of all bins will be used, 
          binSmooth will be set to 0, and doNorm to False
        Default: 0
      dndzMode: indicate which method to use to select dNdz curves
        1: use observational DES-SV data from Crocce et al
        2: use LSST model distribution, divided into bins (Default)
      doNorm: set to True to normalize dN/dz (unless binNum == 0)
        Default: True
      Parameters only used in dndzMode = 2:
        nBins: number of bins to create
          Default: 10
        z0: controls width of full dNdz distribution
          Default: 0.3 (LSST-like)
        zmin,zmax: the min,max redshift to use
          Defaults: 0,4
        useWk: set to True to use W^kappa as dN/dz
          Defalut: False
        binSmooth: controls smoothing of tophat-stamped bins (unless binNum == 0)
          Default: 0 (no smoothing)
    Returns:
      array of W^galaxies values evaluated at input myPk.chis values
  """
  # get redshift array, etc.
  PK,chistar,chis,dchis,zs,dzs,pars = myPk.getPKinterp()

  if biases is None:
    biases = np.ones(zs.size)
  if dndzMode != 1 and dndzMode != 2:
    print 'wrong dNdz mode selected.'
    return 0
  if binNum == 0:
    binSmooth = 0
    doNorm = False
    print 'winGalaxies: binNum set to 0, doNorm set to False.'

  # get dz/dchi as ratio of deltaz/deltachi
  #dzdchi = dzs/dchis
  dzdchi = myPk.Hs

  # extend Z range for smoothing
  extraZ,extraBins = extendZrange(zmin,zmax,nBins,binSmooth)
  zmax += extraZ
  nBins += extraBins

  # get dNdz according to dndzMode,useWk,doNorm,binSmooth settings
  if useWk and dndzMode == 2:
    # do not use biases here since they are multiplied in later
    if doNorm:
      myDNDZ = getNormalizedWinKbin(myPk,binNum,zs,zmin=zmin,zmax=zmax,BPZ=BPZ,
                                    nBins=nBins,binSmooth=binSmooth,dndzMode=dndzMode)
    else:
      rawWinK = getWinKinterp(myPk,binNum=binNum,zmin=zmin,zmax=zmax,nBins=nBins,
                        dndzMode=dndzMode,binSmooth=binSmooth)
      myDNDZ = rawWinK(zs)
  else: # not W_kappa or not dndzMode 2
    if doNorm:
      # this function will do its own dndzMode selection
      myDNDZ = getNormalizedDNDZbin(binNum,zs,z0,zmax,nBins,dndzMode=dndzMode,
                                    BPZ=BPZ,zmin=zmin,binSmooth=binSmooth)
    else: # do not do normalization
      if dndzMode == 1:
        zmin = 0.0
        zmax = 1.5 # hard coded to match domain in dndz files
      rawDNDZ = getDNDZinterp(binNum=binNum,BPZ=BPZ,zmin=zmin,zmax=zmax,z0=z0,
                              dndzMode=dndzMode,binSmooth=binSmooth,nBins=nBins)
      myDNDZ = rawDNDZ(zs) 


  return dzdchi*myDNDZ*biases


def winKappa(myPk,biases=None):
  """
    window function for CMB lensing convergence
    Inputs:
      myPk: MatterPower object
      biases: amplitude of lensing, array corresponding to zs values
        default: all equal to 1
    Returns:
      array of W^kappa values evaluated at input myPk.chis
      should cover slightly larger region than bin of interest for full interpolation
  """
  # get redshift array, etc.
  PK,chistar,chis,dchis,zs,dzs,pars = myPk.getPKinterp()

  if biases is None:
    biases = np.ones(zs.size)

  # Get lensing window function (flat universe)
  lightspeed = 2.99792458e5 # km/s
  myH0 = pars.H0/lightspeed # get H0 in Mpc^-1 units
  #myOmegaM = pars.omegab+pars.omegac #baryonic+cdm
  myOmegaM = pars.omegab+pars.omegac+pars.omegan #baryonic+cdm+neutrino
  myAs = 1/(1.+zs) #should have same indices as chis
  winK = chis*((chistar-chis)/(chistar*myAs))*biases
  winK *= (1.5*myOmegaM*myH0**2)

  return winK


def getWinKinterp(myPk,biases=None,binNum=0,zmin=0,zmax=4,nBins=10,BPZ=True,
                  dndzMode=2,binSmooth=0):
  """
    Purpose:
      get interpolation function for winKappa(z)
    Inputs:
      myPk: MatterPower object
      biases: amplitude of lensing, array corresponding to zs values
        default: all equal to 1
      binNum=0: index indicating which redshift distribution to use
        For individual bin use {1,2,...,nBins}
        Default: 0, for sum of all bins, including outside (zmin, zmax)
      dndzMode: indicate which method is used for corresponding dNdz curves
        1: use observational DES-SV data from Crocce et al
        2: use LSST model distribution, divided into bins (Default)
      BPZ: set to True to use BPZ curves for dNdz, False for TPZ
        Default: True
      Parameters only used in dndzMode = 2:
        nBins: number of bins to create
          Default: 10
        zmin,zmax: the min,max redshift to use
          Defaults: 0,4
      binSmooth: controls smoothing of tophat-stamped bins (unless binNum == 0)
        Default: 0 (no smoothing)
    Returns:
      function for interpolating winKappa(z) result
  """
  # get CMB lensing window
  winK = winKappa(myPk,biases=biases)
  zs = myPk.zs

  # insert (0,0) at beginning
  zs = np.insert(zs,0,0)
  winK = np.insert(winK,0,0)

  # slice out the bin of interest
  if dndzMode == 1: # use observational dndz
    nBins = 5
    mode1zmin = 0
    mode1zmax = 1.5 # match the dndz files
    if binNum != 0:
      binIndices = np.where(np.logical_and( zs>=mode1zmin, zs<=mode1zmax ))
      notBinIndices = np.where(np.logical_or( zs<mode1zmin, zs>mode1zmax ))
      if binNum > nBins: # just return everything over the defined bins
        winK[binIndices] = 0
      else: # one of the bins has been indicated
        binZs = zs[binIndices]
        dndzRatioInterp = getDNDZratio(binNum=binNum,BPZ=BPZ,
                                       zmin=mode1zmin,zmax=mode1zmax)
        dndzRatios = dndzRatioInterp(binZs)
        # use ratios to define winKappa bin
        winK[binIndices] *= dndzRatios
        winK[notBinIndices] = 0
  elif dndzMode == 2: # use dndz model
    if binNum != 0:
      # add bin edges into array
      binEdges =np.linspace(zmin,zmax,nBins+1)
      lowEdgeZ  = binEdges[binNum-1]
      highEdgeZ = binEdges[binNum]
      winKinterp = interp1d(zs,winK,assume_sorted=True,kind='slinear')
      lowEdgeWinK = winKinterp(lowEdgeZ)
      highEdgeWinK = winKinterp(highEdgeZ)
      if zs[0] >= lowEdgeZ:
        indicesBelowBin = [[0,-1],[0,0]]
      else:
        indicesBelowBin = np.where(zs< lowEdgeZ)
      if zs[-1] <= highEdgeZ:
        indicesAboveBin = [[-1,0],[0,0]]
      else:
        indicesAboveBin = np.where(zs> highEdgeZ)
      zs   = np.insert(zs,  indicesAboveBin[0][0],highEdgeZ)
      winK = np.insert(winK,indicesAboveBin[0][0],highEdgeWinK)
      zs   = np.insert(zs,  indicesBelowBin[0][-1]+1,lowEdgeZ)
      winK = np.insert(winK,indicesBelowBin[0][-1]+1,lowEdgeWinK)

    # create tophat bin
    winK = tophat(winK,zs,zmin,zmax,nBins,binNum)

  else: # really, dndzMode has to be 1 or 2.
    print 'covfefe!'
    return 0

  # do smoothing
  if binSmooth != 0 and dndzMode == 2:
    winK = gSmooth(zs,winK,binSmooth)

  return interp1d(zs,winK,assume_sorted=True,kind='slinear')



def winKappaBin(myPk,biases=None,binNum=0,zmin=0,zmax=4,nBins=10,BPZ=True,
                dndzMode=2,binSmooth=0,**kwargs):
  """
    Purpose:
      get one bin from CMB lensing kernel
      (wrapper around getWinKinterp and winKappa)
      (had to do this to get smoothing in getWinKinterp)
    Inputs:
      myPk: MatterPower object
      biases: amplitude of lensing, array corresponding to zs values
        default: all equal to 1
      binNum=0: index indicating which redshift distribution to use
        For individual bin use {1,2,...,nBins}
        Default: 0, for sum of all bins, including outside (zmin, zmax)
      dndzMode: indicate which method is used for corresponding dNdz curves
        1: use observational DES-SV data from Crocce et al
        2: use LSST model distribution, divided into bins (Default)
      BPZ: set to True to use BPZ curves for dNdz, False for TPZ
        Default: True
      Parameters only used in dndzMode = 2:
        nBins: number of bins to create
          Default: 10
        zmin,zmax: the min,max redshift to use
          Defaults: 0,4
        binSmooth: controls smoothing of tophat-stamped bins (unless binNum == 0)
          Default: 0 (no smoothing)
      **kwargs: place holder so that winKappaBin and winGalaxies can have
        same parameter list
    Returns:
      array of W^kappa values evaluated at input myPk.chis
      should cover slightly larger region than bin of interest for full interpolation
  """
  # extend Z range for smoothing
  extraZ,extraBins = extendZrange(zmin,zmax,nBins,binSmooth)
  zmax += extraZ
  nBins += extraBins

  # get Wk(z)
  rawWinK = getWinKinterp(myPk,biases=biases,binNum=binNum,zmin=zmin,zmax=zmax,
                          nBins=nBins,dndzMode=dndzMode,binSmooth=binSmooth,BPZ=BPZ)
  # evaluate at target redshifts
  myWinK = rawWinK(myPk.zs)

  return myWinK


def getNormalizedWinKbin(myPk,binNum,zs,zmin=0.0,zmax=4.0,nBins=1,dndzMode=2,
                         BPZ=True,biases=None,normPoints=1000,binSmooth=0,
                         verbose=False):
  """
    Purpose:
      return normalized WinK array for one particular bin
    Note:
      Left bin edges are used for bin height when normalizing
    Inputs:
      myPk: a MatterPower object
      binNum: index indicating which redshift distribution to use
        {1,2,...,nBins}
      zs: the redshifts to evaluate WinK at
        must all be zmin <= zs <= zmax
      dndzMode: indicate which method used to select corresponding dNdz curves
        1: use observational DES-SV data from Crocce et al
        2: use LSST model distribution, divided into rectagular bins (Default)
      BPZ: set to True to use BPZ curves for dNdz, False for TPZ
        Default: True
      Parameters only used in dndzMode = 2:
        nBins: number of bins to create
          Default: 10
        zmin,zmax: the min,max redshift to use
          Defaults: 0,4
        binSmooth: controls smoothing of tophat-stamped bins (unless binNum == 0)
          Default: 0 (no smoothing)
      biases=None: lensing amplitudes indexed the same as redshift points
        Default = None, indicating amplitude = 1 everywhere.
      normPoints=100: number of points per zs interval to use when normalizing
      verbose: control ammount of output from normBin
    Returns:
      Normalized array of winK corresponding to redshifts zs
  """
  # check binNum
  if binNum < 1 or binNum > nBins:
    print 'die screaming'
    return 0

  # extend Z range for smoothing
  extraZ,extraBins = extendZrange(zmin,zmax,nBins,binSmooth)
  zmax += extraZ
  nBins += extraBins

  # get Wk(z)
  rawWinK = getWinKinterp(myPk,biases=biases,binNum=binNum,zmin=zmin,zmax=zmax,
                          nBins=nBins,dndzMode=dndzMode,binSmooth=binSmooth,BPZ=BPZ)

  # get binZmin, binZmax for normalization range
  if dndzMode == 1:
    zmin = 0.0
    zmax = 1.5 # hard coded to match dndz files
    binZmin = zmin
    binZmax = zmax
  elif dndzMode == 2:
    # divide redshift range into bins and select desired bin
    binEdges = np.linspace(zmin,zmax,nBins+1)
    binZmin = binEdges[binNum-1] # first bin starting at zmin has binNum=1
    binZmax = binEdges[binNum]

    if binSmooth != 0:
      # adjust binZmin, binZmax to appropriate cutoff for smoothed bin
      numSigmas = 4 # should match what is in gSmooth function
      binZmin -= numSigmas*binSmooth
      binZmax += numSigmas*binSmooth
      if binZmin < 0:
        binZmin = 0

  else: # really, dndzMode should be 1 or 2
    print 'covfefe!'
    return 0

  # select bin indices
  binIndices = np.where(np.logical_and( zs>=binZmin, zs<=binZmax ))

  # get normalization factor
  normFac = normBin(rawWinK,binZmin,binZmax,zs[binIndices],normPoints,verbose=verbose)

  # get non-normalized winK  
  binWinK = rawWinK(zs[binIndices])
  myWinK = np.zeros(zs.size)
  myWinK[binIndices] = binWinK

  if verbose:
    myZs = zs[np.where(np.logical_and( zs>=binZmin, zs<binZmax ))]
    print 'zs in getNormalizedWinKbin: ',myZs

  # normalize
  normWinK = normFac*myWinK

  return normWinK




################################################################################
# the angular power spectrum


def getCl(myPk,biases1=None,biases2=None,winfunc1=winKappaBin,winfunc2=winKappaBin,
          dndzMode=2,binNum1=0,binNum2=0,lmax=2500,zmin=0.0,zmax=4.0,nBins=10,
          z0=0.3,doNorm=True,useWk=False,binSmooth=0,BPZ=True):
  """
    Purpose: get angular power spectrum
    Inputs:
      myPk: a matterPower object
      biases1,biases2: array of matter amplitude (A) or 
        galaxy bias * matter amplitude (bA) at each redshift in myPk.zs, 
        depending on which winfunc is selected
        default: None; indicates that all biases equal to 1
      winfunc1,winfunc2: the window functions
        should be winKappaBin or winGalaxies
      dndzMode: select which dNdz scheme to use for winfuncs
        Default: 2
      binNum1,binNum2: index indicating which bin to use
        If dndzMode = 1:
          integer in {0,1,2,3,4,5}
          curves from fig.3 of Crocce et al 2016.
        if dndzMode = 2:
          integer in {0,1,...,nBins-1,nBins}
        Index=0 indicates sum of all other curves
      lmax: highest ell to return.  (lowest will be 2)
      doNorm: set to True to normalize dN/dz.
        Default: True
      BPZ: set to true to use BPZ dNdz in dndzMode 1, False for TPZ
        Default: True
      Parameters only used in dndzMode = 2: (same for both winfuncs)
        zmin,zmax: lowest,highest z to use creating bins
        nBins: number of bins to use
        z0: width of full galaxy distribution
          Default: 0.3 (from LSST science book)
        useWk: set to True to use W^kappa as dN/dz
          Defalut: False
        binSmooth: parameter that controls the amount of smoothing of bin edges
          Default: 0 (no smoothing)
    Returns: 
      l,  the ell values (same length as Cl array)
      Cl, the power spectrum array
  """

  # confirm inputs
  def wincheck(winfunc,num):
    if winfunc == winKappaBin:
      if num == 1:
        print 'window ',num,': kappa ',binNum1
      else:
        print 'window ',num,': kappa ',binNum2
    elif winfunc == winGalaxies:
      if num == 1:
        print 'window ',num,': galaxies ',binNum1
      else:
        print 'window ',num,': galaxies ',binNum2
    else:
      print 'error with input'
      return 0
    return 1
  
  if wincheck(winfunc1,1)==0: return 0,0
  if wincheck(winfunc2,2)==0: return 0,0
  
  # get matter power spectrum P_k^delta
  PK,chistar,chis,dchis,zs,dzs,pars = myPk.getPKinterp()

  # get window functions
  # note that winKappaBin will ignore extra keywords due to use of **kwargs 
  win1=winfunc1(myPk,biases=biases1,z0=z0,dndzMode=dndzMode,binNum=binNum1,
                zmin=zmin,zmax=zmax,nBins=nBins,doNorm=doNorm,useWk=useWk,
                binSmooth=binSmooth,BPZ=BPZ)
  win2=winfunc2(myPk,biases=biases2,z0=z0,dndzMode=dndzMode,binNum=binNum2,
                zmin=zmin,zmax=zmax,nBins=nBins,doNorm=doNorm,useWk=useWk,
                binSmooth=binSmooth,BPZ=BPZ)

  #Do integral over chi
  ls = np.arange(2,lmax+1, dtype=np.float64)
  cl = np.zeros(ls.shape)
  w = np.ones(chis.shape) #this is just used to set to zero k values out of range of interpolation
  for i, l in enumerate(ls):
      k=(l+0.5)/chis
      w[:]=1
      w[k<1e-4]=0
      w[k>=myPk.kmax]=0
      cl[i] = np.dot(dchis, w*PK.P(zs, k, grid=False)*win1*win2/(chis**2))

  #print 'ls: ',ls,', Cl: ',cl
  return ls, cl







################################################################################
# plotting functions


def plotBias():
  """
    plot bias data and polynomial fits
  """
  zs,bs,es = gBias()
  biasFunc1 = getBiasFit(deg=1)
  biasFunc2 = getBiasFit(deg=2)

  zsToPlot = np.linspace(0,1.5,100)

  plt.plot(zsToPlot,biasFunc1(zsToPlot))
  plt.plot(zsToPlot,biasFunc2(zsToPlot))
  plt.errorbar(zs,bs,yerr=es,fmt='o')

  plt.xlabel('redshift')
  plt.ylabel('bias')
  plt.title('polynomial fits to bias data')
  plt.show()


def replotDigitizedDNDZ(zmin=0.0,zmax=1.5):
  """
    Purpose:
      replot the digitized dNdz curves from Crocce et al 2016, fig. 3
      uses interpolation functions from getDNDZinterp
    Inputs:
      zmin,zmax: the lowest,highest redshift points in interpolator and plot

  """
  # points for plotting
  zs = np.linspace(zmin,zmax,1000) 

  for BPZ in (True,False):
    for binNum in range(1,6):
      myInterp = getDNDZinterp(binNum=binNum,BPZ=BPZ,zmin=zmin,zmax=zmax,dndzMode=1)
      plt.plot(zs,myInterp(zs))
    if BPZ:
      title='BPZ'
    else:
      title='TPZ'
    plt.title(title)
    plt.show()


def plotDNDZsum(zmin=0.0,zmax=1.5):
  """
  plot the sum of em...
  """
  # points for plotting
  zs = np.linspace(zmin,zmax,1000) 

  for BPZ in (True,False):
    sumInterp = getDNDZinterp(binNum=0,BPZ=BPZ,dndzMode=1) # 0 for sum
    plt.plot(zs,sumInterp(zs))
    if BPZ:
      title='BPZ'
    else:
      title='TPZ'
    plt.title(title)
    plt.show()


def plotDNDZratio(zmin=0.0,zmax=1.5):
  """
  plot the ratio of bins to sum of em...
  """
  # points for plotting
  zs = np.linspace(zmin,zmax,1000) 

  for BPZ in (True,False):
    for binNum in range(1,6):
      ratioInterp = getDNDZratio(binNum=binNum,BPZ=BPZ)
      plt.plot(zs,ratioInterp(zs))
    if BPZ:
      title='BPZ'
    else:
      title='TPZ'
    plt.title(title)
    plt.show()


def plotRedshiftAppx(myPk,dndzMode=1,BPZ=True,nBins=5):
  """
    for plotting the equivalent redshift bins in the approximation to the integral
      over distance from here to last scattering, compared to g,k kernels
    Inputs:
      myPk: matterpower object; contains redshift points
      dndzMode: which type of dndz to plot
      BPZ: true for BPZ, false for TPZ
      nBins: number of bins to use
        use 5 for dndzMode = 1 (default)

  """
  # get matter power spectrum P_k^delta
  #PK,chistar,chis,dchis,zs,dzs,pars = myPk.getPKinterp()
  zs = myPk.zs

  # get and plot window functions
  winK  = winKappa(myPk)
  plt.plot(zs,winK)
  for binNum in range(1,nBins+1):
    winG = winGalaxies(myPk,BPZ=BPZ,binNum=binNum,dndzMode=dndzMode)
    plt.plot(zs,winG)


  # loop over zs
  for zbin in zs:
    plt.axvline(zbin)
  
  plt.show()


def plotCl(ls,Cl):
  """
    plot the angular power spectrum
    Note:
      Convergence power spectra C_l^kappakappa are plotted directly.
      Convention for plotting lensing power spectra C_l^phiphi is to multiply
        C_l by [l(l+1)]^2/2pi, but that is not done for kappa.  (why not?)
        (Maybe because kappa already has l(l+1) factor in it?)
    Inputs:
      ls: the ell values
      Cl: the power spectrum
  """ 

  plt.semilogy(ls,Cl , color='b')
  #plt.loglog(ls,Cl , color='b')
  #plt.loglog(ls,Cl2, color='r')
  plt.xlim([1,2000])
  #plt.legend(['potential Pk','matter Pk'])
  #plt.ylabel(r'$[\ell(\ell+1)]^2C_\ell^{\phi}/2\pi$')
  plt.ylabel(r'$C_\ell^{\kappa\kappa}$') #needs bigger font!
  plt.xlabel(r'$\ell$')
  plt.title('CMB lensing convergence power spectrum')
  plt.show()


def plotKG(myPk,biasesK=None,biasesG=None,lmax=2500,
           dndzMode=1,binNum=0,zmax=4,nBins=10,z0=0.3,
           doNorm=True,useWk=False,binSmooth=0):
  """
  Purpose:
    to plot each C_l for kappa, galaxy combinations
    Uses just one dNdz for all C_l
  Inputs:
    myPk: a matterPower object
    biasesK: array of lensing amplitudes
      must be same length as myPk.zs
    biasesG, array of galaxy biases
      must be same length as myPk.zs
    lmax: highest ell to plot
    dndzMode: which mode to use for creating dNdz functions
      1: uses DES-SV bins from Crocce et al
      2: uses LSST model dNdz from LSST science book
      Default: 1
    binNum: 
      index defining which bin to use
      if dndzMode is 1: binNum in {1,2,3,4,5}
      if dndzMode is 2: binNum in {1,2,...,nBins-1,nBins}
      Default: 0 for sum of all bins
    doNorm:
    Parameters only used in dndzMode =2:
      zmax: highest redshift to include in bins
      nBins: number of bins to divide 0<z<zmax into
      z0: controls the width of the total galaxy distribution
      useWk:
      binSmooth:

  """
  ls1, Cl1 = getCl(myPk,biases1=biasesK,biases2=biasesK,winfunc1=winKappaBin,   winfunc2=winKappaBin,
                   doNorm=doNorm,useWk=useWk,binSmooth=binSmooth)
  ls2, Cl2 = getCl(myPk,biases1=biasesK,biases2=biasesG,winfunc1=winKappaBin,   winfunc2=winGalaxies, 
                   dndzMode=dndzMode,               binNum2=binNum,
                   zmax=zmax,nBins=nBins,z0=z0,
                   doNorm=doNorm,useWk=useWk,binSmooth=binSmooth)
  #ls3, Cl3 = getCl(myPk,biases1=biasesG,biases2=biasesK,winfunc1=winGalaxies,winfunc2=winKappaBin,   
  #                 dndzMode=dndzMode,binNum1=binNnum,
  #                 zmax=zmax,nBins=nBins,z0=z0,
  #                 doNorm=doNorm,useWk=useWk,binSmooth=binSmooth)
  ls4, Cl4 = getCl(myPk,biases1=biasesG,biases2=biasesG,winfunc1=winGalaxies,winfunc2=winGalaxies,
                   dndzMode=dndzMode,binNum1=binNum,binNum2=binNum,
                   zmax=zmax,nBins=nBins,z0=z0,
                   doNorm=doNorm,useWk=useWk,binSmooth=binSmooth)  

  p1=plt.semilogy(ls1,Cl1,label='$\kappa\kappa$')
  p2=plt.semilogy(ls2,Cl2,label='$\kappa g$')
  #p3=plt.semilogy(ls3,Cl3,label='$g \kappa$')
  p4=plt.semilogy(ls4,Cl4,label='$gg$')
  plt.xlim(0,lmax)
  plt.xlabel(r'$\ell$')
  plt.ylabel(r'$C_{\ell}$')
  plt.title('CMB convergence and DES-SV expected power spectra')
  plt.legend()
  plt.show()


def plotGG(myPk,biases1=None,biases2=None,lmax=2000,
           doNorm=True,useWk=False,binSmooth=0):
  """
  Purpose:
    plot all Cl^{g_i g_j} for i,j in {1,2,3,4,5}
  Inputs:
    myPk: a matterPower object
    biases1,biases2: array of galaxy bias or lensing amplitudes
      must be same length as myPk.zs
      same pair of biases used for each x-cor (this should be improved)
    lmax: highest l to plot
    doNorm:
    useWk:
    binSmooth:
  """
  for i in range(1,6):
    for j in range(i,6):
      print 'starting g_',i,' g_',j
      ls,Cl = getCl(myPk,biases1=biases1,biases2=biases2,winfunc1=winGalaxies,winfunc2=winGalaxies,binNum1=i,binNum2=j,doNorm=doNorm,useWk=useWk,binSmooth=binSmooth)
      plt.semilogy(ls,Cl,label='g_'+str(i)+', g_'+str(j))
  plt.xlim(0,lmax)
  plt.xlabel(r'$\ell$')
  plt.ylabel(r'$C_{\ell}$')
  plt.title('galaxy angular power spectra with DES-SV kernels')
  plt.legend()
  plt.show()


def plotGGsum(myPk,biases=None,doNorm=True,useWk=False,binSmooth=0):
  """
  Note: this is basically included in plotKG
  Note: does not have option for getCl to use anything but dndzMode=1
  plot Cl^gg from sum of al dNdz
  Inputs:
    myPk: a matterPower object
    biases: array of galaxy biases
      must be same length as myPk.zs
    doNorm:
    useWk:
    binSmooth:

  """
  ls,Cl = getCl(myPk,biases1=biases,biases2=biases,winfunc1=winGalaxies,winfunc2=winGalaxies,binNum1=0,binNum2=0,doNorm=doNorm,useWk=useWk,binSmooth=binSmooth)
  plt.semilogy(ls,Cl)
  plt.title('C_l^gg for sum of all dNdz curves')
  plt.show()
  

def plotModelDNDZ(z0=0.3,zmax=4):
  """
  plot the DNDZ model
  Inputs:
    z0: the width control
      Default: 0.3 (used by LSST sci.book)
    zmax: maximum z to plot

  """
  zs = np.linspace(0,zmax,100)
  plt.plot(zs,modelDNDZ(zs,z0))
  plt.show()


def plotModelDNDZbins(z0=0.3,zmin=0,zmax=4,nBins=10,doNorm=False,normPoints=100,
                      useWk=False,binSmooth=0,BPZ=True,nZvals=100):
  """
  plot model DNDZ cut up into bins
  Inputs:
    z0: controls width of total dist.
    zmin,zmax: min,max z for dividing up bins
    nBins: number of bins to divide up
    doNorm = False: set to True to normalize bins
    normPoints: number of points per zs interval to use when normalizing
    binSmooth: controls smoothing of tophat-stamped bins
      Default: 0
    BPZ: 
    nZvals:
  """
  # extend Z range for smoothing
  extraZ,extraBins = extendZrange(zmin,zmax,nBins,binSmooth)
  zmax += extraZ
  nBins += extraBins

  zs = np.linspace(zmin,zmax,normPoints*nBins+1)
  for bNum in range(1,nBins+1):
    if doNorm:
      dndzBin = getNormalizedDNDZbin(bNum,zs,z0,zmax,nBins,normPoints=normPoints,
                                     binSmooth=binSmooth,dndzMode=2) #2 for model
    else:
      rawDNDZ = getDNDZinterp(binNum=bNum,BPZ=BPZ,zmin=zmin,zmax=zmax,dndzMode=2,
                              binSmooth=binSmooth,z0=z0,nBins=nBins,nZvals=nZvals)
      dndzBin = rawDNDZ(zs)
    plt.plot(zs,dndzBin)
  plt.show()


def plotWinKbins(myPk,zmin=0.0,zmax=4.0,nBins=10,doNorm=False,normPoints=100,
                 binSmooth=0,dndzMode=2,verbose=False,BPZ=True):
  """
  plot winKappa cut up into normalized bins
  Inputs:
    myPk: a MatterPower object
    zmin,zmax: min,max z for dividing up bins
    nBins: number of bins to divide up
    doNorm=False: set to True to normalize bins
    normPoints: number of points per zs interval to use when normalizing
    binSmooth: controls smoothing of tophat-stamped bins (unless dndzMode == 1)
      Default: 0
    BPZ: set to True to use BPZ curves for dNdz, False for TPZ
      Default: True
    dndzMode: indicate which method to use to select corresponding dNdz curves
      1: use observational DES-SV data from Crocce et al
      2: use LSST model distribution, divided into rectagular bins (Default)
    verbose: control how much output is given by normBin
      Default: False
  """
  # extend Z range for smoothing
  extraZ,extraBins = extendZrange(zmin,zmax,nBins,binSmooth)
  zmax += extraZ
  nBins += extraBins

  zs = np.linspace(zmin,zmax,normPoints*nBins+1)
  biases = None
  if dndzMode == 1:
    nBins = 5 # to match dndz files
  for binNum in range(1,nBins+1):
    if doNorm:
      winKbin = getNormalizedWinKbin(myPk,binNum,zs,zmin=zmin,zmax=zmax,
                      nBins=nBins,biases=biases,normPoints=normPoints,BPZ=BPZ,
                      binSmooth=binSmooth,dndzMode=dndzMode,verbose=verbose)
    else:
      rawWinK = getWinKinterp(myPk,biases=biases,binNum=binNum,zmin=zmin,zmax=zmax,
                              nBins=nBins,dndzMode=dndzMode,binSmooth=binSmooth,BPZ=BPZ)
      winKbin = rawWinK(zs)

    plt.plot(zs,winKbin)

  plt.show()

  
def plotGaussians():
  """
    plot some gaussians to test the code
  """
  xs = np.linspace(-5,10,100)
  ys1 = gaussian(xs,0,1)
  ys2 = gaussian(xs,0,2)
  ys3 = gaussian(xs,3,1)
  ys4 = gaussian(xs,-0.5,2)
  plt.plot(xs,ys1)
  plt.plot(xs,ys2)
  plt.plot(xs,ys3)
  plt.plot(xs,ys4)
  plt.show()


def plotSmoothing(z0=1.5,zmin=0,zmax=4,nBins=10,doNorm=False,normPoints=100,
                  useWk=False,binSmoothSet=[0,0.05,0.1,0.2,0.4],
                  BPZ=True,nZvals=100,binNum=5):
  """
    get a dndz bin and smooth it
  Inputs:
    z0: controls width of total dist.
    zmin,zmax: min,max z for dividing up bins
    nBins: number of bins to divide up
    doNorm = False: set to True to normalize bins
    normPoints: number of points per zs interval to use when normalizing
    binSmoothSet: set of binSmooth values to control smoothing of tophat-stamped bins
      Default: [0,0.05,0.1,0.2,0.4]
    BPZ: 
    nZvals:
    binNum: which bin to watch melt
      Default: 5
  """
  # expand zs range to accomodate smoothing
  binSmooth = binSmoothSet[-1]
  extraZ,extraBins = extendZrange(zmin,zmax,nBins,binSmooth)
  zmax += extraZ
  nBins += extraBins

  zs = np.linspace(zmin,zmax,normPoints*nBins+1)
  #print zs
  for binSmooth in binSmoothSet:
    #print binSmooth,binNum
    if doNorm:
      dndzBin = getNormalizedDNDZbin(binNum,zs,z0,zmax,nBins,normPoints=normPoints,
                                     binSmooth=binSmooth,dndzMode=2) #2 for model
    else:
      rawDNDZ = getDNDZinterp(binNum=binNum,BPZ=BPZ,zmin=zmin,zmax=zmax,dndzMode=2,
                              binSmooth=binSmooth,z0=z0,nBins=nBins,nZvals=nZvals)
      dndzBin = rawDNDZ(zs)
    plt.plot(zs,dndzBin)
  plt.show()



################################################################################
# testing code

def test(doPlot = True):
  """

  """

  # test __file__
  print 'file: ',__file__,'\n'

  # replot digitized dNdz
  #replotDigitizedDNDZ()
  
  # plot their sum and ratio
  #plotDNDZsum()
  #plotDNDZratio()

  # create matterPower object with default parameters
  #print 'creating myPk...'
  #myPk = matterPower()

  # test getCl with no extra inputs
  #print 'testing getCl'
  #ls, Cl_kappa = getCl(myPk)
  #plotCl(ls,Cl_kappa)


  # test getBiasFit
  #plotBias()

  # getCl for other parameter values; use dNdz 0
  #print 'testing with plotKG'
  #plotKG(myPk,binNum=0,lmax=2000)

  # getCl for gg for all g
  #print 'testing with plotGG and plotGGsum'
  #plotGG(myPk)
  #plotGGsum(myPk)

  # test getCl using different kwargs values
  # the default parameters
  """cosParams = {
      'H0'    : 67.51,
      'ombh2' : 0.022,
      'omch2' : 0.119,
      'mnu'   : 0.06,
      'omk'   : 0,
      'tau'   : 0.06  }"""
  # modified parameters for testing
  """cosParams = {
      'H0'    : 42, #this one changed
      'ombh2' : 0.022,
      'omch2' : 0.119,
      'mnu'   : 0.06,
      'omk'   : 0,
      'tau'   : 0.06  }
  print 'creating myPk2 (H_0=42)... '
  myPk2 = matterPower(**cosParams)
  print 'testing plotKG with different params.'
  plotKG(myPk2,binNum=0,lmax=2000)
  """
  
  # test matterPower with more points
  myNz = 10000 # default value is 1000
  print 'creating myPk3 (nz=',myNz,')... '
  myPk3 = matterPower(nz=myNz)
  plotKG(myPk3,binNum=0,lmax=2000)

  
  # test with different bias model
  biasFunc = getBiasFit()
  biases = biasFunc(myPk3.zs)
  plotKG(myPk3,biasesG=biases,binNum=0,lmax=2000)


  # check redshift points compared to kernels
  myNz = 100 # default value is 1000
  print 'creating myPk4 (nz=',myNz,')... '
  myPk4 = matterPower(nz=myNz)
  plotRedshiftAppx(myPk4,dndzMode=2,nBins=10)

  # check model DNDZ, winK bins
  #plotModelDNDZ()
  #plotWinKbins(myPk)


if __name__=='__main__':
  test()





