#! /usr/bin/env python
"""
  Name:
    crosspower.py
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

class matterPower:
  """
    Purpose: 
      create and manipulate matter power spectrum objects
    Description:

    Data:
      cosParams: the cosmological parameters for camb's set_cosmology
        Default: 
          cosParams = {
            'H0'    : 67.51,
            'ombh2' : 0.022,
            'omch2' : 0.119,
            'mnu'   : 0.06,
            'omk'   : 0,
            'tau'   : 0.06  }
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

    Methods:
      __init__
      getPars: gets camb parameters object
      makePKinterp: make the Pk interpolator
      getPKinterp: return the Pk interpolator and other related info

  """

  def __init__(self,nz=1000,**cos_kwargs):
    """
    
      Inputs:
        nz: the number of z points to use between here and last scattering surface
          Important usage is as the number of points to use in approximation of
            C_l integrals
        **cos_kwargs: the cosmological parameters for camb's set_cosmology

    """
    # set cosmological parameters
    self.cosParams = {
        'H0'    : 67.51,
        'ombh2' : 0.022,
        'omch2' : 0.119,
        'mnu'   : 0.06,
        'omk'   : 0,
        'tau'   : 0.06  }
    self.updateParams(**cos_kwargs)
    self.ns = 0.965
    self.r  = 0
    #self.pars = self.getPars(self.ns,self.r,**self.cosParams) #also called by makePKinterp
    self.nz = nz # this will actually be 2 more than the number of z points
    self.kmax = 10
    self.makePKinterp(newPk=True,nz=self.nz,kmax=self.kmax) #hey... why passing self.stuff?



  def updateParams(self,**cos_kwargs):
    """
      modify object's cosParams variable
      Inputs:
        **cos_kwargs: dictionary of modifications to be passed to set_cosmology
    """
    self.cosParams.update(cos_kwargs)


  def getPars(self,ns=0.965,r=0,**kwargs):
    """
      Purpose:
        quickly get camb parameters object
        follows example code from http://camb.readthedocs.io/en/latest/CAMBdemo.html
          but with slightly different parameters
      Notes:
        This method uses object's parameters by default, but these can be overriden.
        This method does not overwrite object's stored parameters
      Inputs:
        ns: initial power spectrum running for set_params
        n:  parameter for set_params
        **kwargs: keyword args to pass to set_cosmology 
          if not included, object defaults will be used
      Returns:
        the pars object
    """
    # get cosmological parameters
    cosParams = self.cosParams
    cosParams.update(kwargs)

    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    #pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06) #why 0.122?
    #pars.set_cosmology(H0=67.51, ombh2=0.022, omch2=0.119, mnu=0.06, omk=0, tau=0.06)
    pars.set_cosmology(**cosParams)
    pars.set_dark_energy() #re-set defaults
    pars.InitPower.set_params(ns=ns, r=r)

    return pars


  def makePKinterp(self,newPk=True,nz=1000,kmax=10,myVar=model.Transfer_tot,**kwargs):
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
        kmax: kmax to use
        myVar: the variable to get autopower spectrum of
          default: model.Transfer_tot for delta_tot
        **kwargs: keyword args to pass to getPars
          if not used, getPars will use object defaults
      Outputs:
        sets various object variables

    """
    if newPk:
      self.updateParams(**kwargs)
      self.pars = self.getPars(self.ns,self.r)

    #For Limber result, want integration over \chi (comoving radial distance), from 0 to chi_*.
    #so get background results to find chistar, set up arrage in chi, and calculate corresponding redshifts
    results= camb.get_background(self.pars)
    self.chistar = results.conformal_time(0)- model.tau_maxvis.value
    self.chis = np.linspace(0,self.chistar,nz)
    self.zs=results.redshift_at_comoving_radial_distance(self.chis)
    #Calculate array of delta_chi, and drop first and last points where things go singular
    self.dchis = (self.chis[2:]-self.chis[:-2])/2 #overkill since chis are evenly spaced
    self.dzs   = (  self.zs[2:]-  self.zs[:-2])/2 #not as nice as with chi since zs not evenly spaced
    self.chis = self.chis[1:-1]
    self.zs = self.zs[1:-1]
    print 'zs.size: ',self.zs.size

    #Get the matter power spectrum interpolation object (based on RectBivariateSpline). 
    #Here for lensing we want the power spectrum of the Weyl potential.
    self.PK = camb.get_matter_power_interpolator(self.pars, nonlinear=True, 
        hubble_units=False, k_hunit=False, kmax=kmax,
        var1=myVar,var2=myVar, zmax=self.zs[-1])


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


def getDNDZinterp(binNum=1,BPZ=True,zmin=0.0,zmax=1.5,nZvals=100):
  """
    Purpose:
      get interpolator for dNdz(z) data points
    Inputs:
      binNum: integer in {1,2,3,4,5}.
        each corresponds to a redshift bin of width Delta.z=0.2:
        0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0, 1.0-1.2, 1.2-1.4;
        except for binNum=0, which indicates a sum of all other bins
      BPZ: selects which photo-z result to use
        True: use BPZ result
        False: use TPZ result
      zmin,zmax: the lowest,highest redshift points in the interpolation
        (will be added with dndz=0)
      nZvals: number of z values to use for interpolating sum of all curves 
        when binNum=0 (for sum of all) is selected
    Returns:
      interpolation function dNdz(z)
  """
  if binNum==0:
    myZvals = np.linspace(zmin,zmax,nZvals)
    myDNDZvals = np.zeros(nZvals)
    for bN in range(1,6): #can not include 0 in this range
      myFunc = getDNDZinterp(binNum=bN,BPZ=BPZ,zmin=zmin,zmax=zmax)
      myDNDZvals += myFunc(myZvals)
    z = myZvals
    dNdz = myDNDZvals
  else:
    z,dNdz = getDNDZ(binNum=binNum,BPZ=BPZ)
    z    = np.concatenate([[zmin],z,[zmax]])
    dNdz = np.concatenate([[0.],dNdz,[0.] ])

  return interp1d(z,dNdz,assume_sorted=True,kind='slinear')#'quadratic')


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
  numeratorInterp = getDNDZinterp(binNum=binNum,BPZ=BPZ,zmin=zmin,zmax=zmax,nZvals=nZvals)
  denominatorInterp = getDNDZinterp(binNum=0,BPZ=BPZ,zmin=zmin,zmax=zmax,nZvals=nZvals)

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


def modelDNDZbin(z,z0,zmax,nz,binNum,zmin=0.0,binSmooth=0):
  """
  Purpose:
    create redshift distribution with vertical edges based on model dN/dz
  Inputs:
    z: redshift(s) at which to evaluate DNDZ function
    z0: parameter controlling width of total dNdz dist.
    zmax: maximum z to use when slicing total dNdz into sections
    nz: number of evenly spaced sections to divide (0,zmax) into
    binNum: which bin to return dN/dz for.  bin starting at z=0 is number 1,
      bin ending at z=zmax is bin number nz
      Must have 0 < bNum <= nz, 
        or if bNum = 0, the sum of all bins will be returned 
    zmin = 0.0: minimum z for bins
    binSmooth: parameter to control smoothing of tophat-stamped bins
      Default: 0 (no smoothing)
  Returns:
    Amplitude at redshift(s) z.  If within bin, will be model DNDZ.  
      If outside bin, will be zero.
  
  """
  binEdges = np.linspace(zmin,zmax,nz+1)
  myDNDZ = modelDNDZ(z,z0)
  if binNum != 0:
    myDNDZ[np.where(z< binEdges[binNum-1])] = 0
    myDNDZ[np.where(z>=binEdges[binNum  ])] = 0
  
  # do smoothing here

  return myDNDZ


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
        if binNum == 0, sum of all bins will be used, 
          and binSmooth will be set to 0
        Default: 0
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
  if dndzMode == 1:
    zmax = 1.5 # hard coded to match domain in dndz files
    rawDNDZ = getDNDZinterp(binNum=binNum,BPZ=BPZ,zmin=zmin,zmax=zmax)
    binZmin = zmin
    binZmax = zmax
  elif dndzMode == 2:
    smoothDNDZ = modelDNDZbin(zs,z0,zmax,nBins,binNum,zmin=zmin,binSmooth=binSmooth)
    rawDNDZ = interp1d(zs,smoothDNDZ,assume_sorted=True,kind='slinear')

    # divide redshift range into bins and select desired bin
    binEdges = np.linspace(zmin,zmax,nBins+1)
    binZmin = binEdges[binNum-1] # first bin starting at zmin has binNum=1
    binZmax = binEdges[binNum]

    # accomodate binNum = 0 for entire range
    if binNum == 0:
      binZmin = zs[0]
      binZmax = zs[-1]
      binSmooth = 0

    if binSmooth != 0:
      # adjust binZmin, binZmax to appropriate cutoff for smoothed bin
      # bin already smoothed by modelDNDZbin

      # not yet implemented
      pass

  else: # really, it should be dndzMode = 1 or 2
    print 'covfefe!'
    return 0

  # select bin indices
  binIndices = np.where(np.logical_and( zs>=binZmin, zs<=binZmax ))
  #notBinIndices = np.where(np.logical_or( zs<binZmin, zs>binZmax ))
  #print 'binIndices: ',binIndices
  #print 'notBinIndices: ',notBinIndices

  # get normalization factor
  normFac = normBin(rawDNDZ,binZmin,binZmax,zs[binIndices],normPoints,verbose=verbose)

  # get non-normalized DNDZ
  binDNDZ = rawDNDZ(zs[binIndices])
  myDNDZ = np.zeros(zs.size)
  myDNDZ[binIndices] = binDNDZ
  
  #if binNum != 0 and dndzMode == 2:
  #  myDNDZ[np.where(zs< binZmin)] = 0
  #  myDNDZ[np.where(zs>=binZmax)] = 0

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


def gSmooth():
  """
    Purpose:
      smooth a tophat-stamped bin by convolution with a Gaussian
    Inputs:

    Returns:

  """


  return 0


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
          and binSmooth will be set to 0
        Default: 0
      dndzMode: indicate which method to use to select dNdz curves
        1: use observational DES-SV data from Crocce et al
        2: use LSST model distribution, divided into bins (Default)
      doNorm: set to True to normalize dN/dz.
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


  # get dz/dchi as ratio of deltaz/deltachi
  #  why not calculate as 1/H(z)?
  dzdchi = dzs/dchis

  # get dNdz according to dndzMode,useWk,doNorm,binSmooth settings
  if useWk and dndzMode == 2:
    # do not use biases here since they are multiplied in later
    if doNorm:
      myDNDZ = getNormalizedWinKbin(myPk,binNum,zs,zmin=zmin,zmax=zmax,BPZ=BPZ,
                                    nBins=nBins,binSmooth=binSmooth,dndzMode=dndzMode)
    else:
      myDNDZ = winKappa(myPk,binNum=binNum,zmin=zmin,zmax=zmax,nBins=nBins,
                        dndzMode=dndzMode,binSmooth=binSmooth)
  else: # not W_kappa or not dndzMode 2
    if doNorm:
      # this function will do its own dndzMode selection
      myDNDZ = getNormalizedDNDZbin(binNum,zs,z0,zmax,nBins,dndzMode=dndzMode,
                                    BPZ=BPZ,zmin=zmin,binSmooth=binSmooth)
    else: # do not do normalization
      if dndzMode == 1:
        zmin = 0.0
        zmax = 1.5 # hard coded to match domain in dndz files
        rawDNDZ = getDNDZinterp(binNum=binNum,BPZ=BPZ,zmin=zmin,zmax=zmax)
        myDNDZ = rawDNDZ(zs) 
      else: # dndzMode == 2:
        myDNDZ = modelDNDZbin(zs,z0,zmax,nBins,binNum,zmin=zmin,binSmooth=binSmooth)

  return dzdchi*myDNDZ*biases


def winKappa(myPk,biases=None,binNum=0,zmin=0,zmax=4,nBins=10,BPZ=True,
             dndzMode=2,binSmooth=0,includeEdges=False,**kwargs):
  """
    window function for CMB lensing convergence
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
        includeEdges: set to True to add one more myPk.zs point outside of bin range
          on each end of bin in set of nonzero values returned; 
          useful for interpolation over entire bin
          Default: False
      **kwargs: place holder so that winKappa and winGalaxies can have
        same parameter list
    Returns:
      array of W^kappa values evaluated at input myPk.chis
  """
  # get redshift array, etc.
  PK,chistar,chis,dchis,zs,dzs,pars = myPk.getPKinterp()

  if biases is None:
    biases = np.ones(zs.size)

  # Get lensing window function (flat universe)
  lightspeed = 2.99792458e5 # km/s
  myH0 = pars.H0/lightspeed # get H0 in Mpc^-1 units
  myOmegaM = pars.omegab+pars.omegac #baryonic+cdm
  myAs = 1/(1.+zs) #should have same indices as chis
  winK = chis*((chistar-chis)/(chistar*myAs))*biases
  winK *= (1.5*myOmegaM*myH0**2)

  # slice out the bin of interest
  if dndzMode == 1: # use observational dndz
    nBins = 5
    mode1zmin = 0
    mode1zmax = 1.5 # match the dndz files
    if binNum != 0:
      binIndices = np.where(np.logical_and( zs>=mode1zmin, zs<=mode1zmax ))
      notBinIndices = np.where(np.logical_or( zs<mode1zmin, zs>mode1zmax ))
      #print 'binIndices: ',binIndices
      #print 'notBinIndices: ',notBinIndices
      if binNum > nBins: # just return everything over the defined bins
        winK[binIndices] = 0
      else: # one of the bins has been indicated
        binZs = zs[binIndices]
        dndzRatioInterp = getDNDZratio(binNum=binNum,BPZ=BPZ,
                                       zmin=mode1zmin,zmax=mode1zmax)
        dndzRatios = dndzRatioInterp(binZs)
        # use ratios to define winKappa bin
        #print 'dndzRatios nan indices: ',np.where(np.isnan(dndzRatios)==True)
        winK[binIndices] *= dndzRatios
        winK[notBinIndices] = 0

  elif dndzMode == 2: # use dndz model
    # create tophat bins
    binEdges = np.linspace(zmin,zmax,nBins+1)
    #print binEdges
    if binNum != 0:
      #winK[np.where(zs<binEdges[binNum-1])] = 0
      #winK[np.where(zs>binEdges[binNum  ])] = 0
      indicesBelowBin = np.where(zs<binEdges[binNum-1])
      indicesAboveBin = np.where(zs>binEdges[binNum  ])
      if includeEdges: # this is here to allow interp over entire bin
        if zmin != 0:
          indicesBelowBin = indicesBelowBin[0][:-1]
        #if binNum != nBins:
        #print indicesAboveBin[0][1:]
        indicesAboveBin = indicesAboveBin[0][1:]
      winK[indicesBelowBin] = 0
      winK[indicesAboveBin] = 0
      if binSmooth != 0: #do smoothing
        # convolve tophat bins with gaussian
        # smooth winK array
        # write a separate function for this
        # may be better to have interp function call smoothing rather than this one

        pass
    #print winK

  else: # really, dndzMode has to be 1 or 2.
    print 'covfefe!'
    return 0

  #help(winK)
  #print 'winK: ',winK
  #print 'winK nan indices: ',np.where(np.isnan(winK)==True)

  return winK


def getWinKinterp(myPk,**kwargs):
  """
    Purpose:
      get interpolation function for winKappa(z)
    Inputs:
      myPk: MatterPower object
      binNum:
      zmin:
      **kwargs: keyword args to pass along to winKappa
    Returns:
      function for interpolating winKappa(z) result
  """
  winK = winKappa(myPk,includeEdges=True,**kwargs)
  # insert (0,0) at beginning
  zs = np.insert(myPk.zs,0,0)
  winK = np.insert(winK,0,0)
  return interp1d(zs,winK,assume_sorted=True,kind='slinear')


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
        if binNum == 0, sum of all bins will be used, 
          and binSmooth will be set to 0
        Default: 0
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
  # get Wk(z)
  rawWinK = getWinKinterp(myPk,biases=biases,binNum=binNum,zmin=zmin,zmax=zmax,
                          nBins=nBins,dndzMode=dndzMode,binSmooth=binSmooth,BPZ=BPZ)

  # get binZmin, binZmax for normalization range
  if dndzMode == 1:
    zmin = 0.0
    zmax = 1.5 # hard coded to match dndz files
    binZmin = zmin
    binZmax = zmax
    pass
  elif dndzMode ==2:
    # divide redshift range into bins and select desired bin
    binEdges = np.linspace(zmin,zmax,nBins+1)
    binZmin = binEdges[binNum-1] # first bin starting at zmin has binNum=1
    binZmax = binEdges[binNum]

    # accomodate binNum = 0 for entire range
    if binNum == 0:
      binZmin = zmin
      binZmax = zmax
      binSmooth = 0

    if binSmooth != 0:
      # adjust binZmin, binZmax to appropriate cutoff for smoothed bin

      # not yet implemented
      pass

  else: # really, dndzMode should be 1 or 2
    print 'covfefe!'
    return 0

  # get normalization factor
  normFac = normBin(rawWinK,binZmin,binZmax,zs,normPoints,verbose=verbose)

  # get non-normalized winK  
  myWinK = rawWinK(zs)
  if binNum != 0 and dndzMode == 2:
    myWinK[np.where(zs< binZmin)] = 0
    myWinK[np.where(zs>=binZmax)] = 0

  if verbose:
    myZs = zs[np.where(np.logical_and( zs>=binZmin, zs<binZmax ))]
    print 'zs in getNormalizedWinKbin: ',myZs

  # normalize
  normWinK = normFac*myWinK

  return normWinK




################################################################################
# the angular power spectrum


def getCl(myPk,biases1=None,biases2=None,winfunc1=winKappa,winfunc2=winKappa,
          dndzMode=1,binNum1=0,binNum2=0,lmax=2500,zmin=0.0,zmax=4.0,nBins=10,
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
        should be winKappa or winGalaxies
      dndzMode: select which dNdz scheme to use for winfuncs
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
    if winfunc == winKappa:
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
  # note that winKappa will ignore extra keywords due to use of **kwargs 
  win1=winfunc1(myPk,biases=biases1,z0=z0,dndzMode=dndzMode,binNum=binNum1,
                zmin=zmin,zmax=zmax,nBins=nBins,doNorm=doNorm,useWk=useWk,
                binSmooth=binSmooth,BPZ=BPZ)
  win2=winfunc2(myPk,biases=biases2,z0=z0,dndzMode=dndzMode,binNum=binNum2,
                zmin=zmin,zmax=zmax,nBins=nBins,doNorm=doNorm,useWk=useWk,
                binSmooth=binSmooth,BPZ=BPZ)

  #Do integral over chi
  ls = np.arange(2,lmax+1, dtype=np.float64)
  cl_kappa=np.zeros(ls.shape)
  w = np.ones(chis.shape) #this is just used to set to zero k values out of range of interpolation
  for i, l in enumerate(ls):
      k=(l+0.5)/chis
      w[:]=1
      w[k<1e-4]=0
      w[k>=myPk.kmax]=0
      cl_kappa[i] = np.dot(dchis, w*PK.P(zs, k, grid=False)*win1*win2/(chis**2))

  #print 'ls: ',ls,', Cl: ',cl_kappa
  return ls, cl_kappa







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
      myInterp = getDNDZinterp(binNum=binNum,BPZ=BPZ,zmin=zmin,zmax=zmax)
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
    sumInterp = getDNDZinterp(binNum=0,BPZ=BPZ) # 0 for sum
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
  ls1, Cl1 = getCl(myPk,biases1=biasesK,biases2=biasesK,winfunc1=winKappa,   winfunc2=winKappa,
                   doNorm=doNorm,useWk=useWk,binSmooth=binSmooth)
  ls2, Cl2 = getCl(myPk,biases1=biasesK,biases2=biasesG,winfunc1=winKappa,   winfunc2=winGalaxies, 
                   dndzMode=dndzMode,               binNum2=binNum,
                   zmax=zmax,nBins=nBins,z0=z0,
                   doNorm=doNorm,useWk=useWk,binSmooth=binSmooth)
  #ls3, Cl3 = getCl(myPk,biases1=biasesG,biases2=biasesK,winfunc1=winGalaxies,winfunc2=winKappa,   
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


def plotModelDNDZbins(z0=0.3,zmax=4,nBins=10,doNorm=False,normPoints=100,
                      useWk=False,binSmooth=0):
  """
  plot model DNDZ cut up into bins
  Inputs:
    z0: controls width of total dist.
    zmax: max z for dividing up bins
    nBins: number of bins to divide up
    doNorm = False: set to True to normalize bins
    normPoints: number of points per zs interval to use when normalizing
    binSmooth: controls smoothing of tophat-stamped bins
      Default: 0
  """
  zs = np.linspace(0,zmax,normPoints)
  for bNum in range(1,nBins+1):
    if doNorm:
      dndzBin = getNormalizedDNDZbin(bNum,zs,z0,zmax,nBins,normPoints=normPoints,
                                     binSmooth=binSmooth,dndzMode=2) #2 for model
    else:
      dndzBin = modelDNDZbin(zs,z0,zmax,nBins,bNum,binSmooth=binSmooth)
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
  zs = np.linspace(zmin,zmax,normPoints)
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





