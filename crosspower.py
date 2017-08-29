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
# the matterPower object

class matterPower:
  """
    Purpose: 
      create and manipulate matter power spectrum objects
    Description:

    Data:
      cosParams: the cosmological parameters for camb's set_cosmology
        Default: 
          cosParams = {
            'H0' : 67.51,
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
        'H0' : 67.51,
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
        **cos_kwargs: dictionary of modifications
          to be passed to set_cosmology
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
    function to load dN/dz data points from file
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
      get interpolator for dNdz data points
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
      interpolation function dNdz
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


def winGalaxies(chistar,chis,zs,dchis,dzs,pars,dndzNum,biases=None,BPZ=True):
  """
    window function for galaxy distribution
    Inputs: (should have same parameter list as winKappa)
      chistar: chi of CMB (really just a place holder here)
      chis: array of chi values to be used along chi integral
      zs: corresponding redshift value array
      dchis:
      dzs:
      pars:
      dndzNum: index indicating which redshift distribution to use
      biases: array of galaxy biases corresponding to zs values
        default: all equal to 1
      BPZ: flag to indicate BPZ or TPZ; True for BPZ
    Returns:
      array of W^galaxies values evaluated at input chi values
  """
  if biases is None:
    biases = np.ones(zs.size)

  # get dz/dchi as ratio of deltaz/deltachi
  dzdchi = dzs/dchis

  # get and normalize dN/dz
  zmin=0.0; zmax=1.5; normPoints=1000 # number of points in integral appx.
  deltaZ = (zmax-zmin)/(normPoints-1) # -1 for fenceposts to intervals
  rawDNDZ = getDNDZinterp(binNum=dndzNum,BPZ=BPZ,zmin=zmin,zmax=zmax)
  zArray = np.linspace(zmin,zmax,normPoints) # includes one point at each end
  dndzArray = rawDNDZ(zArray)
  area = dndzArray.sum()*deltaZ # ignore endpoints since they're both zero anyway
  normFac=1./area

  """
  #bias: fiducial biases from Giannantonio et al 2016, table 2, gal-gal, real space
  #  the index=0 bias is a place holder
  #  for case where doing sum of all kernels, want weighted avg. to put there.
  #gBias = (0,1.03,1.28,1.32,1.57,1.95)
  biasFunc = getBiasFit()
  biases = biasFunc(zs)
  """

  # get normalized dNdz
  myDNDZ = np.zeros(zs.size)
  myIndices = np.where(np.logical_and(zs>=zmin,zs<=zmax))
  myDNDZ[myIndices] = normFac*rawDNDZ(zs[myIndices])

  #win = dzdchi*myDNDZ*gBias[dndzNum]
  win = dzdchi*myDNDZ*biases
  return win


def winKappa(chistar,chis,zs,dchis,dzs,pars,dndz,biases=None):
  """
    window function for CMB lensing convergence
    Inputs: (should have same parameter list as winGalaxies)
      chistar: chi of CMB
      chis: array of chi values to be used along chi integral
      zs: corresponding redshift value array
      dchis:
      dzs:
      pars:
      dndz: (just a placeholder)
      biases: amplitude of lensing, array corresponding to zs values
        default: all equal to 1
    Returns:
      array of W^kappa values evaluated at input chi
  """
  if biases is None:
    biases = np.ones(zs.size)

  #Get lensing window function (flat universe)
  lightspeed = 2.99792458e5 # km/s
  myH0 = pars.H0/lightspeed # get H0 in Mpc^-1 units
  myOmegaM = pars.omegab+pars.omegac #baryonic+cdm
  myAs = 1/(1.+zs) #should have same indices as chis
  win = chis*((chistar-chis)/(chistar*myAs))*biases
  win *= (1.5*myOmegaM*myH0**2)

  return win





################################################################################
# the angular power spectrum


def getCl(myPk,biases1=None,biases2=None,winfunc1=winKappa,winfunc2=winKappa,dndz1=0,dndz2=0):
  """
    Purpose: get angular power spectrum
    Inputs:
      myPk: a matterPower object
      biases1,biases2: array of galaxy bias or tomographic CMB lensing amplitudes 
        at each redshift in myPk.zs, depending on which winfunc is selected
        default: all equal to 1
      winfunc1,winfunc2: the window functions
        should be winKappa or winGalaxies
      dndz1,dndz2: index indicating which dndz function to use
        integer in {0,1,2,3,4,5},  dndz1 for winfunc1, etc.
        curves from fig.3 of Crocce et al 2016.
        Index=0 indicates sum of all other curves
    Returns: 
      l,  the ell values (same length as Cl array)
      Cl, the power spectrum array


  """
  # confirm inputs
  def wincheck(winfunc,num):
    if winfunc == winKappa:
      print 'window ',num,': kappa'
    elif winfunc == winGalaxies:
      if num == 1:
        print 'window ',num,': galaxies ',dndz1
      else:
        print 'window ',num,': galaxies ',dndz2
    else:
      print 'error with input'
      return 0
    return 1
  
  if wincheck(winfunc1,1)==0: return 0,0
  if wincheck(winfunc2,2)==0: return 0,0
  
  #if gBias is None:
  #  gBias = np.ones(myPk.nz-2) # -2 due to excision of end points
  #if aLens is None:
  #  aLens = np.ones(myPk.nz-2)


  # get matter power spectrum P_k^delta
  PK,chistar,chis,dchis,zs,dzs,pars = myPk.getPKinterp()

  # get window functions
  win1=winfunc1(chistar,chis,zs,dchis,dzs,pars,dndz1,biases=biases1)
  win2=winfunc2(chistar,chis,zs,dchis,dzs,pars,dndz2,biases=biases2)

  #Do integral over chi
  ls = np.arange(2,2500+1, dtype=np.float64)
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


def plotKG(myPk,biasesK=None,biasesG=None,DNDZnum=0,lmax=2500):
  """
  Purpose:
    to plot each C_l for kappa, galaxy combinations
    Uses just one dNdz for all C_l
  Inputs:
    myPk: a matterPower object
    biasesK, array of galaxy bias
    biasesG: array of lensing amplitudes
      must be same length as myPk.zs
    DNDZnum: index in {0,1,2,3,4,5} defining which dNdz to use
      Default: 0 for sum of all others

  """
  ls1, Cl1 = getCl(myPk,biases1=biasesK,biases2=biasesK,winfunc1=winKappa,   winfunc2=winKappa)
  ls2, Cl2 = getCl(myPk,biases1=biasesK,biases2=biasesG,winfunc1=winKappa,   winfunc2=winGalaxies,              dndz2=DNDZnum)
  #ls3, Cl3 = getCl(myPk,biases1=biasesG,biases2=biasesK,winfunc1=winGalaxies,winfunc2=winKappa,   dndz1=DNDZnum)
  ls4, Cl4 = getCl(myPk,biases1=biasesG,biases2=biasesG,winfunc1=winGalaxies,winfunc2=winGalaxies,dndz1=DNDZnum,dndz2=DNDZnum)

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


def plotGG(myPk,biases1=None,biases2=None,lmax=2000):
  """
  Purpose:
    plot all Cl^{g_i g_j} for i,j in {1,2,3,4,5}
  Inputs:
    myPk: a matterPower object
    biases1,biases2: array of galaxy bias or lensing amplitudes
      must be same length as myPk.zs
      same pair of biases used for each x-cor (this should be improved)
    lmax: highest l to plot
  """
  for i in range(1,6):
    for j in range(i,6):
      print 'starting g_',i,' g_',j
      ls,Cl = getCl(myPk,biases1=biases1,biases2=biases2,winfunc1=winGalaxies,winfunc2=winGalaxies,dndz1=i,dndz2=j)
      plt.semilogy(ls,Cl,label='g_'+str(i)+', g_'+str(j))
  plt.xlim(0,lmax)
  plt.xlabel(r'$\ell$')
  plt.ylabel(r'$C_{\ell}$')
  plt.title('galaxy angular power spectra with DES-SV kernels')
  plt.legend()
  plt.show()


def plotGGsum(myPk,biases=None):
  """
  plot Cl^gg from sum of al dNdz
  Inputs:
    myPk: a matterPower object
    biases: array of galaxy biases
      must be same length as myPk.zs
  """
  ls,Cl = getCl(myPk,biases1=biases,biases2=biases,winfunc1=winGalaxies,winfunc2=winGalaxies,dndz1=0,dndz2=0)
  plt.semilogy(ls,Cl)
  plt.title('C_l^gg for sum of all dNdz curves')
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
  
  # plot their sum
  #plotDNDZsum()

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
  #plotKG(myPk,DNDZnum=0,lmax=2000)

  # getCl for gg for all g
  #print 'testing with plotGG and plotGGsum'
  #plotGG(myPk)
  #plotGGsum(myPk)

  # test getCl using different kwargs values
  # the default parameters
  """cosParams = {
      'H0' : 67.51,
      'ombh2' : 0.022,
      'omch2' : 0.119,
      'mnu'   : 0.06,
      'omk'   : 0,
      'tau'   : 0.06  }"""
  # modified parameters for testing
  """cosParams = {
      'H0' : 42, #this one changed
      'ombh2' : 0.022,
      'omch2' : 0.119,
      'mnu'   : 0.06,
      'omk'   : 0,
      'tau'   : 0.06  }
  print 'creating myPk2 (H_0=42)... '
  myPk2 = matterPower(**cosParams)
  print 'testing plotKG with different params.'
  plotKG(myPk2,DNDZnum=0,lmax=2000)
  """
  
  # test matterPower with more points
  myNz = 1000
  print 'creating myPk3 (nz=',myNz,')... '
  myPk3 = matterPower(nz=myNz)
  plotKG(myPk3,DNDZnum=0,lmax=2000)

  
  # test with different bias model
  biasFunc = getBiasFit()
  biases = biasFunc(myPk3.zs)
  plotKG(myPk3,biasesG=biases,DNDZnum=0,lmax=2000)

if __name__=='__main__':
  test()





