#! /usr/bin/env python
"""
  Name:
    crosspower.py
  Purpose:
    calculate various theoretical angular power spectra for gravitational lensing and galaxies
  Uses:
    pycamb (aka camb)
  Modification History:
    Written by Z Knight, 2017.07.31
    Added dNdz interpolation; ZK, 2017.08.04

"""

#import sys, platform, os
import numpy as np
import matplotlib.pyplot as plt
#import scipy.integrate as sint
from scipy.interpolate import interp1d
import camb
from camb import model, initialpower


################################################################################
# some functions

def getPars():
  """
    Purpose:
      quickly get camb parameters object
      follows example code from http://camb.readthedocs.io/en/latest/CAMBdemo.html
        but with slightly different parameters
    Inputs:

    Returns:
      the pars object
  """
  #Set up a new set of parameters for CAMB
  pars = camb.CAMBparams()
  #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
  #pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06) #why 0.122?
  pars.set_cosmology(H0=67.51, ombh2=0.022, omch2=0.119, mnu=0.06, omk=0, tau=0.06)
  pars.set_dark_energy() #re-set defaults
  pars.InitPower.set_params(ns=0.965, r=0)

  return pars


def getPKinterp(nz=100,kmax=10,myVar=model.Transfer_tot):
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
      nz:   number of steps to use for the radial/redshift integration
      kmax: kmax to use
      myVar: the variable to get autopower spectrum of
        default: model.Transfer_tot for delta_tot
    Returns:
      the PK(z,k) interpolator
      chistar
      chis (array of chi values)
      dchis (delta chi array)
      zs (redshift array)
      dzs (delta z array)
      pars (CAMB parameters)
  """
  pars = getPars()

  #For Limber result, want integration over \chi (comoving radial distance), from 0 to chi_*.
  #so get background results to find chistar, set up arrage in chi, and calculate corresponding redshifts
  results= camb.get_background(pars)
  chistar = results.conformal_time(0)- model.tau_maxvis.value
  chis = np.linspace(0,chistar,nz)
  zs=results.redshift_at_comoving_radial_distance(chis)
  #Calculate array of delta_chi, and drop first and last points where things go singular
  dchis = (chis[2:]-chis[:-2])/2 #overkill since chis are evenly spaced
  dzs   = (  zs[2:]-  zs[:-2])/2 #not as nice as with chi since zs not evenly spaced
  chis = chis[1:-1]
  zs = zs[1:-1]

  #Get the matter power spectrum interpolation object (based on RectBivariateSpline). 
  #Here for lensing we want the power spectrum of the Weyl potential.
  PK = camb.get_matter_power_interpolator(pars, nonlinear=True, 
      hubble_units=False, k_hunit=False, kmax=kmax,
      var1=myVar,var2=myVar, zmax=zs[-1])

  return PK,chistar,chis,dchis,zs,dzs,pars


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


def getDNDZinterp(binNum=1,BPZ=True,zmin=0.0,zmax=1.5):
  """
    Purpose:
      get interpolator for dNdz data points
    Inputs:
      binNum: integer in {1,2,3,4,5}.
        each corresponds to a redshift bin of width Delta.z=0.2:
        0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0, 1.0-1.2, 1.2-1.4
      BPZ: selects which photo-z result to use
        True: use BPZ result
        False: use TPZ result
      zmin,zmax: the lowest,highest redshift points in the interpolation
        (will be added with dndz=0)
    Returns:
      interpolation function dNdz
  """
  z,dNdz = getDNDZ(binNum=binNum,BPZ=BPZ)
  z    = np.concatenate([[zmin],z,[zmax]])
  dNdz = np.concatenate([[0.],dNdz,[0.] ])

  return interp1d(z,dNdz,assume_sorted=True,kind='slinear')#'quadratic')


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




def winGalaxies(chistar,chis,zs,dchis,dzs,pars,dndzNum,BPZ=True):
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
      BPZ: flag to indicate BPZ or TPZ; True for BPZ
    Returns:
      array of W^galaxies values evaluated at input chi values
  """
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

  #bias: fiducial biases from Giannantonio et al 2016, table 2, gal-gal, real space
  #  the index=0 bias is a place holder
  gBias = (0,1.03,1.28,1.32,1.57,1.95)
  
  # get normalized dNdz
  myDNDZ = np.zeros(zs.size)
  myIndices = np.where(np.logical_and(zs>=zmin,zs<=zmax))
  myDNDZ[myIndices] = normFac*rawDNDZ(zs[myIndices])

  win = dzdchi*myDNDZ*gBias[dndzNum]
  return win


def winKappa(chistar,chis,zs,dchis,dzs,pars,dndz):
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
    Returns:
      array of W^kappa values evaluated at input chi
  """
  #Get lensing window function (flat universe)
  lightspeed = 2.99792458e5 # km/s
  myH0 = pars.H0/lightspeed # get H0 in Mpc^-1 units
  myOmegaM = pars.omegab+pars.omegac #baryonic+cdm
  myAs = 1/(1.+zs) #should have same indices as chis2
  win = ((chistar-chis)/(chistar*myAs))
  win *= (1.5*myOmegaM*myH0**2)

  return win


def getCl(winfunc1=winKappa,winfunc2=winKappa,dndz1=1,dndz2=1):
  """
    Purpose: get angular power spectrum
    Inputs:
      winfunc1,winfunc2: the window functions
        should be winKappa or winGalaxies
      dndz1,dndz2: index indicating which dndz function to use
        integer in {1,2,3,4,5},  dndz1 for winfunc1, etc.
        curves from fig.3 of Crocce et al 2016.
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
  
  # get matter power spectrum P_k^delta
  nz = 100 #number of steps to use for the radial/redshift integration
  kmax=10  #kmax to use
  PK,chistar,chis,dchis,zs,dzs,pars = getPKinterp(nz=nz,kmax=kmax)

  # get window functions
  win1=winfunc1(chistar,chis,zs,dchis,dzs,pars,dndz1)
  win2=winfunc2(chistar,chis,zs,dchis,dzs,pars,dndz2)

  #Do integral over chi
  ls = np.arange(2,2500+1, dtype=np.float64)
  cl_kappa=np.zeros(ls.shape)
  w = np.ones(chis.shape) #this is just used to set to zero k values out of range of interpolation
  for i, l in enumerate(ls):
      k=(l+0.5)/chis
      w[:]=1
      w[k<1e-4]=0
      w[k>=kmax]=0
      cl_kappa[i] = np.dot(dchis, w*PK.P(zs, k, grid=False)*win1*win2)

  return ls, cl_kappa

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


def plotKG(DNDZnum=1):
  """
  Purpose:
    to plot each C_l for kappa, galaxy combinations
    Uses just one dNdz for all C_l
  Inputs:
    DNDZnum: index in {1,2,3,4,5} defining which dNdz to use

  """
  ls1, Cl1 = getCl(winfunc1=winKappa,   winfunc2=winKappa)
  ls2, Cl2 = getCl(winfunc1=winKappa,   winfunc2=winGalaxies,              dndz2=DNDZnum)
  ls3, Cl3 = getCl(winfunc1=winGalaxies,winfunc2=winKappa,   dndz1=DNDZnum)
  ls4, Cl4 = getCl(winfunc1=winGalaxies,winfunc2=winGalaxies,dndz1=DNDZnum,dndz2=DNDZnum)

  p1=plt.semilogy(ls1,Cl1,label='kk')
  p2=plt.semilogy(ls2,Cl2,label='kg')
  p3=plt.semilogy(ls3,Cl3,label='gk')
  p4=plt.semilogy(ls4,Cl4,label='gg')
  plt.legend()
  plt.show()

def plotGG():
  """
  Purpose:
    plot all Cl^{g_i g_j} for i,j in {1,2,3,4,5}

  """
  for i in range(1,6):
    for j in range(i,6):
      print 'starting g_',i,' g_',j
      ls,Cl = getCl(winfunc1=winGalaxies,winfunc2=winGalaxies,dndz1=i,dndz2=j)
      plt.semilogy(ls,Cl,label='g_'+str(i)+', g_'+str(j))
  plt.legend()
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

  # test getCl with no inputs
  #ls, Cl_kappa = getCl()
  #plotCl(ls,Cl_kappa)


  # getCl for other parameter values; use dNdz 1
  #plotKG(DNDZnum=1)

  # getCl for gg for all g
  plotGG()


if __name__=='__main__':
  test()





