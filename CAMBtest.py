#! /usr/bin/env python
"""
  Name:
    CAMBtest.py
  Purpose:
    Explore how to use pycamb for P(z,k) interpolation
      and create C_l^{phi,phi}, C_l^{kappa,kappa}, C_l^{g,kappa},etc.

  Written by Z Knight, 2017.06.28
  Added galaxy kernels and galaxy-lensing cross power; ZK, 2017.07.28


"""

#import sys, platform, os
import numpy as np
import matplotlib.pyplot as plt
#import scipy.integrate as sint
#from scipy.interpolate import interp1d
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


def CAMBdemoCl():
  """
    example code from http://camb.readthedocs.io/en/latest/CAMBdemo.html

  """

  #Set up a new set of parameters for CAMB
  #pars = camb.CAMBparams()
  #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
  #pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
  #pars.InitPower.set_params(ns=0.965, r=0)

  pars = getPars()
  pars.set_for_lmax(2500, lens_potential_accuracy=0); # this line for C_l, not P_k

  #calculate results for these parameters
  results = camb.get_results(pars)

  #get dictionary of CAMB power spectra
  powers =results.get_cmb_power_spectra(pars)
  for name in powers: print name

  #plot the total lensed CMB power spectra versus unlensed, and fractional difference
  totCL=powers['total']
  unlensedCL=powers['unlensed_scalar']
  print totCL.shape
  #Python CL arrays are all zero based (starting at L=0), Note L=0,1 entries will be zero by default.
  #The different CL are always in the order TT, EE, BB, TE (with BB=0 for unlensed scalar results).
  ls = np.arange(totCL.shape[0])
  fig, ax = plt.subplots(2,2, figsize = (12,12))
  ax[0,0].plot(ls,totCL[:,0], color='k')
  ax[0,0].plot(ls,unlensedCL[:,0], color='r')
  ax[0,0].set_title('TT')
  ax[0,1].plot(ls[2:], 1-unlensedCL[2:,0]/totCL[2:,0]);
  ax[0,1].set_title(r'$\Delta TT$')
  ax[1,0].plot(ls,totCL[:,1], color='k')
  ax[1,0].plot(ls,unlensedCL[:,1], color='r')
  ax[1,0].set_title(r'$EE$')
  ax[1,1].plot(ls,totCL[:,3], color='k')
  ax[1,1].plot(ls,unlensedCL[:,3], color='r')
  ax[1,1].set_title(r'$TE$');
  for ax in ax.reshape(-1): ax.set_xlim([2,2500])
  plt.show()


def CAMBdemoPzk(redshifts=[0., 0.8]):
  """
    example code from http://camb.readthedocs.io/en/latest/CAMBdemo.html
      modified to have redshifts as a parameter
    Purpose:
      creates linear and nonlinear power spectra at multiple redshifts, and plots them
    Inputs:
      redshifts: the redshifts to find power spectra at
        Note: must be two redshift tuple

  """

  #Now get matter power spectra and sigma8 at redshift 0 and 0.8
  #pars = camb.CAMBparams()
  #pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
  #pars.set_dark_energy() #re-set defaults
  #pars.InitPower.set_params(ns=0.965)

  pars = getPars()
  #Not non-linear corrections couples to smaller scales than you want
  #pars.set_matter_power(redshifts=[0., 0.8], kmax=2.0)
  pars.set_matter_power(redshifts=redshifts, kmax=2.0)

  #Linear spectra
  pars.NonLinear = model.NonLinear_none
  results = camb.get_results(pars)
  kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)
  s8 = np.array(results.get_sigma8())

  #Non-Linear spectra (Halofit)
  pars.NonLinear = model.NonLinear_both
  results.calc_power_spectra(pars)
  kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)

  print 'redshifts:       : ',np.sort(redshifts)[::-1]
  print 'Linear sigma_8   : ',s8
  print 'NonLinear sigma_8: ',results.get_sigma8()

  for i, (redshift, line) in enumerate(zip(z,['-','--'])):
      plt.loglog(kh, pk[i,:], color='k', ls = line)
      plt.loglog(kh_nonlin, pk_nonlin[i,:], color='r', ls = line)
  plt.xlabel('k/h Mpc');
  plt.legend(['linear','non-linear'], loc='lower left');
  plt.title('Matter power at z=%s and z= %s'%tuple(z));
  #plt.title('Matter power');# at z=%s and z= %s'%tuple(z));
  plt.show()


def getDzk(redshifts=[0., 0.8], minkh=1e-4, maxkh=1, npoints=200, doPlot=True):
  """
    Purpose:
      Runs CAMB to create P(z,k) power spectra
      Uses P(z,k) = (D(z,k))**2 * P(z=0,k) to define D(z,k)
    Note: uses cosmological parameters as mix of Jan-2017 defaults (Planck 2015) and 
      those found in Planck 2015 XIII (1502.01589v2), table 4, last column
      However, these are giving a caluclated 
        sigma_8(z=0) = 0.7913, rather than the expected 0.8159 +- 0.0086 (2.9 sigma diff.)
    Inputs:
      redshifts: the redshifts to find power spectra at
        Must include 0.0 in list, otherwise lowest redshift will be used as ref. P(z,k)
      minkh: minimum value of k/h for output grid (very low values < 1e-4 may not be calculated)
      maxkh: maximum value of k/h (check consistent with input params.Transfer.kmax)
      npoints: number of points equally spaced in log k
      doPlot: set to True to make plot of power spectra
    Returns: kh,z,Dzk,kh_nonlin,z_nonlin,Dzk_nonlin (linear and nonlinear versions)
      kh : k/h values for Dzk  
      z  : redshift values for Dzk 
      Dzk: growth factor
      kh_nonlin :
      z_nonlin  :
      Dzk_nonlin:

  """

  #Get matter power spectra and sigma8 at redshifts specified
  #pars = camb.CAMBparams()
  #pars.set_cosmology(H0=67.74, ombh2=0.022, omch2=0.119)
  #pars.set_dark_energy() #re-set defaults
  #pars.InitPower.set_params(ns=0.967)

  pars = getPars()
  #Not non-linear corrections couples to smaller scales than you want
  pars.set_matter_power(redshifts=redshifts, kmax=2.0)

  #Linear spectra
  pars.NonLinear = model.NonLinear_none
  results = camb.get_results(pars)
  kh, z, pk = results.get_matter_power_spectrum(minkh=minkh, maxkh=maxkh, npoints = npoints)
  s8linear = np.array(results.get_sigma8())

  #Non-Linear spectra (Halofit)
  pars.NonLinear = model.NonLinear_both
  results.calc_power_spectra(pars)
  kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=minkh, maxkh=maxkh, npoints=npoints)
  s8nonlinear = np.array(results.get_sigma8())

  print 'redshifts:       : ',np.sort(redshifts)[::-1]
  print 'Linear sigma_8   : ',s8linear
  print 'NonLinear sigma_8: ',s8nonlinear


  # plot power spectra
  nRedshifts = redshifts.__len__()
  colors = ['b','g','r','c','m','y']
  nColors = colors.__len__()
  if doPlot:
    fig, ax = plt.subplots(1,2, figsize = (12,6))
    for redIndex in range(nRedshifts):
      ax[0].loglog(kh, pk[redIndex,:],  color=colors[redIndex%nColors])
      ax[1].loglog(kh_nonlin, pk_nonlin[redIndex], color=colors[redIndex%nColors]);
    ax[0].set_title('Linear power spectra')
    ax[1].set_title('Non-Linear power spectra')
    ax[0].set_xlabel('k/h Mpc');
    ax[1].set_xlabel('k/h Mpc');
    plt.legend(redshifts, loc='lower left')
    plt.show()

  # create ratios and find D(z,k)
  # lowest redshift (0.0) should be at end of list
  # no... looks like it's at the beginning (reversed for sigma_8 list)
  redshiftZeroIndex = 0 #-1
  Pratios        = np.empty(pk.shape)
  Pratios_nonlin = np.empty(pk_nonlin.shape)
  for zNum in range(nRedshifts):
    Pratios[zNum]        = pk[zNum]/pk[redshiftZeroIndex]
    Pratios_nonlin[zNum] = pk_nonlin[zNum]/pk_nonlin[redshiftZeroIndex]
  Dzk        = np.sqrt(Pratios)
  Dzk_nonlin = np.sqrt(Pratios_nonlin)

  if doPlot:
    fig, ax = plt.subplots(1,2, figsize = (12,6))
    for redIndex in range(nRedshifts):
      ax[0].loglog(kh, Dzk[redIndex,:],  color=colors[redIndex%nColors])
      ax[1].loglog(kh_nonlin, Dzk_nonlin[redIndex], color=colors[redIndex%nColors]);
    ax[0].set_title('sqrt(Linear power spectra ratio)')
    ax[1].set_title('sqrt(Non-Linear power spectra ratio)')
    ax[0].set_xlabel('k/h Mpc');
    ax[1].set_xlabel('k/h Mpc');
    plt.legend(redshifts, loc='lower left')
    plt.show()

  return kh,z,Dzk,kh_nonlin,z_nonlin,Dzk_nonlin

def getPKinterp(nz=100,kmax=10,myVar=model.Transfer_tot):
  """
    example code from http://camb.readthedocs.io/en/latest/CAMBdemo.html
      (modified to have nz,kmax,myVar as inputs)
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
      pars (CAMB parameters)
  """
  #First set up parameters as usual
  #pars = camb.CAMBparams()
  #pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
  #pars.InitPower.set_params(ns=0.965)

  pars = getPars()

  #For Limber result, want integration over \chi (comoving radial distance), from 0 to chi_*.
  #so get background results to find chistar, set up arrage in chi, and calculate corresponding redshifts
  results= camb.get_background(pars)
  chistar = results.conformal_time(0)- model.tau_maxvis.value
  chis = np.linspace(0,chistar,nz)
  zs=results.redshift_at_comoving_radial_distance(chis)
  #Calculate array of delta_chi, and drop first and last points where things go singular
  dchis = (chis[2:]-chis[:-2])/2
  chis = chis[1:-1]
  zs = zs[1:-1]

  #Get the matter power spectrum interpolation object (based on RectBivariateSpline). 
  #Here for lensing we want the power spectrum of the Weyl potential.
  PK = camb.get_matter_power_interpolator(pars, nonlinear=True, 
      hubble_units=False, k_hunit=False, kmax=kmax,
      var1=myVar,var2=myVar, zmax=zs[-1])

  return PK,chistar,chis,dchis,zs,pars


def CAMBdemoLimber():
  """
    example code from http://camb.readthedocs.io/en/latest/CAMBdemo.html
    Purpose:
      For calculating large-scale structure and lensing results yourself, get a power spectrum
      interpolation object. In this example we calculate the CMB lensing potential power
      spectrum using the Limber approximation, using PK=camb.get_matter_power_interpolator() function.
      calling PK(z, k) will then get power spectrum at any k and redshift z in range.

    Inputs:

  """
  nz = 100 #number of steps to use for the radial/redshift integration
  kmax=10  #kmax to use
  PK,chistar,chis,dchis,zs,pars = getPKinterp(nz=nz,kmax=kmax,myVar=model.Transfer_Weyl)

  #Have a look at interpolated power spectrum results for a range of redshifts
  #Expect linear potentials to decay a bit when Lambda becomes important, and change from non-linear growth
  plt.figure(figsize=(8,5))
  k=np.exp(np.log(10)*np.linspace(-4,2,200))
  zplot = [0, 0.5, 1, 4 ,20]
  for z in zplot:
      plt.loglog(k, PK.P(z,k))
  plt.xlim([1e-4,kmax])
  plt.xlabel('k Mpc')
  plt.ylabel('$P_\Psi\, Mpc^{-3}$')
  plt.legend(['z=%s'%z for z in zplot])
  plt.show()

  #Get lensing window function (flat universe)
  win = ((chistar-chis)/(chis**2*chistar))**2
  #Do integral over chi
  ls = np.arange(2,2500+1, dtype=np.float64)
  cl_kappa=np.zeros(ls.shape)
  w = np.ones(chis.shape) #this is just used to set to zero k values out of range of interpolation
  for i, l in enumerate(ls):
      k=(l+0.5)/chis
      w[:]=1
      w[k<1e-4]=0
      w[k>=kmax]=0
      cl_kappa[i] = np.dot(dchis, w*PK.P(zs, k, grid=False)*win/k**4)
  cl_kappa*= (ls*(ls+1))**2

  #Compare with CAMB's calculation:
  #note that to get CAMB's internal calculation accurate at the 1% level at L~2000, 
  #need lens_potential_accuracy=2. Increase to 4 for accurate match to the Limber calculation here
  pars.set_for_lmax(2500,lens_potential_accuracy=2)
  results = camb.get_results(pars)
  cl_camb=results.get_lens_potential_cls(2500) 
  #cl_camb[:,0] is phi x phi power spectrum (other columns are phi x T and phi x E)

  #Make plot. Expect difference at very low-L from inaccuracy in Limber approximation, and
  #very high L from differences in kmax (lens_potential_accuracy is only 2, though good by eye here)
  cl_limber= 4*cl_kappa/2/np.pi #convert kappa power to [l(l+1)]^2C_phi/2pi (what cl_camb is)
  plt.loglog(ls,cl_limber, color='b')
  plt.loglog(np.arange(2,cl_camb[:,0].size),cl_camb[2:,0], color='r')
  plt.xlim([1,2000])
  plt.legend(['Limber','CAMB hybrid'])
  plt.ylabel('$[L(L+1)]^2C_L^{\phi}/2\pi$')
  plt.xlabel('$L$')
  plt.show()


def PkPkTest():
  """
    Purpose: comparing matter Pk to potential Pk from pycamb


  """
  # get (from demo) potential power spectrum Pk^psi
  nz = 100 #number of steps to use for the radial/redshift integration
  kmax=10  #kmax to use
  PK,chistar,chis,dchis,zs,pars = getPKinterp(nz=nz,kmax=kmax,myVar=model.Transfer_Weyl)

  # get matter power spectrum Pk^delta
  PK2,chistar2,chis2,dchis2,zs2,pars2 = getPKinterp(nz=nz,kmax=kmax,myVar=model.Transfer_tot)
  #PK2,chistar2,chis2,dchis2,zs2,pars2 = getPKinterp(nz=nz,kmax=kmax,myVar=model.Transfer_cdm)
  #PK2,chistar2,chis2,dchis2,zs2,pars2 = getPKinterp(nz=nz,kmax=kmax,myVar=None) # to force delta_tot

  # with potential Pk^psi
  #Get lensing window function (flat universe)
  win = ((chistar-chis)/(chis**2*chistar))**2
  #Do integral over chi
  ls = np.arange(2,2500+1, dtype=np.float64)
  cl_kappa=np.zeros(ls.shape)
  w = np.ones(chis.shape) #this is just used to set to zero k values out of range of interpolation
  for i, l in enumerate(ls):
      k=(l+0.5)/chis
      w[:]=1
      w[k<1e-4]=0
      w[k>=kmax]=0
      cl_kappa[i] = np.dot(dchis, w*PK.P(zs, k, grid=False)*win/k**4)
  cl_kappa*= (ls*(ls+1))**2

  # with matter Pk^delta
  #Get lensing window function (flat universe)
  #kmPerMpc = 3.086e19
  #myH0 = pars.H0/kmPerMpc # get H0 in s^-1 units
  lightspeed = 2.99792e5 # km/s
  myH0 = pars.H0/lightspeed # get H0 in Mpc^-1 units
  myOmegaM = pars.omegab+pars.omegac #baryonic+cdm
  myAs = 1/(1.+zs2) #should have same indices as chis2
  win2 = ((chistar2-chis2)/(chistar2*myAs))**2

  #Do integral over chi
  ls = np.arange(2,2500+1, dtype=np.float64)
  cl_kappa2=np.zeros(ls.shape)
  w = np.ones(chis.shape) #this is just used to set to zero k values out of range of interpolation
  for i, l in enumerate(ls):
      k=(l+0.5)/chis
      w[:]=1
      w[k<1e-4]=0
      w[k>=kmax]=0
      cl_kappa2[i] = np.dot(dchis, w*PK2.P(zs2, k, grid=False)*win2)#/k**4)
  cl_kappa2*= (1.5*myOmegaM*myH0**2)**2

  print 'median factor: ',np.median(cl_kappa/cl_kappa2)

  # compare Cl values in plot
  cl_limber  = 4*cl_kappa /2/np.pi #convert kappa power to [l(l+1)]^2C_phi/2pi (what cl_camb is)
  cl_limber2 = 4*cl_kappa2/2/np.pi #convert kappa power to [l(l+1)]^2C_phi/2pi (what cl_camb is)
  plt.loglog(ls,cl_limber , color='b')
  plt.loglog(ls,cl_limber2, color='r')
  plt.xlim([1,2000])
  plt.legend(['potential Pk','matter Pk'])
  plt.ylabel('$[L(L+1)]^2C_L^{\phi}/2\pi$')
  plt.xlabel('$L$')
  plt.show()




################################################################################
# testing code

def test(doPlot = True):
  """

  """

  # test __file__
  print 'file: ',__file__,'\n'

  # do examples from demo website
  #CAMBdemoCl()
  #CAMBdemoPzk()

  #redshifts=[0.,3.]
  #CAMBdemoPzk(redshifts=redshifts)

  # my version for calculating D(z,k)
  #redshifts=[0,1,10,100,1000,1200]
  #kh,z,Dzk,kh_nonlin,z_nonlin,Dzk_nonlin = getDzk(redshifts=redshifts)

  # demo for the Limber approximation
  #CAMBdemoLimber()

  # my version for comparing matter Pk to potential Pk versions
  PkPkTest()


if __name__=='__main__':
  test()



