#! /usr/bin/env python
"""
Name:
  precisiontes
Purpose:
  test the comoving distance and growth functions to find optimal parameters that control precision
Uses:
  cosmography.py

Inputs:

Outputs:

Modification History:
  Written by Z Knight, 2017.06.05

"""

import numpy as np
import matplotlib.pyplot as plt
#import healpy as hp
import time # for measuring duration
import cosmography as cg


################################################################################
# helping functions

def photonParam(T=2.726,h=0.678):
  """
    finds the density parameter of photon radiation 
    Inputs:
      T: the photon temperature
      h: the hubble parameter
    Returns:
      the density parameter for photons
  """
  pi = 3.14159
  #k  = 1.3806e-23 #J/K
  #c  = 2.9978e8   #m/s
  #h  = 6.6261e-34 #Js
  #G  = 6.6741e-11 #Nm^2/kg^2

  # as in Phys 262, hw 2.2
  # use k=hbar=c=1 units...
  KperGeV    = 1.1605e13
  GeVper_h   = 2.1332e-42
  m_planck   = 1.22e19  #GeV

  ro_c = 3*m_planck**2/(8*pi)*(h*GeVper_h)**2
  ro_r = pi**2/15 * (T/KperGeV)**4

  return ro_r/ro_c


################################################################################
# testing code

def test(params='Gian',zMax=10,nSteps=1000,loglog=0):
  """
    code for testing cosmography and growth
    Inputs:
      zMax: the maximum redshift to compute comoving distance out to
      nSteps: the number of steps to take in computing distance
        This will be the number of computed values as well as part of 
        the determination of step size: zMax/nSteps in summation appx. to int.
      loglog: controls plotting of comoving distance
        0: linear
        1: semilog
        2: loglog

  """
  # define cosmological parameters
  if params == 'Gian':
    # Following Giannantonio et al, 2016, using Planck 2013 (+WMAP pol + ACT/SPT + BAO) best-fitting flat LambaCDM+nu (1 massive neutrino) model
    omega_b  = 0.0222
    omega_c  = 0.119
    omega_nu = 0.00064
    h        = 0.678
    tau      = 0.0952
    A_s      = 2.21e-9
    n_s      = 0.961
    kbar     = 0.05 #Mpc^-1; pivot scale of n_s
    sigma_8  = 0.829

    omega_L  = 0.317826 # ~from Planck 2013 XVI, not Gian.
  else: 
    print '... covfefe ...'
    return 0

  # convert params to form expected by cosmography.py
  H0 = h*100
  #Omega_m  = (omega_c+omega_b) /h**2 
  Omega_m  = (omega_c+omega_b+omega_nu) /h**2
  Omega_nu = omega_nu/h**2

  #Omega_k = 0.0 (implicit in code)
  #um... what about omega_lambda, omega_r?
  Omega_r = photonParam(T=2.726,h=h)
  print 'Omega_r = ',Omega_r

  #N_eff = 3.046
  #nufac = N_eff*(7/8.)*(4/11.)**(4/3.)
  #Omega_r2 = Omega_nu/nufac #guessing here.  need to confirm.
  #print 'Omega_r2 = ',Omega_r2
  
  #Omega_L = 1-Omega_m-Omega_r-Omega_nu
  Omega_L = omega_L/h**2


  # test comoving distance
  # At this point, Omega_nu is exclusively matter.  Fix this please.

  startTime = time.time()
  z_vals, comDist = cg.ComovingDistance(zMax,Omega_m,Omega_L,nSteps,H0,omegaR=Omega_r)
  endTime = time.time()
  print 'time elapsed: ',(endTime-startTime),' seconds'
  cg.makePlot(z_vals,comDist,"Comoving Distance","Mpc",loglog)

  # compare results with and without radiation
  startTime = time.time()
  z_vals2, comDist2 = cg.ComovingDistance(zMax,Omega_m,Omega_L,nSteps,H0)
  endTime = time.time()
  print 'time2 elapsed: ',(endTime-startTime),' seconds'
  cg.makePlot(z_vals,comDist2-comDist,"Radiation Contribution to Comoving Distance","Mpc",loglog)




  # test for convergence
  # use same zMax parameter
  # refine nSteps to higher levels
  doConvergence = False
  if doConvergence:
    fac3=10
    fac4=100
    fac5=1000

    startTime = time.time()
    z_vals3, comDist3 = cg.ComovingDistance(zMax,Omega_m,Omega_L,nSteps*fac3,H0,omegaR=Omega_r)
    endTime = time.time()
    print 'time3 elapsed: ',(endTime-startTime),' seconds'

    startTime = time.time()
    z_vals4, comDist4 = cg.ComovingDistance(zMax,Omega_m,Omega_L,nSteps*fac4,H0,omegaR=Omega_r)
    endTime = time.time()
    print 'time4 elapsed: ',(endTime-startTime),' seconds'

    startTime = time.time()
    z_vals5, comDist5 = cg.ComovingDistance(zMax,Omega_m,Omega_L,nSteps*fac5,H0,omegaR=Omega_r)
    endTime = time.time()
    print 'time5 elapsed: ',(endTime-startTime),' seconds'

    # extract the indices for comparison
    indices3 = np.arange(nSteps+1)*fac3
    indices4 = np.arange(nSteps+1)*fac4
    indices5 = np.arange(nSteps+1)*fac5

    # plot results
    plt.semilogx(z_vals,np.zeros(nSteps+1)) #has to be zero
    plt.semilogx(z_vals,comDist3[indices3]-comDist)
    plt.semilogx(z_vals,comDist4[indices4]-comDist)
    plt.semilogx(z_vals,comDist5[indices5]-comDist)
    plt.title('comoving distance differences from base case')
    plt.xlabel('redshift')
    plt.ylabel('Mpc')
    plt.show()
    






if __name__=='__main__':
  test()



