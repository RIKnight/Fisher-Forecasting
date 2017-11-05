#! /usr/bin/env python
"""
Name:
  linearGrowth
Purpose:
  implement the linear growth function for cosmic structure formation
  uses Dodelson eq.n 7.77
Uses:
  cosmography.py

Inputs:

Outputs:

Modification History:
  Originally written as part of ISWprofile.ClusterVoid class by Z Knight, 2015
  Extracted from ClusterVoid by Z Knight, 2017.06.06

"""

import numpy as np
import matplotlib.pyplot as plt
#import healpy as hp
import time # for measuring duration
import cosmography as cg
import scipy.integrate as sint
#from scipy.interpolate import interp1d


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
# linear growth functions


  def hubbleparam(self,a):
    """ the hubble parameter as a function of scale factor and cosmological 
        parameters.  Curvature is assumed to be flat and is omitted from
        the calculation.
        returns hubble parameter in units km/s/Mpc
    """
    omega_scaled = self.Omega_M*a**-3 + self.Omega_r*a**-4 + self.Omega_L
    return self.H_0 * np.sqrt(omega_scaled)

  def conftime(self,a):
    """ the conformal time / horizon distance for a given scale factor a 
        a can be a single scale factor or an array of them
        returns conftime in units Mpc/c (where c=1 so units are Mpc)
        I call this a conformal time but really it's a comoving distance in Mpc
          from a'=0 to a'=a.
        To convert to Mpc/h/c, multiply by H_0/100
    """
    a = np.array(a)
    eta = np.zeros(a.size)
    int_eta = lambda a: a**-2*self.hubbleparam(a)**-1
    for aindex in range(a.size):
      if a.size == 1:
        result = sint.quad(int_eta,0,a)
      else:
        result = sint.quad(int_eta,0,a[aindex])
      eta[aindex] = result[0]
    return eta*self.c_light



  def int_D1(self,a):
    """ the integrand for the growth factor equation """
    H = self.hubbleparam(a)
    return (a*H/self.H_0)**-3

  def D1(self,a):
    """ growth factor; Dodelson eq.n 7.77 
        a can be a single scale factor or an array of them
    """
    a = np.array(a)
    prefac = 5*self.Omega_M*self.hubbleparam(a)/(2*self.H_0)
    growth = np.zeros(a.size)
    for index in range(a.size):
      if a.size == 1:
        result = sint.quad(self.int_D1, 0, a)
      else:
        result = sint.quad(self.int_D1, 0, a[index])
      growth[index] = result[0]
    return prefac*growth

  def dD1_deta(self,a):
    """ derivative of the growth factor wrt conformal time eta 
        returns dD1/deta in units c/Mpc
    """
    H = self.hubbleparam(a)
    return ( -a*H*self.D1(a) + 5*self.Omega_M*self.H_0**2/(2*a*H) )/self.c_light




################################################################################
# testing code

def test(params='Gian'):
  """


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


"""
  # test D1 (growth factor)
  adomain = np.logspace(-7,0,100)
  D1range = myCluster.D1(adomain)
  plt.loglog(adomain,D1range)
  plt.xlabel('scale factor')
  plt.ylabel('growth factor')
  plt.show()

  # test dD1_deta (derivative of growth factor)
  adomain = np.logspace(-7,0,100)
  dD1range = myCluster.dD1_deta(adomain)
  plt.loglog(adomain,dD1range)
  plt.xlabel('scale factor')
  plt.ylabel('d(growth factor) / d(conformal time) [Mpc**-1]')
  plt.show()

"""





if __name__=='__main__':
  test()



