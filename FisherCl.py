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

"""

#import sys, platform, os
import numpy as np
import matplotlib.pyplot as plt
#import scipy.integrate as sint
#from scipy.interpolate import interp1d
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

  """
  # fiducial cosmological parameters
  H_0        = 67.04   # km/sec/Mpc
  c_light    = 2.998e5 # km/sec
  km_per_Mpc = 3.086e19
  Omega_L    = 0.6817
  Omega_M    = 0.3183  # sum of DM and b
  Omega_DM   = 0.2678
  Omega_b    = 0.04902
  Omega_r    = 8.24e-5
  # Omega_k explicitly assumed to be zero in formulae and omitted

  def __init__(self,massfile,zcenter):
    """

    """



################################################################################
# testing code

def test(doPlot = True):
  """
    function for testing the FisherMatrix object
  """

  # test __file__
  print 'file: ',__file__,'\n'

  # create and initialize object
  Fmatrix = FisherMatrix()


if __name__=='__main__':
  test()





