#! /usr/bin/env python
"""
  Name:
    getPk.py
  Purpose:
    This program loads matter power spectra, plots them, and creates ratios of power spectra
    Created for extraction of linear growth function D(z,k) from P(z,k) CAMB output

  Written by Z Knight, 2017.06.23
  Fixed indexing error; ZK, 2017.06.27

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sint
from scipy.interpolate import interp1d


################################################################################
# some functions

def loadPk(filenames, doPlot=False, verbose=False):
  """
    Purpose:
      loads power spectra from CAMB matterpower.dat files
    Inputs:
      filenames: string or list of strings containing filenames to load
      doPlot: set to True to plot each (P,k) on loglog axes
      verbose: set to True for extra output
    Outputs:
      Returns numpy array with indices (n,Pk)
        where n=0 is for k values, otherwise is index from filenames list+1
    Note: CAMB matter power spectrum output against k/h in units of h^{-3} Mpc^3

  """
  # question: do all matterpower files use the same k values?  Assume they do for now.
  # in CAMB params.ini, transfer_interp_matterpower = T
  #  => regular interpolated grid in log k

  # check for single string
  if filenames[0].__len__() == 1:
    #single = True
    filenames = [filenames]
    nFiles = 1
  else:
    #single = False
    nFiles = filenames.__len__()
  print nFiles,' file names in list.'

  # load data
  powspec = np.loadtxt(filenames[0])
  nWaveNums = powspec.shape[0]
  specCube = np.empty((nFiles+1,nWaveNums))

  for fileNum in range(nFiles):
    powspec = np.loadtxt(filenames[fileNum])
    if verbose:
      print 'Power spectrum data shape: ',powspec.shape
      kmin = powspec[0,0]
      kmax = powspec[-1,0]
      print 'kmin = ',kmin,', kmax = ',kmax

    if fileNum == 0:
      specCube[0] = powspec[:,0]
    specCube[fileNum+1] = powspec[:,1]

    if doPlot:
      # display power spectrum plot
      plt.loglog(powspec[:,0],powspec[:,1])
      plt.title('CAMB matter power spectrum')
      plt.xlabel('Wavenumber k [h/Mpc]')
      plt.ylabel('Power Spectrum P(k) [(Mpc/h)^3]')
      plt.show()
    

  return specCube


################################################################################
# testing code

def test(doPlot = True):
  """

  """

  # test __file__
  print 'file: ',__file__,'\n'

  # load data from files: P(k,z)

  # try with just one filename
  folder = 'matterpower/'
  filename = 'zrange_matterpower00.dat'
  
  prefix = 'zrange_'
  basename = 'matterpower'
  nDigits = 2 # number of digits in numerical distinguisher in filename
  suffix = '.dat'
  nFiles = 11 # should match up with file descriptions in CAMB params.ini
  filenames = []
  for fileNum in range(nFiles):
    myNumString = str(fileNum).zfill(nDigits)
    filenames.append(folder+prefix+basename+myNumString+suffix)
  print 'Power spectrum files: ',filenames


  Pzk = loadPk(filenames)
  print 'shape of Pzk: ',Pzk.shape


  # create power spectrum interpolation function
  #powspecinterp = interp1d(powspec[:,0],powspec[:,1],kind='cubic')

  if doPlot:
    # display power spectrum plot
    for filenum in range(1,nFiles+1):
      plt.loglog(Pzk[0],Pzk[filenum])
    plt.title('CAMB matter power spectrum')
    plt.xlabel('Wavenumber k [h/Mpc]')
    plt.ylabel('Power Spectrum P(k) [(Mpc/h)^3]')
    plt.show()

  # create ratios to Pzk(z=1) 
  redshiftZeroIndex = 11 # should match up with params.ini value +1 (for k in 0th place)
  Pratios = np.empty(Pzk.shape)
  Pratios[0] = Pzk[0] # k/h values
  for fileNum in range(1,nFiles+1):
    Pratios[fileNum] = Pzk[fileNum]/Pzk[redshiftZeroIndex]
    print filenames[fileNum-1],': ',Pratios[fileNum,0]

  if doPlot:
    # display power ratio plot
    for filenum in range(1,nFiles+1):
      plt.loglog(Pratios[0],Pratios[filenum])
    plt.title('CAMB matter power spectrum ratio')
    plt.xlabel('Wavenumber k [h/Mpc]')
    plt.ylabel('Power Spectrum ratio P(k,z)/P(k,z=0)')
    plt.show()


if __name__=='__main__':
  test()



