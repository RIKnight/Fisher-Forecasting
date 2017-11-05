#!/usr/bin/env python
"""
  NAME:
    cosmography.py
  PURPOSE:
    calculate various distance measures in an expanding universe
  PROCEDURE:
    Several formulae taken from Hogg, astro-ph/9905116
  MODIFICATION HISTORY:
    Originally written for Phys 267 HW1 By Z Knight, 2013.04.10
    Turned into portable module; Z Knight, 2015.08.14
    Added comoving distance; improved documentation; 
      replaced Hubble Time with 1/H_0; ZK, 2015.08.19
    Added radiation component to E and all functions that call it; 
      Modified all functions that use nSteps to return nSteps+1 data points
      and include z=0 and z=zMax at ends; ZK, 2017.06.06

"""

import numpy as np
import matplotlib.pyplot as plt
#import sympy as sp


def E(z,omegaM,omegaK,omegaL,omegaR=0.0):
  """ 
    equation 14, Hogg
      (modified by addition of radiation component)
    Friedmann equation: H(z) = H_0 * E(z) 
    In late universe uses omega_r ~ 0
    returns E(z)
  """
  if omegaR == 0.0:
    return np.sqrt(                omegaM*(1+z)**3+omegaK*(1+z)**2+omegaL)
  else:
    return np.sqrt(omegaR*(1+z)**4+omegaM*(1+z)**3+omegaK*(1+z)**2+omegaL)

def DAngular(zMax,omegaM,omegaL,nSteps,DH,omegaR=0.0):
  """ 
    equations 15, 16, 18, Hogg
    DH is the Hubble distance c/H_0 in Mpc
    uses omega_k = 0 (for simplicity in eq 16)
    implements numeric integration by adding up small increments
    Note: this can also be calculated as comovingDistance/(1+z)
    returns two arrays: one of redshifts and another of angular diameter distances in Mpc
      of length nSteps+1, including z=0 and z=zMax
  """
  zVals = np.zeros(nSteps+1)
  distances = np.zeros(nSteps+1)
  stepSize = (zMax+0.0)/nSteps
  # calculate integral values
  for i in range(1,nSteps+1):
    zVals[i] = i*stepSize
    distances[i] = distances[i-1] + stepSize/E(zVals[i],omegaM,0,omegaL,omegaR)
  # convert D_C to D_A
  for i in range(1,nSteps+1):
    distances[i] = distances[i]/(1+zVals[i])*DH
  return zVals, distances

def DLuminosity(zMax,omegaM,omegaL,nSteps,DH,omegaR=0.0):
  """ 
    equation 21, Hogg
    DH is the Hubble distance c/H_0 in Mpc
    uses DAngular
    returns two arrays: one of redshifts and another of luminosity distances in Mpc
      of length nSteps+1, including z=0 and z=zMax
  """
  zVals, distances = DAngular(zMax,omegaM,omegaL,nSteps,DH,omegaR=omegaR)
  for i in range(1,nSteps+1):
    distances[i] = distances[i]*(1+zVals[i])**2
  return zVals, distances

def ApMagnitude(zMax,omegaM,omegaL,nSteps,DH,omegaR=0.0):
  """ 
    equations 25, 26, Hogg
    DH is the Hubble distance c/H_0 in Mpc
    uses DLuminosity
    returns two arrays: one of redshifts and another of apparent magnitudes
      of length nSteps+1, including z=0 and z=zMax
  """
  MStar = -20.6
  zVals, distances = DLuminosity(zMax,omegaM,omegaL,nSteps,DH,omegaR=omegaR)
  magnitudes = np.zeros(nSteps+1)
  for i in range(1,nSteps+1):
    # convert Mpc to pc and divide by standard value of 10pc
    magnitudes[i] = MStar+5*np.log10(distances[i]*1e6/10)
  return zVals, magnitudes

def AngSize(zMax,omegaM,omegaL,nSteps,DH,omegaR=0.0):
  """ 
    angsize = size/D_A 
    DH is the Hubble distance c/H_0 in Mpc
    uses DAngular
    returns two arrays: one of redshifts and another of angle in arcseconds subtended by 1 kpc
      of length nSteps+1, including z=0 and z=zMax
  """
  linearSize = 0.001 # Mpc (where 0.001 Mpc = 1 kpc)
  zVals, distances = DAngular(zMax,omegaM,omegaL,nSteps,DH,omegaR=omegaR)
  angSize = np.zeros(nSteps+1)
  for i in range(1,nSteps+1):
    angSize[i] = linearSize/distances[i]*206265
  return zVals, angSize

def LookBackTime(zMax,omegaM,omegaL,nSteps,H_0,omegaR=0.0):
  """
    equation 30, Hogg
    H_0 is the Hubble constant in km/s/Mpc
    implements numeric integration by adding up small increments
    note: this is not conformal time.  
      For conformal time, use comoving distance and divide by c
    returns two arrays: one of redshifts and another of lookback times in Gyrs
      of length nSteps+1, including z=0 and z=zMax
  """
  kmPerMpc = 3.086e19
  secPerYr = 3.157e7
  conv = kmPerMpc/secPerYr/1e9

  zVals = np.zeros(nSteps+1)
  times = np.zeros(nSteps+1)
  stepSize = (zMax+0.0)/nSteps
  # calculate integral values
  for i in range(1,nSteps+1):
    zVals[i] = i*stepSize
    times[i] = times[i-1] + stepSize/E(zVals[i],omegaM,0,omegaL,omegaR)/(1+zVals[i])/H_0
  # convert from seconds*Mpc/km to Gyrs
  return zVals, times*conv

def ComovingDistance(zMax,omegaM,omegaL,nSteps,H_0,omegaR=0.0):
  """
    H_0 is the Hubble constant in km/s/Mpc
    implements numeric integration by adding up small increments
    returns two arrays: one of redshifts and another of comoving distances in comoving Mpc
      of length nSteps+1, including z=0 and z=zMax
  """
  c_light = 2.998e5 #km/s

  zVals = np.zeros(nSteps+1)
  distances = np.zeros(nSteps+1)
  stepSize = (zMax+0.0)/nSteps
  # calculate integral values
  for i in range(1,nSteps+1):
    zVals[i] = i*stepSize
    distances[i] = distances[i-1] + stepSize/E(zVals[i],omegaM,0,omegaL,omegaR)/H_0
  # convert from seconds*Mpc/km to Mpc
  return zVals, distances*c_light




def makePlot(x,y,title,ylabel,loglog):
  """
    x,y are numpy arrays to plot against each other
    title : the title of the plot
    ylabel : the ylabel
    set loglog = 0 to use linear-linear axes, 
                 1 to use semilog (x log) axes,
                 otherwise will plot log-log axes
  """

  plt.subplot(1,1,1)
  if loglog == 0:
    plt.plot(x,y)
  elif loglog == 1:
    plt.semilogx(x,y)
  else:
    plt.loglog(x,y)
  plt.title(title)
  plt.xlabel("redshift z")
  plt.ylabel(ylabel)
  plt.show()


###############################################################################
# testing code

def test():
  """
    function for creating plots for Phys 267
  """

  # Cosmological parameters taken from WMAP paper
  #  Jarosik et al, 2011, ApJS, 192, 14
  H0 = 71.0 # Hubble constant in km/sec/Mpc
  omegaM = 0.222+0.0449 # omega_c + omega_b for DM and baryons
  omegaL = 0.734
  t0 = 13.75 # age of universe in Gyr

  # Cosmological parameters not mentioned in WMAP paper
  omegaK = 0  # zero curvature
  c = 2.998e5 # speed of light in kilometers per second

  # Derived parameters
  DH = c/H0  # hubble distance
  #tH = 1/H0  # hubble time



  print "This program plots several cosmological variables as a function"
  print "of redshift, z."

  zMax = 10
  nSteps = 1000
  # Create and plot angular diameter distances
  z_vals,D_ang = DAngular(zMax,omegaM,omegaL,nSteps,DH)
  makePlot(z_vals,D_ang,"Angular Diameter distance","Mpc",1)

  # Create and plot luminosity distances
  z_vals,D_lum = DLuminosity(zMax,omegaM,omegaL,nSteps,DH)
  makePlot(z_vals,D_lum,"Luminosity Distance","Mpc",1)

  # Create and plot apparent magnitudes
  z_vals,ap_magnitude = ApMagnitude(zMax,omegaM,omegaL,nSteps,DH)
  makePlot(z_vals,ap_magnitude,"apparent magnitude of M* = -20.6","apparent magnitude",2)
  #  next line for testing
  #makePlot(D_lum,ap_magnitude,"apparent magnitude of M* = -20.6","apparent magnitude",1)

  # Create and plot angular size in arcseconds for 1 kpc length
  z_vals,ang_size = AngSize(zMax,omegaM,omegaL,nSteps,DH)
  makePlot(z_vals,ang_size,"Angular Size of 1 kpc distance in arcsec","arcsec",2)

  # Create and plot lookback time
  z_vals, lbTime = LookBackTime(zMax,omegaM,omegaL,nSteps,H0)
  makePlot(z_vals,lbTime,"Lookback time","years * 10^9",1)
  ages = np.zeros(nSteps+1)
  for i in range(nSteps+1):
    ages[i]=t0-lbTime[i]
  makePlot(z_vals,ages,"Age of the universe","years * 10^9",1)


  # This section not part of Phys 267
  # Create and plot comoving distance
  z_vals, comDist = ComovingDistance(zMax,omegaM,omegaL,nSteps,H0)
  makePlot(z_vals,comDist,"Comoving Distance","Mpc",1)



if __name__=='__main__':
  test()


