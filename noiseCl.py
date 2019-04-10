#! /usr/bin/env python
"""
  Name:
    noiseCl
  Purpose:
    Calculate noise levels for power spectra: 
      kappa-kappa (lensing reconstruction)
      galaxy-galaxy (shot noise)
      temperature and polarization (detector noise)
  Uses:
    quicklens (for reconstruction noise)
      https://github.com/dhanson/quicklens
    crosspower (for galaxy bin structure)
  Modification History:
    Written by Z Knight, 2018.03.10
    Cosmetic change in noisePower; ZK, 2018.04.27
    Fixed nbar vs nbarSr typo in shotNoise function; ZK, 2018.05.08
    
"""
import numpy as np
import matplotlib.pyplot as plt
import crosspower as cp
import quicklens as ql

################################################################################
# reconstruction noise
# copy and minimized code using quicklens (from demo)

def getRecNoise(lmax,nlev_t,nlev_p,beam_fwhm):
    """
    Purpose:
        Calculate reconstruction noise
        Currently default noise is for EB reconstruction
    Inputs:
        lmax: maximum ell to return  !!!!!!!!!!!!!!!!!!!!!! re-do with different lmaxes to match M's paper !!!!!!!!!
        nlev_t: temperature noise level, in uK.arcmin.
        nlev_p: polarization noise level, in uK.arcmin.
        beam_fwhm: Gaussian beam full-width-at-half-maximum.
    Returns:
        ells: the ell values of the EB_noise array
        EB_noise: the EB noise power spectrum
    """
    # calculation parameters.
    #lmax       = 3000 # maximum multipole for T, E, B and \phi.
    #nx         = 512  # number of pixels for flat-sky calc.
    #dx         = 1./60./180.*np.pi # pixel width in radians.

    # params from Hu and Okamoto, 2002; "near perfect reference experiment"
    #nlev_t     = 1.   # temperature noise level, in uK.arcmin.
    #nlev_p     = 1.414   # polarization noise level, in uK.arcmin.
    #beam_fwhm  = 4.   # Gaussian beam full-width-at-half-maximum.

    # actually, use the same noise levels as a possible CMB-S4 design:
    #  CMBS4 v1
    #  fwhm = 1; ST = 1; SP = ST*1.414
    #nlev_t     = 1.   # temperature noise level, in uK.arcmin.
    #nlev_p     = 1.414   # polarization noise level, in uK.arcmin.
    #beam_fwhm  = 1.   # Gaussian beam full-width-at-half-maximum.


    prefix = 'FisherCl_match'
    #cl_unl     = ql.spec.get_camb_scalcl(lmax=lmax,prefix=prefix) # unlensed theory spectra.
    cl_len     = ql.spec.get_camb_lensedcl(lmax=lmax,prefix=prefix) # lensed theory spectra.

    bl         = ql.spec.bl(beam_fwhm, lmax) # transfer function.
    #pix        = ql.maps.pix(nx,dx)

    # noise spectra
    #nltt       = (np.pi/180./60.*nlev_t)**2 / bl**2
    nlee=nlbb  = (np.pi/180./60.*nlev_p)**2 / bl**2

    # signal spectra
    #sltt       = cl_len.cltt
    #slte       = cl_len.clte
    slee       = cl_len.clee
    slbb       = cl_len.clbb
    zero       = np.zeros(lmax+1)

    # signal+noise spectra
    #cltt       = sltt + nltt
    clee       = slee + nlee
    clbb       = slbb + nlbb

    # filter functions
    #flt        = np.zeros( lmax+1 ); flt[2:] = 1./cltt[2:]
    fle        = np.zeros( lmax+1 ); fle[2:] = 1./clee[2:]
    flb        = np.zeros( lmax+1 ); flb[2:] = 1./clbb[2:]

    # intialize quadratic estimators
    #qest_TT    = ql.qest.lens.phi_TT(sltt)
    #qest_EE    = ql.qest.lens.phi_EE(slee)
    #qest_TE    = ql.qest.lens.phi_TE(slte)
    #qest_TB    = ql.qest.lens.phi_TB(slte)
    qest_EB    = ql.qest.lens.phi_EB(slee)

    watch = ql.util.stopwatch()
    # take some pieces from function calc_nlqq
    qest, clXX, clXY, clYY, flX, flY = qest_EB, clee, zero, clbb, fle, flb 

    errs = np.geterr(); np.seterr(divide='ignore', invalid='ignore')

    print "[%s]"%watch.elapsed(), "calculating full-sky noise level for estimator of type", type(qest)
    clqq_fullsky = qest.fill_clqq(np.zeros(lmax+1, dtype=np.complex), clXX*flX*flX, clXY*flX*flY, clYY*flY*flY)
    resp_fullsky = qest.fill_resp(qest, np.zeros(lmax+1, dtype=np.complex), flX, flY)
    nlqq_fullsky = clqq_fullsky / resp_fullsky**2

    np.seterr(**errs)

    nlpp_EB_fullsky = nlqq_fullsky
    # end of copies from calc_nkqq

    ls         = np.arange(0,lmax+1)
    #print ls
    #print 'done.'
    
    return ls, nlpp_EB_fullsky




################################################################################
# CMB noise

# These implement equation 26 from CosmicFish implementation notes V1.0
def beamSig2(fwhm1,fwhm2):
    """
    Input:
        fwhm1,2: the beam FWHMs in arcminutes
          For TT or EE, these will be the same, for TE they may not be
    Returns:
        fwhm scaled and squared
    """
    return fwhm1*fwhm2* (np.pi/10800)**2 / (8*np.log(2))

def beamVar(fwhm1,fwhm2,ST1,ST2):
    """
    Inputs:
        fwhm1,2: the beam FWHMs in arcminutes
          For TT or EE, these will be the same, for TE they may not be
        ST1,2: DeltaT/T, the temperature sensitivities in 10^-6 microK/K
          (See table 1.3 from Planck bluebook)
          Or, this can be used to represent SP, the polarization sensitivity
          For TT or EE, these will be the same, for TE they may not be
    Returns:
        sensitivity squared, in (10^-6 microK*radians)^2 (huh?)
    """
    Tcmb = 2.7260   # +/- 0.0013; Fixsen 2009
    return ST1*fwhm1* ST2*fwhm2* (Tcmb*np.pi/10800)**2

def noisePowerMultiChannel(fwhm1s,fwhm2s,ST1s,ST2s,ells):
    """
    Purpose:
        Find the total noise power for a multi-channel CMB instrument
    Inputs:
        fwhm1s,2s: arrays of FWHMs
          Length is for the number of channels
        ST1s,S2s: arrays of STs
          Length is for the number of channels
        ells: array of ell values to calculate noise at
    Returns:
        Array of total noise power at each ell value
    """
    nChannels = fwhm1s.__len__()
    mySums = np.zeros(ells.__len__()) 

    for chInd in range(nChannels):
        myBeamSig2 = beamSig2(fwhm1s[chInd],fwhm2s[chInd])
        myBeamVar  = beamVar(fwhm1s[chInd],fwhm2s[chInd],ST1s[chInd],ST2s[chInd])
        for ellIndex,ell in enumerate(ells):
            mySums[ellIndex] += np.exp(-1*ell*(ell+1)*myBeamSig2)/myBeamVar
    return 1./mySums
    

# this function is for noise on the single temperature map level
#  from CMB-S4 science book (first edition) equation 8.23
def noisePower(ST1,ST2,fwhm,ells):
    """
    Purpose:
        Find the total noise power for a single-channel CMB instrument
    Inputs:
        ST1,ST2: the map sensitivity DeltaT (or DeltaP) in microK-arcmin
          For TT or EE these will be the same, for TE they will not
        fwhm: the FWHM of the beam in arcmin
        ells: array of ell values to calculate noise at
    Returns:
        Array of total noise power at each ell value in microK^2
    """
    arcminPerRad = 60*180/np.pi

    noise_l = np.empty(ells.__len__(),dtype='float64')
    fwhmFac = (fwhm/arcminPerRad)**2/(8*np.log(2))
    #print 'fwhmFac: ',fwhmFac
    for ellIndex, ell in enumerate(ells):
        noise_l[ellIndex] = ST1*ST2/arcminPerRad**2 *np.exp(ell*(ell+1)*fwhmFac)

    return noise_l



################################################################################
# galaxy shot noise

def shotNoise(nbar,binEdges,myDNDZ=cp.modelDNDZ3,nz=100000,verbose=False):
    """
    Purpose: calculate galaxy shot noise power by bin
        Nl^gg = 1/n_bar, where n_bar is number of galaxies per steradian
    Example:
        From Schaan et. al.: LSST n_source = 26/arcmin^2 for full survey
    Inputs:
        nbar: galaxy density in arcmin^-2
        binEdges: the edges of the bins.  
          There must be no bin overlap, and no space in between them.
        myDNDZ: Which dn/dz function to use
          Note: this can be a funciton of only one parameter: z
          Default: cp.modelDNDZ3
        nz: The number of z points to use in approximating the integrals 
          for measuring areas
    Returns:
        N_gg: an array containing the noise level for each bin
          Will be the same length as binEdges-1
    """
    nBins = binEdges.__len__()-1
    binAreas = np.zeros(nBins)

    # convert arcmin^-2 to sr^-1 : use (arcmin/rad)^2
    arcminPerRad = 60*180/np.pi
    nbarSr = nbar*(arcminPerRad)**2
    if verbose:
        print 'nbar: ',nbar,'nbarSr: ',nbarSr

    myZs = np.linspace(binEdges[0],binEdges[-1],nz)
    normPoints = 1  #number of points between points in myZs
    for binNum in range(nBins):
        binAreas[binNum] = 1./cp.normBin(myDNDZ,binEdges[binNum],
                                         binEdges[binNum+1],myZs,normPoints)
    if verbose:
        print 'binAreas: ',binAreas

    areaSum = np.sum(binAreas)
    
    binFractions = binAreas/areaSum
    if verbose:
        print 'bin fractions: ',binFractions,', total: ',np.sum(binFractions)
    
    # get n_bar and Noise for each bin
    binNbars = nbarSr*binFractions
    N_gg = 1./binNbars
    if verbose:
        print 'N_gg: ',N_gg
    
    return N_gg


################################################################################
# plotting code

def plotRecNoise():
    """
    Purpose: plot reconstruction noise power
    
    """
    # noise levels from a possible CMB-S4 design:
    nlev_t     = 1.   # temperature noise level, in uK.arcmin.
    nlev_p     = 1.414   # polarization noise level, in uK.arcmin.
    beam_fwhm  = 1.   # Gaussian beam full-width-at-half-maximum.
    
    lmax = 3000
    ells,EB_noise = getRecNoise(lmax,nlev_t,nlev_p,beam_fwhm)
    
    t = lambda l: (l*(l+1.))**2/(2.*np.pi)  # scaling to apply to cl_phiphi when plotting.

    #cl_unl.plot( 'clpp', t=t, color='k' ) # this would plot the cl_phiphi power spectrum if cl_unl is defined
    plt.loglog(ells, t(ells)*EB_noise, color='m', label=r'EB' )

    plt.xlim(2,2048)
    plt.ylim(5e-10,5e-7)
    plt.legend(loc='upper left', ncol=2)
    plt.setp( plt.gca().get_legend().get_frame(), visible=False)

    plt.xlabel(r'$L$')
    plt.ylabel(r'$[L(L+1)]^2 C_L^{\phi\phi} / 2\pi$') # should be dd, not phiphi?

    #plt.ion()
    plt.show()


def plotNoisePower():
    """
    Purpose: plot noise powers to verify code accuracy
        Data from table 1 and comparison is against figure 1 from
          Ma, 2017; arxiv:1707.03348

    """
    # select some ell values to use; don't need them all!
    myElls = np.floor(np.logspace(0,4,100))

    def atr(arcmin):
        # converts arcmin to radians
        return arcmin*np.pi/10800

    # get noise powers
    print 'calculating temperature noise powers...'
    # Planck
    fwhm = 5; ST = 47
    noisePlanck = noisePower(ST,ST,fwhm,myElls)
    # CMBS4 v1
    fwhm = 3; ST = 3
    noiseCMBS41 = noisePower(ST,ST,fwhm,myElls)
    # CMBS4 v2
    fwhm = 1; ST = 3
    noiseCMBS42 = noisePower(ST,ST,fwhm,myElls)
    # CMBS4 v3
    fwhm = 3; ST = 1
    noiseCMBS43 = noisePower(ST,ST,fwhm,myElls)
    # CMBS4 v4
    fwhm = 1; ST = 1
    noiseCMBS44 = noisePower(ST,ST,fwhm,myElls)

    # get CMB T spectrum
    #print 'skipping CMB T for now.'

    # plot them
    ellConv = myElls*(myElls+1)/(2*np.pi)
    plt.loglog(myElls,noisePlanck*ellConv,label='Planck')
    plt.loglog(myElls,noiseCMBS41*ellConv,label='CMB-S4 v1')
    plt.loglog(myElls,noiseCMBS42*ellConv,label='CMB-S4 v2')
    plt.loglog(myElls,noiseCMBS43*ellConv,label='CMB-S4 v3')
    plt.loglog(myElls,noiseCMBS44*ellConv,label='CMB-S4 v4')


    plt.ylim((1e-8,3e4))
    plt.xlim((2,1e4))
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\ell(\ell+1)N_{\ell} / 2\pi [\mu K^2]$')
    plt.title('Temperature noise model for Planck and CMB-S4 versions')
    plt.legend()
    print 'plotting'
    plt.show()
    print 'done'


def plotNoisePowerMulti():
    """
    Purpose: plot noise powers to verify code accuracy
    Uses: noisePowerMultiChannel

    """
    # some testing data for the most sensitive / smallest beam Planck detectors
    # Similar to Planck Bluebook values, but actually from CosmicFish
    # 100 GHz channel
    ST100 = 2.5
    PT100 = 4.0
    FWHM100 = 10.0
    # 143 GHz channel
    ST143 = 2.2
    PT143 = 4.2
    FWHM143 = 7.1
    # 217 GHz channel
    ST217 = 4.8
    PT217 = 9.8
    FWHM217 = 5.0

    # form into vectors that noisePowerMultiChannel wants
    ST_vec = [ST100,ST143,ST217]
    PT_vec = [PT100,PT143,PT217]
    FWHM_vec = [FWHM100,FWHM143,FWHM217]

    # select some ell values to use; don't need them all!
    myElls = np.floor(np.logspace(0,4,100))

    #TT
    NlTT = noisePowerMultiChannel(FWHM_vec,FWHM_vec,ST_vec,ST_vec,myElls)
    #TP
    NlTP = noisePowerMultiChannel(FWHM_vec,FWHM_vec,ST_vec,PT_vec,myElls)
    #PP
    NlPP = noisePowerMultiChannel(FWHM_vec,FWHM_vec,PT_vec,PT_vec,myElls)

    # plot them
    ellConv = myElls*(myElls+1)/(2*np.pi)
    plt.loglog(myElls,NlTT*ellConv,label=r'$N_\ell^{TT}$')
    plt.loglog(myElls,NlTP*ellConv,label=r'$N_\ell^{TP}$')
    plt.loglog(myElls,NlPP*ellConv,label=r'$N_\ell^{PP}$')

    plt.ylim((1e-8,3e4))
    plt.xlim((2,1e4))
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\ell(\ell+1)N_{\ell} / 2\pi [\mu K^2]$')
    plt.title('Temperature noise model for Planck multi-channel T,P')
    plt.legend()
    plt.show()

def plotShotNoise(nbar=66):
    """
    Purpose: plot the bins used to calculate shot noise and print related stuff
    Inputs:
        nbar: the galaxy density in arcmin^-2
    
    """
    #From Schaan et. al.: LSST n_source = 26/arcmin^2 for full survey
    #nbar = 26*(arcminPerRad)**2 # sr^-1
    #nbar = 66*(arcminPerRad)**2 # sr^-1  # 66 to match Bye's value
    #nbar = 66 # arcmin^-2
    
    # For consistency with Fisher_7_6X.obj: be compatible with cp.tophat beesBins=True; modelDNDZ3
    #binEdges = [0.0,0.5,1.0,2.0,3.0,4.0,7.0]
    binEdges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 
                  2.3, 2.6, 3.0, 3.5, 4.0, 7.0]
    nBins = binEdges.__len__()-1
    myDNDZ = cp.modelDNDZ
    #myDNDZ = cp.modelDNDZ3
    
    N_gg = shotNoise(nbar,binEdges,myDNDZ=myDNDZ,verbose=True)
    print 'N_gg = ',N_gg

    # plot the bins
    myZs = np.linspace(0,7,100000)
    fullDNDZ = myDNDZ(myZs)
    for binNum in range(nBins):
        myBin = cp.tophat(fullDNDZ,myZs,binEdges[0],binEdges[-1],nBins,binNum+1,beesBins=True)
        #plt.plot(myZs,myBin)
        plt.semilogy(myZs,myBin)
        plt.xlabel('redshift')
        plt.ylabel('dN/dz')
        plt.title('galaxy distribution')
    plt.show()


################################################################################
# testing code

def test(doPlot = True):
    """

    """

    # test __file__
    print 'file: ',__file__,'\n'

    # test the reconstruction noise
    plotRecNoise()
    
    # test the detector noise
    plotNoisePower()
    plotNoisePowerMulti()

    # test the shot noise
    plotShotNoise()

if __name__=='__main__':
    test()


