from lines_library import *

import numpy as np 
from scipy.constants import h,k,c
from iminuit import Minuit
from matplotlib import pylab as plt 
from astropy.modeling.models import BlackBody1D
from scipy import stats
from scipy.optimize import curve_fit


                            #####################################################
                            #                                                   #
                            #                                                   #
                            #          Base functions of the Spec fitter        #
                            #                                                   #
                            #                                                   #
                            #####################################################




def get_blackbody(lbda_a, temperature_k, ampliture=1, redshift=0, norm = 1e-14):
    """ This function creates a black body spectrum based on a wavelength array from the considered spectrum, 
        temperature and redshift (and a potential normalisation factor)
    
    Parameters
    ----------
    lbda_a: [array]
        wavelength array in angstrom
        
    temperature_k: [float]
        black body temperature in kelvin
        
    amplitude: [float] -optional-
        amplitude of the blackbody flux
        
    redshift: [float] -optional-
        redshift correction applied to the wavelength array. 
        /!\ Does not affect the flux since we consider that redshifts are << 1
    
    norm: [float] -optional-
        normalisation factor of the flux to correct for very small numbers
        
    Returns
    -------
    array (flux in erg/s/cm2/m)
    """
    lam = 1e-10*lbda_a*(1+redshift) # convert to metres and correct for redshift
    #lam = lbda_a*(1+redshift) # don't convert to meters only correct for redshidt
    #we should not use the redshift since we are correcting the spectrum for redshift before the fitting

    #idea : in the get blackbody, multuply it by a legendre polynom and see what comes out? The ocontinuum will never be an actual blackbody radiation
    
    return ampliture*norm* (2*h*c**2 / lam**5) * (1/ (np.exp(h*c / (lam*k*temperature_k)) - 1))




def get_3rd_ord_pol_continuum(lbda_a, coef_0, coef_1, coef_2, coef_3):
    '''
    This function creates a 3rd order polynomial to fit the continuum of a spectrum whenever it is not clear that it is a blackbody emission spectrum

    Parameters
    ----------


    Returns
    -------
    '''

    return coef_0*(lbda_a)**3   +  coef_1*(lbda_a)**2  +  coef_2*lbda_a  +  coef_3





def get_emission_lines(lbda_a, sigma, redshift, 
                       include=["HeII_1","Hbeta"], 
                       amplitude=1):
    """ (cf. LINES, supposedly for flash features mainly and Halpha)
        Creates an emission line profile with a gaussian shape for the
        inputted wavelength array for each relevant lines (see include) and adds them. 
        (Verfies that you're providing for each line requested an amplitude, a sigma). 
        It returns a spectrum with each requested lines on a flat continuum (within the wavelength range specified
        by lbda_a)
        
    
    Parameters
    ----------
    lbda_a: [array]
        wavelength array in angstrom
        
    sigma: [float]
        scale of the gaussian lines (FWHM?)
        
    redshift: [float]
        reshift of the spectrum to be applied to match the potential lines in the spectrum
        
    include: [list of string] -optional-
        Lines you want to use. (See LINES)
        
    amplitude: [float or list of] -optional-
        the amplitude of each line included (see include)
        remark: if only a single float given, all lines will have the same
        
    Returns
    -------
    array (flux in erg/s/cm2/A)  <- à vérifier, a priori ça fait sens, mais pas convaincue des unités 
    """
    nlines    = len(include)
    # Manage amplitudes
    amplitude = np.atleast_1d(amplitude) # transforms amplitude in an array of at least 1d for further uses
    if len(amplitude)!=1 and len(amplitude) != nlines:
        raise ValueError("you gave %d amplitude, but there are %d lines"%(len(amplitude),nlines) )
        
    # Manage sigma
    sigma = np.atleast_1d(sigma)
    if len(sigma)!=1 and len(sigma) != nlines:
        raise ValueError("you gave %d sigma, but there are %d lines"%(len(sigma),nlines) )
        
    # Get the centroids at the right redshift
    mus = np.asarray([LINES[k] for k in include])*(1+redshift) # <- redshift corecction
    # All gaussian lines (mus[:,None] -> broadcasting)
    lines = stats.norm.pdf( lbda_a, loc=mus[:,None], scale=sigma[:,None]) * amplitude[:,None]
    #lines = stats.cauchy.pdf( lbda_a, loc=mus[:,None], scale=sigma[:,None]) * amplitude[:,None]
    
    return np.sum(lines, axis=0)


def get_bump(lbda_a, amplit_bump, redshift):
    '''
    This function is based on the estimation of the flash bump from the early WHT spectrum of ZTF19abeajml.
    The bump is described as a polynom of order 3 between 4228Å and 4794Å (REST FRAME). 
    We are trying to test the appearance of this feature for an early time spectrum. The function is such
    that it returns the polynom for this interval and 0 otherwise.
    This is for now puyrely empirical and will be subject to change.
    
    parameters
    ----------
    lbda_a [array] wavelength input in angstrom
    amplit_bump [float] amplitude of the bump 
    redshift [float] redshift of the supernova. This is a FIXED parameter 
    
    returns
    -------
    flux [array] of the bump. POlynomial(3) bump for [4228*(1+z), 4794*(1+z)] Å. 
    
    '''
    
    #polyf = np.poly1d([-6.69441966e-09,  8.90864300e-05, -3.94497402e-01,  5.81400190e+02])
    polyf = np.poly1d([-5.59098658e-09,  7.40214783e-05, -3.25939489e-01,  4.77393907e+02])
    bump = polyf(lbda_a)
    
    b =(lbda_a> 4100*(1+redshift))    
    a =(lbda_a< 4800*(1+redshift))

    bump = bump*a*b
    
    return amplit_bump*bump


    #need to set boundaries... I don't trust the idea of extrapolation with this...
    
    #polyf = np.poly1d([-4.01007130e-09,  5.49359465e-05, -2.50347644e-01,  3.79595401e+02])
    


def get_model_wbump(lbda_a, sigma, redshift, line_amplitudes, bb_temp, bb_amplitude, bump_amplitude,
             include=["HeII_1","Hbeta"]):
    
    """ This function creates the full spectral model (blackbody continuum + lines). 
        For a specified wavelength aray and blackbody temperature, it returns a (synthetic) of a BB with specified 
        lines (see. include)
    
    Parameters
    ----------
    lbda_a: [array]
        wavelength array in angstrom
        
    sigma: [float]
        scale of the gaussian lines (FWHM?)
        
    redshift: [float]
        reshift of the spectrum to be applied to match the potential lines in the spectrum
        
    line_amplitudes: [float or array?] 
        amplitude for each selected lines. if only a float is inputted all the lines will have the same strength
        
    bb_amplitudes: [float] 
        amplitude of the BB
    
    bb_temp: [float]
        Temperature of the BB spectrum
        
    bump_amplitude: [float]
        AMplitude of the flash bump
        
    include: [list of strings] - optional -
        Lines you want to use. (See LINES) by default it's ["HeII_1","CIV","Halpha","Hbeta"]
    
    
    Returns
    -------
        array (flux in erg/s/cm2/A)  
    
    """
    bb    = get_blackbody(lbda_a, bb_temp ,ampliture=bb_amplitude, redshift = redshift)
    lines = get_emission_lines(lbda_a, sigma, redshift, amplitude=line_amplitudes, 
                              include=include)
    bump  = get_bump(lbda_a, bump_amplitude, redshift )
    return bb + lines + bump



def get_model_wobump(lbda_a, sigma, redshift, line_amplitudes, bb_temp, bb_amplitude,
             include=["HeII_1","Hbeta"]):
    
    """ This function creates the full spectral model (blackbody continuum + lines). 
        For a specified wavelength aray and blackbody temperature, it returns a (synthetic) of a BB with specified 
        lines (see. include)
    
    Parameters
    ----------
    lbda_a: [array]
        wavelength array in angstrom
        
    sigma: [float]
        scale of the gaussian lines (FWHM?)
        
    redshift: [float]
        reshift of the spectrum to be applied to match the potential lines in the spectrum
        
    line_amplitudes: [float or array?] 
        amplitude for each selected lines. if only a float is inputted all the lines will have the same strength
        
    bb_amplitudes: [float] 
        amplitude of the BB
    
    bb_temp: [float]
        Temperature of the BB spectrum
        
    bump_amplitude: [float]
        AMplitude of the flash bump
        
    include: [list of strings] - optional -
        Lines you want to use. (See LINES) by default it's ["HeII_1","CIV","Halpha","Hbeta"]
    
    
    Returns
    -------
        array (flux in erg/s/cm2/A)  
    
    """
    bb    = get_blackbody(lbda_a, bb_temp ,ampliture=bb_amplitude, redshift = redshift)
    lines = get_emission_lines(lbda_a, sigma, redshift, amplitude=line_amplitudes, 
                              include=include)
   
    return bb + lines


def get_model_polybump(lbda_a, redshift, coefs, bump_ampl):
    '''
    rgus fynction blabla

    parameters
    ----------


    returns
    -------

    '''

    bump       = get_bump(lbda_a,amplit_bump = bump_ampl , redshift = redshift)
    continuum  = get_3rd_ord_pol_continuum(lbda_a, coefs[0], coefs[1], coefs[2], coefs[3])
    return bump+continuum   





def get_param_model_wbump_(lbda_a, param):
    '''looks for the parameters and returns the models using the function "get_model" 
    
    Parameters
    ----------
    lbda_a [array] : wavelentgth
    
    param [list?]  : list of all the parameters needed to create the model (sigma, redshift, amplitudes,  
                     blackbody amplitude, blackbody temperature) 
    
    Returns
    -------
    model flux [array] : a model of BB continuum + lines for the given parameters
    
    
    '''
  
    sigma, redshift, *amplitudes, bb_temp, bb_amplitude , bump_amplitude = param
    return get_model_wbump(lbda_a, sigma=sigma, redshift=redshift,line_amplitudes=amplitudes, bb_temp=bb_temp, bb_amplitude=bb_amplitude, bump_amplitude = bump_amplitude )


def get_param_model_wo_bump_(lbda_a, param):
    '''looks for the parameters and returns the models using the function "get_model" 
    
    Parameters
    ----------
    lbda_a [array] : wavelentgth
    
    param [list?]  : list of all the parameters needed to create the model (sigma, redshift, amplitudes,  
                     blackbody amplitude, blackbody temperature) 
    
    Returns
    -------
    model flux [array] : a model of BB continuum + lines for the given parameters
    
   
    '''
  
    sigma, redshift, *amplitudes, bb_temp, bb_amplitude  = param
    return get_model_wobump(lbda_a, sigma=sigma, redshift=redshift,line_amplitudes=amplitudes, bb_temp=bb_temp, bb_amplitude=bb_amplitude )



def get_param_model_poly_(lbda_a, param):
    '''
    this functions...bla

    parameters
    ----------

    returns
    -------
    '''
    redshift, *coefs = param
    return get_3rd_ord_pol_continuum(lbda_a, coefs[0], coefs[1], coefs[2], coefs[3])


def get_param_model_polybump_(lbda_a, param):
    '''
    this functions...bla

    parameters
    ----------

    returns
    -------
    '''
    redshift, *coefs, bump_ampl = param
    return get_model_polybump(lbda_a, redshift = redshift, coefs = coefs, bump_ampl = bump_ampl)

    


##########################################################################################################################

def get_model_wbump_wolines(lbda_a,  redshift,  bb_temp, bb_amplitude, bump_amplitude):
    
    """ This function creates the full spectral model (blackbody continuum + lines). 
        For a specified wavelength aray and blackbody temperature, it returns a (synthetic) of a BB with specified 
        lines (see. include)
    
    Parameters
    ----------
    lbda_a: [array]
        wavelength array in angstrom
        
    sigma: [float]
        scale of the gaussian lines (FWHM?)
        
    redshift: [float]
        reshift of the spectrum to be applied to match the potential lines in the spectrum
        
    line_amplitudes: [float or array?] 
        amplitude for each selected lines. if only a float is inputted all the lines will have the same strength
        
    bb_amplitudes: [float] 
        amplitude of the BB
    
    bb_temp: [float]
        Temperature of the BB spectrum
        
    bump_amplitude: [float]
        AMplitude of the flash bump
        
    include: [list of strings] - optional -
        Lines you want to use. (See LINES) by default it's ["HeII_1","CIV","Halpha","Hbeta"]
    
    
    Returns
    -------
        array (flux in erg/s/cm2/A)  
    
    """
    bb    = get_blackbody(lbda_a, bb_temp ,ampliture=bb_amplitude, redshift = redshift)
#     lines = get_emission_lines(lbda_a, sigma, redshift, amplitude=line_amplitudes, 
#                               include=include)
    bump  = get_bump(lbda_a, bump_amplitude, redshift )
    return bb + bump



def get_model_wobump_wolines(lbda_a, redshift, bb_temp, bb_amplitude):
    
    """ This function creates the full spectral model (blackbody continuum + lines). 
        For a specified wavelength aray and blackbody temperature, it returns a (synthetic) of a BB with specified 
        lines (see. include)
    
    Parameters
    ----------
    lbda_a: [array]
        wavelength array in angstrom
        

    redshift: [float]
        reshift of the spectrum to be applied to match the potential lines in the spectrum
        
    bb_amplitudes: [float] 
        amplitude of the BB
    
    bb_temp: [float]
        Temperature of the BB spectrum
        
    bump_amplitude: [float]
        AMplitude of the flash bump
        

    
    Returns
    -------
        array (flux in erg/s/cm2/A)  
    
    """
    bb    = get_blackbody(lbda_a, bb_temp ,ampliture=bb_amplitude, redshift = redshift)
#     lines = get_emission_lines(lbda_a, sigma, redshift, amplitude=line_amplitudes, 
#                               include=include)
   
    return bb 


def get_param_model_wbump_wolines(lbda_a, param):
    '''looks for the parameters and returns the models using the function "get_model" 
    
    Parameters
    ----------
    lbda_a [array] : wavelentgth
    
    param [list?]  : list of all the parameters needed to create the model (sigma, redshift, amplitudes,  
                     blackbody amplitude, blackbody temperature) 
    
    Returns
    -------
    model flux [array] : a model of BB continuum + lines for the given parameters
    
    Examples:
    ---------
    '''
  
    redshift, bb_temp, bb_amplitude , bump_amplitude = param
    return get_model_wbump(lbda_a, redshift=redshift,bb_temp=bb_temp,bb_amplitude=bb_amplitude, bump_amplitude = bump_amplitude)


def get_param_model_wo_bump_wolines(lbda_a, param):
    '''looks for the parameters and returns the models using the function "get_model" 
    
    Parameters
    ----------
    lbda_a [array] : wavelentgth
    
    param [list?]  : list of all the parameters needed to create the model (sigma, redshift, amplitudes,  
                     blackbody amplitude, blackbody temperature) 
    
    Returns
    -------
    model flux [array] : a model of BB continuum + lines for the given parameters
    
    Examples:
    ---------
    '''
  
    redshift, bb_temp, bb_amplitude  = param
    return get_model_wobump(lbda_a, redshift=redshift,bb_temp=bb_temp, bb_amplitude=bb_amplitude )
   







###########################################################################################################################################################################################################################






                                        #####################################################
                                        #                                                   #
                                        #                                                   #
                                        #          Class for Model definition               #
                                        #            BB continuum + Lines                   #
                                        #                                                   #
                                        #####################################################




class BBLinesModel( object ):
    
    '''
    BBlinesModel 
    what type of object: it's a flux / lambda defined here like a continuum+lines
    the object itself will be an array, given a set a wl (?)
    '''
    FREEPARAMETERS = ["bb_temp","bb_ampl",
                     "sigma","redshift","amplitudes"]
    
    
    # ============== #
    #  Methods       #
    # ============== #
    
    def __init__(self, lbda=None, guess=None):
        """ This function initialises the parameters of the object (the wavelength range and e.g. reshdift, 
            sigma of the gaussian which define the lines...etc)
            
        Parameters
        ----------
        
        self [object] 
        
        lbda [array]  : wavelength range to consider
        
        guess [array] : initial guesses in sigma, amplitude and redshift that the model will take into account.
        
        
        Returns
        -------
        
        self [array]: parameters and wavelength range to consider
        
        """
        if lbda is not None:
            self.set_lbda(lbda)
            
        if guess is not None:
            self.set_param(guess)
            
    # ------- #
    # SETTER  #
    # ------- #
    
    def set_usedlines(self, line_to_use):
        """
        Sets the lines considered to include to the model (which need to be fit)
        
        Parameters
        ----------
        self []: the object
        line_to_use [list of strings] : Name of the lines. cf. LINES 
        
        
        Returns
        -------
        Internal variable(?)
        self._usedlines [attribute ? list of strings?]
        
        
        
        """
        self._usedlines = line_to_use
        
        
    def set_lbda(self, lbda):
        """
        Sets the wavelength range over which the model will be computed
        
        Parameters
        ----------
        self [object]
        
        lbda [array] : wavelength
        
        Returns
        -------
        
        self.lbda [array] : attribute of the object (from the class)
        
        """
        
        self.lbda = np.asarray(lbda)
        
    def set_parameters(self, param):
        """
        
        """
        sigma, redshift, *amplitudes ,bb_temp,bb_amplitude  = param
        #redshift = z
        self.parameters = {"sigma":sigma,
                           "amplitudes":amplitudes,
                           "redshift":redshift,
                           "bb_temp":bb_temp,
                           "bb_amplitude":bb_amplitude 
                          }
        
        
    # ------- #
    # GETTER  #
    # ------- #
    def get_model(self, params=None):
        """ 
        This function returns the full model used to fit a spectrum (A*BB(lbda,T)+B*Bump(lbda)+C*lines(lbda,sigma))
        
        
        parameters
        ----------
        
        returns
        -------
        
        
        """
        if params is not None:
            self.set_parameters(params)
            
        return self.get_continuum()+self.get_emission_lines()
        
    def get_continuum(self, params=None):
        """ 
        This function gets the blackbody continuum emissoin for a given set of parameters. Note that the blackbody was
        normalized to 10**0 scale
        
        parameters
        ----------
        
        returns
        -------
        
        """
        if params is not None:
            temp, ampl, redshift = params
        else:
            temp, ampl, redshift = self.parameters["bb_temp"], self.parameters["bb_amplitude"], self.parameters["redshift"]
            
        return get_blackbody( self.lbda, temp, ampl, redshift)
    
    def get_emission_lines(self, params=None):
        """ 
        This function calls an external function of the same name (this should be changed), based on the params
        provided, it gets the name of the lines we want to use and sets their parameters (FWHM, amplitudes, redshift)
        
        Disclaimer: the shape is not optimizable. you have to change it yourself (gaussian or cauchy)
        
        parameters
        ----------
        params [array] : [SIGMA *AMPLITUDE REDSHIFT]
        
        returns
        -------
        flux of emission lines [array] sizes like self.lbda, 
        
        """
        if params is not None:
            sigma, *amplitudes, redshift = params
        else:
            sigma, amplitudes, redshift = self.parameters["sigma"],self.parameters["amplitudes"], self.parameters["redshift"]
            
        return get_emission_lines(self.lbda, sigma, redshift,
                                   include=self.usedlines, 
                                  amplitude=amplitudes)
    
    
    
    
    # ------- #
    # PLOTTER #
    # ------- #
    
    #not sure if necessary, here we are just defining a model. makes more sense to have a plot option with the fitter object
        
    
    
    # ============== #
    #   Properties   #
    # ============== #
    @property
    def usedlines(self):
        """ 
        
        """
        if not hasattr(self, "_usedlines"):  
            self.set_usedlines(["HeII_1","Hbeta"]) #from the set
        return self._usedlines
    
    @property
    def nparam(self):
        """ 
        
        """
        return len(self.FREEPARAMETERS)

###########################################################################################################################################################################################################################  








                                    #####################################################
                                    #                                                   #
                                    #                                                   #
                                    #          Class for Model definition               #
                                    #          BB continuum + Lines + Bump              #
                                    #                                                   #
                                    #####################################################






class BBLinesBumpModel( object ):
    
    '''
    BBLinesBumpModel 
    This model is the bump model which includes a blackbody continuum, lines and the 
    theorised bump which corresponds to flash
    what type of object: it's a flux / lambda defined here like a continuum+lines + bump
    the object itself will be an array, given a set a wl (?)
    '''
    FREEPARAMETERS = ["bb_temp","bb_ampl",
                     "sigma","redshift", "amplitudes", "bump_amplitude"]
    
    
    # ============== #
    #  Methods       #
    # ============== #
    
    def __init__(self, lbda=None, guess=None):
        """ This function initialises the parameters of the object (the wavelength range and e.g. reshdift, 
            sigma of the gaussian which define the lines...etc)
            
        Parameters
        ----------
        
        self [object] 
        
        lbda [array]  : wavelength range to consider
        
        guess [array] : initial guesses in sigma, amplitude and redshift that the model will take into account.
        
        
        Returns
        -------
        
        self [array]: parameters and wavelength range to consider
        
        """
        if lbda is not None:
            self.set_lbda(lbda)
            
        if guess is not None:
            self.set_param(guess)
            
    # ------- #
    # SETTER  #
    # ------- #
    
    def set_usedlines(self, line_to_use):
        """
        Sets the lines considered to include to the model (which need to be fit)
        
        Parameters
        ----------
        self []: the object
        line_to_use [list of strings] : Name of the lines. cf. LINES 
        
        
        Returns
        -------
        Internal variable(?)
        self._usedlines [attribute ? list of strings?]
        
        
        
        """
        self._usedlines = line_to_use
        
        
    def set_lbda(self, lbda):
        """
        Sets the wavelength range over which the model will be computed
        
        Parameters
        ----------
        self [object]
        
        lbda [array] : wavelength
        
        Returns
        -------
        
        self.lbda [array] : attribute of the object (from the class)
        
        """
        
        self.lbda = np.asarray(lbda)
        
    def set_parameters(self, param):
        """
        this function sets the parameters of the model BB + Bump + Lines

        parameters 
        ----------
        param [array] : sigma          [float]
                        redshift       [float]
                        amplitudes     [list]
                        bb_temp        [float]
                        bb_amplitude   [float]    
                        bump_amplitude [float]
                    
        returns
        -------
        none, only sets the paramters of the object 
                        
        
        """
        sigma, redshift, *amplitudes ,bb_temp,bb_amplitude,bump_amplitude  = param
        #redshift = z
        self.parameters = {"sigma":sigma,
                           "amplitudes":amplitudes,
                           "redshift":redshift,
                           "bb_temp":bb_temp,
                           "bb_amplitude":bb_amplitude,
                           "bump_amplitude":bump_amplitude
                          }
    # ------- #
    # GETTER  #
    # ------- #
    def get_model(self, params=None):
        """ 
        This function returns the full model used to fit a spectrum (A*BB(lbda,T)+B*Bump(lbda)+C*lines(lbda,sigma))
        This requires the file "external_functions.py" which has the function
        
        parameters
        ----------
        
        returns
        -------
        
        
        """
        if params is not None:
            self.set_parameters(params)
            
        return self.get_continuum()+self.get_emission_lines()+self.get_bump()
        
    def get_continuum(self, params=None):
        """ 
        This function gets the blackbody continuum emissoin for a given set of parameters. Note that the blackbody was
        normalized to 10**0 scale
        
        parameters
        ----------
        
        returns
        -------
        
        """
        if params is not None:
            temp, ampl, redshift = params
        else:
            temp, ampl, redshift = self.parameters["bb_temp"], self.parameters["bb_amplitude"], self.parameters["redshift"]
            
        return get_blackbody( self.lbda, temp, ampl, redshift)
    
    def get_emission_lines(self, params=None):
        """ 
        This function calls an external function of the same name (this should be changed), based on the params
        provided, it gets the name of the lines we want to use and sets their parameters (FWHM, amplitudes, redshift)
        
        Disclaimer: the shape is not optimizable. you have to change it yourself (gaussian or cauchy)
        
        parameters
        ----------
        params -optional- [array] : [SIGMA *AMPLITUDE REDSHIFT]
        
        returns
        -------
        flux of emission lines [array] sizes like self.lbda, 
        
        """
        if params is not None:
            sigma, *amplitudes, redshift = params
        else:
            sigma, amplitudes, redshift = self.parameters["sigma"],self.parameters["amplitudes"], self.parameters["redshift"]
            
        return get_emission_lines(self.lbda, sigma, redshift,
                                   include=self.usedlines, 
                                  amplitude=amplitudes)
    
    
    def get_bump(self, params=None):
        '''
        this function gets the bump functionwhich constitutes the test for the presence of "flash" feature
        Spectra were not always obtained at the same period of time from explosion. Also, progenitors do not alwys
        have the same amount of CSM (from which flash features are supposed to originate from), hence we can't
        clearly identify which lines are present and it which quantity. We also supose that around 4700Å, there are
        several blended lines which are not always discernable. 
        
        parameters
        ----------
        params -optional- [array] : [AMPLITUDE_BUMP REDHSIFT]
        
        returns
        -------
        flux of the bump [array] with the same size as lbda. It should return only the polynom
        and padded by zero on the size.  (Again, because I do not trust extrapolation, we consider
        only the polynom in the interval it was defined)
        
        '''
        
        if params is not None:
            ampli_bump, redshift = params
            
        else:
            ampli_bump, redshift = self.parameters['bump_amplitude'], self.parameters['redshift']
            
        return get_bump(self.lbda, ampli_bump, redshift)
    
    
    # ------- #
    # PLOTTER #
    # ------- #
    
    # def plot_model_spectrum(self, lbda ,params, **kwargs):
    #     '''
    #     This plots a flux model based on a continuum and chosen emission lines
        
    #     Parameters
    #     ----------
        
    #     lbda [array] wavelenfth array 
        
    #     params [array]
        
        
    #     Returns
    #     -------
        
        
    #     '''
    #     self.spectrum = _get_param_model_wbump_(lbda, params)
    #     #plt.plot()
        
    
    
    # ============== #
    #   Properties   #
    # ============== #
    @property
    def usedlines(self):
        """ 
        
        """
        if not hasattr(self, "_usedlines"):  
            self.set_usedlines(["HeII_1","Hbeta"]) #from the set
        return self._usedlines
    
    @property
    def nparam(self):
        """ 
        
        """
        return len(self.FREEPARAMETERS)






################################################################################################################################################################






                                    #####################################################
                                    #                                                   #
                                    #                                                   #
                                    #          Class for Model definition               #
                                    #                BB continuum                       #
                                    #                                                   #
                                    #####################################################








class BBModel( object ):
    
    '''
    BBlinesModel 
    what type of object: it's a flux / lambda defined here like a continuum+lines
    the object itself will be an array, given a set a wl (?)
    '''
    FREEPARAMETERS = ["bb_temp","bb_ampl","redshift"]
    
    
    # ============== #
    #  Methods       #
    # ============== #
    
    def __init__(self, lbda=None, guess=None):
        """ This function initialises the parameters of the object (the wavelength range and e.g. reshdift, 
            sigma of the gaussian which define the lines...etc)
            
        Parameters
        ----------
        
        self [object] 
        
        lbda [array]  : wavelength range to consider
        
        guess [array] : initial guesses in sigma, amplitude and redshift that the model will take into account.
        
        
        Returns
        -------
        
        self [array]: parameters and wavelength range to consider
        
        """
        if lbda is not None:
            self.set_lbda(lbda)
            
        if guess is not None:
            self.set_param(guess)
            
    # ------- #
    # SETTER  #
    # ------- #
    

        
    def set_lbda(self, lbda):
        """
        Sets the wavelength range over which the model will be computed
        
        Parameters
        ----------
        self [object]
        
        lbda [array] : wavelength
        
        Returns
        -------
        
        self.lbda [array] : attribute of the object (from the class)
        
        """
        
        self.lbda = np.asarray(lbda)
        
    def set_parameters(self, param):
        """
        
        """
        redshift ,bb_temp,bb_amplitude  = param
        #redshift = z
        self.parameters = {
                           "redshift":redshift,
                           "bb_temp":bb_temp,
                           "bb_amplitude":bb_amplitude 
                          }
        
        
    # ------- #
    # GETTER  #
    # ------- #
    def get_model(self, params=None):
        """ 
        This function returns the full model used to fit a spectrum (A*BB(lbda,T)+B*Bump(lbda)+C*lines(lbda,sigma))
        
        
        parameters
        ----------
        
        returns
        -------
        
        
        """
        if params is not None:
            self.set_parameters(params)
            
        return self.get_continuum()
        
    def get_continuum(self, params=None):
        """ 
        This function gets the blackbody continuum emissoin for a given set of parameters. Note that the blackbody was
        normalized to 10**0 scale
        
        parameters
        ----------
        
        returns
        -------
        
        """
        if params is not None:
            temp, ampl, redshift = params
        else:
            temp, ampl, redshift = self.parameters["bb_temp"], self.parameters["bb_amplitude"], self.parameters["redshift"]
            
        return get_blackbody( self.lbda, temp, ampl, redshift)
    

    
    
    # ------- #
    # PLOTTER #
    # ------- #
    
    
    # ============== #
    #   Properties   #
    # ============== #

    @property
    def nparam(self):
        """ 
        
        """
        return len(self.FREEPARAMETERS)



######################################################################################################################################################




                                    #####################################################
                                    #                                                   #
                                    #                                                   #
                                    #          Class for Model definition               #
                                    #             BB continuum + Bump                   #
                                    #                                                   #
                                    #####################################################



class BBBumpModel( object ):
    
    '''
    BBLinesBumpModel 
    This model is the bump model which includes a blackbody continuum, lines and the 
    theorised bump which corresponds to flash
    what type of object: it's a flux / lambda defined here like a continuum+lines + bump
    the object itself will be an array, given a set a wl (?)
    '''
    FREEPARAMETERS = ["bb_temp","bb_ampl",
                     "redshift", "bump_amplitude"]
    
    
    # ============== #
    #  Methods       #
    # ============== #
    
    def __init__(self, lbda=None, guess=None):
        """ This function initialises the parameters of the object (the wavelength range and e.g. reshdift, 
            sigma of the gaussian which define the lines...etc)
            
        Parameters
        ----------
        
        self [object] 
        
        lbda [array]  : wavelength range to consider
        
        guess [array] : initial guesses in sigma, amplitude and redshift that the model will take into account.
        
        
        Returns
        -------
        
        self [array]: parameters and wavelength range to consider
        
        """
        if lbda is not None:
            self.set_lbda(lbda)
            
        if guess is not None:
            self.set_param(guess)
            
    # ------- #
    # SETTER  #
    # ------- #
    

        
    def set_lbda(self, lbda):
        """
        Sets the wavelength range over which the model will be computed
        
        Parameters
        ----------
        self [object]
        
        lbda [array] : wavelength
        
        Returns
        -------
        
        self.lbda [array] : attribute of the object (from the class)
        
        """
        
        self.lbda = np.asarray(lbda)
        
    def set_parameters(self, param):
        """
        
        """
        redshift,bb_temp,bb_amplitude,bump_amplitude  = param
        #redshift = z
        self.parameters = {
                           "redshift":redshift,
                           "bb_temp":bb_temp,
                           "bb_amplitude":bb_amplitude,
                           "bump_amplitude":bump_amplitude
                          }
    # ------- #
    # GETTER  #
    # ------- #
    def get_model(self, params=None):
        """ 
        This function returns the full model used to fit a spectrum (A*BB(lbda,T)+B*Bump(lbda)+C*lines(lbda,sigma))
        
        
        parameters
        ----------
        
        returns
        -------
        
        
        """
        if params is not None:
            self.set_parameters(params)
            
        return self.get_continuum()+self.get_bump()
        
    def get_continuum(self, params=None):
        """ 
        This function gets the blackbody continuum emissoin for a given set of parameters. Note that the blackbody was
        normalized to 10**0 scale
        
        parameters
        ----------
        
        returns
        -------
        
        """
        if params is not None:
            temp, ampl, redshift = params
        else:
            temp, ampl, redshift = self.parameters["bb_temp"], self.parameters["bb_amplitude"], self.parameters["redshift"]
            
        return get_blackbody( self.lbda, temp, ampl, redshift)
    

    
    
    def get_bump(self, params=None):
        '''
        this function gets the bump functionwhich constitutes the test for the presence of "flash" feature
        Spectra were not always obtained at the same period of time from explosion. Also, progenitors do not alwys
        have the same amount of CSM (from which flash features are supposed to originate from), hence we can't
        clearly identify which lines are present and it which quantity. We also supose that around 4700Å, there are
        several blended lines which are not always discernable. 
        
        parameters
        ----------
        params -optional- [array] : [AMPLITUDE_BUMP REDHSIFT]
        
        returns
        -------
        flux of the bump [array] with the same size as lbda. It should return only the polynom
        and padded by zero on the size.  (Again, because I do not trust extrapolation, we consider
        only the polynom in the interval it was defined)
        
        '''
        
        if params is not None:
            ampli_bump, redshift = params
            
        else:
            ampli_bump, redshift = self.parameters['bump_amplitude'], self.parameters['redshift']
            
        return get_bump(self.lbda, ampli_bump, redshift)
    
    

    
    # ============== #
    #   Properties   #
    # ============== #

    @property
    def nparam(self):
        """ 
        
        """
        return len(self.FREEPARAMETERS)

    
########################################################################################################################################################################################################################### 









                                    #####################################################
                                    #                                                   #
                                    #                                                   #
                                    #          Class for Model definition               #
                                    #       Poly(3)continuum + Bump + Lines              #
                                    #                                                   #
                                    #####################################################











class Poly3BumpModel( object ):
    
    
    ''' Poly3BumpModel 
    This model is the bump model which includes a blackbody continuum, lines and the 
    theorised bump which corresponds to flash
    what type of object: it's a flux / lambda defined here like a continuum+lines + bump
    the object itself will be an array, given a set a wl (?)

    '''
    
    FREEPARAMETERS = ["pol_coefs","redshift", "bump_amplitude"]
    
    
    # ============== #
    #  Methods       #
    # ============== #
    
    def __init__(self, lbda=None, guess=None):
        """ This function initialises the parameters of the object (the wavelength range and e.g. reshdift, 
            sigma of the gaussian which define the lines...etc)
            
        Parameters
        ----------
        
        self [object] 
        
        lbda [array]  : wavelength range to consider
        
        guess [array] : initial guesses in sigma, amplitude and redshift that the model will take into account.
        
        
        Returns
        -------
        
        self [array]: parameters and wavelength range to consider
        
        """
        if lbda is not None:
            self.set_lbda(lbda)
            
        if guess is not None:
            self.set_param(guess)
            
    # ------- #
    # SETTER  #
    # ------- #
    
        
        
    def set_lbda(self, lbda):
        """
        Sets the wavelength range over which the model will be computed
        
        Parameters
        ----------
        self [object]
        
        lbda [array] : wavelength
        
        Returns
        -------
        
        self.lbda [array] : attribute of the object (from the class)
        
        """
        
        self.lbda = np.asarray(lbda)
        
    def set_parameters(self, param):
        """
        this function sets the parameters of the model BB + Bump + Lines

        parameters 
        ----------
        param [array] 
                        redshift       [float]
                   
                        pol_coefs      [list]
                        bump_amplitude [float]
                    
        returns
        -------
        none, only sets the paramters of the object 
                        
        
        """
        redshift, *pol_coefs ,bump_amplitude  = param
        #redshift = z
        self.parameters = {
                           "redshift":redshift,
                           "pol_coefs":pol_coefs,
                           "bump_amplitude":bump_amplitude
                          }
    # ------- #
    # GETTER  #
    # ------- #
    def get_model(self, params=None):
        """ 
        This function returns the full model used to fit a spectrum (A*BB(lbda,T)+B*Bump(lbda)+C*lines(lbda,sigma))
        This requires the file "external_functions.py" which has the function
        
        parameters
        ----------
        
        returns
        -------
        
        
        """
        if params is not None:
            self.set_parameters(params)
            
        return self.get_continuum()+self.get_bump()
        
    def get_continuum(self, params=None):
        """ 
        This function gets the blackbody continuum emissoin for a given set of parameters. Note that the blackbody was
        normalized to 10**0 scale
        
        parameters
        ----------
        
        returns
        -------
        
        """
        if params is not None:
            pol_coefs = params
        else:
            pol_coefs = self.parameters["pol_coefs"]
            
        return get_3rd_ord_pol_continuum( self.lbda, *pol_coefs)
    
    
    def get_bump(self, params=None):
        '''
        this function gets the bump functionwhich constitutes the test for the presence of "flash" feature
        Spectra were not always obtained at the same period of time from explosion. Also, progenitors do not alwys
        have the same amount of CSM (from which flash features are supposed to originate from), hence we can't
        clearly identify which lines are present and it which quantity. We also supose that around 4700Å, there are
        several blended lines which are not always discernable. 
        
        parameters
        ----------
        params -optional- [array] : [AMPLITUDE_BUMP REDHSIFT]
        
        returns
        -------
        flux of the bump [array] with the same size as lbda. It should return only the polynom
        and padded by zero on the size.  (Again, because I do not trust extrapolation, we consider
        only the polynom in the interval it was defined)
        
        '''
        
        if params is not None:
            ampli_bump, redshift = params
            
        else:
            ampli_bump, redshift = self.parameters['bump_amplitude'], self.parameters['redshift']
            
        return get_bump(self.lbda, ampli_bump, redshift)
    
    

    
    
    # ============== #
    #   Properties   #
    # ============== #

    
    @property
    def nparam(self):
        """ 
        
        """
        return len(self.FREEPARAMETERS)





###########################################################################################################################################################################################################################



    

                                    #####################################################
                                    #                                                   #
                                    #                                                   #
                                    #          Class for Model definition               #
                                    #           Poly(3)continuum +( Lines )               #
                                    #                                                   #
                                    #####################################################











class Poly3Model( object ):
    
    
    ''' Poly3Model 
    This model is the bump model which includes a blackbody continuum, lines and the 
    theorised bump which corresponds to flash
    what type of object: it's a flux / lambda defined here like a continuum+lines + bump
    the object itself will be an array, given a set a wl (?)

    '''
    
    FREEPARAMETERS = ["pol_coef0","pol_coef1","pol_coef2","pol_coef3","redshift"]
    
    
    # ============== #
    #  Methods       #
    # ============== #
    
    def __init__(self, lbda=None, guess=None):
        """ This function initialises the parameters of the object (the wavelength range and e.g. reshdift, 
            sigma of the gaussian which define the lines...etc)
            
        Parameters
        ----------
        
        self [object] 
        
        lbda [array]  : wavelength range to consider
        
        guess [array] : initial guesses in sigma, amplitude and redshift that the model will take into account.
        
        
        Returns
        -------
        
        self [array]: parameters and wavelength range to consider
        
        """
        if lbda is not None:
            self.set_lbda(lbda)
            
        if guess is not None:
            self.set_param(guess)
            
    # ------- #
    # SETTER  #
    # ------- #

        
        
    def set_lbda(self, lbda):
        """
        Sets the wavelength range over which the model will be computed
        
        Parameters
        ----------
        self [object]
        
        lbda [array] : wavelength
        
        Returns
        -------
        
        self.lbda [array] : attribute of the object (from the class)
        
        """
        
        self.lbda = np.asarray(lbda)
        
    def set_parameters(self, param):
        """
        this function sets the parameters of the model BB + Bump + Lines

        parameters 
        ----------
        param [array] :
                        redshift       [float]
                        amplitudes     [list]
                        pol_coefs      [list]
                     

        returns
        -------
        none, only sets the paramters of the object 
                        
        
        """
        redshift ,*pol_coefs   = param
        #redshift = z
        self.parameters = {
                           "redshift":redshift,
                           "pol_coef0":pol_coefs[0],
                           "pol_coef1":pol_coefs[1],
                           "pol_coef2":pol_coefs[2],
                           "pol_coef3":pol_coefs[3]
                         
                          }
    # ------- #
    # GETTER  #
    # ------- #
    def get_model(self, params=None):
        """ 
        This function returns the full model used to fit a spectrum (A*BB(lbda,T)+B*Bump(lbda)+C*lines(lbda,sigma))
        This requires the file "external_functions.py" which has the function
        
        parameters
        ----------
        
        returns
        -------
        
        
        """
        if params is not None:
            self.set_parameters(params)
            
        return self.get_continuum()
        
    def get_continuum(self, params=None):
        """ 
        This function gets the blackbody continuum emissoin for a given set of parameters. Note that the blackbody was
        normalized to 10**0 scale
        
        parameters
        ----------
        
        returns
        -------
        
        """
        if params is not None:
            pol_coefs = params
        else:
            pol_coef0, pol_coef1, pol_coef2, pol_coef3 = self.parameters["pol_coef0"], self.parameters["pol_coef1"], self.parameters["pol_coef2"], self.parameters["pol_coef3"]
            
        return get_3rd_ord_pol_continuum( self.lbda, pol_coef0, pol_coef1, pol_coef2, pol_coef3)
    


    
    
    # ============== #
    #   Properties   #
    # ============== #

    @property
    def nparam(self):
        """ 
        
        """
        return len(self.FREEPARAMETERS)





    
###########################################################################################################################################################################################################################  
###########################################################################################################################################################################################################################  







                                    #####################################################
                                    #                                                   #
                                    #                                                   #
                                    #            Class Fitter definition                #
                                    #               Use of iMinuit                      #
                                    #                                                   #
                                    #####################################################







class SpecFitter( object ):
    
    """ 
     
     This class is defined to fit a spectrum based on the model of BB continuum + lines

    """
        
    def __init__(self, lbda, flux, variance=None, fitmodel="BBLinesModel"):
        '''
        Function of initialisation 
    
    Parameters
    ----------
    lbda      [array] wavelength range over which we perform the fit
    flux      [array] flux of the corresponding wavelength array 
    variance  [array]  -optional- whether there is an error spectrum
    fitmodel  [string] -optional-
    
    Returns
    -------
    

        
        '''
        self.set_specdata(lbda, flux, variance=variance)
        if fitmodel is not None:
            self.set_model(fitmodel)
            
    # ------- #
    # SETTER  #
    # ------- # 
    def set_specdata(self, lbda, flux, variance=None):
        
        """ sets the attributed of the object. Here the object is the real spectrum and attributes are the wavelength
         and the associated flux
    
    Parameters
    ----------
    self : object
    lbda [array/ column from asropytable] : wavelength data
    
    flux [array/ column from astropytable] : flux data
    
    variance [array/ column from table] -optional- : variance on the measurement of the flux
    
    Returns
    -------
    habille l'objet

        """
        self.lbda = np.asarray(lbda)
        self.flux = np.asarray(flux)
        if variance is not None:
            self.variance = np.asarray(variance)
        else:
            self.variance = None
            
            
            
    def set_model(self, modelname, **kwargs):
        """ 
    this functions sets the model to use and the wavelength range on which the model
    will be evaluated
    
    Parameters
    ----------
    modelname [string] Poly3BumpModel,Poly3Model ,BBLinesBumpModel, BBBumpModel, BBModel, BBLinesModel
    
    Returns
    -------
    void

        """
        self._modelname = modelname
        self.model = eval("%s(**kwargs)"%modelname) #eval: makes a string into a "command"#def var model, it remains a string though...?
        self.model.set_lbda(self.lbda)  #<- definition de la variable lambda dans var model



        
    # ------- #
    # GETTER  #
    # ------- # 
    
    def get_chi2(self, param):
        """ chi_2 calculation of the model vs data
    
    Parameters
    ----------
    param [array] (or list of var?) 
    parameters needed to compute spectrumfollowing the specified model
    
    Returns
    -------
    the chi_2 of the model vs data. 

        """
        self.model.set_parameters(param)
        model_ = self.model.get_model()
        
        
        if self.has_variance:
            pull_ = (model_ - self.flux)**2/self.variance
        else:
            pull_ = (model_ - self.flux)**2
            
        return np.sum(pull_)
    
    




    #######################
    
    #     BLACKBODY

     #######################

    
    #===============#
    # BBLines
    #===============#  
    
    def _get_chi2minuit_wobump_(self, sigma, redshift, 
                         ampl_1,ampl_2,
                         bb_temp, bb_ampl):
        """ 
            chi_2 for minuit which does not accept as input an array of amplitude each amplitude has 
            to be set on its own. This is for the model without the bump

            Parameters
            ----------
            sigma [float] width of the lines used
            redshift [float] redshift odf the supernova. This is apriori fixed
            ampl_i [float] amplitude of the lines
            bb_temp [float] blackbody temperature
            bb_ampl [float] blackbody amplitude


            Returns
            -------
            Chi2 of the model computed with the params  vs. the data
    
        """

        
        
        param = [sigma, redshift, 
                         ampl_1,ampl_2,
                         bb_temp, bb_ampl]

        
        
        return self.get_chi2(param)
  


    def _setup_minuit_wobump(self, guess, boundaries=None, fixed=None, errordef=1, print_level=0):
        """ 
            this function sets up the minuit optimizer. Minuit needs to have explicit
            declaration of each variable on which one needs to perform the fit

            parameters
            ----------
            boundaries must be dict = vaname:[min,max]
            fixed list of param to fix
            guess [array]: first guess for the spectrum

            returns
            -------   

        """

        sigma, redshift, ampl_1,ampl_2,bb_temp, bb_ampl = guess
        self._paramname = "sigma,redshift,ampl_1,ampl_2,bb_temp,bb_ampl".split(",")
        
        local_ = locals() # dict with the local variables
        self.minuit_kwargs = {}
        for p in self._paramname:
            self.minuit_kwargs[p] = local_[p]
            
        if boundaries is not None:
            for k,v in boundaries.items():
                self.minuit_kwargs["limit_"+k] = v
                
        if fixed is not None:
            for k in fixed:
                self.minuit_kwargs["fix_"+k] = True
        
        self.minuit = Minuit(self._get_chi2minuit_wobump_, errordef=errordef, print_level=print_level, **self.minuit_kwargs)
        
        



    #===============#
    # BBlinesBump
    #===============# 
    
    
    def _get_chi2minuit_bump_(self, sigma, redshift, 
                         ampl_1,ampl_2,
                         bb_temp, bb_ampl, bump_ampl):
        """ 
            chi_2 for minuit which does not accept as input an array of amplitude each amplitude has 
            to be set on its own

            Parameters
            ----------
            sigma [float] width of the lines used
            redshift [float] redshift odf the supernova. This is apriori fixed
            ampl_i [float] amplitude of the lines
            bb_temp [float] blackbody temperature
            bb_ampl [float] blackbody amplitude
            bump_ampl [float] bump amplitude


            Returns
            -------
            Chi2 of the model computed with the params  vs. the data
    
    
        """
        
        param = [sigma, redshift, 
                         ampl_1,ampl_2,
                         bb_temp, bb_ampl, bump_ampl]

        
        
        
        return self.get_chi2(param)

       

    def _setup_minuit_wbump(self, guess, boundaries=None, fixed=None, errordef=1, print_level=0):
        """ 
            this function sets up the minuit optimizer. Minuit needs to have explicit
            declaration of each variable on which one needs to perform the fit

            parameters
            ----------
            boundaries must be dict = vaname:[min,max]
            fixed list of param to fix
            guess [array]: first guess for the spectrum

            returns
            -------


        """

    
       
        sigma, redshift, ampl_1,ampl_2,bb_temp, bb_ampl, bump_ampl = guess
        self._paramname = "sigma,redshift,ampl_1,ampl_2,bb_temp,bb_ampl,bump_ampl".split(",")

        
        local_ = locals() # dict with the local variables
        self.minuit_kwargs = {}
        for p in self._paramname:
            self.minuit_kwargs[p] = local_[p]
            
        if boundaries is not None:
            for k,v in boundaries.items():
                self.minuit_kwargs["limit_"+k] = v
                
        if fixed is not None:
            for k in fixed:
                self.minuit_kwargs["fix_"+k] = True
        
        self.minuit = Minuit(self._get_chi2minuit_bump_, errordef=errordef, print_level=print_level, **self.minuit_kwargs)






    #######################
    
    #     POLY

     #######################



    
    #===============#
    # poly3Bump
    #===============# 
    
    
    def _get_chi2minuit_polybump_(self, redshift, pol_coef0, pol_coef1, pol_coef2, pol_coef3,bump_ampl):
        """ 
            chi_2 for minuit which does not accept as input an array of amplitude each amplitude has 
            to be set on its own

            Parameters
            ----------
            sigma [float] width of the lines used
            redshift [float] redshift odf the supernova. This is apriori fixed
            ampl_i [float] amplitude of the lines
            pol_ci  [float] coefficient of the polynom used as a continuum for the sepctrum
            bump_ampl [float] bump amplitude


            Returns
            -------
            Chi2 of the model computed with the params  vs. the data
    
    
        """
        
        param = param = [redshift,pol_coef0,pol_coef1,pol_coef2,pol_coef3,bump_ampl]

        
        
        
        return self.get_chi2(param)


    def _setup_minuit_polybump(self, guess, boundaries=None, fixed=None, errordef=1, print_level=0):
        """ 
            this function sets up the minuit optimizer. Minuit needs to have explicit
            declaration of each variable on which one needs to perform the fit

            parameters
            ----------
            boundaries must be dict = vaname:[min,max]
            fixed list of param to fix
            guess [array]: should be [sigma, redshift]+[list amplitudes?]+[listcoef_polynom]+[bump_ampl]

            returns
            -------   
                
        """

        

        redshift, pol_coef0, pol_coef1, pol_coef2, pol_coef3, bump_ampl = guess
        self._paramname = "redshift,pol_coef0,pol_coef1,pol_coef2,pol_coef3,bump_ampl".split(",")


        
        local_ = locals() # dict with the local variables
        self.minuit_kwargs = {}
        for p in self._paramname:
            self.minuit_kwargs[p] = local_[p]
            
        if boundaries is not None:
            for k,v in boundaries.items():
                self.minuit_kwargs["limit_"+k] = v
                
        if fixed is not None:
            for k in fixed:
                self.minuit_kwargs["fix_"+k] = True
        
        self.minuit = Minuit(self._get_chi2minuit_polybump_, errordef=errordef, print_level=print_level, **self.minuit_kwargs)



    #===============#
    # poly3
    #===============# 

    def _get_chi2minuit_poly_(self, redshift, pol_coef0, pol_coef1, pol_coef2, pol_coef3):
        """ 
              chi_2 for minuit which does not accept as input an array of amplitude each amplitude has 
             to be set on its own

             Parameters
             ----------
             sigma [float] width of the lines used
             redshift [float] redshift odf the supernova. This is apriori fixed
             ampl_i [float] amplitude of the lines
             pol_ci  [float] coefficient of the polynom used as a continuum for the sepctrum
             bump_ampl [float] bump amplitude


             Returns
             -------
             Chi2 of the model computed with the params  vs. the data
    
    
        """
        
        param = [redshift,pol_coef0,pol_coef1,pol_coef2,pol_coef3]

        
        
        
        return self.get_chi2(param)


    def _setup_minuit_poly(self, guess, boundaries=None, fixed=None, errordef=1, print_level=0):
        """ 
        this function sets up the minuit optimizer. Minuit needs to have explicit
        declaration of each variable on which one needs to perform the fit
        
        parameters
        ----------
        boundaries must be dict = vaname:[min,max]
        fixed list of param to fix
        guess [array]: should be [sigma, redshift]+[list amplitudes?]+[listcoef_polynom]+[bump_ampl]
        
        returns
        -------   
                
        """

        

        redshift, pol_coef0, pol_coef1, pol_coef2, pol_coef3 = guess
        self._paramname = "redshift,pol_coef0,pol_coef1,pol_coef2,pol_coef3".split(",")


        
        local_ = locals() # dict with the local variables
        self.minuit_kwargs = {}
        for p in self._paramname:
            self.minuit_kwargs[p] = local_[p]
            
        if boundaries is not None:
            for k,v in boundaries.items():
                self.minuit_kwargs["limit_"+k] = v
                
        if fixed is not None:
            for k in fixed:
                self.minuit_kwargs["fix_"+k] = True
        
        self.minuit = Minuit(self._get_chi2minuit_poly_, errordef=errordef, print_level=print_level, **self.minuit_kwargs)







    
    
    # ------- #
    # Fitter  #
    # ------- # 
    
    
    def fit_minuit(self, guess, boundaries=None, fixed=None):
        
        """ Fitter of data with a model based on the iminuit library. 
    
    Parameters
    ----------
    guess [array] initial guesses for the fit
    guess needs to have the following form :
    for BBLinesBumpModel :
    [sigma, redshift, ampl_1,ampl_2,ampl_3,ampl_4, bb_temp, bb_ampl, bump_ampl]
    
    for BBLinesModel:
    [sigma, redshift, ampl_1,ampl_2,ampl_3,ampl_4, bb_temp, bb_ampl]
    
    boundaries -optional- [] (cf. _setup_minuit)
    
    fixed -optional- [list of strings]
    name of variables which we require to be fixed (like the redshift)
    
    Returns
    -------
    pour l'instant voir la documentation de iminuit,  pas sure de comprendre ce que ca retourne vraiment
    
        """
        if self._modelname == 'BBLinesBumpModel' :
            self._setup_minuit_wbump(guess, boundaries, fixed)
            self.minuit_output = self.minuit.migrad()
        
            
        elif self._modelname == 'BBLinesModel':
            self._setup_minuit_wobump(guess, boundaries, fixed)
            self.minuit_output = self.minuit.migrad()
        

        elif self._modelname == 'Poly3BumpModel':
            self._setup_minuit_polybump(guess, boundaries, fixed)
            self.minuit_output = self.minuit.migrad()

        elif self._modelname == 'Poly3Model':
            self._setup_minuit_poly(guess, boundaries, fixed)
            self.minuit_output = self.minuit.migrad()

        
        


    # -------------- #
    # SETUP MINUIT   #
    # -------------- # 



    
        



        
        
    # ============== #
    #    Plotter     #
    # ============== #

    # def show_fit(self):
    #     '''
    #     This function displays the plot of the spectra and the fit of the spectra

    #     '''

    #     if self._modelname == 'BBLinesBumpModel' :

    #     bestmodel_minuit = _get_param_model_wbump_(spectral_data_1["wl"].data,parame)
    #     #bestmodel_minuit = _get_param_model_wo_bump_(spectral_data_1["wl"].data,parame)

    #     self.model.set_parameters(param)
    #     model_ = self.model.get_model()

    #     fig = plt.figure()
    #     ax = fig.add_axes([0.1,0.1,0.8,0.8])

    #     ax.plot(spectral_data_1["wl"].data, spectral_data_1["flux"].data, label = 'data')
        
    #     ax.plot(spectral_data_1["wl"].data, bestmodel_minuit, label = 'Fit Minuit')
        
    #     plt.axvline(LINES.get("HeII_1"))
    #     plt.axvline(LINES.get("Hbeta"))

    #     ax.legend()
    
    
    
        
    # ============== #
    #   Properties   #
    # ============== #
    @property
    def has_variance(self):
        """ returns True if it has a variance """
        return self.variance is not None
#     def fixed_redshift(self):
#         """sets the redshift if it is known"""
#         if 
    
    
    
###########################################################################################################################################################################################################################  
