# Last updated 12 October 2017

import numpy as np
import pylab as plt
#import astropy as astro
from astropy import units as u
from astropy import constants as c
from scipy.integrate import odeint, solve_ivp
import scipy.integrate as integrate
from scipy.signal import fftconvolve as fconv
#from astropy.analytic_functions import blackbody_lambda
#from astropy.analytic_functions import blackbody_nu
from astropy.modeling.models import BlackBody
#blackbody_lambda, blackbody_nu
from decimal import Decimal

# Some decorators for dealing with astropy units
#def requireUnits(original_function):
#	def new_function(*args, **kwargs):
#		for index, arg in enumerate(args):
#			if not hasattr(arg, 'unit'):
#				raise ValueError("Argument at index " + str(index) + " does not have units")
#            original_function(*args, **kwargs)
#	return returnval

def xor(p, q):
    """ Re-inventin' the wheel """
    return ((p and not q) or (not p and q))

def cart2pol(x,y):
    r = np.sqrt(np.power(x,2)+np.power(y,2))

    #theta = np.unwrap(np.arctan2(y,x))
    theta = np.arctan2(y,x)
    
    return (r,theta)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def EllipsePolar(theta, a, e):

    r = a * (1-np.power(e,2))/(1+e*np.cos(theta))

    return r

# Some pretty ways to print astropy units
def Qprint(quantity_in,sigfig=3,style='s'):
    """ wrap up the syntax for printing astropy Quantity objects in a pretty 
    way. Other options for style inclue 'latex' """
    quantity = quantity_in.copy()
    # If no unit supplied, explicitly make it dimensionless
    if not hasattr(quantity,'unit'):
        quantity = quantity * u.dimensionless_unscaled
    # Need to make a zero padded string with the number of significant figures
    sf = str(sigfig).zfill(3)
    if style == 'hw':
        print('Not supported')
    else:
        fmtstr = '{0.value:0.'+sf+'g} {0.unit:'+style+'}'
    x = fmtstr.format(quantity) 
    return x

def Uprint(quantity,style='latex'):
    """ Print the unit from an astropy Quantity object in a nice format 
    (for axes labels, for example) """
    fmtstr = '{0.unit:'+style+'}'
    x = fmtstr.format(quantity)
    return x

@u.quantity_input
def B_lambda(T : u.K, wavelength : u.m) -> u.W/u.m**3/u.steradian:
    """ This is just a thin wrapper to ensure SI units upon output and return
    of values rather than a function.  Obviously, you're free to use the functional form 
    native to astropy.  Speaking of, astropy can't seem to decide where they want to put this
    particular function; this is compatible with 5.1.  """

    B = BlackBody(temperature = T, scale = 1*u.W/u.m**3/u.steradian)(wavelength)
    return B

@u.quantity_input
def B_nu(T : u.K, frequency : u.Hz) -> u.W/u.m**2/u.steradian/u.Hz:
    """ This is just a thin wrapper to ensure SI units upon output and return
    of values rather than a function.  Obviously, you're free to use the functional form 
    native to astropy.  Speaking of, astropy can't seem to decide where they want to put this
    particular function; this is compatible with 5.1.  """

    B = BlackBody(temperature = T, scale = 1*u.W/u.m**2/u.steradian/u.Hz)(frequency)
    return B

"""This is a template for a function that does all the things I want:
ensure quantities have appropriate input units, be able to calculate
any quantity from an algebraic equation given the other N-1
qauntities, and ensure the output unit is sensible for any set of
inputs, but also allow the output unit to be whatever they specify.
But, it's clunky.  First, because the output units differ depending on
the inputs, I can't use astropy's way of checking the output units.
Second, numerically comparing against zero is a bad idea, and not
clear if there is any numerical value which is good in general.
Finally, there's a lot of re-typing things for every new function of
this type.  It would be nice to have a generalized template that
specifies the inputs and their units, and switches to the appropriate
algebraic expression depending on the inputs.

"""

@u.quantity_input(M=u.kg, a=u.m, P=u.s)
def Keplers3rdNewton(M=0*u.kg, a=0*u.m, P=0*u.s, output_unit = None):
    """ Calculate Newton's version of Kepler's Third Law, given any two of the
    three variables mass M, semimajor axis a, period P """
    fac = np.power(2.*np.pi,2)/c.G
    if ((M.value > 0) & (a.value > 0) & (P.value == 0)):
        P = np.sqrt(fac*np.power(a,3)/M)
        # Surely this can be generalized a bit
        if (output_unit is None):
            P = P.to(u.s)
        else:
            P = P.to(output_unit)
        returnval = P
    if ((M.value > 0) & (a.value == 0) & (P.value > 0)):
        a = np.power(np.power(P,2)/(fac/M),1./3.)
        if (output_unit is None):
            a = a.to(u.m)
        else:
            a = a.to(output_unit)
        returnval = a
    if ((M.value == 0) & (a.value > 0) & (P.value > 0)):
        M = np.power(a,3)/np.power(P,2)*fac
        if (output_unit is None):
            M = M.to(u.kg)
        else:
            M = M.to(output_unit)
        returnval = M
    return returnval

@u.quantity_input(T=u.K, wavelength=u.m)
def WienLaw(T=0*u.K,wavelength=0*u.m):
    const = 0.0029*u.K*u.m   

    if (T.value > 0 and wavelength.value == 0):
        val = const / T
        val = val.to(u.m)

    if (T.value == 0 and wavelength.value > 0):
        val = const / wavelength
        val = val.to(u.K)

    return val

@u.quantity_input(R=u.m, M=u.kg)
def SchwarzschildRadius(R=0*u.m, M=0.*u.kg):
    if (R.value > 0 and M.value == 0):
        val = (R*np.power(c.c,2)/(2.*c.G)).to(u.kg)
    if (M.value > 0 and R.value == 0):
        val = (2.*c.G*M/np.power(c.c,2)).to(u.m)
    return val
               
def GravForce(m1,m2,r):
    F = c.G * m1 * m2 / np.power(r,2)
    F = F.to(u.Newton)
    return F

def alphaLaneEmden(n,P_c,rho_c):
    alpha2 = (n+1)*P_c/(4.*np.pi*c.G*np.power(rho_c,2))
    return np.sqrt(alpha2)

def EscapeVelocity(M,R):
    v_esc = np.sqrt(2*c.G*M/R)
    v_esc = v_esc.to(u.meter/u.second)
    return v_esc



def orbit( Mstar_Solmass, aAU, e ):
    """ orbit : [ P, t, r, theta, x, y ] = orbit( Mstar_Solmass, aAU, e )
    Given the mass of a massive central object Mstar_Solmass (given
    in solar masses) around which a smaller mass is orbiting at a semi-major
    axis aAU (in AU) with eccentricity e, this program calculates the period
    P of the orbit (in seconds) and also generates a  timestep t (in seconds), 
    the r and theta coordinates corresponding to t, and the x and y 
    coordinates corresponding to t.  Note that r, x, and y are in SI units 
    (meters), and theta is in radians.  The origin of coordinates is the 
    focus of the ellipse. """

    # We are going to make our output quantities objects with
    # associated dimensions
    # First, associate the dimension, and then convert to SI units
    Mstar = (Mstar_Solmass * u.solMass).to(u.kilogram)
    a = (aAU * u.AU).to(u.m)

    # Period of the orbit, in seconds
    P = (np.sqrt(4*np.power(np.pi,2)*np.power(a,3)/(c.G*Mstar))).to(u.second)

    # Number of time steps.  This is a variable we can control.  The larger the
    # number of time steps, the smaller the time interval, and the smaller the
    # error in calculating numerical derivatives.
    n = 10000;
    # Make a list of time steps from 0 to the period with equal spacing given
    # by dt.
    time,dt = np.linspace(0,P,num=n,retstep=True)
    # The theta step is not uniform; we will calculate it below.  Here we just
    # define a list to hold the values of theta we'll calculate.
    theta = np.zeros(n+1);
    # Similarly for r.  Make r a dimensional quantity
    r = np.zeros(n) * u.meter;

    LoM = np.sqrt(c.G*Mstar*a*(1 - np.power(e,2)))

    for i,t in enumerate(time):
        # First, calculate the radial position corresponding to the current
        # theta.  We start (by definition) with theta = 0 at perihelion.
        r[i] = a*(1 - np.power(e,2))/(1 + e*np.cos(theta[i]))
        # Compute the next value for theta using the fixed time step by 
        # combining Eq. (2.31) with Eq. (2.32), which is Kepler's Second Law.
        dtheta = (LoM/np.power(r[i],2)*dt).value
        theta[i+1] = theta[i]+dtheta

    theta = theta[0:-1]
        
    return {'period':P,'time':time,'r':r,'theta':theta}

# Newton's Laws in 2D for a slightly more general force law
#def Orbital2DCart(X,t):
#    x = X[0]
#    vx = X[1]
#    y = X[2]
#    vy = X[3]    
#    r2 = np.power(x,2) + np.power(y,2)
#    r = np.sqrt(r2)
#    CentralForce = 1./np.power(r,2.1)
#    theta = np.arctan2(y,x)
#    dX= [0,0,0,0]
#    dX[0] = vx # dx/dt = v_x
#    dX[1] = - CentralForce * np.cos(theta) # dv/dt = Fx
#    dX[2] = vy
#    dX[3] = - CentralForce * np.sin(theta)
#    return dX

def TwoBody2DCart(X,t):
    # Rename the values in the input X to be more readable    
    x = X[0]
    vx = X[1]
    y = X[2]
    vy = X[3]  
    # Calculate r^2
    r2 = np.power(x,2) + np.power(y,2)
    dX= [0,0,0,0]
    dX[0] = vx # dx/dt = v_x
    dX[1] = -1./np.power(r2,3./2.) * x # dv_x/dt = F_x / m
    dX[2] = vy # dy/dt = v_y
    dX[3] = -1./np.power(r2,3./2.) * y # dv_y/dt = F_y / m
    return dX

def TwoBodyOrbit(t,init):
    X = odeint(TwoBody2DCart,init,t)
    x = X[:,0]
    vx = X[:,1]
    y = X[:,2]
    vy = X[:,3]
    return {'x':x,'y':y,'vx':vx,'vy':vy}

#def OrbitODE(t,init):
#    X = odeint(Orbital2DCart,init,t)
#    x = X[:,0]
#    vx = X[:,1]
#    y = X[:,2]
#    vy = X[:,3] 
#    r = np.sqrt(np.power(x,2)+np.power(y,2))
#    theta = np.arctan2(y,x)
#    return {'x':x,'y':y,'vx':vx,'vy':vy,'r':r,'theta':theta}
    
# --- Useful relations for the two body problem ---

@u.quantity_input
def ReducedMass(m1 : u.kg, m2 : u.kg) -> u.kg:
    
    return m1 * m2/(m1 + m2)
 
def TwoBodyEnergy(m1,m2,r,v):
    mu = ReducedMass(m1,m2)
    M = m1 + m2
    E = (0.5 * mu * np.power(v,2) - c.G * M * mu / r).to(u.J)
    return E

def SemiMajorAxis(m1,m2,E):
    a = c.G * m1 * m2 / (2* np.abs(E))
    a = a.to(u.meter)
    return a

@u.quantity_input
def EfromSemiMajorAxis(m1 : u.kg, m2 : u.kg, a : u.m) -> u.J:
    
    E = - c.G * m1 * m2 / (2* np.abs(a))
    
    return E
    
def Eccentricity(m1,m2,E,L):
    M = m1 + m2
    mu = ReducedMass(m1,m2)
    e = (np.sqrt(1+2*E*np.power(L,2)/(np.power(c.G*M,2)*np.power(mu,3)))).to(u.dimensionless_unscaled)
    return e

@u.quantity_input
def Period(m1 : u.kg, m2 : u.g, a : u.m) -> u.s:
    
    P = np.sqrt(np.power(2*np.pi,2) * np.power(a,3) / (c.G * (m1+m2)))
    
    return P
    
def PhysicalOrbit(m1,m2,a,e):
    """ Calculates the two-body orbit """
    m1 = m1.to(u.kg)
    m2 = m2.to(u.kg)
    a = a.to(u.m)
    mu = ReducedMass(m1,m2)
    M = m1+m2
    # Period, in seconds
    P = Period(m1,m2,a)
    # Energy, in Joules
    E = EfromSemiMajorAxis(m1,m2,a)
    # Angular momentum, in Joule-seconds
    L = (mu*np.sqrt(c.G*M*a*(1-np.power(e,2)))).to(u.J*u.s)
    # Factor to make non-dimensional time
    t0 = (np.sqrt(np.power(a,3)/(c.G*M))).to(u.s)
    # Perihelion velocity
    v_p = (np.sqrt(c.G*M/a *(1+e)/(1-e))).to(u.m/u.s)
    # Perihelion distance
    r_p = a * (1-e)
    # Factor to make non-dimensional length
    r0 = a
    # Factor to make non-dimensional velocity
    v0 = (r0/t0).to(u.m/u.s)
    # Time.  For bound orbits, 2 pi times the characteristic time t0
    t = np.linspace(0,P/t0,num=3000)
    # Initial conditions, in dimensionless units
    init = [r_p/r0,0,0,v_p/v0]
    # Solve the differential equation
    soln = TwoBodyOrbit(t,init)
    """ I know it's ugly, but the *= notation just ... doesn't work for creating
    Quantity objects """
    # Re-dimensionalize
    soln['x'] = (soln['x']*r0).to(u.m)
    soln['y'] = (soln['y']*r0).to(u.m)
    soln['vx'] = (soln['vx']*v0).to(u.m/u.s)
    soln['vy'] = (soln['vy']*v0).to(u.m/u.s)
    soln['t'] = t*t0
    soln['E'] = E
    soln['L'] = L
    soln['P'] = P
    # Go from the fictitious problem back to the orbits of the two 
    list = ['x','y','vx','vy']
    for l in list:
        soln[l+'1']=-soln[l]*m2/M
        soln[l+'2']=soln[l]*m1/M
    return soln
    
# ----- Hydrogen atom quantities
@u.quantity_input
def HAtomEnergy(n, Z=1) -> u.J:
    """ Returns the energy of the level with principal quantum number n in the
    Bohr model of the hydrogen atom.  The energy is negative (relative to 0, 
    which is unbound), and given in eV.  The optional argument Z gives the 
    charge on the nucleus, in units of the proton charge. """
    E = c.m_e * np.power(c.c,2) / 2. * np.power(c.alpha,2) * np.power(Z,2)/np.power(n,2)
    
    return -E
    
def HAtomDeltaE(n_i,n_f,Z=1,wavelength=False):
    """ Calculates the change in energy in the Bohr model of the hydrogen atom 
    between initial quantum number n_i and final quantum number n_f.  The 
    optional argument Z gives the charge on the nucleus, in units of the proton 
    charge.  If the optional argument wavelength is set to True, the energy change is
    returned in units of the wavelength of light that would be emitted.  Positive
    energy denotes that energy must be added, negative that energy is emitted.
    Wavelengths are always positive.  """
    E_i = HAtomEnergy(n_i,Z=Z)
    E_f = HAtomEnergy(n_f,Z=Z)
    deltaE = E_f - E_i
    if wavelength:
        deltaE = (c.h * c.c / np.abs(deltaE)).to(u.m)
    return deltaE

def HAtomRadius(n,Z=1):
    """ Compute the radius of the Bohr hydrogen atom (this is the most probable
    radius quantum mechanically).  
    n is the principal quantum number and the optional argument Z gives the 
    charge on the nucleus, in units of the proton charge.    
    Note that
    c.a0 = (c.hbar/(c.m_e*c.c*c.alpha)).to(u.m) 
    """
    r = c.a0 * np.power(n,2)/Z
    return r

@u.quantity_input
def Larmor(a : u.m/np.square(u.s)) -> u.W:
    P = 2./3. * np.square(c.e.si)/(4.*np.pi*c.eps0) * np.square(a)/np.power(c.c,3)
    return P

# ----- Thermal physics
def MeanMolecMass(molmass,frac,type=None):
    assert (type=='number' or type=='mass')
    assert (len(frac) == len(molmass))
    assert(np.array(frac).sum() == 1)
    if (type=='number'):
        mu = (np.array(frac)*np.array(molmass)).sum()
    if (type=='mass'):
        mu = 1./((np.array(frac)/np.array(molmass)).sum())
    return mu

@u.quantity_input
def BoltzmannFactor(g, E : u.J, T : u.K) -> u.dimensionless_unscaled:
    
    """ Returns the Boltzmann factor for degeneracy g, energy E and temperature T """   
    
    f = g * np.exp(-E/(c.k_B*T))
    
    return f

def gH(n):
    
    return 2*np.square(n)

@u.quantity_input
def PartitionFunction_H(T : u.K, nmax=5) -> u.dimensionless_unscaled:
    
    """ Calculate an approximation to the hydrogen partition function (truncated at nmax) """
    
    partition_function = 0
    
    for n in np.arange(1,nmax+1):
        
        En = HAtomEnergy(n)
        
        partition_function += BoltzmannFactor(gH(n), En, T)
        
    return partition_function

def Probability_H(n_array, T):
    
    q_H = PartitionFunction_H(T, nmax=n_array.max())
    
    P_H = BoltzmannFactor(gH(n_array), HAtomEnergy(n_array), T) / q_H
    
    return P_H

@u.quantity_input
def BBAvgE(T : u.K) -> u.J:
    
    E = np.power(np.pi, 4)/(30*zeta(3)) * c.k_B * T
    
    #E2 = 3.729e-23 * u.J/u.K * T 
    
    return E1

@u.quantity_input
def Boltzmann(g1,g2,E1 : u.J,E2 : u.J,T : u.K) -> u.dimensionless_unscaled:
    """ Returns the ratio n2/n1 for states at energy E1 and E2 """   
    f = g2/g1 * np.exp(-(E2-E1).to(u.J)/(c.k_B*T))
    return f

@u.quantity_input
def SigmaV(T : u.K,mu) -> (u.m/u.s):
    """ Behavior assumes T is in Kelvin. """
    m = mu*c.m_p
    sigma_v = np.sqrt(c.k_B * T / m)
    return sigma_v

@u.quantity_input
def AvgV(T,mu) -> (u.m/u.s):
    """ Compute the average velocity of a particle of molecular mass mu at 
    temperature T.  SI units are assumed, but inputs should NOT be Quantity 
    objects """
    return np.sqrt(8/np.pi)*SigmaV(T,mu)

@u.quantity_input
def MostLikelyV(T : u.K,mu) -> (u.m/u.s):
    return np.sqrt(2.)*SigmaV(T,mu)

def MaxwellBoltzmann(v,mu,T):#,units=False):
    """ Default behavior is to assume T is in Kelvin an v in meters/second, but
    that these are not astropy Quantity obects """   
    sigma_v = SigmaV(T*u.K,mu).value
    x = v/sigma_v
    Pv = np.sqrt(2./np.pi)/sigma_v*np.power(x,2)*np.exp(-np.power(x,2)/2.)
    return Pv

def AtmEscParam(v,mu,T):#,units=False):
    """ Default behavior is to assume T is in Kelvin an v in meters/second, but
    that these are not astropy Quantity obects """   
    sigma_v = SigmaV(T,mu)
    x = v/sigma_v
    Pv = 1./16.*np.sqrt(2./np.pi)/sigma_v*np.power(x,2)*v*np.exp(-np.power(x,2)/2.)
    return Pv

@u.quantity_input#(n_e=np.power(u.m,-3),P_e=u.Pa)
def Saha(T : u.K,Z1,Z2,chi : u.J,n_e=None, P_e=None) -> u.dimensionless_unscaled:
    if (not xor(n_e,P_e)):
        print('Must give one of either electron number density n_e or electron pressure P_e as keyword input arguments.')
        return
    if P_e:
        n_e = P_e/(c.k_B*T)
    x = (chi/(c.k_B*T)).to(u.dimensionless_unscaled)
    prefac = 2.*np.pi*c.m_e*c.k_B*T/np.power(c.h,2)
    f = 2 * Z2/(n_e*Z1) * np.power(prefac,1.5) * np.exp(-x)
    #f = f.to(u.dimensionless_unscaled)
    return f

# The astropy decorator u.quantity_input does it better
def CheckUnit(var,unit):
    """ Checks that the input variable has a unit, and that is convertible to a specified type """
    if not hasattr(var,'unit'):
        raise ValueError("Variable must have a unit")
    if not var.unit.is_equivalent(unit):
        raise ValueError('Variable not convertible to '+unit.to_string())
    return

@u.quantity_input
def SahaClosedBoxH(rho : u.kg*np.power(u.m,-3), T : u.K) -> u.dimensionless_unscaled :
    """ Calculate the ionized fraction of Hydrogen in a closed box of mass density rho and temperature T """
    #CheckUnit(rho,u.kg*np.power(u.m,-3))
    #CheckUnit(T,u.K)
    chi = 13.6*u.eV
    prefac = 2.*np.pi*c.m_e*c.k_B*T/np.power(c.h,2)
    a_coeff = (c.m_p/rho) * np.power(prefac,1.5) * np.exp(-(chi/(c.k_B*T)).to(u.dimensionless_unscaled))
    x = a_coeff/2. * (np.sqrt(1+4./a_coeff) - 1)
    #x = x.to(u.dimensionless_unscaled)
    return x

# Line profiles
def Lorentz(nu,nu0,gamma_n):
    k = gamma_n/(4.*np.pi)
    x = nu - nu0
    phi = 1/np.pi * k / (np.power(x,2)+np.power(k,2))
    return phi
    
def Doppler(nu,nu0,T,mu):
    C = c.c.value
    #m_p = c.m_p.value    
    s = SigmaV(T,mu)
    x = nu - nu0
    exparg = - np.power(x * C / ( nu0 * s), 2)/2.
    phi = C/nu0 * np.power(np.sqrt(2*np.pi) * s, -1) * np.exp(exparg)
    return phi

def Voigt(lambda0,gamma_n,mu,T):
    """ Enter line wavelength lambda0 in meters (not as a unit), and gamma_n,
    mu, and T as for the natural and Doppler profiles """
    nu0 = c.c.value/lambda0
    frac = 0.02
    nu = np.linspace(nu0*(1-frac),nu0*(1+frac),num=np.power(2,20))
    
    phi_L = Lorentz(nu,nu0,gamma_n)
    phi_D = Doppler(nu,nu0,T,mu)
    phi_V = fconv(phi_L,phi_D,mode='same')
    Phi = phi_V/phi_V.max()
    
    return {'Phi':Phi,'nu':nu}

def PhotonCrossSection(f):
    """ Calculate the cross section for a given oscillator strength f and line
    profile phi """
    sigma = np.power(c.e.si,2)/(4.*c.eps0*c.m_e*c.c) * f * phi
    return sigma

def HO(y,t):
    dy = [0,0]
    dy[0] = y[1] # dx/dt = v_x
    dy[1] = -y[0] # dv_x/dt = -x
    return dy

def LaneEmden(y,x,n=1):
    dy = [0,0]
    # x = 0 requires special handling
    if x == 0:
        dy[0] = 1
    else:
        dy[0] = -y[1] / np.power(x,2)
    dy[1] = np.power(y[0],n) * np.power(x,2)
    return dy

def FreeFall(y,t,mass=1*c.M_earth.value):
    #mass = (1*c.M_earth.value)
    dy = [0,0]
    dy[0] = y[1]
    dy[1] = - c.G.value * mass / np.power(y[0],2)
    return dy

def nonDemFreeFall(y,t):
    dy = [0,0]
    dy[0] = y[1]
    dy[1] = -1./np.power(y[0],2)
    return dy

def FreeFallTime(M,R):
    t_ff = np.pi/2. * np.power(R,3/2.) / np.power( 2 * c.G * M, 1./2)
    return t_ff

def FindNearest(array,val):
    diff = np.abs(array-val)
    indx = (np.where(diff == diff.min()))[0]
    return indx
    
def IsScalar(x):
    # A dumb way to figure out if we've been handed a scalar.  Got to
    # be a better way
    Scalar = False
    try:
        len(x)
    except:
        Scalar = True
    return Scalar

def Luminosity(Mass):
    """ Mass is assumed to be in stellar masses, and luminosity is returned in stellar luminosity """
    L = []
    if (IsScalar(Mass)):
        Mass = np.array([Mass])
    for M in Mass:
        if M <= 0.7:
            L.append(0.35 * np.power(M,2.62))
        if M > 0.7:
            L.append(1.02 * np.power(M,3.92))
    return np.array(L)

def Radius(Mass):
    """ Mass is assumed to be in stellar masses, and radius is returned in stellar radii """
    R = []
    if (IsScalar(Mass)):
        Mass = np.array([Mass])
    for M in Mass:
        if M < 1.33:
            R.append(1.06 * np.power(M,0.945))
        if M > 1.33:
            R.append(1.33 * np.power(M,0.555))
    return np.array(R)

@u.quantity_input
def ScaleHeight(g : u.m/u.s**2, T : u.K, mu) -> u.m:
    
    mbar = mu * c.m_p
    z_s = c.k_B * T / (mbar * g)
    
    return z_s


def P_c(M,R):
    P = 3.*c.G*np.power(M,2)/(8.*np.pi*np.power(R,4))
    P = P.to(u.Pa)
    return P

def P_rad(T):
    P = 4 * c.sigma_sb / (3.*c.c) * np.power(T,4)
    P = P.to(u.Pa)
    return P

def eps_pp(rho,T,X):
    eps_0pp = 1.08e-12 * u.W * np.power(u.m,3) * np.power(u.kg,-2)
    T6 = 1e6*u.K
    eps = eps_0pp * np.power(X,2) * rho * np.power(T/T6,4)
    return eps

def T_no_greenhouse(d,A,L=c.L_sun):
    return np.power(L/(16.*np.pi*c.sigma_sb*np.power(d,2))*(1-A),1./4.).to(u.K)

def RadioactiveDating(t_half,age=None,fraction=None):
    if (age is None) and (fraction is not None):
        # User wants the age
        age = t_half * np.log10(fraction) / np.log10(0.5)
        returnval = age
    if (age is not None) and (fraction is None):
        # User wants the fraction
        fraction = np.power(0.5,age/t_half)
        returnval = fraction
    return returnval

def Quadratic(a, b, c):
    
    rt = np.sqrt(np.power(b, 2) - 4*a*c)

    return ((-b + rt)/(2*a), (-b - rt)/(2*a))

def ae_from_init(X0, VY0):
    
    a = 0.5/np.abs(0.5 * np.power(VY0, 2) - 1/X0)
    
    e = np.sqrt(1 - np.power(X0*VY0, 2)/a)
    
    return (a, e)

def init_from_ae(a, e):
    
    X0 = a*(1-e) # assume we always start at perihelion
    VY0 = np.sqrt( a * (1-np.power(e, 2)) ) / X0
    
    return(X0, VY0)

def TwoBody2DCartEqMo(tau, Vec):
    
    """ The nondimensional two-body problem in Cartesian coordinates.  
    The order of the components of the input vector Vec is 
    X, X/dtau, Y, dY/dtau.  
    
    Appropriate for use with scipy.integrate.solve_ivp
    Also backwards compatible with scipy.integrate.odeint using tfirst=True
    
    """
    
    # Rename the values in the input vector Vec to be more readable by humans   
    X = Vec[0]
    VX = Vec[1]
    Y = Vec[2]
    VY = Vec[3]  
    
    # Calculate r^2
    r2 = np.power(X, 2) + np.power(Y, 2)
    
    # Initialize the right hand side
    dVecdtau = [0,0,0,0]
    
    dVecdtau[0] = VX 
    dVecdtau[1] = -1./np.power(r2, 3./2.) * X 
    dVecdtau[2] = VY 
    dVecdtau[3] = -1./np.power(r2, 3./2.) * Y 
    
    return dVecdtau

def TwoBody2DCartIVP(tau, init):
    
    orbit = solve_ivp(TwoBody2DCartEqMo, [tau[0], tau[-1]], init, t_eval = tau, method='LSODA', rtol=1e-6)
    
    X = orbit.y[0,:]
    Y = orbit.y[2,:]
    R = np.sqrt(np.power(X,2)+np.power(Y,2))
    theta = np.unwrap(np.arctan2(Y, X))
    
    return {'X': X, 'VX': orbit.y[1,:], 'Y': Y, 'VY': orbit.y[3,:], 'tau': orbit.t, 'init': init, 'R': R, 'theta': theta}

def MSLuminosity(mass : u.kg) -> u.W:
    
    # express in terms of solar mass; ensure array functions behave properly by converting to an array
    m = (np.array(mass)*u.kg / c.M_sun).to(u.dimensionless_unscaled).value
    
    # a function to smoothly transition between the two power laws
    smooth_transition = 1/2*(np.tanh((m-0.7)/0.1)+1)
       
    exponent = 2.62 + smooth_transition*(3.92-2.62)
    prefac = (0.35 + smooth_transition*(1.02 - 0.35))*c.L_sun
    
    return prefac * np.power(m, exponent)

def MSRadius(mass : u.kg) -> u.m:
    
    # express in terms of solar mass; ensure array functions behave properly by converting to an array
    m = (np.array(mass)*u.kg / c.M_sun).to(u.dimensionless_unscaled).value
    
    smooth_transition = 1/2*(np.tanh((m-1.66)/0.1)+1)
       
    exponent = 0.945 + smooth_transition*(0.555 - 0.945)
    prefac = (1.06 + smooth_transition*(1.33 - 1.06))*c.R_sun
    
    return prefac * np.power(m, exponent)

def MSTemperature(mass : u.kg) -> u.K:
    
    L = MSLuminosity(mass)
    R = MSRadius(mass)
    
    T = np.power(L / (4.*np.pi*np.power(R,2)*c.sigma_sb), 1/4)
    
    return T