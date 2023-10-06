import numpy as np
from astropy import units as u, constants as c
from scipy.integrate import solve_ivp

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

# The two-body dimensionless differential equation solver
def TwoBody2DCartEqMo(tau, Vec):
    
    """ The nondimensional two-body problem in Cartesian coordinates.  
    The order of the components of the input vector Vec is 
    X, dX/dtau, Y, dY/dtau.  
    
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

def TwoBody2DCart(tau, init):
    
    orbit = solve_ivp(TwoBody2DCartEqMo, [tau[0], tau[-1]], init, t_eval = tau, method='LSODA', rtol=1e-6)
    
    return {'X': orbit.y[0,:], 'VX': orbit.y[1,:], 'Y': orbit.y[2,:], 'VY': orbit.y[3,:], 'tau': orbit.t}

def Vec2DMag(vec):
    
    """ Calculate the length of a 2-D vector """
    
    # Could also do this as (for a 2-D vector)
    return np.sqrt(np.power(vec[0], 2) + np.power(vec[1], 2))

@u.quantity_input
def ReducedMass(m1 : u.kg, m2 : u.kg) -> u.kg:
    
    return m1 * m2 / (m1 + m2)

@u.quantity_input 
def GravPotEnergy(m1 : u.kg, m2 : u.kg, rvec : u.m):
    
    r = Vec2DMag(rvec)
    
    U = - (c.G * m1 * m2 / r).to(u.J)
    
    return U

def KineticEnergy(m1, m2, vvec):
    
    mu = ReducedMass(m1, m2)
    
    K = (0.5 * mu * np.power(Vec2DMag(vvec), 2)).to(u.J)
    
    return K
    
def TwoBodyEnergy(m1, m2, rvec, vvec):
    
    E = (GravPotEnergy(m1, m2, rvec) + KineticEnergy(m1, m2, vvec)).to(u.J)
   
    return E

def SemiMajorAxis(m1, m2, E):
    
    a = c.G * m1 * m2 / (2* np.abs(E))
    
    # This will return an error if the units of m1, m2, and E are not convertible to kg, kg, and J
    a = a.to(u.meter)
    
    return a

# --- Circular orbit formulae
@u.quantity_input
def V_circ(M : u.kg, R : u.m) -> u.m/u.s:

    ''' M is the sum of the masses
        R is their separation '''

    v = (np.sqrt(c.G * M / R)).to(u.m / u.s)

    return v

@u.quantity_input
def E_circ(M : u.kg, mu : u.kg, R: u.m) -> u.J:

    E = - (c.G * M * mu / (2 * R)).to(u.J)

    return E

@u.quantity_input
def L_circ(M : u.kg, mu : u.kg, R: u.m) -> u.J*u.s:

    L = (mu * np.sqrt(c.G * M * R)).to(u.J*u.s)

    return L


