from astropy import units as u

# Some pretty ways to print astropy units
def Qprint(quantity_in, sigfig=3, style='s'):
    """ wrap up the syntax for printing astropy Quantity objects in a pretty 
    way. Other options for style inclue 'latex' """
    quantity = quantity_in.copy()
    # If no unit supplied, explicitly make it dimensionless
    if not hasattr(quantity, 'unit'):
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
