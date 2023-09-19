import numpy as np
from astropy import stats

def ImageStats(_image, display=False):
    
    """ Collect summary statistics of the input image """
    
    image = _image[np.isfinite(_image)]
    
    stat = {}
    stat['min'] = image.min()
    stat['max'] = image.max()
    stat['mean'] = image.mean()
    stat['median'] = np.median(image)
    stat['std'] = image.std()
    stat['mad_std'] = stats.mad_std(image)
    
    # Notice that it's kind of absurd to print many significant figures for these numbers; 
    # let's use formatting to get a nicer looking output
    if display:
        for key in stat.keys():
            print(key, ':', '{:.5g}'.format(stat[key]))
    
    return stat

def Histogram(data, bins = None):
    
    """ The default behavior of both np.histogram and plt.bar requires a lot of extra typing """
    if bins is None:
        bins = 1000
    
    hist, bin_edges = np.histogram(data, bins)
    
    bin_centers = (bin_edges[1:] + bin_edges[0:-1])/2
    
    bin_widths = (bin_edges[1:] - bin_edges[0:-1])
    
    return {'counts': hist, 'bin_centers': bin_centers, 'bin_widths': bin_widths}

def PlotHistogram(ax, myhist):
    
    ax.bar(myhist['bin_centers'], myhist['counts'], myhist['bin_widths'])
    
    return

def ImageXY(image):
    
    ny, nx = image.shape
    
    y, x = np.mgrid[:ny, :nx]
    #y, x = np.meshgrid(np.arange(0,ny), np.arange(0,nx))
    
    return x, y 