import matplotlib.pyplot as plt
from matplotlib import rc

def setstyle():

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ["Palatino"]
    plt.rcParams['font.size'] = 18
    plt.rcParams['figure.figsize'] = (12,9)
    plt.rcParams['text.usetex'] = True
