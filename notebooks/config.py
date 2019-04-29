import matplotlib.pyplot as plt

SMALL_SIZE = 8
MEDIUM_SIZE = 8
BIGGER_SIZE = 8
plt.rcParams['font.family'] = 'Helvetica Neue LT Com'
plt.rcParams['font.weight'] = '500'
plt.rcParams['figure.autolayout'] = True

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

width = 3.5
height = width / 1.618

colors_dict = {
    'green': '#009E73',
    'grey': '#A8A496',
    'blue': '#0072B2',
    'orange': '#D55E00',
    'pink': '#CC79A7'
}