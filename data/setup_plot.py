import matplotlib.pyplot as plt 
from matplotlib import font_manager as fm 
from matplotlib.pyplot import gca
import matplotlib as mpl
from cycler import cycler
import math

linestyle_tuple = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot'),
     
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
     
     ]

linestyle_dict = {k : v for k,v in linestyle_tuple}

def setup_global():
    font_entry = fm.FontEntry(
        fname = './gillsans.ttf',
        name='gill-sans')

    # set font
    fm.fontManager.ttflist.insert(0, font_entry) 
    mpl.rcParams['font.family'] = font_entry.name 

    mpl.use('Agg')
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.markersize'] = 5
    #mpl.rcParams['hatch.linewidth'] = 0.05

    ## Use the following two lines of code if you want type-1 fonts
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

# def setup_local():
#     plt.clf()
#     gca().yaxis.grid(linestyle='dotted', which="both")
#     gca().spines['left'].set_color('#606060')
#     gca().spines['bottom'].set_color('#606060')
#     global_cycler = cycler(color=get_colors()[:len(get_linestyles())]) + cycler(linestyle=get_linestyles()) 
#     gca().set_prop_cycle(global_cycler)
    
def setup_local(axis=None):
    if axis is None:
        plt.clf()
        axis = gca()
    
    axis.yaxis.grid(linestyle='dotted', which="both")
    axis.spines['left'].set_color('#606060')
    axis.spines['bottom'].set_color('#606060')
    global_cycler = cycler(color=get_colors()[:len(get_linestyles())]) + cycler(linestyle=get_linestyles()) 
    axis.set_prop_cycle(global_cycler)

def set_aspect_ratio(ratio=3/5, logx=None, logy=None, axis=None):
    if axis is None:
        axis = gca() 
    xleft, xright = axis.get_xlim()
    if logx is not None:
        xleft = math.log(xleft, logx)
        xright = math.log(xright, logx)
    ybottom, ytop = axis.get_ylim()
    if logy is not None:
        ytop = math.log(ytop, logy)
        print(ytop, ybottom)
        ybottom = math.log(ybottom, logy)
    axis.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

# def set_aspect_ratio(ratio=3/5, logx=None, logy=None): 
#     xleft, xright = gca().get_xlim()
#     if logx is not None:
#         xleft = math.log(xleft, logx)
#         xright = math.log(xright, logx)
#     ybottom, ytop = gca().get_ylim()
#     if logy is not None:
#         ytop = math.log(ytop, logy)
#         print(ytop, ybottom)
#         ybottom = math.log(ybottom, logy)
#     gca().set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

def get_colors():
    return ['#D55E00', 
            '#009E73',
            '#0072B2', 
            '#CC79A7', 
            '#000000', 
            '#E03A3D',
            '#F0E442',]

def get_colors_dict():
    return {
            'orange': '#D55E00', 
            'green': '#009E73',
            'blue': '#0072B2', 
            'purple': '#CC79A7', 
            'black': '#000000', 
            'red': '#E03A3D',
            'yellow': '#F0E442',
            }

def get_hatches():
    return ['xxx', '|||', '\\\\', 'xxxxxx',  '///////', '+']

def get_linestyles():
    return [
            linestyle_dict['solid'], 
            linestyle_dict['dotted'], 
            linestyle_dict['dashdotted'],
            linestyle_dict['dashed'],
            linestyle_dict['dashdotdotted'],
            linestyle_dict['densely dashdotted'],
            
    ]

def get_linestyles_dict():
    return linestyle_dict

def get_markers():
    return ['^', 'd', 's', 'o', 'x']