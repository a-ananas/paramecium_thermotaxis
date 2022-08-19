# -*- coding: utf-8 -*-


#################
### LIBRARIES ###
#################

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from scipy import stats
mpl.rcParams['agg.path.chunksize'] = 10000

#-------------------------------------------------------

#################
### CONSTANTS ###
#################

# list of control, grad on and grad off runs number 

# /!\ RUNS 5 6 7 are a bit odd, some problems occured during recording, might want to 
#remove them from the analysis

ALL_RUNS = range(1,20)
CONTROLS = list(range(5, 20, 3))
GRAD_ON = [1,3] + list(range(6, 20, 3))
GRAD_OFF = [2, 4] + list(range(7, 20, 3))

# acquisition period : time between 2 frames taken in s 
T = 0.06
# length scale : 68.5 pi/mm : dpi in pi/mm
dpi = 68.5
#middle pixel of the pool (data on ImageJ)
x_m, y_m = (1438/2)+222, (1028/2)+64

#-------------------------------------------------------
[1, 3] + list(range(6, 20, 3))
#################
### FUNCTIONS ###
#################

def histo_position_total(df, title):
    fig, ax = plt.subplots(2, figsize=(15,10))
    fig.suptitle(title)
    ax[0].hist(df['xBody'], bins=1920//8)
    ax[0].set_title('x position')
    ax[1].hist(df['yBody'], bins=1200//8)
    ax[1].set_title('y position')
    
def calculate_mean_displacement(df):
    """
    df : DataFrame that needs to have an xBody and a yBody column
    """
    xs,ys = df["xBody"].values, df["yBody"].values
    d = np.zeros_like(xs)
    d[1:] = np.sqrt( np.square(xs[1:]-xs[:-1]) + np.square(ys[1:]-ys[:-1]))
    return d.mean()

def plot_mean_position(df, tilte):
    """
    df : DataFrame that needs to have an xBody and a yBody column
    applied on group by id
    """
#    x,y = df["xBody"].values.mean(), df["yBody"].values.mean()
    
def plot_trajectories(df, title, with_label=False):
    """
    df : DataFrame that needs to have an xBody and a yBody column
    """
    #Need to flip the y axis 
    xs,ys = df["xBody"].values,  1200-df["yBody"].values
    if with_label :
        label = df['id'].unique()[0]
        plt.plot(xs,ys, lw=0.5, 
             label=label,
             )
        plt.legend()
    else : 
        plt.plot(xs,ys, lw=0.5) 
    plt.title(title)

#%%
#Loading dataframes into a dictionnary {#of_run : [df, group_id, group_img]}
#converting pixels to mm distance from the center of the pool (negative x -> hot side, positive x -> cold side)

dataframes = {}

for n in ALL_RUNS:
    
    path = "/home/antoine/Bureau/Antoine/"+str(n)+"/Tracking_Result/tracking.txt"
    data = pd.read_csv(path, sep='\t')
    data["xBody"] = (data["xBody"] - x_m) / dpi
    data["yBody"] = (data["yBody"] - y_m) / dpi
    data_group_id = data.groupby('id')
    data_group_img = data.groupby('imageNumber')
    
    dataframes[n] = (data, data_group_id, data_group_img)

#%%

for n, (data, data_group_id, data_group_img)  in dataframes.items():
    
    title_hist = "Run "+str(n)+" : histogram of positions"
    title_traj = "Run "+str(n)+" : trajectories of all objects"
    
    path = "F:\\Antoine\\"+str(n)+"\\"
    
    histo_position_total(data, title_hist)
    plt.show()
    data_group_id.apply(plot_trajectories, title_traj)
    plt.show()

print("DONE!!!")
    
#%%

# divide each batch in 30 sec period

number_of_frames_30s = 30000//T

for n in ALL_RUNS: 
    data = dataframes[n][0]
    
    number_of_frames_data = data['imageNumber'].unique().size
    
    mean_x_positions = []
    mean_y_positions = []
    
    std_x = []
    std_y = []
    
    names = []
    
    for i in range(0, number_of_frames_data//number_of_frames_30s):
        
        low_bound = i*number_of_frames_30s
        up_bound = (i+1)*number_of_frames_30s
        
        df = data[data["imageNumber"].between(low_bound, up_bound)]
        
        mean_x_positions.append(df['xBody'].mean())
        mean_y_positions.append(df['yBody'].mean())
        
        std_x.append(df['xBody'].std())
        std_y.append(df['yBody'].std())
        
        names.append(str(low_bound)+" - "+str(up_bound))
        
    fig, ax = plt.subplots(2, figsize=(15,10))
    fig.suptitle("Mean evolution of position in run "+ str(n))
    ax[0].bar(range(0, number_of_frames_data//number_of_frames_30s), mean_x_positions, yerr=std_x)
    ax[0].set_ylabel('Mean x positions across 30sec periods')
    ax[0].set_xticks(range(0, number_of_frames_data//number_of_frames_30s))
    ax[0].set_xticklabels(names)
    ax[0].set_title('x position')
    ax[1].bar(range(0, number_of_frames_data//number_of_frames_30s), mean_y_positions, yerr=std_y)
    ax[1].set_ylabel('Mean y positions across 30sec periods')
    ax[1].set_xticks(range(0, number_of_frames_data//number_of_frames_30s))
    ax[1].set_xticklabels(names)
    ax[1].set_title('y position')
    
#%%

# constants for plotting and statistical tests

number_stripes = 200
time_period = 30000
number_of_frames_time_period = time_period//T

#%%

# divide the image field into 5 separate stripes and count the number of cells in each stripes
# time dependance : fuse with code above to see the evolution on 30 seconds time periods

for n in ALL_RUNS:
        
    data = dataframes[n][0]
    
    x_min = data['xBody'].min()
    x_max = data['xBody'].max()
    
    dx = (x_max - x_min)/number_stripes
    
    number_of_frames_data = data['imageNumber'].unique().size
    
    if n in GRAD_OFF:
        cmap = plt.cm.rainbow_r
        title = "ON -> OFF"
    elif n in GRAD_ON:
        cmap = plt.cm.rainbow
        title = "OFF -> ON"
    else:
        cmap = plt.cm.rainbow
        title = "CONTROL"
        
    fig, ax = plt.subplots(1, figsize=(15,10))
    fig.suptitle("Number of cells in "+str(number_stripes)+" areas in run "+ str(n)+" - "+title)
    
    cs = cmap(np.linspace(0, 1, number_of_frames_data//number_of_frames_time_period))

    
    for j in range(0, number_of_frames_data//number_of_frames_time_period):
        
        low_bound = j*number_of_frames_time_period
        up_bound = (j+1)*number_of_frames_time_period
        
        number_cells_per_stripes = []
        
        data_tw = data[data["imageNumber"].between(low_bound, up_bound)]
        
        number_of_traj = data_tw['id'].unique().size
    
        for i in range(number_stripes):
            low_bound = i*dx
            up_bound = (i+1)*dx
            
            df = data_tw[data_tw["xBody"].between(low_bound, up_bound)]
            
            #number of unique cells detected by fasttrack
            number_cells_per_stripes.append(df['id'].unique().size/number_of_traj)
        
        bars = ax.bar(range(0,number_stripes), number_cells_per_stripes, edgecolor=cs[j], fill=False, alpha=0, linewidth=5)
        ax.set_ylabel("Number of cells on"+str(time_period//1000)+" seconds periods/Total number of cells")
        ax.set_xlabel("Stripe number (from left to right on x axis)")
        ax.set_title(str(time_period//1000)+" seconds periods represented (red gradient on, blue gradient off)")
        for bar in bars:
            x, y = bar.get_xy()
            w, h = bar.get_width(), bar.get_height()
            ax.plot([x, x + w], [y + h, y + h], color=cs[j], lw=2.5, alpha=0.75)
    
print("DONE!!!")
            
#%%

# attempt to use KS test and Wilcoxon test (stats.wilcoxon and stats.ranksums)
# to compare first 30sec distrib to last 30 sec distrib in every run

for n in ALL_RUNS:
    data = dataframes[n][0]
    
    x_min = data['xBody'].min()
    x_max = data['xBody'].max()
    
    dx = (x_max - x_min)/number_stripes
    
    number_of_frames_data = data['imageNumber'].unique().size
    
    number_cells = []
    
    if n in CONTROLS:
        title = "CONTROL"
    else:
        title = "GRADIENT"
    
    for j in range(0, 2):
        
        if j == 0:
            low_bound = j*number_of_frames_time_period
            up_bound = (j+1)*number_of_frames_time_period
        else:
            low_bound = number_of_frames_data - number_of_frames_time_period
            up_bound = number_of_frames_data
            
        number_cells_per_stripes = []        
        
        data_tw = data[data["imageNumber"].between(low_bound, up_bound)]
        
        number_of_traj = data_tw['id'].unique().size
    
        for i in range(number_stripes):
            low_bound = i*dx
            up_bound = (i+1)*dx
            
            df = data_tw[data_tw["xBody"].between(low_bound, up_bound)]
            
            #number of unique cells detected by fasttrack
            number_cells_per_stripes.append(df['id'].unique().size/number_of_traj)
        
        number_cells.append(number_cells_per_stripes)
        
    ks = stats.kstest(number_cells[0], number_cells[1])
    wx = stats.wilcoxon(number_cells[0], number_cells[1])
    rs = stats.ranksums(number_cells[0], number_cells[1])
    
    print(80*"#")
    print("For run "+str(n)+" of type "+title)
    
    
    print("\t-> Ks Test :" ,ks.statistic, "p-val :", ks.pvalue)
    print("\t-> Wilcoxon Test :", wx.statistic, "p-val :", wx.pvalue)
    print("\t-> Wilcoxon Test (rs):", rs.statistic, "p-val :", rs.pvalue)

    print("\n")
            
#%%

# plot standard deviation for each frame

def std_by_frame(df):
    xs = df['xBody'].values
    
    return(xs.std())

N = 17

for n in range(N, N+3):
    frms = dataframes[n][2]

    std = frms.apply(std_by_frame)
    
    i = n-N
    s = std.size

    plt.plot(range(i*s, (i+1)*s), std)
    
plt.show()

#%%

# plot mean for each frame

def mean(df):
    xs = df['xBody'].values
    
    return(xs.mean())

N = 17


for N in range(11, 18, 3):
    max_x = 0
    min_x = 0
    for n in range(N, N+3):
        frms = dataframes[n][2]
        data = dataframes[n][0]
        
        max_x = max(data['xBody'].max(), max_x)
        min_x = min(data['xBody'].min(), min_x)
    
        m = frms.apply(mean)
        
        i = n-N
        s = m.size
    
        plt.plot(range(i*s, (i+1)*s), m)
    
    plt.ylim((min_x, max_x))
    plt.title(f"Enregistrements {N}-{N+1}-{N+2}")
    plt.xlabel("Temps (s)")
    plt.ylabel("Distance (mm)")
    plt.show()

#%%

def mean(df):
    xs = df['xBody'].values
    
    return(xs.mean())

for N in range(11, 18, 3):
    i0 = 0
    t, x = [], []
    
    fig, ax = plt.subplots()
    
    for n in range(N, N+3):
        
        data = dataframes[n][0]
        frames_group = dataframes[n][2]
        
        t += list((data["imageNumber"] + i0)*T)
        x += list(data["xBody"])
        
        frames_to_time = (data["imageNumber"].unique() + i0) * T
    
        i0 += max(data["imageNumber"]+1)
        
        ax.plot(frames_to_time, frames_group.apply(mean), c='k')
        ax.axvline(i0*T,c='r', linestyle="dashed")
    
    hist = ax.hist2d(t, x, bins=(60,60), cmap=plt.cm.jet, density=True, range=np.array([(0, 900), (-10, 10)]))
    fig.colorbar(hist[3], ax=ax)
    ax.set_ylim((-10, 10))
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Distance en x par rapport au centre (mm)")
    #plt.show()
    plt.savefig(f"x_hist2D_{N}_{N+1}_{N+2}.png", dpi=300)
    plt.clf()
#%%

def mean_y(df):
    ys = df['yBody'].values
    
    return(ys.mean())

for N in range(11, 18, 3):
    i0 = 0
    t, y = [], []
    
    fig, ax = plt.subplots()
    
    for n in range(N, N+3):
        
        data = dataframes[n][0]
        frames_group = dataframes[n][2]
        
        t += list((data["imageNumber"] + i0)*T)
        y += list(data["yBody"])
        
        frames_to_time = (data["imageNumber"].unique() + i0) * T
    
        i0 += max(data["imageNumber"]+1)
        
        ax.plot(frames_to_time, frames_group.apply(mean_y), c='k')
        ax.axvline(i0*T,c='r', linestyle="dashed")
    
    hist = ax.hist2d(t, y, bins=(60,60), cmap=plt.cm.jet, density=True, range=np.array([(0, 900), (-10, 10)]))
    fig.colorbar(hist[3], ax=ax)
    ax.set_ylim((-10, 10))
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Distance en y par rapport au centre  (mm)")
    #plt.show()
    plt.savefig(f"y_hist2D_{N}_{N+1}_{N+2}.png", dpi=300)
    plt.clf()
    
#%%

#histograms on x 

for N in range(11, 18, 3):
    i0 = 0
    t, y = [], []
    
    fig, ax = plt.subplots()
    
    for n in range(N, N+3):
        
        data = dataframes[n][0]
        x = data["xBody"]
        ax.hist(x, label=f"enr. {n}", density=True, alpha=0.5, bins=30)
    
    ax.set_xlabel("Distance en x par rapport au centre  (mm)")
    ax.set_ylabel("Densité de position")
    ax.set_xlim((-10, 10))
    ax.legend(loc='center left', bbox_to_anchor=(0.9, 0.89))    
    #plt.show()
    plt.savefig(f"x_hists_{N}_{N+1}_{N+2}.png", dpi=300)
    plt.clf()

#%%

#histograms on y

for N in range(11, 18, 3):
    i0 = 0
    t, y = [], []
    
    fig, ax = plt.subplots()
    
    for n in range(N, N+3):
        
        data = dataframes[n][0]
        y = data["yBody"]
        ax.hist(y, label=f"enr. {n}", density=True, alpha=0.5, bins=30)
    
    ax.set_xlabel("Distance en y par rapport au centre  (mm)")
    ax.set_ylabel("Densité de position")
    ax.set_xlim((-10, 10))
    ax.legend(loc='center left', bbox_to_anchor=(0.9, 0.89))    
    #plt.show()
    plt.savefig(f"y_hists_{N}_{N+1}_{N+2}.png", dpi=300)
    plt.clf()

#%%
############################
### TEST CELLS TEMP CODE ###
############################

###  testing error bars on scatter plot  ###

df = pd.read_csv("D:\\Alexis\\20220805\\20220805_15\\Tracking_Result\\tracking.txt", sep='\t')

xs = df.groupby('id').mean()['xBody']
x = xs.mean()
ys = df.groupby('id').mean()['yBody']
y = ys.mean()

xserr = df.groupby('id').sem()['xBody'].values
yserr = df.groupby('id').sem()['xBody'].values

xerr = xs.sem()
yerr = ys.sem()

plt.errorbar(xs.values, ys.values, xerr=xserr, yerr=yserr, fmt='.')
plt.errorbar(x,y, xerr=xerr, yerr=yerr, fmt='.', c='r')
#%%
fig, ax = plt.subplots()
bars = ax.bar(np.arange(10), np.random.randint(2, 50, 10), fill=False, edgecolor='w')
for bar in bars:
    x, y = bar.get_xy()
    w, h = bar.get_width(), bar.get_height()
    ax.plot([x, x + w], [y + h, y + h], color='black', lw=2.5)
ax.margins(x=0.02)
plt.show()
#%%
# create data
x = np.random.normal(size=50000)
y = x * 3 + np.random.normal(size=50000)
 
# Big bins
plt.hist2d(x, y, bins=(50, 50), cmap=plt.cm.jet)
plt.show()