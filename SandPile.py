#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import random
import scipy as sp


params = {
   'axes.labelsize': 30,
   'font.size': 8,
   'legend.fontsize': 8,
   'xtick.labelsize': 20,
   'ytick.labelsize': 20,
   'figure.figsize': [20, 15]
   }



plt.rcParams.update(params)
def get_slope(h, i):    #Retreiving the slopes of each site from heights of neighbours
    z_i = h[i] - h[i+1]
    return z_i




def gen_slope_th(prob_zth_1):         #Generating the random threshold slope values (1,2 with 50% probability for each in Oslo model)
    val = random.uniform(0,1)
    if val <= prob_zth_1:
        z_th = 1
    else:
        z_th = 2
    return z_th



def oslo(grains, L, prob_zth_1):    #Runs the Oslo model for 'grains' grains, ssytem size L, P(z_threshold = 1) = prob_zth_1 
    """
    This will run the Oslo model and output values such as the h(1) for all time,
    the average height run across all time, total set of configurations of height
    and slopes, and full set of avalanches run 
    
    
    -------------
    Parameters:
    grains : number of grains added to the system (starts from empty configuration)
    
    L : system size chosen
    
    Prob_zth_1 : Value for the probability that a given slop has a threshold slope
    of 1. i.e. for Oslo model prob_zth_1 = 0.5, but for BTW model prob_zth_1 = 1
    
    -------------
    Returns:
        final z configuration
        final height configuration
        all height configurations in time
        labelled sites in the system
        the average size of h(0) (starting from empty system)
        height of site 1 at all times
        sum of all gradients at each time (equal to height of i=1)
        time as a list (from 0 up to grains)
        crossover time tc
        height of site 1 at final time step
        list of all avalanches in the system
    
    
    """
    sites = [i for i in range(L+1)]           #Placement of each site, from 0 to L+1
    h = [0 for i in range(L+1)]               #Heights of piles at each site
    z = [0 for i in range(L)]               #Gradients of each site
    z_th = [gen_slope_th(prob_zth_1) for i in range(L)]            #Threshold gradients
    all_h = []
   
    #To determine if the system is in steady state
   
   
    #Task 2 elements
    h_0_avg = []
    h_0_t = []                     #Height of site 0 after relaxing (i = 1 in notes) at some time (grain added)
    z_i_sum = []
    time = []
    h_0_t_2 = 0
   
    t_c = 0
   
   
    full_s = []
    test = 0

    for g in range(grains):
       
        h[0] += 1
        h[-1] = 0          
        z[0] = get_slope(h, 0)
        checklist = [0]
        if test > 0:
            full_s.append(s)
        s = 0

        while len(checklist) != 0:
            for i in checklist:
                while z[i] > z_th[i]:
                    if i == 0 :
                        z[0] += -2
                        h[0] += -1
                        z[1] += 1
                        h[1] += 1
                        s += 1
                        if z[1] > z_th[1]:
                            checklist.append(1)
                   
                    elif i == L-1:
                        z[L-1] += -1
                        h[L-1] += -1
                        z[L-2] += 1
                        h[L] += 1
                        s += 1
                        if z[-2] > z_th[-2]:
                            checklist.append(L-2)
                        test += 1
                    else:
                        z[i] += -2
                        h[i] += -1
                        z[i+1] += 1
                        h[i+1] += 1
                        s += 1
                        if z[i+1] > z_th[i+1]:
                            checklist.append(i+1)
                        z[i-1] += 1
                        if z[i-1] > z_th[i-1]:
                            checklist.append(i-1)                            
                         
                           
                    z_th[i] = gen_slope_th(prob_zth_1)
                checklist.remove(i)
               
            if test ==  1:
                t_c  = g
        h[-1] = 0
        for p in range(len(z)):
            if z[p] > z_th[p]:
                print("Error at ", p)
            elif h[p] - h[p+1] > 2:
                print("error at ", p)
               
        copy_h_2 = np.copy(h)
        copy_h = copy_h_2.tolist()
        all_h.append(copy_h)
               
        h_0_avg.append(h[0])  
       
        h_0_t.append(h[0])
       
        z_i_sum.append(np.sum(z))
       
        time.append(g)
        h_0_t_2 = h[0]
        h[-1] = 0
       
    return z, h, all_h, sites, np.mean(h_0_avg), h_0_t, z_i_sum, time, t_c, h_0_t_2, full_s


def height_set_all_time(grains, L, prob_zth_1):
   
    h_0_all_t = oslo(grains, L, prob_zth_1)[5]
   
    return h_0_all_t


def oslo_t_c(grains, L, prob_zth_1):     #Producing the Oslo model, where grains added represent the number of additions to the 0th site
    """
    This runs the Oslo model only until the first grain leaves the system, upon
    which it ends the program and returns the crossover time
    
    ----------------
    Inputs:
        grains to drive the system 
        system size L
        P(z_th = 1) = prob_zth_1
        
    ----------------
    Outputs:
        crossover time tc
    """

    h = [0 for i in range(L+1)]               #Heights of piles at each site
    z = [0 for i in range(L)]               #Gradients of each site
    z_th = [gen_slope_th(prob_zth_1) for i in range(L)]            #Threshold gradients
   
    #2b
    test = 0
   
   
    for g in range(grains):
        #Adding grain to 0th site and driving the avalanches
        h[0] += 1
        h[-1] = 0           #This is fixed at 0
        z[0] = get_slope(h, 0)
        checklist = [0]



        #Relaxation process
        while len(checklist) != 0:
            for i in checklist:
                while z[i] > z_th[i]:
                    if i == 0 :
                        z[0] += -2
                        h[0] += -1
                        z[1] += 1
                        h[1] += 1
                        if z[1] > z_th[1]:
                            checklist.append(1)
                           
                   
                    elif i == L-1:
                        z[L-1] += -1
                        h[L-1] += -1
                        z[L-2] += 1
                        h[L] += 1
                        if z[-2] > z_th[-2]:
                            checklist.append(L-2)
                        test += 1
                           
                    else:
                        z[i] += -2
                        h[i] += -1
                        z[i+1] += 1
                        h[i+1] += 1
                        if z[i+1] > z_th[i+1]:
                            checklist.append(i+1)
                        z[i-1] += 1
                        if z[i-1] > z_th[i-1]:
                            checklist.append(i-1)                            
                         
                           
                    z_th[i] = gen_slope_th(prob_zth_1)
                checklist.remove(i)
        if test > 0:
            break
       
       
    return g

def oslo_t_c_e(grains, L, prob_zth_1):     #Producing the Oslo model, where grains added represent the number of additions to the 0th site
    """
    This runs the Oslo model from an empty system, but only until the crossover 
    time has been reached
    
    ----------
    Inputs:
        grains to drive the system
        system size L
        P(z_th_1) = prob_zth_1
        
    ----------
    Outputs:
        crossover time
        last configuration of heights
        last configuration of slopes
        last configuration of threshold slopes
    
    """
    

    h = [0 for i in range(L+1)]               #Heights of piles at each site
    z = [0 for i in range(L)]               #Gradients of each site
    z_th = [gen_slope_th(prob_zth_1) for i in range(L)]            #Threshold gradients
   
    #2b
    test = 0
   
   
    for g in range(grains):
        #Adding grain to 0th site and driving the avalanches
        h[0] += 1
        h[-1] = 0           #This is fixed at 0
        z[0] = get_slope(h, 0)
        checklist = [0]



        #Relaxation process
        while len(checklist) != 0:
            for i in checklist:
                while z[i] > z_th[i]:
                    if i == 0 :
                        z[0] += -2
                        h[0] += -1
                        z[1] += 1
                        h[1] += 1
                        if z[1] > z_th[1]:
                            checklist.append(1)
                           
                   
                    elif i == L-1:
                        z[L-1] += -1
                        h[L-1] += -1
                        z[L-2] += 1
                        h[L] += 1
                        if z[-2] > z_th[-2]:
                            checklist.append(L-2)
                        test += 1
                           
                    else:
                        z[i] += -2
                        h[i] += -1
                        z[i+1] += 1
                        h[i+1] += 1
                        if z[i+1] > z_th[i+1]:
                            checklist.append(i+1)
                        z[i-1] += 1
                        if z[i-1] > z_th[i-1]:
                            checklist.append(i-1)                            
                         
                           
                    z_th[i] = gen_slope_th(prob_zth_1)
                checklist.remove(i)
        if test > 0:
            break
       
       
    return g, h, z, z_th


def M_averaging(grains, L, prob_zth_1, M):
    h_tilda = [0 for i in range(grains)]
    for i in range(M):
        h_0_new = height_set_all_time(grains, L, prob_zth_1)
        for j in range(grains):
            h_tilda[j] += h_0_new[j] / M
   
    return h_tilda
       
 
     
def e_average_height(grains, L, prob_zth_1, t_0, h_input, z_input, z_th_input):

    h = h_input  
    z = z_input          
    z_th = z_th_input      
    h_0_t = []
    #2b
   
   
    for g in range(grains):
        #Adding grain to 0th site and driving the avalanches
        h[0] += 1
        h[-1] = 0           #This is fixed at 0
        z[0] = get_slope(h, 0)
        checklist = [0]



        #Relaxation process
        while len(checklist) != 0:
            for i in checklist:
                while z[i] > z_th[i]:
                    if i == 0 :
                        z[0] += -2
                        h[0] += -1
                        z[1] += 1
                        h[1] += 1
                        if z[1] > z_th[1]:
                            checklist.append(1)
                           
                   
                    elif i == L-1:
                        z[L-1] += -1
                        h[L-1] += -1
                        z[L-2] += 1
                        h[L] += 1
                        if z[-2] > z_th[-2]:
                            checklist.append(L-2)
                           
                    else:
                        z[i] += -2
                        h[i] += -1
                        z[i+1] += 1
                        h[i+1] += 1
                        if z[i+1] > z_th[i+1]:
                            checklist.append(i+1)
                        z[i-1] += 1
                        if z[i-1] > z_th[i-1]:
                            checklist.append(i-1)                            
                         
                    h[-1] = 0    
                    z_th[i] = gen_slope_th(prob_zth_1)
                checklist.remove(i)
        h_0_t.append(h[0])
 
    return np.sum(h_0_t), h_0_t



def h_bra(grains, L, prob_zth_1):
    t_0_alt, h_input, z_input, z_th_input =  oslo_t_c_e(grains, L, prob_zth_1)
    t_0 = t_0_alt + 5000
    h_bra = 0
    T = grains - t_0
    h_bra = e_average_height(T, L, prob_zth_1, t_0, h_input, z_input, z_th_input)[0] / T
    return h_bra, T
   
   
def e_sigma(grains, L, prob_zth_1):
    t_0_alt, h_input, z_input, z_th_input =  oslo_t_c_e(grains, L, prob_zth_1)
    t_0 = t_0_alt + 5000
    T = grains - t_0
    avg_height_list = e_average_height(T, L, prob_zth_1, t_0, h_input, z_input, z_th_input)[1]
    return T, avg_height_list

def get_rec_configs(grains, L, prob_zth_1, t_0, h_input, z_input, z_th_input):
    """
    This produces the number of recurrent configurations, as it only runs the Olso
    model for t > tc
    
    --------------
    Inputs:
        grains to drive the sysetem for t > tc
        system size L
        P(z_th = 1) 
        time input to continue adding to
        height configuration to continue from
        slop configuration to continue from
        z_th configuration to continue from
        
    --------------
    Outputs:
        length of unique recurrent configurations
    
    """
    h = h_input  
    z = z_input          
    z_th = z_th_input      
 
    configs = []
    counter_h_of_size_L = [0 for i in range(2*L + 1)]

    for g in range(grains):
        h[0] += 1
        h[-1] = 0           #This is fixed at 0
        z[0] = get_slope(h, 0)
        checklist = [0]


        #Relaxation process
        while len(checklist) != 0:
            for i in checklist:
                while z[i] > z_th[i]:
                    if i == 0 :
                        z[0] += -2
                        h[0] += -1
                        z[1] += 1
                        h[1] += 1

                        if z[1] > z_th[1]:
                            checklist.append(1)

                 
                    elif i == L-1:
                        z[L-1] += -1
                        h[L-1] += -1
                        z[L-2] += 1
                        h[L] += 1
                        if z[-2] > z_th[-2]:
                            checklist.append(L-2)
                         
                    else:
                        z[i] += -2
                        h[i] += -1
                        z[i+1] += 1
                        h[i+1] += 1
                        if z[i+1] > z_th[i+1]:
                            checklist.append(i+1)
                        z[i-1] += 1
                        if z[i-1] > z_th[i-1]:
                            checklist.append(i-1)  
                    z_th[i] = gen_slope_th(prob_zth_1)
                checklist.remove(i)
                h[-1] = 0
        included_edy = 0
        for b in range(len(configs)):
            if h == configs[b]:
                included_edy += 1
                break
        if included_edy == 0:
            q = np.copy(h)
            configs.append(q.tolist())
            counter_h_of_size_L[h[0]] += 1
       
    return len(configs)

def Task_1_rec_configs(grains, L, prob_zth_1):
   
    t_0_alt, h_input, z_input, z_th_input =  oslo_t_c_e(grains, L, prob_zth_1)
    T = 20000
   
    Nr = get_rec_configs(grains, L, prob_zth_1, T, h_input, z_input, z_th_input)
    return Nr
   
   
   
   
   
def get_configs(grains, L, prob_zth_1, t_0, h_input, z_input, z_th_input):
    h = h_input  
    z = z_input          
    z_th = z_th_input      
   
    counter_h_of_size_L = [0 for i in range(2*L + 1)]

    for g in range(grains):
        h[0] += 1
        h[-1] = 0          
        z[0] = get_slope(h, 0)
        checklist = [0]


        #Relaxation process
        while len(checklist) != 0:
            for i in checklist:
                while z[i] > z_th[i]:
                    if i == 0 :
                        z[0] += -2
                        h[0] += -1
                        z[1] += 1
                        h[1] += 1

                        if z[1] > z_th[1]:
                            checklist.append(1)

                   
                    elif i == L-1:
                        z[L-1] += -1
                        h[L-1] += -1
                        z[L-2] += 1
                        h[L] += 1
                        if z[-2] > z_th[-2]:
                            checklist.append(L-2)
                           
                    else:
                        z[i] += -2
                        h[i] += -1
                        z[i+1] += 1
                        h[i+1] += 1
                        if z[i+1] > z_th[i+1]:
                            checklist.append(i+1)
                        z[i-1] += 1
                        if z[i-1] > z_th[i-1]:
                            checklist.append(i-1)  
                    z_th[i] = gen_slope_th(prob_zth_1)
                checklist.remove(i)
                h[-1] = 0
            counter_h_of_size_L[h[0]] += 1
    return counter_h_of_size_L



def Task_2_g(grains, L, prob_zth_1):
    t_0_alt, h_input, z_input, z_th_input =  oslo_t_c_e(grains, L, prob_zth_1)
    t_0 = t_0_alt + 5000
    T = 2000000

    counter_h_of_size_L = get_configs(T, L, prob_zth_1, t_0, h_input, z_input, z_th_input)
    return counter_h_of_size_L

def logbin(data, scale = 1., zeros = False):
    """
    logbin(data, scale = 1., zeros = False)

    Log-bin frequency of unique integer values in data. Returns probabilities
    for each bin.

    Array, data, is a 1-d array containing full set of event sizes for a
    given process in no particular order. For instance, in the Oslo Model
    the array may contain the avalanche size recorded at each time step. For
    a complex network, the array may contain the degree of each node in the
    network. The logbin function finds the frequency of each unique value in
    the data array. The function then bins these frequencies in logarithmically
    increasing bin sizes controlled by the scale parameter.

    Minimum binsize is always 1. Bin edges are lowered to nearest integer. Bins
    are always unique, i.e. two different float bin edges corresponding to the
    same integer interval will not be included twice. Note, rounding to integer
    values results in noise at small event sizes.

    Parameters
    ----------

    data: array_like, 1 dimensional, non-negative integers
          Input array. (e.g. Raw avalanche size data in Oslo model.)

    scale: float, greater or equal to 1.
          Scale parameter controlling the growth of bin sizes.
          If scale = 1., function will return frequency of each unique integer
          value in data with no binning.

    zeros: boolean
          Set zeros = True if you want binning function to consider events of
          size 0.
          Note that output cannot be plotted on log-log scale if data contains
          zeros. If zeros = False, events of size 0 will be removed from data.

    Returns
    -------

    x: array_like, 1 dimensional
          Array of coordinates for bin centres calculated using geometric mean
          of bin edges. Bins with a count of 0 will not be returned.
    y: array_like, 1 dimensional
          Array of normalised frequency counts within each bin. Bins with a
          count of 0 will not be returned.
    """
    if scale < 1:
        raise ValueError('Function requires scale >= 1.')
    count = np.bincount(data)
    tot = np.sum(count)
    smax = np.max(data)
    if scale > 1:
        jmax = np.ceil(np.log(smax)/np.log(scale))
        if zeros:
            binedges = scale ** np.arange(jmax + 1)
            binedges[0] = 0
        else:
            binedges = scale ** np.arange(1,jmax + 1)
            # count = count[1:]
        binedges = np.unique(binedges.astype('uint64'))
        x = (binedges[:-1] * (binedges[1:]-1)) ** 0.5
        y = np.zeros_like(x)
        count = count.astype('float')
        for i in range(len(y)):
            y[i] = np.sum(count[binedges[i]:binedges[i+1]]/(binedges[i+1] - binedges[i]))
    else:
        x = np.nonzero(count)[0]
        y = count[count != 0].astype('float')
        if zeros != True and x[0] == 0:
            x = x[1:]
            y = y[1:]
    y /= tot
    x = x[y!=0]
    y = y[y!=0]
    return x,y





def s_k(grains, L, prob_zth_1, h_input, z_input, z_th_input, k):
    """
    This runs the Oslo model for t > tc and returns the total number of avalanches
    
    ----------
    Inputs:
        grains to drive the system
        system size L
        P(z_th = 1)
        Input height configuration 
        Input slop configuration
        Input z_th configuration 
        kth moment
 
    """
    
    h = h_input  
    z = z_input          
    z_th = z_th_input      
    full_s = []
    for g in range(grains):
        h[0] += 1
        h[-1] = 0          
        z[0] = get_slope(h, 0)
        checklist = [0]
        if g > 0:
            full_s.append(s_float**k)
        s = 0
       


        #Relaxation process
        while len(checklist) != 0:
            for i in checklist:
                while z[i] > z_th[i]:
                    if i == 0 :
                        z[0] += -2
                        h[0] += -1
                        z[1] += 1
                        h[1] += 1
                        if z[1] > z_th[1]:
                            checklist.append(1)
                        s += 1
                           
                   
                    elif i == L-1:
                        z[L-1] += -1
                        h[L-1] += -1
                        z[L-2] += 1
                        h[L] += 1
                        if z[-2] > z_th[-2]:
                            checklist.append(L-2)
                        s += 1
                    else:
                        z[i] += -2
                        h[i] += -1
                        z[i+1] += 1
                        h[i+1] += 1
                        if z[i+1] > z_th[i+1]:
                            checklist.append(i+1)
                        z[i-1] += 1
                        if z[i-1] > z_th[i-1]:
                            checklist.append(i-1)                            
                        s += 1
                           
                    z_th[i] = gen_slope_th(prob_zth_1)
                checklist.remove(i)
        s_float = float(s)
        if s < 0:
            print("Negative avalanche size at grain ", g)
   
   
    return np.sum(full_s)



def Task_3c(grains, L, prob_zth_1, k):
    t_0_alt, h_input, z_input, z_th_input =  oslo_t_c_e(grains, L, prob_zth_1)
    t_0 = t_0_alt + 5000
    T = 100000
    bra_s_k = s_k(T, L, prob_zth_1, h_input, z_input, z_th_input, k) / T
    return bra_s_k

def std_dev(a):
    T= len(a)
    val_1_p = []
    for i in range(len(a)):
        val_1_p.append(a[i]**2 / T)
    val_2 = (sum(a) / T) ** 2
    sig = np.sqrt(sum(val_1_p) - val_2)
   
    return sig


def find_trans_configs(grains, L, prob_zth_1):
    full_configs = oslo(grains, L, prob_zth_1)[2]
    trans_counter = 0
    for i in range(len(full_configs)):
        rec_test = 0
        for p in range(len(full_configs)):
            if full_configs[i] == full_configs[p]:
                rec_test += 1
        if rec_test == 1:
            trans_counter += 1
    return trans_counter, full_configs


def full_config_counter(grains, L, prob_zth_1):
    Nt = find_trans_configs(grains, L, prob_zth_1)[0]
    Nr = Task_1_rec_configs(grains, L, prob_zth_1)
    Ns = Nt + Nr
   
    return Nr, Nt, Ns

 
 
labels = ['L = 4', 'L = 8','L = 16','L = 32','L = 64','L = 128','L = 256','L = 512']

k_labels = ['k = 1', 'k = 2', 'k = 3', 'k = 4']

#%%

L = 16
grains = 10000
z_1, h_1, all_h_1, sites_1, h_0_avg_1, h_0_t_1, z_i_sum_1, time_1, t_c_1, h_0_t_2_1, s_1 = oslo(grains, L, 0.5)


#Visualising the grain system


fig = plt.figure()
plt.xlim([-2, 32])
plt.ylim([-2, 32])

n = grains
bar = plt.bar(sites_1, all_h_1[-1], width = 0.9)


def animate(i):
    y = all_h_1[i+1]
    for i, b in enumerate(bar):
        b.set_height(y[i])
       
anim = animation.FuncAnimation(fig, animate, repeat = True, blit = False, frames=n,
                               interval = 1)

plt.show()

#%%
#Task 1
#L and grains can be changed 
grains = 1000
L = 4
#Finding Transient Configurations in the BTW Model
Nt = find_trans_configs(grains, L, 1)[0]

#Finding recurrent Configurations in BTW Model
Nr = full_config_counter(grains, L, 1)[0]

#Finding recurrent Configurations in Oslo Model
Nr_Oslo = full_config_counter(grains, 2, 0.5)[0]
#%%
#Task 2a)
#From here is advisable to set grains = 1000000 for decent statistics
#May find 'grain sets' that are more specific for given calculations


grains = 1000000


L_val = [4,8,16,32,64,128,256, 512]
h_0_t = [[0 for i in range(grains)] for i in range(len(L_val))]
M = 2
tc_2_a = [0 for i in range(len(L_val))]

 
#With averaging
for j in range(M):
    u = 0
    for i in L_val:
        a, b, h_comp_val, c, d, h_0_val, r, g, tc_av, fdgfd, io = oslo(grains, i, 0.5)
        h_0_t_test = [0 for i in range(len(h_0_val))]
        for p in range(len(h_0_val)):
            h_0_t_test[p] += h_0_val[p] / M
        tc_2_a[u] += tc_av / M
        for r in range(len(h_0_t_test)):
            h_0_t[u][r] += h_0_t_test[r]
           
        print(i, " done")
        u += 1

Task_2a_h_0_t = h_0_t
       
       







#%%
#Task 2b)
#Finding values of tc (with averaging)

M = 5
L_val = [4,8,16,32,64,128,256, 512]
t_c_val_2 = [0 for i in range(len(L_val))]

## With averaging
for i in range(M):
    t = 0
    for j in L_val:

        t_c_new_2 = oslo_t_c(grains, j, 0.5)
        t_c_val_2[t] += (t_c_new_2 / M)
        t += 1
       
#%%
#Plotting results of 2b)
#Quadratic plot of tc with system size L
plt.figure()

plt.plot(L_val, t_c_val_2, 'o', ms = 7, color = 'black', label = 'Mean crossover time $\\langle t_{c} \\rangle$')

fit, cov = np.polyfit(L_val, t_c_val_2, 2, cov = True)

x = np.linspace(0, max(L_val), 1000)
y = []
for i in range(len(x)):
    y.append(fit[0]*x[i]**2 + fit[1]*x[i] + fit[2])

plt.plot(x, y, '--', color = 'red', label = 'Fitted Quadratic')

plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 27, markerscale = 1)
   
plt.xlabel('L', fontsize = 30)
plt.ylabel('$\\langle t_{c}\\rangle$', fontsize = 30)

plt.show()

#%%
#Finding scaling form of tc

tc_fit, tc_cov = np.polyfit(np.log(L_val), np.log(t_c_val_2), 1, cov = True)

x_tc_fit = np.linspace(0, max(L_val) + 5)
y_tc_fit = []
for i in range(len(x_tc_fit)):
    new = np.exp(tc_fit[1]) * (x_tc_fit[i] ** tc_fit[0])
    y_tc_fit.append(new)
   
plt.plot(x_tc_fit, y_tc_fit, '--')
plt.plot(L_val, t_c_val_2, 'o')

#%%
#Reducing tc by scaling form

tc_scaled_y = []
plt.xlabel("L")
plt.ylabel("$\\ t_{c}}$ / $ 0.9  L^{1.99}$")
for i in range(len(L_val)):
    new_val = t_c_val_2[i] / (np.exp(tc_fit[1]) * (L_val[i] ** tc_fit[0]))
    tc_scaled_y.append(new_val)
   
plt.plot(L_val, tc_scaled_y, 'o', color = 'black',  label = '$\\ t_{c}}$ / $ 0.9  L^{1.99}$ Data points')

t_c_fit_horiz, t_c_fit_horiz_const = np.polyfit(L_val, tc_scaled_y, 0, cov = True)
x_horiz_tc = np.linspace(min(L_val), max(L_val), 1000)
y_horiz_tc = []
for i in range(len(x_horiz_tc)):
    new_val = t_c_fit_horiz[0] * (x_horiz_tc[i]**0)
    y_horiz_tc.append(new_val)
plt.plot(x_horiz_tc, y_horiz_tc, '--', color = 'red', label = 'Best fit line')



plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 25, markerscale = 1)
       
#%%
#Plotting results from 2a): Seeing where the critical point occurs in time  
   

plt.figure()
time_1 = []
for i in range(len(Task_2a_h_0_t[0])):
    time_1.append(i)
time_pre_tc = [[] for i in range(len(Task_2a_h_0_t))]
h_pre_tc = [[] for i in range(len(Task_2a_h_0_t))]
time_post_tc = [[] for i in range(len(Task_2a_h_0_t))]
h_post_tc = [[] for i in range(len(Task_2a_h_0_t))]
for j in range(len(Task_2a_h_0_t)):
    for i in range(len(Task_2a_h_0_t[0])):
        if i < t_c_val_2[j]:
        #if i < tc_2_a[j]:
            time_pre_tc[j].append(i)
            h_pre_tc[j].append(Task_2a_h_0_t[j][i])
        else:
            time_post_tc[j].append(i)
            h_post_tc[j].append(Task_2a_h_0_t[j][i])
   

##For black and red of just one length to see where tc occurs 
#for j in range(6, 7):
#
#    plt.plot(time_pre_tc[j], h_pre_tc[j], ms=1, color = 'black', label = "h before $t_{c}$")
#    plt.plot(time_post_tc[j], h_post_tc[j], ms = 1, color = 'red', label = "h after $t_{c}$")  
#
#    plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
#                                     fontsize = 15, markerscale = 3)
#  
           
###For all L
for i in range(len(Task_2a_h_0_t)):
    plt.plot(time_1, Task_2a_h_0_t[len(Task_2a_h_0_t) - 1 - i],ms = 1, label = L_val[len(Task_2a_h_0_t) - 1 - i])            
    plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 17, markerscale = 1)
       
           
           
plt.xlabel('Time', fontsize = 30)
plt.ylabel('Height', fontsize = 30)

plt.show()




#%%
#Task 2d)
#Producing sets of averaged statistics to better produce the data collapse
L_val = [4,8,16,32,64, 128, 256, 512]
h_tilda_set = []

t = np.linspace(0, grains, grains)

for i in L_val:
    new_h = M_averaging(grains, i, 0.5, 5)
    print("Finished i is ", i)
    h_tilda_set.append(new_h)
   
#%%
#Plotting form of scaling function F(x)

#Scaling the y-axis; y -> y/L
h_tilda_set_over_L = [[0 for i in range(len(h_tilda_set[0]))] for y in range(len(L_val))]
for j in range(len(L_val)):
    for i in range(len(h_tilda_set[j])):
        new_h = (h_tilda_set[j][i]) / (L_val[j] )
        h_tilda_set_over_L[j][i] += (new_h)
       
#Scaling the x-axis; t -> t/L^2
t_over_L_D = [[] for i in range(len(L_val))]
for j in range(len(L_val)):    
    for i in range(len(t)):
        t_over_L_D[j].append(t[i] / (L_val[j]**2))
       

plt.figure()

for i in range(len(L_val)):
    plt.loglog((t_over_L_D[i][10:1000000]), (h_tilda_set_over_L[i][10:1000000]), label =  labels[i])    
    plt.xlabel('t / $L^2$ ', fontsize = 30)
    plt.ylabel('$\\bar h$ / $L$', fontsize = 30)
    plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 20, markerscale = 1)
plt.show()

#%%
#Finding h dependence on t from gradient  the Data Collapsed data
#L_val_to_inspect is simply the value of L to take the gradient from

L_val_to_inspect = 7 

h_tilda_t_prop = h_tilda_set[L_val_to_inspect][100:40000]
h_tilda_t_prop_t = t[100:40000]
plt.xlabel("t")
plt.ylabel("$\\bar h$")
plt.plot(np.log(h_tilda_t_prop_t), np.log(h_tilda_t_prop), label = L_val[L_val_to_inspect])

h_dep_on_t_grad, end_2_d_cov = np.polyfit(np.log(h_tilda_t_prop_t), np.log(h_tilda_t_prop), 1, cov = True)


print("h scales as L^", h_dep_on_t_grad[0])
#%%
#Task 2e)
#Producing values for average height

h_avg_set = []

for i in L_val:
    h_bra_new = h_bra(grains, i, 0.5)[0]
    print(i, " done")
    h_avg_set.append(h_bra_new)
       
   
   
#%%
#Plotting <h> against L, showing a linear relationship
   
   
plt.xlabel("L")
plt.ylabel("$\\langle h(t;L) \\rangle_{t} $")
#plt.plot(L_val, h_avg_set, '--', color = 'black')
fit_2_e, cov_2_e = np.polyfit(L_val, h_avg_set, 1, cov = True)
x_2_e = np.linspace(0, max(L_val), 1000)
plt.plot(L_val, h_avg_set, 'x', ms = 8, color = 'red', label = 'Measured values of $\\langle h(t;L) \\rangle_{t} $')
y_2_e = []
for i in range(len(x_2_e)):
    new_val = (fit_2_e[0]*x_2_e[i])+(fit_2_e[1])
    y_2_e.append(new_val)
plt.plot(x_2_e, y_2_e, '--', label = 'Linear fit')
plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 15, markerscale = 1)





#%%
#Finding corrections due to scaling by dividing <h> by L


new_h_over_L = []
for i in range(len(L_val)):
    new_h_over_L.append(h_avg_set[i] / L_val[i])
   
plt.xlabel("L")
plt.ylabel("$\\frac{\\langle h \\rangle}{L}$")
plt.plot(L_val, new_h_over_L)
plt.plot(L_val, new_h_over_L, 'x')




#%%
#Finding a0 roughly
plt.plot(L_val, h_avg_set)
fit, cov = np.polyfit(L_val, h_avg_set, 1, cov = True)
a0 = fit[0]



#%%
#Covariance test to find a0
a_0_test = np.linspace(1.73, 1.8, 10000)
cov_2_e_set = []



cov_set = []
fit_set = []

for j in range(len(a_0_test)):
    final_y = []

    for i in range(len(L_val)):
        new_y = (1-(h_avg_set[i] / (L_val[i]*a_0_test[j])))
        final_y.append(new_y)
       
    test_with_this_x = []
    test_with_this_y = []
    for i in range(len(L_val)):
        new_x = np.log(L_val[i])
        test_with_this_x.append(new_x)
        new_y = np.log(final_y[i])
        test_with_this_y.append(new_y)
    fit, cov = np.polyfit(test_with_this_x, test_with_this_y, 1, cov = True)
    fit_set.append(cov[0][0])



best_val = 0
for i in range(len(fit_set)):
    if fit_set[i] == min(fit_set):
        best_val = i


#%%
#Plot to retrieve Figure 4
print("Best value of a0 is ", a_0_test[best_val])
plt.ylabel("Covariance Entry")
plt.xlabel("Value of $a_{0}$")
plt.plot(a_0_test, fit_set, '--')

#%%

#Plotting with a0 to retrieve Figure 5

cov_2_e_set = []
final_y = []
for i in range(len(L_val)):
    new_y = (1-(h_avg_set[i] / (L_val[i]*a_0_test[best_val])))
    #new_y = (1-(full_h_avg[i] / (full_L[i]*1.7286104610461046)))

    final_y.append(new_y)
plt.plot(np.log(L_val), np.log(final_y))
plt.ylabel("ln (1 - $\\frac{\\langle h \\rangle}{L a_{0}}$)")
plt.xlabel("ln (L)")
fit_1, cov_1 = np.polyfit(np.log(L_val), np.log(final_y), 1, cov = True)
cov_2_e_set.append(cov[0][0])
w1 = -fit_1[0]
a1 = fit_1[1]
x_2g_val = np.linspace(min(np.log(L_val)-1), max(np.log(L_val)+1))

y_val_2_e = []
for i in range(len(x_2g_val)):
    new_val = (x_2g_val[i]*fit_1[0]) + fit_1[1]
    y_val_2_e.append(new_val)

#plt.loglog((full_L), (final_y))
plt.plot(np.log(L_val), np.log(final_y), 'x', ms = 15, color = 'black', label = "Calculated values of ln (1 - $\\frac{\\langle h \\rangle}{L a_{0}}$)" )
plt.plot(x_2g_val, y_val_2_e, '--', color = 'red', label = "Fitted polynomial of order 1")


plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 20, markerscale = 1)



#%%
#Task 2f)
#Showing how std. dev. of height scales with L, will be used to find Figures 6, 7

M = 1
L_val_2_f = [4,8,16,32, 64,128, 256, 512]

sigma_oslo = []
for m in range(M):
    j = 0
    for i in L_val_2_f:
        new_sigma, sigma_oslo_new = e_sigma(grains, i, 0.5)
        T = len(sigma_oslo_new)
        test_1 = []
        for p in range(len(sigma_oslo_new)):
            test_1.append(sigma_oslo_new[p]**2 / T)
        test_2 = (sum(sigma_oslo_new) / T) ** 2
       
        sigma_oslo.append(np.std(sigma_oslo_new))
        print(L_val_2_f[j], " done")
        j += 1
   
#%%
#Log graph fit of sigma against L
plt.xlabel("L")
plt.ylabel("$\sigma_{h}$")
sigma_fit, sigma_cov = np.polyfit(np.log(L_val_2_f),np.log(sigma_oslo), 1, cov = True)

x_2f_val = np.linspace(min(np.log(L_val_2_f)-1), max(np.log(L_val_2_f)+1))

y_val_2_f = []
for i in range(len(x_2f_val)):
    new_val = (x_2f_val[i]*sigma_fit[0]) + sigma_fit[1]
    y_val_2_f.append(new_val)
   
plt.plot(np.log(L_val_2_f), np.log(sigma_oslo), 'x', ms = 15, color = 'black', label = "Measured values of $\\sigma _{L}$" )
plt.plot(x_2f_val, y_val_2_f, '--', color = 'red', label = "Fitted polynomial of order 1")


plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 25, markerscale = 1)


print("sigma is prop. to  ", sigma_fit[0])
print("sigma a is ", np.exp(sigma_fit[1]))



#%%
#Non log graph of sigma against L
x_val_2f = (np.linspace(min(L_val_2_f), max(L_val_2_f), 100)).tolist()
y_val_2f = []
for i in range(len(x_val_2f)):
    new_val = np.exp(sigma_fit[1]) * (x_val_2f[i] ** sigma_fit[0])
    y_val_2f.append(new_val)
plt.plot(x_val_2f, y_val_2f, '--', color = 'black', label = "Fit of form $ \\propto L^{0.24}$")

plt.xlabel("L")
plt.ylabel("$\\sigma _{h} (L)$")
for i in range(8):
    plt.plot(L_val_2_f[i], sigma_oslo[i], 'x', ms = 20, label = labels[i])

   
plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 20, markerscale = 0.5)      

#%%
#Showing that sigma/a L^beta is approximately constant
sigma_over_L = []

for i in range(len(sigma_oslo)):
    new_val = sigma_oslo[i] / (np.exp(sigma_fit[1]) * (L_val_2_f[i] ** sigma_fit[0]))
    sigma_over_L.append(new_val)

plt.ylabel("$\\sigma _{L}$ / $a L^{\\beta}$")
plt.xlabel("L")

plt.plot(L_val_2_f, sigma_over_L, 'o',  label = '$\\sigma _{L}$ / $a L^{\\beta}$ Data points')

sigma_const_fit, sigma_const_cov = np.polyfit(L_val_2_f, sigma_over_L, 0, cov = True)

x_sig_con = np.linspace(0, max(L_val_2_f))
y_sig_con = []
for i in range(len(x_sig_con)):
    new_val = sigma_const_fit[0]
    y_sig_con.append(new_val)
   
plt.plot(x_sig_con, y_sig_con, '--',  label = 'Best fit line')

plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 27, markerscale = 1)


#%%
##Task 2g)
#Finding configuration of height h probabilities
#Note this will run for 2000000 grains once in steady state to ensure good statistics 
L_val = [4,8,16, 32, 64, 128, 256, 512]

grains = 200000

full_x_val_2g = []
full_P = []
full_config = []
full_counter = []
for i in L_val:
    counter_h_of_size_L = Task_2_g(grains, i, 0.5)
    P = []
    x_val = []
    for j in range(len(counter_h_of_size_L)):
            P.append(counter_h_of_size_L[j] / (sum(counter_h_of_size_L)))
            x_val.append(j)
    full_x_val_2g.append(x_val)
    full_P.append(P)

    full_counter.append(counter_h_of_size_L)
    print(i, " done")

plt.figure()






#%%
#Plotting results of config. height probs. -> expect Gaussian forms

for i in range(len(full_P)):
    plt.xlabel("h")
    plt.ylabel("P(h;L)")
    plt.plot(full_x_val_2g[i], full_P[i], label = L_val[i])
    plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 15, markerscale = 1)

plt.show()


#%%
#Producing <h> for the following data collapse

h_bras = [0 for i in range(len(L_val))]

j = 0
for i in (L_val):
    new_val = h_bra(grains, i, 0.5)[0]
    h_bras[j] += new_val
    print(L_val[j], " done")
    j += 1


#%%
#Data collapsing
P_adj = [[] for i in range(len(full_P))]
x_adj = [[] for i in range(len(full_P))]

for i in range(len(full_P)):
    for j in range(len(full_P[i])):
        new_val = full_P[i][j] * sigma_oslo[i]

        P_adj[i].append(new_val)
        new_val_2 = (full_x_val_2g[i][j] - h_bras[i]) / sigma_oslo[i]

        x_adj[i].append(new_val_2)
#%%
#Plotting result of data collapse -> expect overlapping general Gaussian form
       
plt.figure()

labels = ['L = 4', 'L = 8','L = 16','L = 32','L = 64','L = 128','L = 256','L = 512']


plt.xlabel("(h - $\\langle h \\rangle$) / $\\sigma_{h}$")
plt.ylabel("$\\sigma_{h}$P(h;L)")
axes = plt.gca()

axes.set_xlim([-3.5,3.5])

for i in range(len(P_adj)):
    plt.plot(x_adj[i], P_adj[i], label = labels[i])
    plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 20, markerscale = 1)
plt.show()  
   
   

#%%
#Task 3a)
#Producing sets of avalanches for different L

L_val = [4,8,16,32,64,128, 256, 512]
grains = 1500000

full_s = []
full_s_norm = []

#grains_set = [700000,700000,700000,700000,700000,700000,800000,1500000]

full_x_val_3a = []

full_ava = []
p = 0
for i in L_val:
    avalanches = oslo(grains, i, 0.5)[10]
    full_ava.append(avalanches)
    p += 1
    s = [0 for j in range(max(avalanches)+1)]
    s_norm = []


    for j in avalanches:
        s[avalanches[j]] += 1
  
  
    x_val = []
    for j in range(len(s)):
        x_val.append(j)
  
    full_x_val_3a.append(x_val)
    print(i, " done")
  



#%%
#Plotting log-binned distribution for L = 256 in order to extract tau alone

a_x_full = []
a_y_full = []
plt.figure()

x_512 = []
y_512 = []

plt.xlabel("s")
plt.ylabel("P(s; L)")


for i in range(len(L_val)):
#for i in range(7,8):
    s, P = logbin(full_ava[i], scale = 1.2)
    x, y = logbin(full_ava[i], scale = 1)
    a_x_full.append(s)
    a_y_full.append(P)

    plt.loglog((s), (P), label = labels[i])
    
    
   ###To show for just L=512, change range to (7,8) and uncomment
#    plt.loglog((s), (P), label = 'L = 512 - log-binned distribution', color = 'blue')
#    plt.loglog(x, y, 'o', label = 'L = 512 - Constant bin sizes' , ms = 1, color = 'red')


    if L_val[i] == 512:
        x_512.append(s.tolist())
        y_512.append(P.tolist())

    plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 20, markerscale = 1)
    #plt.plot(np.log(full_x_val_3a[i]), np.log(full_s_norm[i]))
plt.show


#%%
#Finding tau from linear region of 256 log-binned distribution

x_tau_val = x_512[0][10:55]
y_tau_val = y_512[0][10:55]
plt.loglog(x_tau_val, y_tau_val)
new_fit, new_cov = np.polyfit(np.log10(x_tau_val), np.log10(y_tau_val), 1, cov = True)
tau = -new_fit[0]

#%%
#Task 3b)
#Vertically aligning bumps for data collapse


tau = 1.56
L_val_3b = L_val[2:8]
labels_3b = labels[2:8]
full_ava_3b = full_ava[2:8]

plt.xlabel("s")
plt.ylabel("$s^{\\tau_{s}}$P(s;L)")

for i in range(len(L_val_3b)):
    s, P = logbin(full_ava_3b[i], scale = 1.2)
    sP = []
    print("For ", i, " len(s) is ", len(s))
    for j in range(len(s)):
        sP.append(P[j] * (s[j] ** tau))
    s_L_D = []
    for j in range(len(s)):
        new_s = s[j] / L_val[i]
        s_L_D.append(new_s)
    plt.loglog((s), (sP), label = labels_3b[i])
#    plt.loglog((s), (sP), 'x')

    plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 25, markerscale = 1)
 

plt.show()

#%%
#Horizontally aligning distributions for data collapse (now fully collapsed)

D = 2.25
tau = 1.56


plt.xlabel("s / $L^D$")
plt.ylabel("$s^{\\tau_{s}}$P(s;L)")

for i in range(len(L_val_3b)):
    s, P = logbin(full_ava_3b[i], scale = 1.2)
    sP = []
    for j in range(len(s)):
        sP.append(P[j] * (s[j] ** tau))
    s_L_D = []
    for j in range(len(s)):
        new_s = s[j] / (L_val[i]**D)
        s_L_D.append(new_s)
    #plt.loglog((s_L_D), (sP), 'x')
    plt.loglog((s_L_D), (sP),  label = labels_3b[i])

    plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 25, markerscale = 1)
 

plt.show()



#%%
#Task 3c)
#Finding kth moment for avalanche sizes

L_val = [4, 8,16, 32, 64, 128, 256, 512]
s_k_full = [[] for i in range(len(L_val))]
grains_set = [1000000,1000000,1000000,3000000,3000000,4000000]

k = [1,2,3,4]
M = 1
s_k_full_test = [[] for i in range(len(L_val))]

for i in range(M):
    m = 0

    for j in L_val:
        temp_sum = []
        for i in k:
            s_new = Task_3c(grains, j, 0.5, i)
            temp_sum.append(s_new / M)
            print("k ", i, " done")
#        s_k_full[m].append(temp_sum)
        s_k_full_test[m].append(temp_sum)
        print("L ", j, " done")
        m += 1
   
#Rearranging data to work quicker
   
y_3_end = [[] for i in range(len(k))]
for i in range(len(s_k_full_test[0][0])):
    for j in range(len(s_k_full_test)):
        y_3_end[i].append((s_k_full_test[j][0][i]))
         
#%%
#Plotting kth moment results 

       
k_labels = ['k = 1', 'k = 2', 'k = 3', 'k = 4']
s_k_grads = [0 for i in range(len(k))]

#y_3_end_shortened = []
#for i in range(len(y_3_end)):
#    y_3_end_shortened.append(y_3_end[i][3:7])
axes = plt.gca()

axes.set_xlim([10,600])

for i in range(len(k)):
    plt.loglog((L_val[2:8]), (y_3_end[i][2:8]),label = k_labels[i])
    plt.loglog((L_val[2:8]), (y_3_end[i][2:8]), 'x', color = 'black')
    plt.xlabel("L")
    plt.ylabel("$\\langle s^{k} \\rangle$")
    fit, cov = np.polyfit(np.log(L_val[2:8]), np.log(y_3_end[i][2:8]), 1, cov = True)
    s_k_grads[i] += (fit[0])
    plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 25, markerscale = 3)

plt.show()




#%%
#Calculating D and tau graphically
plt.xlabel('k', fontsize = 30)

plt.ylabel('D(1 + k - $\\tau_{s}$)', fontsize = 30)  

D_calc, D_cov = np.polyfit(k[0:4],s_k_grads[0:4],1, cov = True)
x_3_c = np.linspace(0, 5, 1000)
y_3_c = []
for i in range(len(x_3_c)):
    new_val = (x_3_c[i] * D_calc[0]) + D_calc[1]
    y_3_c.append(new_val)
plt.plot(x_3_c, y_3_c, '--', label = 'Fitted polynomial')
plt.plot(k[0:4], s_k_grads[0:4], 'x', color = 'red',ms = 20,label = 'Points of D(1 + k - $\\tau_{s}$)')
#plt.plot(k, s_k_grads, 'x')
print("D is ", D_calc[0])
print("$\tau^{s}$ is ", (1-D_calc[1] / D_calc[0]))
plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 25, markerscale = 0.5)

x_inter = -D_calc[1] / D_calc[0]

#%%
#Showing another method of finding tau from each k respectively
D = D_calc[0]

y_3_end_over_L = [[] for i in range(len(y_3_end))]

for i in range(len(y_3_end)):
    for j in range(len(y_3_end[i])):
        new_val = (y_3_end[i][j] / (L_val[j] ** (D * (1 + k[i]))))
        y_3_end_over_L[i].append(new_val)
       
for i in range(len(y_3_end_over_L)):
    plt.xlabel("L")
    plt.ylabel("$\\langle s^{k} \\rangle$ / $L^{D(1 + k)}$")
    plt.loglog((L_val), (y_3_end_over_L[i]),label = k_labels[i])
    plt.loglog((L_val), (y_3_end_over_L[i]), 'x')
    plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 15, markerscale = 3)

   
#%%
tau_m_2 = []
L_b = L_val
y_b = []
for i in range(len(y_3_end_over_L)):
    y_new = y_3_end_over_L[i]

    y_b.append(y_new)
   
   
   
for i in range(len(y_3_end_over_L)):
   fit, cov = np.polyfit(np.log(L_val[0:5]), np.log(y_3_end_over_L[i][0:5]), 1, cov = True)
   plt.loglog((L_val), (y_3_end_over_L[i]), label = k_labels[i])
   plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 15, markerscale = 3)
   tau_m_2.append(-fit[0] / D)
   

   
   
   
   


#%%
#Investigating any corrections due to scaling by dividing <s^k> by L^D(1+k-tau)
   
#Justifying picking L=16 as smallest length scale

y_corr_check = [[] for i in range(len(y_3_end))]
for i in range(len(y_3_end)):
    for j in range(len(y_3_end[i])):
        y_corr_check[i].append(y_3_end[i][j] / (L_val[j] ** (D*(1 + k[i] - 1.55))))
       
for i in range(len(y_corr_check)):
    plt.xlabel("L")
    plt.ylabel("$\\langle s^{k} \\rangle$ / $L^{D(1+k-\\tau_{s})}$")
    plt.plot(L_val, y_corr_check[i], label = k_labels[i])


    plt.legend(numpoints=3, handler_map={tuple: HandlerTuple(ndivide=None)},
                                     fontsize = 30, markerscale = 3)
   