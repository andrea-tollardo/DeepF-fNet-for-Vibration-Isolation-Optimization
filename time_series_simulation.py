import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import scipy.io as sio

#==========================================#
#                                          #
#   DeepF-fNet: tri-chiral honeycomb LRM   #
#                                          #
#       SICE4 time series simulation       #
#                                          #
#         Author: Andrea Tollardo          #
#                                          #
#==========================================#

# SIMULATION PARAMETERS TO CHANGE
#=====================================================================
n1 = 75 # length of first steady state frequency
n12 = 25 # length of first transient
n2 = 100 # length of second steady state frequency
n23 = 50 # length of second transient
n3 = 50 # length of third transient

f1 = 250.0 #[Hz] first steady state frequency
f2 = 750.0 #[Hz] second steady state frequency
f3 = 500.0 #[Hz] third steady state frequency

noise = True # activate/deactivate gaussian noise to the input signal
sigma_noise = 100.0 #[Hz] standard deviation of gaussian noise

N_database = 1000 # number of elements of the dataset to form the datatbase (suggested: unseen data during training)
#=====================================================================

# Build input target frequency signal
n_steps = n1 + n12 + n2 + n23 + n3 # length of time series
A = (f1*2*3.14)**2 #[rad2/s2] convert to SI units
B = (f2*2*3.14)**2 #[rad2/s2] convert to SI units
C = (f3*2*3.14)**2 #[rad2/s2] convert to SI units
a = 0.0 # initialize gaussian noise as deactivated
if noise:
	a = 1.0 # activate gaussian noise
In1 = A*np.ones(n1,dtype=np.float32) + a*np.random.normal(0,(sigma_noise*2*3.14)**2,n1) # first steady state frequency
In2 = np.linspace(A,B,n12,dtype=np.float32) + a*np.random.normal(0,(sigma_noise*2*3.14)**2,n12) # first transient frequency
In3 = B*np.ones(n2,dtype=np.float32) + a*np.random.normal(0,(sigma_noise*2*3.14)**2,n2) # second steady state frequency
In4 = np.linspace(B,C,n23,dtype=np.float32) + a*np.random.normal(0,(sigma_noise*2*3.14)**2,n23) # second transient frequency
In5 = C*np.ones(n3,dtype=np.float32) + a*np.random.normal(0,(sigma_noise*2*3.14)**2,n3) # third steady state frequency
Inn = np.concatenate([In1,In2,In3,In4,In5]) # put signal pieces together
plt.plot(range(n_steps),np.sqrt(Inn)/2/np.pi) # plot input signal frequency
plt.grid(visible=True)
plt.xlabel(r'$Sampling\,\,\,steps\,\,\,[-]$')
plt.ylabel(r'$\hat{f}\,\,\,[Hz]$')
plt.show()

# Find the closest bandgaps to the targets exploiting first bandgap between 2nd and 3rd eigenfrequency
BG = np.load('Bandgaps.npy') # load bandgaps and corresponding frequencies
gaps,f_bg = BG[0][:N_database,1:2],BG[1][:N_database,1:2] # extract gap extensions and average frequencies between 2nd and 3rd eigenfrequency  
f_best = np.array(f_bg) #[rad2/s2] best stop frequencies for the target (i.e., the 1st bandgap)

# Calculate MSE of the bandgaps wrt the targets (initialization)
MSE_comp = [] # initialize mean squared error
id_best = [] # initialize best database element index
stop_freq = Inn.copy() # initialize target stop frequencies  
for i in range(n_steps):
    MSE_comp1 = (stop_freq[i] - f_best[i,:])**2 # [rad4/s4] squared difference between target at step i and stop frequencies database
    MSE_comp = np.array(MSE_comp1) # convert to numpy array
    MSE = (1/(MSE_comp.shape[0]*stop_freq.shape[0]))*np.sum(MSE_comp,axis=0) #[rad4/s4] compute MSE
    id_best.append(np.argmin(MSE)) # index of the best dataset element to select for SICE4

# Load the previously trained model
IEPS = keras.models.load_model("IEPS_model.h5") # load trained IEPS

# Load dispersion curve database
omega2_data = tuple(np.load('omega2_database.npy')) # dispersion curve database
NC = np.load('Norm_Constants.npy',allow_pickle=True) # normalization constants
mu_B_omega2 = NC[0][0] # dispersion curve means
sigma2_B_omega2 = NC[0][1] # dispersion curves standard deviations
mu_B_geomFEM = NC[1][0] # parameters means
sigma2_B_geomFEM = NC[1][1] # parameters standard deviations
e = NC[2] # standardization bias

# Correct initialized dispersion curves and estimate new parameters
r_list = [] # r initialization
L_list = [] # L initialization
s_list = [] # s initialization
for i in range(n_steps):
    # Correction
    omega2 = np.expand_dims(omega2_data[id_best[i]],axis=0).copy() # prepare dispersion curves for manipulation
    omega2[0,1:,:] = omega2[0,1:,:] + stop_freq[i] - f_bg[id_best[i],0] # shift all the eigenrequencies starting from the 2nd one
    omega2[0,0,:] = omega2[0,0,:]*np.min(omega2[0,1,:])/np.max(omega2[0,0,:]) # scale first eigenfrequency to not impose an unphysical stop-band at 0 Hz
    # Estimation
    omega2n = (omega2 - mu_B_omega2[id_best[i]:id_best[i]+1,:])/tf.sqrt(sigma2_B_omega2[id_best[i]:id_best[i]+1,:] + e) # normalize IEPS input
    geomn = IEPS(omega2n) # estimate normalized parameters
    geom_val = tf.sqrt(sigma2_B_geomFEM[id_best[i]:id_best[i]+1,:] + e)*geomn + mu_B_geomFEM[id_best[i]:id_best[i]+1,:] # inverted standard normalization
    r_list.append(geom_val[0,0]) # add obtained r to the list
    L_list.append(geom_val[0,1]) # add obtained L to the list
    s_list.append(geom_val[0,2]) # add obtained s to the list

# Plot behavior of the geometric parameters at each step of the MPC
plt.plot(np.arange(0,n_steps,1),np.array(r_list)*1e3,'-b',label='r')
plt.plot(np.arange(0,n_steps,1),np.array(L_list)*1e3,'-r',label='L')
plt.plot(np.arange(0,n_steps,1),np.array(s_list)*1e3,'-g',label='s')
plt.grid(visible=True)
plt.legend()
plt.xlabel(r'$Sampling\,\,\,steps\,\,\,[-]$')
plt.ylabel(r'$Parameters\,\,\,[mm]$')
plt.show()

#sio.savemat('Sim_Results.mat',dict(signal=Inn,r=r_list,L=L_list,s=s_list)) # save results to post-process in MatLab