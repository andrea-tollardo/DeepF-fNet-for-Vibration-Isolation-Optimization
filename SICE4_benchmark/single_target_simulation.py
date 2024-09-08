import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from time import time
import scipy.io as sio
import matlab.engine

#==========================================#
#                                          #
#   DeepF-fNet: tri-chiral honeycomb LRM   #
#                                          #
#    SICE4 single frequency simulation     #
#                                          #
#         Author: Andrea Tollardo          #
#                                          #
#==========================================#

t0=time() # track required time for SICE4 to run

# SIMULATION PARAMETERS TO CHANGE
#================================================
f_stop = 200 #[Hz] frequency to stop
N_database = 1000 # number of elements of the dataset to form the datatbase (suggested: unseen data during training)
load_IEPS = False # simulate IEPS loading time
#================================================

# Load the previously trained model and constants
delta = 0 # [s] initialize pause time interval for IEPS loading simulation
if load_IEPS:
    print('\nIEPS model loaded in this run\n')
if load_IEPS==False:
    t1 = time() # track pause time interval
IEPS = keras.models.load_model("IEPS_model.h5") # load trained IEPS
# Load dispersion curve database
omega2_data = tuple(np.load('omega2_database.npy')) # dispersion curve database
NC = np.load('Norm_Constants.npy',allow_pickle=True) # normalization constants
mu_B_omega2 = NC[0][0] # dispersion curve means
sigma2_B_omega2 = NC[0][1] # dispersion curves standard deviations
mu_B_geomFEM = NC[1][0] # parameters means
sigma2_B_geomFEM = NC[1][1] # parameters standard deviations
e = NC[2] # standardization bias
if load_IEPS==False:
    t2 = time() # stop track pause time interval
    delta = t2 - t1 # [s] pause time interval

# Find the closest bandgaps to the targets exploiting first bandgap between 2nd and 3rd eigenfrequency
BG = np.load('Bandgaps.npy') # load bandgaps and corresponding frequencies
gaps,f_bg = BG[0][:N_database,1:2],BG[1][:N_database,1:2] # extract gap extensions and average frequencies between 2nd and 3rd eigenfrequency  
f_best = np.array(f_bg) #[rad2/s2] best stop frequencies for the target (i.e., the 1st bandgap)

# Calculate MSE of the bandgaps wrt the targets (initialization)
stop_freq = np.array([(2*np.pi*f_stop)**2]) #[rad2/s2] convert to SI units
MSE_comp = [] # initialize mean squared error
MSE_comp.append((stop_freq - f_best[:,0])**2) # [rad4/s4] squared difference between target at step i and stop frequencies database
MSE_comp = np.array(MSE_comp) # convert to numpy array
MSE = (1/(MSE_comp.shape[1]*stop_freq.shape[0]))*np.sum(MSE_comp,axis=0)#[rad4/s4] compute MSE
id_best = np.argmin(MSE) # index of the best dataset to select for SICE4

# Correct dispersion curve and estimate new parameters
omega2 = np.expand_dims(omega2_data[id_best],axis=0).copy() # initialize dispersion curve
omega2[0,1:,:] = omega2[0,1:,:] + stop_freq - f_bg[id_best,0] # shift all the eigenrequencies starting from the 2nd one
omega2[0,0,:] = omega2[0,0,:]*np.min(omega2[0,1,:])/np.max(omega2[0,0,:]) # scale first eigenfrequency to not impose an unphysical stop-band at 0 Hz
omega2n = (omega2 - mu_B_omega2[id_best:id_best+1,:])/tf.sqrt(sigma2_B_omega2[id_best:id_best+1,:] + e) # normalize IEPS input
geomn = IEPS(omega2n) # estimate normalized parameters
geom_val = tf.sqrt(sigma2_B_geomFEM[id_best:id_best+1,:] + e)*geomn + mu_B_geomFEM[id_best:id_best+1,:] # inverted standard normalization
r = geom_val[0,0] #[m] mean radius of tri-chiral honeycomb circle
L = geom_val[0,1] #[m] ligament length of tri-chiral honeycomb
s = geom_val[0,2] #[m] wall thickness of tri-chiral honeycomb

print(f'\nOptimal parameters:\nr: {r*1e3:.4f} mm\nL: {L*1e3:.4f} mm\ns: {s*1e3:.4f} mm\n') # print optimal parameters
print(f'CPU time: {time()-t0-delta:.4f} s') # print SICE4 run time for a single frequency

# Prompt optimal results to Comsol for validation
sio.savemat('Params.mat',dict(r=r,L=L,s=s)) # export optimal parameters
eng = matlab.engine.start_matlab() # start matlab engine
print('\nGenerating dispersion curve with Comsol ...')
eng.run('IEPS_validation.m',nargout=0) # run IEPS validation
print('\nDispersion curve generated')
eng.quit() # quit matlab engine

# Plot predicted bandgaps versus target frequency
omega2Comsol = sio.loadmat('omega2Comsol.mat')
k_true = tf.convert_to_tensor(omega2Comsol['kF'],dtype=tf.float32)
omega2_true = tf.convert_to_tensor(omega2Comsol['omega2'],dtype=tf.float32)
for i in range(3):
    plt.plot(k_true[0,:],tf.sqrt(omega2_true[i,:])/2/tf.constant(np.pi,dtype=tf.float32),'-b')
plt.plot([k_true[0,0],k_true[0,-1]],[f_stop,f_stop],'--r')
plt.grid(visible=True)
plt.xlabel(r'$\kappa\,\,\,[rad/m]$')
plt.ylabel(r'$f_{n}\,\,\,[Hz]$')
plt.show()