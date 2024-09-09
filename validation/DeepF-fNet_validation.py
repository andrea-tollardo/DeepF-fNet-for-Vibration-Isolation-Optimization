import numpy as np
import mat73
import scipy.io as sio
import matplotlib.pyplot as plt
import math as m
import tensorflow as tf
from tensorflow import keras
import matlab.engine

#==========================================#
#                                          #
#   DeepF-fNet: tri-chiral honeycomb LRM   #
#                                          #
#         IEPS and WES validation          #
#                                          #
#         Author: Andrea Tollardo          #
#                                          #
#==========================================#

# VALIDATION PARAMTERS TO CHANGE
#==============================================
i_val = 101 # dataset element object of validation (from 1 to N_dataset)
#==============================================

# Load the previously trained models
IEPS = keras.models.load_model("IEPS_model.h5") # import trained IEPS
WES = keras.models.load_model("WES_model.h5") # import trained WES

# Load dataset
InputTensor = tf.convert_to_tensor(mat73.loadmat('TrainingTensorNN.mat')['TrainingTensor'],dtype=tf.float32) # Tensor to store values to be used during training, imported from -v7.3 .mat file
InputTensor = tf.transpose(InputTensor,perm=[2,0,1]) # reorder batch dimension to let it be the first one
randData = i_val - 1 # conversion of index of the validation dataset
sio.savemat('Index.mat',dict(i_py=randData)) # save index for WES validation
r_true,L_true,s_true = tf.squeeze(InputTensor[randData:randData+1,1,0:3]) # parameters labels
omega2 = InputTensor[randData:randData+1,:10,3:43] # [rad2/s2] input dispersion curve
k_true = InputTensor[randData:randData+1,:40,43] # [rad/m] wave number label

# Normalization constants
mu_B_omega2 = tf.tile(tf.expand_dims(tf.math.reduce_mean(tf.math.real(InputTensor[:,:10,3:43]),axis=0),0),[InputTensor[:,:10,3:43].get_shape().as_list()[0],1,1]) #[rad2/s2] mean across batch dimension + batch dimension restored for broadcasting
sigma2_B_omega2 = tf.tile(tf.expand_dims(tf.math.reduce_variance(tf.math.real(InputTensor[:,:10,3:43]),axis=0),0),[InputTensor[:,:10,3:43].get_shape().as_list()[0],1,1]) #[rad4/s4] variance across batch dimension + batch dimension restored for broadcasting
mu_B_geomFEM = tf.tile(tf.expand_dims(tf.math.reduce_mean(tf.math.real(InputTensor[:,1,:3]),axis=0),0),[InputTensor[:,1,:3].get_shape().as_list()[0],1]) #[m] mean across batch dimension + batch dimension restored for broadcasting
sigma2_B_geomFEM = tf.tile(tf.expand_dims(tf.math.reduce_variance(tf.math.real(InputTensor[:,1,:3]),axis=0),0),[InputTensor[:,1,:3].get_shape().as_list()[0],1]) #[m2] variance across batch dimension + batch dimension restored for broadcasting
e = tf.constant(1e-7,dtype=tf.float32) # standardization bias

# IEPS validation
omega2n = (omega2 - mu_B_omega2[randData:randData+1,:])/tf.sqrt(sigma2_B_omega2[randData:randData+1,:] + e) # standard gaussian normalization
geomn = IEPS(omega2n) # estimations of normalized parameters
geom_val = tf.sqrt(sigma2_B_geomFEM[randData:randData+1,:] + e)*geomn + mu_B_geomFEM[randData:randData+1,:] # inverted standard normalization

# Print ground-truth vs predicted parameters
r = geom_val[0,0] #[m] mean radius of tri-chiral honeycomb circle
L = geom_val[0,1] #[m] ligament length of tri-chiral honeycomb
s = geom_val[0,2] #[m] wall thickness of tri-chiral honeycomb
print(f'\nIEPS validation\nr -> ground truth: {r_true*1e3:.4f} mm  predicted: {r*1e3:.4f} mm\nL -> ground truth: {L_true*1e3:.4f} mm  predicted: {L*1e3:.4f} mm\ns -> ground truth: {s_true:.4f} mm  predicted: {s:.4f} mm')

# Construction of the abscissa axis, i.e. predicted wave number range
n_discr_k = 40 # number of discrete points of division of the wave number range
n_V = 9 # number of vertical repetitions of the hexagons
n_H = 5 # number of horizontal repetitions of the hexagons
theta = tf.math.atan(2*r/L) #[rad] predicted ligament angle
w_ALR = 3*L*n_H/tf.math.cos(theta) #[m] predicted width of the ALR
lambda_x = 2.5*w_ALR #[m] predicted minimum periodic wave length
k_pred = tf.linspace(-tf.constant(m.pi)/lambda_x,tf.constant(m.pi)/lambda_x,n_discr_k)

# Ground truth lengths
theta_true = tf.math.atan(2*r_true/L_true) #[rad] ligament angle labels
w_ALR_true = 3*L_true*n_H/tf.math.cos(theta_true) #[m] label width of ALR
h_ALR_true = tf.sqrt(3.0)*L_true*n_V/tf.math.cos(theta_true) #[m] label height of ALR
h_HS = tf.constant([0.8e-3],dtype=tf.float32) #[m] height of HS
w_HS_true = 2.5*w_ALR_true #[m] label width of HS

# Generate predicted dispersion curve with Comsol
sio.savemat('Params.mat',dict(r=r,L=L,s=s)) # export optimal parameters
eng = matlab.engine.start_matlab() # start matlab engine
print('\nGenerating dispersion curve with Comsol ...')
eng.run('IEPS_validation.m',nargout=0) # run IEPS validation
print('\nDispersion curve generated')
eng.quit() # quit matlab engine
omega2Comsol = sio.loadmat('omega2Comsol.mat') # load generated results
k_true = tf.convert_to_tensor(omega2Comsol['kF'],dtype=tf.float32) # extract wave number range from FEM
omega2_true = tf.convert_to_tensor(omega2Comsol['omega2'],dtype=tf.float32) # extract dispersion curve from FEM

# Plot IEPS validation results
for i in range(10):
    plt.plot(k_true[0,:],tf.sqrt(omega2_true[i,:])/2/tf.constant(np.pi,dtype=tf.float32),'-b') # FEM results
for i in range(10):
    plt.plot(k_pred,tf.sqrt(omega2[0,i,:])/2/tf.constant(np.pi,dtype=tf.float32),'--r') # IEPS input dispersion curve
plt.grid(visible=True)
plt.xlabel(r'$\kappa\,\,\,[rad/m]$')
plt.ylabel(r'$f_{n}\,\,\,[Hz]$')
plt.show()

# WES validation
w_ALR_true = tf.expand_dims(w_ALR_true,0) # manipulate to build Coord
w_ALR_true = tf.expand_dims(w_ALR_true,1) # manipulate to build Coord
h_ALR_true = tf.expand_dims(h_ALR_true,0) # manipulate to build Coord
h_ALR_true = tf.expand_dims(h_ALR_true,1) # manipulate to build Coord
w_HS_true = tf.expand_dims(w_HS_true,0) # manipulate to build Coord
w_HS_true = tf.expand_dims(w_HS_true,1) # manipulate to build Coord
h_HS = tf.expand_dims(h_HS,1) # manipulate to build Coord
Coord = tf.concat([tf.expand_dims(tf.concat([-w_HS_true/2,tf.zeros([1,1],dtype=tf.float32)],1),1), # creation of the 51 physical nodes of the LRM
                    tf.expand_dims(tf.concat([-w_ALR_true/2,tf.zeros([1,1],dtype=tf.float32)],1),1),
                    tf.expand_dims(tf.concat([-w_ALR_true/4,tf.zeros([1,1],dtype=tf.float32)],1),1),
                    tf.expand_dims(tf.concat([tf.zeros([1,1],dtype=tf.float32),tf.zeros([1,1],dtype=tf.float32)],1),1),
                    tf.expand_dims(tf.concat([w_ALR_true/4,tf.zeros([1,1],dtype=tf.float32)],1),1),
                    tf.expand_dims(tf.concat([w_ALR_true/2,tf.zeros([1,1],dtype=tf.float32)],1),1),
                    tf.expand_dims(tf.concat([w_HS_true/2,tf.zeros([1,1],dtype=tf.float32)],1),1),
                    tf.expand_dims(tf.concat([-w_HS_true/2,h_HS/2],1),1),
                    tf.expand_dims(tf.concat([-w_ALR_true/2,h_HS/2],1),1),
                    tf.expand_dims(tf.concat([-w_ALR_true/4,h_HS/2],1),1),
                    tf.expand_dims(tf.concat([tf.zeros([1,1],dtype=tf.float32),h_HS/2],1),1),
                    tf.expand_dims(tf.concat([w_ALR_true/4,h_HS/2],1),1),
                    tf.expand_dims(tf.concat([w_ALR_true/2,h_HS/2],1),1),
                    tf.expand_dims(tf.concat([w_HS_true/2,h_HS/2],1),1),
                    tf.expand_dims(tf.concat([-w_HS_true/2,h_HS],1),1),
                    tf.expand_dims(tf.concat([-w_ALR_true/2,h_HS],1),1),
                    tf.expand_dims(tf.concat([-w_ALR_true/4,h_HS],1),1),
                    tf.expand_dims(tf.concat([tf.zeros([1,1],dtype=tf.float32),h_HS],1),1),
                    tf.expand_dims(tf.concat([w_ALR_true/4,h_HS],1),1),
                    tf.expand_dims(tf.concat([w_ALR_true/2,h_HS],1),1),
                    tf.expand_dims(tf.concat([w_HS_true/2,h_HS],1),1),
                    tf.expand_dims(tf.concat([-w_ALR_true/2,h_HS+h_ALR_true/6],1),1),
                    tf.expand_dims(tf.concat([-w_ALR_true/4,h_HS+h_ALR_true/6],1),1),
                    tf.expand_dims(tf.concat([tf.zeros([1,1],dtype=tf.float32),h_HS+h_ALR_true/6],1),1),
                    tf.expand_dims(tf.concat([w_ALR_true/4,h_HS+h_ALR_true/6],1),1),
                    tf.expand_dims(tf.concat([w_ALR_true/2,h_HS+h_ALR_true/6],1),1),
                    tf.expand_dims(tf.concat([-w_ALR_true/2,h_HS+h_ALR_true/3],1),1),
                    tf.expand_dims(tf.concat([-w_ALR_true/4,h_HS+h_ALR_true/3],1),1),
                    tf.expand_dims(tf.concat([tf.zeros([1,1],dtype=tf.float32),h_HS+h_ALR_true/3],1),1),
                    tf.expand_dims(tf.concat([w_ALR_true/4,h_HS+h_ALR_true/3],1),1),
                    tf.expand_dims(tf.concat([w_ALR_true/2,h_HS+h_ALR_true/3],1),1),
                    tf.expand_dims(tf.concat([-w_ALR_true/2,h_HS+h_ALR_true/2],1),1),
                    tf.expand_dims(tf.concat([-w_ALR_true/4,h_HS+h_ALR_true/2],1),1),
                    tf.expand_dims(tf.concat([tf.zeros([1,1],dtype=tf.float32),h_HS+h_ALR_true/2],1),1),
                    tf.expand_dims(tf.concat([w_ALR_true/4,h_HS+h_ALR_true/2],1),1),
                    tf.expand_dims(tf.concat([w_ALR_true/2,h_HS+h_ALR_true/2],1),1),
                    tf.expand_dims(tf.concat([-w_ALR_true/2,h_HS+2*h_ALR_true/3],1),1),
                    tf.expand_dims(tf.concat([-w_ALR_true/4,h_HS+2*h_ALR_true/3],1),1),
                    tf.expand_dims(tf.concat([tf.zeros([1,1],dtype=tf.float32),h_HS+2*h_ALR_true/3],1),1),
                    tf.expand_dims(tf.concat([w_ALR_true/4,h_HS+2*h_ALR_true/3],1),1),
                    tf.expand_dims(tf.concat([w_ALR_true/2,h_HS+2*h_ALR_true/3],1),1),
                    tf.expand_dims(tf.concat([-w_ALR_true/2,h_HS+5*h_ALR_true/6],1),1),
                    tf.expand_dims(tf.concat([-w_ALR_true/4,h_HS+5*h_ALR_true/6],1),1),
                    tf.expand_dims(tf.concat([tf.zeros([1,1],dtype=tf.float32),h_HS+5*h_ALR_true/6],1),1),
                    tf.expand_dims(tf.concat([w_ALR_true/4,h_HS+5*h_ALR_true/6],1),1),
                    tf.expand_dims(tf.concat([w_ALR_true/2,h_HS+5*h_ALR_true/6],1),1),
                    tf.expand_dims(tf.concat([-w_ALR_true/2,h_HS+h_ALR_true],1),1),
                    tf.expand_dims(tf.concat([-w_ALR_true/4,h_HS+h_ALR_true],1),1),
                    tf.expand_dims(tf.concat([tf.zeros([1,1],dtype=tf.float32),h_HS+h_ALR_true],1),1),
                    tf.expand_dims(tf.concat([w_ALR_true/4,h_HS+h_ALR_true],1),1),
                    tf.expand_dims(tf.concat([w_ALR_true/2,h_HS+h_ALR_true],1),1)],axis=1)

# Compute normalization constants
mu_B_Coord = tf.tile(tf.expand_dims(tf.math.reduce_mean(tf.math.real(InputTensor[:,:,46:48]),axis=0),0),[InputTensor[:,:,46:48].get_shape().as_list()[0],1,1]) #[m] mean across batch dimension + batch dimension restored for broadcasting
sigma2_B_Coord = tf.tile(tf.expand_dims(tf.math.reduce_variance(tf.math.real(InputTensor[:,:,46:48]),axis=0),0),[InputTensor[:,:,46:48].get_shape().as_list()[0],1,1]) #[m2] variance across batch dimension + batch dimension restored for broadcasting
uTVect = tf.reshape(InputTensor[:,:,48:],[-1,InputTensor[:,:,46].get_shape().as_list()[1]*10*40*2,1]) # reshape ground truth displacements as a vector
uTRe = tf.math.real(uTVect) # extract real part from ground truth displacements
uTIm = tf.math.imag(uTVect) # extract imaginary part from ground truth displacements
displTrue = tf.squeeze(tf.concat([uTRe,uTIm],1)) # concatenate real and imaginary part for standardization. Reduce 1 dimension
mu_B_uFEM = tf.tile(tf.expand_dims(tf.math.reduce_mean(displTrue,axis=0),0),[displTrue[:,:].get_shape().as_list()[0],1]) #[m] mean across batch dimension + batch dimension restored for broadcasting
sigma2_B_uFEM = tf.tile(tf.expand_dims(tf.math.reduce_variance(displTrue,axis=0),0),[displTrue[:,:].get_shape().as_list()[0],1]) #[m2] variance across batch dimension + batch dimension restored for broadcasting
e = tf.constant(1e-7,dtype=tf.float32) # standardization bias

# Estimate nodal displacements
CoordNorm = (Coord - mu_B_Coord[randData:randData+1,:,:])/tf.sqrt(sigma2_B_Coord[randData:randData+1,:,:] + e) # standard gaussian normalization
x,y = CoordNorm[:,:,0:1],CoordNorm[:,:,1:] # extract normalized coordinates
X = tf.Variable(tf.zeros([x.get_shape().as_list()[0],2*x.get_shape().as_list()[1],x.get_shape().as_list()[2]])) # prepare coordinates to host twice their number for manipulation (102)
Y = tf.Variable(tf.zeros([y.get_shape().as_list()[0],2*y.get_shape().as_list()[1],y.get_shape().as_list()[2]])) # prepare coordinates to host twice their number for manipulation (102)
for i  in range(x.get_shape().as_list()[1]):
    x_i = tf.tile(tf.expand_dims(x[:,i,:],1),[1,2,1]) # repeat x twice near each other (u and v are subsequent)
    X[:,i:i+2,:].assign(x_i) # assign to X
    y_i = tf.tile(tf.expand_dims(y[:,i,:],1),[1,2,1]) # repeat y twice near each other (u and v are subsequent)
    Y[:,i:i+2,:].assign(y_i) # assign to Y
X_new = tf.tile(X,[1,10*40*2,1]) # repeat the new tensor n*k*2 times (eigenfrequencies, wave number range and real + imaginary part)
Y_new = tf.tile(Y,[1,10*40*2,1]) # repeat the new tensor n*k*2 times (eigenfrequencies, wave number range and real + imaginary part)
In = tf.concat([X_new,Y_new],2) # stack the new X and Y coordinates one near the other [N_batch,81600,2] to feed to the PDES model as input
w_predNorm = WES(In) # estimate normalized nodal displacements
w_pred = tf.sqrt(sigma2_B_uFEM[randData:randData+1,:] + e)*w_predNorm + mu_B_uFEM[randData:randData+1,:] # inverted standard normalization

# Separate and save displacements
idx = int(w_pred.get_shape().as_list()[1]/2) # index corresponding to half of the total length of the output vector, separating real from imaginary part
u_re = tf.cast(w_pred[:,0:idx:2],dtype=tf.float64) # real part of displacements along x direction (sweeping n first, k second and the position third)
u_im = tf.cast(w_pred[:,idx::2],dtype=tf.float64) # imaginary part of displacements along x direction (sweeping n first, k second and the position third)
v_re = tf.cast(w_pred[:,1:idx:2],dtype=tf.float64) # real part of displacements along y direction (sweeping n first, k second and the position third)
v_im = tf.cast(w_pred[:,idx+1::2],dtype=tf.float64) # imaginary part of displacements along y direction (sweeping n first, k second and the position third)
sio.savemat('uv_val.mat',dict(uRe=u_re.numpy(),vRe=v_re.numpy(),uIm=u_im.numpy(),vIm=v_im.numpy())) # save displacements to matlab file

# Run validation on matlab
eng = matlab.engine.start_matlab() # start matlab engine
print('\nGenerating displacement field ...')
eng.run('WES_validation.m',nargout=0) # run WES validation
print('\nDisplacement field generated ...')
eng.quit() # quit matlab engine
print('\nFigures saved as ground_truth.fig and predicted.fig')
