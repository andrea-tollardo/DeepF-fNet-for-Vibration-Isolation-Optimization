import numpy as np
import mat73
import tensorflow as tf

#==========================================#
#                                          #
#   DeepF-fNet: tri-chiral honeycomb LRM   #
#                                          #
#   Database and normalization constants   #
#           extraction for SICE4           #
#                                          #
#         Author: Andrea Tollardo          #
#                                          #
#==========================================#

# Import dataset and extract the dispersion curves
InputTensor = tf.convert_to_tensor(mat73.loadmat('TrainingTensorNN.mat')['TrainingTensor'],dtype=tf.float32) # Tensor to store values to be used during training, imported from matlab file; for -v7.3 .mat files
InputTensor = tf.transpose(InputTensor,perm=[2,0,1]) # reorder batch dimension to let it be the first one
DispCurve = InputTensor[:,:10,3:43] # dimensions: N x 10 x 40
DispCurve_np = DispCurve.numpy() # convert to numpy matrix

# Normalization constants
mu_B_omega2 = tf.tile(tf.expand_dims(tf.math.reduce_mean(tf.math.real(InputTensor[:,:10,3:43]),axis=0),0),[InputTensor[:,:10,3:43].get_shape().as_list()[0],1,1]) #[rad2/s2] mean across batch dimension + batch dimension restored for broadcasting
sigma2_B_omega2 = tf.tile(tf.expand_dims(tf.math.reduce_variance(tf.math.real(InputTensor[:,:10,3:43]),axis=0),0),[InputTensor[:,:10,3:43].get_shape().as_list()[0],1,1]) #[rad4/s4] variance across batch dimension + batch dimension restored for broadcasting
mu_B_geomFEM = tf.tile(tf.expand_dims(tf.math.reduce_mean(tf.math.real(InputTensor[:,1,:3]),axis=0),0),[InputTensor[:,1,:3].get_shape().as_list()[0],1]) #[m] mean across batch dimension + batch dimension restored for broadcasting
sigma2_B_geomFEM = tf.tile(tf.expand_dims(tf.math.reduce_variance(tf.math.real(InputTensor[:,1,:3]),axis=0),0),[InputTensor[:,1,:3].get_shape().as_list()[0],1]) #[m2] variance across batch dimension + batch dimension restored for broadcasting
e = tf.constant(1e-7,dtype=tf.float32)

# Save database of dispersion curves and normalization constants
np.save('omega2_database',DispCurve_np) # save dispersion curve database
NC_omega2 = [mu_B_omega2,sigma2_B_omega2] # compose dispersion curve normalization constants
NC_geomFEM = [mu_B_geomFEM,sigma2_B_geomFEM] # compose output parameters normalization constants
NC = [NC_omega2,NC_geomFEM,e] # compose normalization constants list
np.save('Norm_Constants',NC) # save normalization constants