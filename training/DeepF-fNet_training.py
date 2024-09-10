# DeepF-fNet_training.py

#==========================================#
#                                          #
#   DeepF-fNet: tri-chiral honeycomb LRM   #
#                                          #
#           DeepF-fNet training            #
#                                          #
#         Author: Andrea Tollardo          #
#                                          #
#==========================================#

print("+-------------------------------------------------+\n|                                                 |\n|      DeepF-fNet: tri-chiral honeycomb LRM       |\n|                                                 |\n|                DeepF-fNet training              |\n|                                                 |\n|              Author: Andrea Tollardo            |\n|                                                 |\n+-------------------------------------------------+\n")
print("Department of Mechanical Engineering - Politecnico di Milano")
print("Master Thesis << DeepF-fNet: a Physics-Informed Neural Network for Vibration Suppression Optimization >>")
# Importation of the modules
#-------------------------------------------------------------------------------------------------------------------------------
import matplotlib as mp
import scipy.io as sio
import mat73
import matplotlib.pyplot as plt
import math as m
import tensorflow as tf
from tensorflow import keras
from collections import defaultdict
from time import  time
import database_extraction
import bandgap_extraction

keras.backend.clear_session()

# Definition of functions and classes
#------------------------------------------------------------------------------------------------------------------------------
# PINN model creation functions
def create_IEPS_model(InputTensor, verbose=False):
    """Definition of the first Convolutional Neural Network to obtain inverse parameters
    
    Args:
    ----
    InputTensor: desired dispersion function (n rows, k columns)
    verbose: boolean, indicate whether to show the model summary
    
    Outputs:
    --------
    IEP_model: the CNN model of the Inverse Eigenvalue Problem Solver (IEPS)
    """
    # Hyperparameters
    input_dim = tf.expand_dims(InputTensor,axis=-1).get_shape().as_list()
    filter_num = 3 # number of convolutional layers
    filter_size = 3 # size of each filter
    pooling = 2 # size of the 2D max pooling
    stride = 1 # stride of each filter and max pooling
    pad = "same" # padding option
    hidden_size = 160 # size of hidden fully connected layer
    IEP_output_size = 3 # number of neurons in the output of the IEP Network: 3 geometric parameters
    dropout = True # activates/deactivates dropout in the convolutional layers
    dropout_rate = 0.5 # rate of the neurons to be deactivated
    conv_act_func = "tanh" # non linear activation function of convolutional filter
    hid_act_func = "tanh" # non linear activation function of hidden fully connected layer
    out_act_func = "linear" # linear activation function of output layer

    # Definition of the model with a Sequential API
    IEP_model = keras.models.Sequential()
    IEP_model._name = 'IEPS'
    # Definition of the input tensor
    IEP_model.add(tf.keras.Input(shape=input_dim))
    # Definition of the convolutional layers
    for i in range(filter_num):
        # Convolutional filter
        IEP_model.add(tf.keras.layers.Conv2D(filters=int(input_dim[-1]), kernel_size=filter_size, strides=stride, padding=pad, activation=conv_act_func))
        IEP_model.layers[-1]._name = 'Convolution_' + str(i)
        # Pooling layer
        IEP_model.add(tf.keras.layers.MaxPooling2D(pool_size=pooling, strides=stride))
        IEP_model.layers[-1]._name = 'Max_Pooling_' + str(i)
    # Flattening of the convolution output
    IEP_model.add(tf.keras.layers.Flatten())
    IEP_model.layers[-1]._name = 'Flatten'
    # Definition of the single hidden, fully connected layer
    IEP_model.add(tf.keras.layers.Dense(units=hidden_size, activation=hid_act_func))
    IEP_model.layers[-1]._name = 'Hidden_Fully_Connected'
    # Dropout
    if dropout:
        IEP_model.add(tf.keras.layers.Dropout(rate=dropout_rate))
        IEP_model.layers[-1]._name = 'Dropout'
    # Definition of the output tensor
    IEP_model.add(tf.keras.layers.Dense(units=IEP_output_size,activation=out_act_func))
    IEP_model.layers[-1]._name = 'Geometric_Parameters'
    
    if verbose:
        IEP_model.summary() # report the summary of the constructed network
        
    return IEP_model   

def create_WES_model(Coord, Nsamples, verbose=False):
    """Definition of the second Convolutional Neural Network to solve the Wave Equation
    The number of rows of the input is equal to the number of elements in the output to allow a simple differentiation with respect to physical coordinates

    Args:
    ----
    Coord: XY Coordinates of the nodes of the Locally Resonant Metamaterial (same number of rows of the output layer, 2 columns)
    Nsamples: number of sample locations where u and v displacements were probed (51)
    verbose: boolean, indicate whether to show the model summary
    
    Outputs:
    --------
    WES_model: the CNN model of the Wave Equation Solver (WES)
    """
    # Hyperparameters
    input_dim = tf.expand_dims(Coord,axis=-1).get_shape().as_list()
    filter_num = 2 # number of convolutional layers
    filter_size = 2 # size of each filter
    pooling = 1 # size of the 2D max pooling
    stride = 1 # stride of each filter and of max pooling
    pad = "same" # padding option
    hidden_size = 20 # size of hidden fully connected layers
    WES_output_size = 2*2*Nsamples*10*40 # number of neurons in the output of the WES Network: 51 couples of displacements for every n and k, real and imaginary part
    dropout = True # activates/deactivates dropout in the convolutional layers
    dropout_rate = 0.5 # rate of the neurons to be deactivated
    conv_act_func = "tanh" # non linear activation function of convolutional filter
    hid_act_func = "tanh" # non linear activation function of hidden fully connected layer
    out_act_func ="linear" # linear activation function of output layer

    # Definition of the model with a Sequential API
    WES_model = keras.models.Sequential()
    WES_model._name = 'WES'
    # Definition of the input tensor
    WES_model.add(keras.Input(shape=input_dim))
    # Definition of the convolutional layers
    for i in range(filter_num):
        # Convolutional filter
        WES_model.add(keras.layers.Conv2D(filters=int(input_dim[-1]),kernel_size=filter_size, strides=stride, padding=pad, activation=conv_act_func))
        WES_model.layers[-1]._name = 'Convolution_' + str(i)
        # Pooling layer
        WES_model.add(keras.layers.MaxPooling2D(pool_size=pooling, strides=stride))
        WES_model.layers[-1]._name = 'Max_Pooling_' + str(i)
    # Flattening of the convolution output
    WES_model.add(keras.layers.Flatten())
    WES_model.layers[-1]._name = 'Flatten'
    # Definition of the single hidden, fully connected layer
    WES_model.add(keras.layers.Dense(units=hidden_size, activation=hid_act_func))
    WES_model.layers[-1]._name = 'Hidden_Fully_Connected'
    # Dropout
    if dropout:
        WES_model.add(keras.layers.Dropout(rate=dropout_rate))
        WES_model.layers[-1]._name = 'Dropout'
    # Definition of the output tensor
    WES_model.add(keras.layers.Dense(units=WES_output_size,activation=out_act_func))
    WES_model.layers[-1]._name = 'Field_Displacements'
    
    if verbose:
        WES_model.summary() # report the summary of the constructed network
        
    return WES_model

# Physics-Informed part
@tf.function
def PDE_calculator(x_n, y_n, nu_s, rho_s, E_s, omega2, PDE_model, mu_B_uFEM, sigma2_B_uFEM, bs):
    """Calculates the residual of the PDE of both the Attachable Local Resonator (ALR) and the underneath Host Structure (HS).
    The PDE applied here is the harmonic wave equation imposing as general solution the one of the Floquet-Bloch theorem.
    The function calculates the residual of the two domains separately, for every n and k
    
    Args:
    ----
    x_n: normalized x-coordinates of the nodes (51*2*2*n*k)
    y_n: normalized y-coordinates of the nodes (51*2*2*n*k)
    nu_s: effective Poisson's coefficient of the ALR, as calculated from IEPS output
    rho_s: effective density of the ALR, as calculated from IEPS output
    E_s: effective Young's Modulus of the ALR, as calculated from IEPS output
    omega2: dispersion curve fed to the IEPS (n rows, k columns)
    PDE_model: the model to solve the PDE (WES)
    mu_B_uFEM: mean of the model output across whole dataset (from training data)
    sigma2_B_uFEM: variance of the model output across whole dataset (from training data)
    bs: mini-batch size
    
    Outputs:
    --------
    PDE_residual: scalar residual of the governing PDE, in the MSE sense among all the coordinates (n and k included) and all the mini-batches
    u_re, v_re: real parts of the displacement field as predicted by the PDES
    u_im, v_im: imaginary parts of the displacement field as predicted by the PDES
    sigma_x_ALR_re,sigma_y_ALR_re: real part of normal stresses in x and y direction
    sigma_x_ALR_im,sigma_y_ALR_im: imaginary part of normal stresses in x and y direction
    sigma_y_HS_re: real part of normal stress in y direction
    sigma_y_HS_im: imaginary part of normal stress in y direction
    """
    # Prediction of the displacement field by the WES
    w_predNorm = PDE_model(tf.transpose(tf.concat([x_n,y_n],1),perm=[0,2,1]),training=True) # apply model to estimate displacement field
    # De-normalize output to impose physics
    e = tf.constant(1e-7, dtype=tf.float32) # standardization bias
    w_pred = tf.sqrt(sigma2_B_uFEM[:bs,:] + e)*w_predNorm + mu_B_uFEM[:bs,:] # inverted standard normalization
    # Separate displacements
    idx = int(w_pred.get_shape().as_list()[1]/2) # index corresponding to half of the total length of the output vector, separating real from imaginary part
    u_re = w_pred[:,0:idx:2] # real part of displacements along x direction (sweeping n first, k second and the position third)
    u_im = w_pred[:,idx::2] # imaginary part of displacements along x direction (sweeping n first, k second and the position third)
    v_re = w_pred[:,1:idx:2] # real part of displacements along y direction (sweeping n first, k second and the position third)
    v_im = w_pred[:,idx+1::2] # imaginary part of displacements along y direction (sweeping n first, k second and the position third)

    # Set material properties and dispersion curve
    rho_ALR_pred = tf.expand_dims(rho_s,1) # expand dimensions for future concatenation in the 2D PDE
    E_ALR_pred =  tf.expand_dims(E_s,1) # expand dimensions for future concatenation in the 2D PDE
    nu_ALR_pred = tf.expand_dims(nu_s,1) # the Poisson's ratio is the same in all the directions since the tri-chiral honeycomb ALR is isotropic
    nu_steel = 0.29 # Poisson's ratio of steel
    rho_steel = 7850 # [kg/m^3] steel density
    E_steel = 2.12e11 # [Pa] steel Young's Modulus

    dim_n = omega2.get_shape().as_list()[1] # extraction of the number of eigenfrequencies
    dim_k = omega2.get_shape().as_list()[2] # extraction of the number of wave number range discrete points
    omega2 = tf.transpose(omega2,perm=(0,2,1)) # prepare omega2 for a correct flattening
    omega2 = tf.reshape(omega2,[-1,dim_n*dim_k]) # first all the n eigenfrequencies are listed for a single k (i.e. sweep n first, k second)
    omega2 = tf.tile(omega2,[1,51]) # dispersion functions are repeated for every coordinate X,Y. It is not repeated twice the number of nodes since the PDEs of the real and imaginary part of the ALR are separated
    omega2 = tf.expand_dims(omega2,1) # expand dimensions for concatenation in the 2D PDE
    omega2 = tf.tile(omega2,[1,2,1]) # repeatition in the same PDE since it is 2D
    
    # Construction of the plane-strain isotropic elasticity tensors
    C11_22_ALR = tf.tile(tf.expand_dims(E_ALR_pred/(1-nu_ALR_pred**2),1),[1,x_n.get_shape().as_list()[1],x_n.get_shape().as_list()[2]]) # first and second diagonal term of the elasticity tensor
    C12_21_ALR = C11_22_ALR*nu_ALR_pred # non null extradiagonal term of the elasticity tensor
    C33_ALR = tf.tile(tf.expand_dims(E_ALR_pred/(2*(1+nu_ALR_pred)),1),[1,x_n.get_shape().as_list()[1],x_n.get_shape().as_list()[2]]) # third diagonal term of the elasticity tensor
    
    C11_22_HS = E_steel/(1-nu_steel**2) # first and second diagonal term of the elasticity tensor (scalar, so broadcasted in every direction)
    C12_21_HS = C11_22_HS*nu_steel # non null extradiagonal term of the elasticity tensor (scalar, so broadcasted in every direction)
    C33_HS = E_steel/(2*(1+nu_steel)) # third diagonal term of the elasticity tensor (scalar, so broadcasted in every direction)

    # Calculate gradients of the displacements with respect to physical coordinates by separating the former into real and imaginary part
    eps_x_re = tf.gradients(u_re, x_n)[0] # normal strain in x direction, real part
    eps_y_re = tf.gradients(v_re, y_n)[0] # normal strain in y direction, real part
    eps_xy_re = 0.5*(tf.gradients(u_re, y_n)[0]+tf.gradients(v_re, x_n)[0]) # shear strain, real part
    
    eps_x_im = tf.gradients(u_im, x_n)[0] # normal strain in x direction, imaginary part
    eps_y_im = tf.gradients(v_im, y_n)[0] # normal strain in y direction, imaginary part
    eps_xy_im = 0.5*(tf.gradients(u_im, y_n)[0]+tf.gradients(v_im, x_n)[0]) # shear strain, imaginary part
    
    # Dot product between strain and elasticity tensor to obtain stresses. Here the two domains of the LRM are considered
    # Attachable Local Resonator (ALR)
    sigma_x_ALR_re = C11_22_ALR*eps_x_re + C12_21_ALR*eps_y_re # normal stress in x direction, real part
    sigma_y_ALR_re = C12_21_ALR*eps_x_re + C11_22_ALR*eps_y_re # normal stress in y direction, real part
    sigma_xy_ALR_re = C33_ALR*eps_xy_re # shear stress, real part

    sigma_x_ALR_im = C11_22_ALR*eps_x_im + C12_21_ALR*eps_y_im # normal stress in x direction, imaginary part
    sigma_y_ALR_im = C12_21_ALR*eps_x_im + C11_22_ALR*eps_y_im # normal stress in y direction, imaginary part
    sigma_xy_ALR_im = C33_ALR*eps_xy_im # shear stress, imaginary part

    # Host Structure (HS)
    sigma_x_HS_re = C11_22_HS*eps_x_re + C12_21_HS*eps_y_re # normal stress in x direction, real part
    sigma_y_HS_re = C12_21_HS*eps_x_re + C11_22_HS*eps_y_re # normal stress in y direction, real part
    sigma_xy_HS_re = C33_HS*eps_xy_re # shear stress, real part

    sigma_x_HS_im = C11_22_HS*eps_x_im + C12_21_HS*eps_y_im # normal stress in x direction, imaginary part
    sigma_y_HS_im = C12_21_HS*eps_x_im + C11_22_HS*eps_y_im # normal stress in y direction, imaginary part
    sigma_xy_HS_im = C33_HS*eps_xy_im # shear stress, imaginary part
    
    # Calculate the gradients of the stresses with respect to the normalized coordinates
    # Attachable Local Resonator (ALR)
    dsig_x_dx_ALR_re = tf.gradients(sigma_x_ALR_re, x_n)[0] # derivative of the normal stress in x direction w.r.t. x, real part
    dsig_y_dy_ALR_re = tf.gradients(sigma_y_ALR_re, y_n)[0] # derivative of the normal stress in y direction w.r.t. y, real part
    dtau_dx_ALR_re = tf.gradients(sigma_xy_ALR_re, x_n)[0] # derivative of the shear stress w.r.t. x, real part
    dtau_dy_ALR_re = tf.gradients(sigma_xy_ALR_re, y_n)[0] # derivative of the shear stress w.r.t. y, real part

    dsig_x_dx_ALR_im = tf.gradients(sigma_x_ALR_im, x_n)[0] # derivative of the normal stress in x direction w.r.t. x, imaginary part
    dsig_y_dy_ALR_im = tf.gradients(sigma_y_ALR_im, y_n)[0] # derivative of the normal stress in y direction w.r.t. y, imaginary part
    dtau_dx_ALR_im = tf.gradients(sigma_xy_ALR_im, x_n)[0] # derivative of the shear stress w.r.t. x, imaginary part
    dtau_dy_ALR_im = tf.gradients(sigma_xy_ALR_im, y_n)[0] # derivative of the shear stress w.r.t. y, imaginary part

    # Host Structure (HS)
    dsig_x_dx_HS_re = tf.gradients(sigma_x_HS_re, x_n)[0]# derivative of the normal stress in x direction w.r.t. x, real part
    dsig_y_dy_HS_re = tf.gradients(sigma_y_HS_re, y_n)[0]# derivative of the normal stress in y direction w.r.t. y, real part
    dtau_dx_HS_re = tf.gradients(sigma_xy_HS_re, x_n)[0]# derivative of the shear stress w.r.t. x, real part
    dtau_dy_HS_re = tf.gradients(sigma_xy_HS_re, y_n)[0]# derivative of the shear stress w.r.t. y, real part
   
    dsig_x_dx_HS_im = tf.gradients(sigma_x_HS_im, x_n)[0]# derivative of the normal stress in x direction w.r.t. x, imaginary part
    dsig_y_dy_HS_im = tf.gradients(sigma_y_HS_im, y_n)[0]# derivative of the normal stress in y direction w.r.t. y, imaginary part
    dtau_dx_HS_im = tf.gradients(sigma_xy_HS_im, x_n)[0]# derivative of the shear stress w.r.t. x, imaginary part
    dtau_dy_HS_im = tf.gradients(sigma_xy_HS_im, y_n)[0]# derivative of the shear stress w.r.t. y, imaginary part

    # Construction of the divergence of the stress stensor
    # Attachable Local Resonator (ALR)
    divS_1_ALR_re = dsig_x_dx_ALR_re[:,:,:idx:2] + dtau_dy_ALR_re[:,:,1:idx:2] # first row of the divergence of the stress tensor, real part
    divS_2_ALR_re = dtau_dx_ALR_re[:,:,:idx:2] + dsig_y_dy_ALR_re[:,:,1:idx:2] # second row of the divergence of the stress tensor, real part

    divS_1_ALR_im = dsig_x_dx_ALR_im[:,:,idx::2] + dtau_dy_ALR_im[:,:,idx+1::2] # first row of the divergence of the stress tensor, imaginary part
    divS_2_ALR_im = dtau_dx_ALR_im[:,:,idx::2] + dsig_y_dy_ALR_im[:,:,idx+1::2] # second row of the divergence of the stress tensor, imaginary part

    # Host Structure (HS)
    divS_1_HS_re = dsig_x_dx_HS_re[:,:,:idx:2] + dtau_dy_HS_re[:,:,1:idx:2] # first row of the divergence of the stress tensor, real part
    divS_2_HS_re = dtau_dx_HS_re[:,:,:idx:2] + dsig_y_dy_HS_re[:,:,:idx:2] # second row of the divergence of the stress tensor, real part

    divS_1_HS_im = dsig_x_dx_HS_im[:,:,idx::2] + dtau_dy_HS_im[:,:,idx+1::2] # first row of the divergence of the stress tensor, imaginary part
    divS_2_HS_im = dtau_dx_HS_im[:,:,idx::2] + dsig_y_dy_HS_im[:,:,idx+1::2] # second row of the divergence of the stress tensor, imaginary part
    
    # PDE for both domains, real and imaginary part
    N_coord_threshold = 14 # after this coordinate, the domain switches from HS to LRM
    
    # ALR domain
    w1_re = tf.concat([tf.expand_dims(u_re[:,:N_coord_threshold*dim_n*dim_k],1),tf.expand_dims(v_re[:,:N_coord_threshold*dim_n*dim_k],1)], axis=1) # column vector of predicted displacements u and v, real part 
    w1_im = tf.concat([tf.expand_dims(u_im[:,:N_coord_threshold*dim_n*dim_k],1),tf.expand_dims(v_im[:,:N_coord_threshold*dim_n*dim_k],1)], axis=1) # column vector of predicted displacements u and v, imaginary part 
    div_sigma_ALR_re = tf.concat([divS_1_ALR_re[:,:,:N_coord_threshold*dim_n*dim_k],divS_2_ALR_re[:,:,:N_coord_threshold*dim_n*dim_k]], axis=1) # column vector of divergence of stress tensor, real part
    div_sigma_ALR_im = tf.concat([divS_1_ALR_im[:,:,:N_coord_threshold*dim_n*dim_k],divS_2_ALR_im[:,:,:N_coord_threshold*dim_n*dim_k]], axis=1) # column vector of divergence of stress tensor, imaginary part
    # Construction of the inertial term
    inertia_ALR_re = tf.tile(tf.expand_dims(rho_ALR_pred,1),[1,2,N_coord_threshold*dim_n*dim_k])*omega2[:,:,:N_coord_threshold*dim_n*dim_k]*w1_re # column vector of inertial term, real part. Density is the same in both directions
    inertia_ALR_im = tf.tile(tf.expand_dims(rho_ALR_pred,1),[1,2,N_coord_threshold*dim_n*dim_k])*omega2[:,:,:N_coord_threshold*dim_n*dim_k]*w1_im # column vector of inertial term, imaginary part. Density is the same in both directions
    # ALR residual for each set of coordinate (including n and k)
    res_ALR = tf.reduce_mean(keras.losses.mean_squared_error(inertia_ALR_re + div_sigma_ALR_re + inertia_ALR_im + div_sigma_ALR_im,0.0),axis=0) # mean squared error computed along the batch dimension. Sign convention of Comsol Eigenfrequency Study

    # HS domain
    w2_re = tf.concat([tf.expand_dims(u_re[:,N_coord_threshold*dim_n*dim_k:],1),tf.expand_dims(v_re[:,N_coord_threshold*dim_n*dim_k:],1)], axis=1) # column vector of predicted displacements u and v, real part
    w2_im = tf.concat([tf.expand_dims(u_im[:,N_coord_threshold*dim_n*dim_k:],1),tf.expand_dims(v_im[:,N_coord_threshold*dim_n*dim_k:],1)], axis=1) # column vector of predicted displacements u and v, imaginary part
    div_sigma_HS_re = tf.concat([divS_1_HS_re[:,:,N_coord_threshold*dim_n*dim_k:],divS_2_HS_re[:,:,N_coord_threshold*dim_n*dim_k:]], axis=1) # column vector of divergence of stress tensor, real part
    div_sigma_HS_im = tf.concat([divS_1_HS_im[:,:,N_coord_threshold*dim_n*dim_k:],divS_2_HS_im[:,:,N_coord_threshold*dim_n*dim_k:]], axis=1) # column vector of divergence of stress tensor, imaginary part
    # Construction of the inertial term
    inertia_HS_re = rho_steel*omega2[:,:,N_coord_threshold*dim_n*dim_k:]*w2_re # column vector of inertial term, real part. Density is the same in both directions and every batch
    inertia_HS_im = rho_steel*omega2[:,:,N_coord_threshold*dim_n*dim_k:]*w2_im # column vector of inertial term, imaginary part. Density is the same in both directions and every batch
    # HS residual for each set of coordinate (including n and k)
    res_HS = tf.reduce_mean(keras.losses.mean_squared_error(inertia_HS_re + div_sigma_HS_re + inertia_HS_im + div_sigma_HS_im,0.0),axis=0) # mean squared error computed along the batch dimension. Sign convention of Comsol Eigenfrequency Study
    
    PDE_residual = tf.reduce_mean(res_ALR + res_HS) # MSE of the residual of the governing PDE, computed across the 2 orthogonal directions
    
    return PDE_residual, u_re, v_re, u_im, v_im, sigma_x_ALR_re, sigma_y_ALR_re, sigma_x_ALR_im, sigma_y_ALR_im, sigma_y_HS_re, sigma_y_HS_im

# IEPS and WES train steps
@tf.function
def IEPS_train_step(omega2, geom_FEM, comp_weight, geom_weight, IEP_model, mu_B_omega2, sigma2_B_omega2, mu_B_geomFEM, sigma2_B_geomFEM, bs):
    """Calculate gradients of the loss with respect to the IEPS parameters.
    
    Args:
    ----
    omega2: dispersion function coming from training dataset (n rows, k columns)
    geom_FEM: ground truth geometric parameters from solution of FEM direct eigenfrequencies problem (1 row, 3 columns)
    comp_weight: weight of the compliance constraints loss
    geom_weight: weight of the geometrical loss
    IEP_model: the inverse eigenvalue problem model (IEPS)
    mu_B_omega2: mean of the model input across whole dataset
    sigma2_B_omega2: variance of the model input across whole dataset
    mu_B_geomFEM: mean of the model output across whole dataset (from training data)
    sigma2_B_geomFEM: variance of the model output across whole dataset (from training data)
    bs: mini-batch size
    
    Outputs:
    --------
    comp_loss: loss function of the compliance constraints (i.e. compliance between r, L and s)
    geom_loss: loss function based on the FEM values of r, L and s
    total_loss: total loss function
    geom: predicted geometric parameters for cascade training
    """

    # Input normalization
    e = tf.constant(1e-7,dtype=tf.float32) # standardization bias
    omega2Norm = (omega2 - mu_B_omega2[:bs,:])/tf.sqrt(sigma2_B_omega2[:bs,:] + e) # standard gaussian normalization

    # Extraction of the prediction of the model
    geomNorm = IEP_model(omega2Norm,training=True) # prediction of [r,L,s] in this order

    # Output de-normalization to impose physics
    geom = tf.sqrt(sigma2_B_geomFEM[:bs,:] + e)*geomNorm + mu_B_geomFEM[:bs,:] # inverted standard normalization

    # Compliance constraints
    comp_degeneration = keras.activations.relu(-geom[:,1]/2+geom[:,0]+geom[:,2]/2) # degenerated pattern compliance
    comp_compenetration = keras.activations.relu(-geom[:,0]+geom[:,2]/2) # self-intersecting geometry compliance

    # Calculate loss
    comp_loss = tf.reduce_mean(tf.square(comp_degeneration + comp_compenetration)) # MSE computed across batch dimension
    geom_loss = tf.reduce_mean(keras.losses.mean_squared_error(geom, geom_FEM)) # MSE computed across batch dimension
    total_loss = comp_weight*comp_loss + geom_weight*geom_loss # weighted total loss

    return comp_loss, geom_loss, total_loss, geom

@tf.function
def WES_train_step(nu_s, rho_s, E_s, omega2, w_HS, k, Coord, uFEM, PDE_weight, BC_weight, data_weight, PDE_model, new_tensor, mask2b, mask3b, mu_B_Coord, sigma2_B_Coord, mu_B_uFEM, sigma2_B_uFEM, bs):
    """Calculate gradients of the loss with respect to the WES parameters.
    In this function Boundary Conditions (BC) are applied, i.e. where the Floquet-Bloch shape of the solution is implied
    
    Args:
    ----
    nu_s: effective Poisson's coefficient of the ALR as predicted by the IEPS (different for each batch)
    rho_s: effective density of the ALR as predicted by the IEPS (different for each batch)
    E_s: effective Young's Modulus of the ALR as predicted by the IEPS (different for each batch)
    omega2: the same dispersion curve fed as input to the IEPS (n rows, k columns)
    w_HS: width of the HS (different for each batch)
    k: wave number range in x direction (k rows, 1 column)
    Coord: XY coordinates coming from training dataset (51 rows, 2 columns)
    uFEM: displacements/eigenvectors resulting from FEM direct eigenfrequencies problem (51 rows, 2 x n x k columns, complex values)
    PDE_weight: weight for the PDE Loss
    BC_weight: weight for the BC Loss
    data_weight: weight for the data Loss
    PDE_model: the PDE model (WES)
    new_tensor: externally created tf.Variable used to modify shape of kx for BC application
    mask2b: externally created boolean mask for tensor slicing used to apply BC2
    mask3b: externally created boolean  mask for tensor slicing used to apply BC3
    mu_B_Coord: mean of the model input across whole dataset
    sigma2_B_Coord: variance of the model input across whole dataset
    mu_B_uFEM: mean of the model output across whole dataset (from training data)
    sigma2_B_uFEM: variance of the model output across whole dataset (from training data)
    bs: mini-batch size
    
    Outputs:
    --------
    PDE_loss: calculated PDE Loss
    BC_loss: calculated boundary conditions loss
    data_loss: calculated data loss
    total_loss: weighted sum of PDE Loss, BC Loss, and data Loss
    """
    global Xa,Ya # externally defined to allow manipulation of X,Y (from training dataset) to feed as input to PDES

    # Normalization of input data
    e = tf.constant(1e-7,dtype=tf.float32) # standardization bias
    CoordNorm = (Coord - mu_B_Coord[:bs,:,:])/tf.sqrt(sigma2_B_Coord[:bs,:,:] + e) # standard gaussian normalization

    dim_n = omega2.get_shape().as_list()[1] # number of eigenfrequencies
    dim_k = omega2.get_shape().as_list()[2] # number of elements in the wavenumber range  

    # Change shape of k to apply Floquet-Bloch BC as a matrix multiplication
    for i in range(k.get_shape().as_list()[1]):
        col = tf.expand_dims(k[:,i], axis=1) # expand dimension of k to concatenate in column direction
        repeated_col = tf.tile(col, [1, dim_n]) # repeat i-th value of k n times (given a fixed wave number, k_i is the same for every eigenfrequency)
        new_tensor = tf.concat([new_tensor, repeated_col], axis=1) # concatenate resulting vectors in column direction (N_batches rows, n*k columns)
    k_extended = tf.convert_to_tensor(new_tensor[:,1:],dtype=tf.float32) # neglect first column since it was an externally defined placeholder
    k_extended = tf.tile(k_extended,[1,3]) # repeat resulting vector times 3 since the coordinates on which Floquet-Bloch BC is applied are three at a time
    expk = tf.math.exp(-tf.complex(0.0,1.0)*(tf.cast(k_extended*tf.tile(tf.expand_dims(w_HS,1),[1,k_extended.get_shape().as_list()[1]]),dtype=tf.complex64))) # prepare term for Floquet-Bloch BC application

    #Extraction of the XY coordinates from the training tensor and repetition in order to fit the number of displacements
    if Xa is None: # code to create Xa,Ya variables as tf.Variable just once
        Xa = tf.Variable(tf.zeros([CoordNorm.get_shape().as_list()[0],102],dtype=tf.float32),trainable=False) # initialize Xa
        Ya = tf.Variable(tf.zeros([CoordNorm.get_shape().as_list()[0],102],dtype=tf.float32),trainable=False) # initialize Ya
        return Xa.assign_add(tf.tile(CoordNorm[:,:,0],[1,2])), Ya.assign_add(tf.tile(CoordNorm[:,:,1],[1,2])) # X,Y are repeated twice to form Xa,Ya (real and imaginary part are separated)
    i = 0 # initialization of while loop index
    while i < tf.shape(CoordNorm)[1]: # sweep all the 51 coordinates
        Xa[:,i:i+2].assign(tf.concat([tf.expand_dims(Xa[:,i],1),tf.expand_dims(Xa[:,i+CoordNorm.get_shape().as_list()[1]],1)],axis=1)) # assign real and imaginary part of the i-th coordinate (i.e., the same one) to 2 adjacent positions of Xa
        Ya[:,i:i+2].assign(tf.concat([tf.expand_dims(Ya[:,i],1),tf.expand_dims(Ya[:,i+CoordNorm.get_shape().as_list()[1]],1)],axis=1)) # assign real and imaginary part of the i-th coordinate (i.e., the same one) to 2 adjacent positions of Ya
        i = i + 2
    X = tf.expand_dims(tf.tile(Xa,[1,dim_n*dim_k*2]),1) # repeat resulting vector times n x k x 2 (2 are the directions of the displacements)
    Y = tf.expand_dims(tf.tile(Ya,[1,dim_n*dim_k*2]),1) # repeat resulting vector times n x k x 2 (2 are the directions of the displacements)

    #Extraction of the displacement field coming from the FEM simulation
    u_FEM = uFEM[:,:,0::2] # extract just u complex displacement (step=2)
    v_FEM = uFEM[:,:,1::2] # extract just v complex displacement (step=2)
    u_FEM = tf.reshape(u_FEM,[-1,dim_n*dim_k*CoordNorm.get_shape().as_list()[1]]) # flatten to fit output (sweep n first, k second, coordinates third), independently for each batch
    v_FEM = tf.reshape(v_FEM,[-1,dim_n*dim_k*CoordNorm.get_shape().as_list()[1]]) # flatten to fit output (sweep n first, k second, coordinates third), independently for each batch

    # Generate PDE residual for PDE_Loss
    residual, uRe, vRe, uIm, vIm, sigma_x_ALR_Re, sigma_y_ALR_Re, sigma_x_ALR_Im, sigma_y_ALR_Im, sigma_y_HS_Re, sigma_y_HS_Im = PDE_calculator(X,Y,nu_s,rho_s,E_s,omega2,PDE_model,mu_B_uFEM,sigma2_B_uFEM,bs) # compute residuals of the PDE
    idx = int(2*uRe.get_shape().as_list()[1]) # index corresponding to half of the total displacement field vector (20400*2 = 81600/2)
    PDE_loss = residual 
    
    # Import predicted stresses to impose BC
    sigma_x_ALR_re = tf.reduce_mean(sigma_x_ALR_Re[:,:,:idx:2],1) # only results corresponding to real part division are meaningful (i < idx). Reduce dimensions from [N_batch,1,81600] to [N_batch,81600]
    sigma_y_ALR_re = tf.reduce_mean(sigma_y_ALR_Re[:,:,1:idx:2],1) # only results corresponding to real part division are meaningful (i < idx). Reduce dimensions from [N_batch,1,81600] to [N_batch,81600]
    sigma_x_ALR_im = tf.reduce_mean(sigma_x_ALR_Im[:,:,idx::2],1) # only results corresponding to imaginary part division are meaningful (i > idx). Reduce dimensions from [N_batch,1,81600] to [N_batch,81600]
    sigma_y_ALR_im = tf.reduce_mean(sigma_y_ALR_Im[:,:,idx+1::2],1) # only results corresponding to imaginary part division are meaningful (i > idx). Reduce dimensions from [N_batch,1,81600] to [N_batch,81600]
    sigma_y_HS_re = tf.reduce_mean(sigma_y_HS_Re[:,:,1:idx:2],1) # only results corresponding to real part division are meaningful (i < idx). Reduce dimensions from [N_batch,1,81600] to [N_batch,81600]
    sigma_y_HS_im = tf.reduce_mean(sigma_y_HS_Im[:,:,idx+1::2],1) # only results corresponding to imaginary part division are meaningful (i > idx). Reduce dimensions from [N_batch,1,81600] to [N_batch,81600]
    
    # Application of BC
    #Neumann BCs - Null normal stresses
    # BC1
    BC1 = sigma_y_HS_re[:,0:7*dim_n*dim_k] + sigma_y_HS_im[:,0:7*dim_n*dim_k] # Lower boundary of the plate
    # BC2
    BC2_a = sigma_x_ALR_re[:,15*dim_n*dim_k:16*dim_n*dim_k] + sigma_x_ALR_im[:,15*dim_n*dim_k:16*dim_n*dim_k] # first nodes where to impose BC2
    for i in range(21,47,5): # remaining nodes where to impose BC2
        mask2b[:,i*dim_n*dim_k:(i+1)*dim_n*dim_k].assign(tf.fill([mask2b.get_shape().as_list()[0],dim_n*dim_k],True)) # switch boolean mask to True only in the wanted coordinates (n*k terms at a time)
    BC2_b = tf.reshape(tf.boolean_mask(sigma_x_ALR_re,mask2b),[-1,6*dim_n*dim_k]) + tf.reshape(tf.boolean_mask(sigma_x_ALR_im,mask2b),[-1,6*dim_n*dim_k]) # BC2 imposed on the remaining nodes. Reshape done to fix the dimensions of the vectors (dynamic by default)
    BC2 = tf.concat([BC2_a,BC2_b],axis=1) # Left boundary of the LRM
    # BC3
    BC3_a = sigma_x_ALR_re[:,19*dim_n*dim_k:20*dim_n*dim_k] + sigma_x_ALR_im[:,19*dim_n*dim_k:20*dim_n*dim_k] # first nodes where to impose BC3
    for i in range(25,51,5): # remaining nodes where to impose BC3
        mask3b[:,i*dim_n*dim_k:(i+1)*dim_n*dim_k].assign(tf.fill([mask3b.get_shape().as_list()[0],dim_n*dim_k],True)) # switch boolean mask to True only in the wanted coordinates (n*k terms at a time)
    BC3_b = tf.reshape(tf.boolean_mask(sigma_x_ALR_re,mask3b),[-1,6*dim_n*dim_k]) + tf.reshape(tf.boolean_mask(sigma_x_ALR_im,mask3b),[-1,6*dim_n*dim_k]) # BC3 imposed on the remaining nodes. Reshape done to fix the dimensions of the vectors (dynamic by default)
    BC3 = tf.concat([BC3_a,BC3_b],axis=1) # Right boundary of the LRM
    # BC 4
    BC4 = sigma_y_ALR_re[:,46*dim_n*dim_k:] + sigma_y_ALR_im[:,46*dim_n*dim_k:] # Upper boundary of the LRM

    #Dirichlet BCs - Floquet periodicity (boundary conditions change depending on the wave number)
    # BC 5
    u_pred = tf.cast(uRe,dtype=tf.complex64) + tf.complex(0.0,1.0)*tf.cast(uIm,dtype=tf.complex64) # reconstruction of the complex u displacement
    v_pred = tf.cast(vRe,dtype=tf.complex64) + tf.complex(0.0,1.0)*tf.cast(vIm,dtype=tf.complex64) # reconstruction of the complex v displacement

    floch_u2sx = u_pred[:,0:dim_n*dim_k] # first node on left side where to apply BC5
    floch_u2sx = tf.concat([floch_u2sx,u_pred[:,7*dim_n*dim_k:8*dim_n*dim_k]],axis=1) # second node on left side where to apply BC5, stack beside the previous one in column direction
    floch_u2sx = tf.concat([floch_u2sx,u_pred[:,14*dim_n*dim_k:15*dim_n*dim_k]],axis=1) # third node on left side where to apply BC5, stack beside the previous one in column direction. Resulting dimensions [N_batch rows, 3*n*k columns]
    floch_u2sx = tf.matmul(floch_u2sx,expk,transpose_b=True) # Floquet_Bloch BC reconstructed as a matrix multiplication. reduce_sum across number of coordinates is intrinsic in matmul
    floch_u2sx = tf.linalg.tensor_diag_part(floch_u2sx) # only diagonal terms of the resulting tensor are meaningful
    
    floch_u2dx = u_pred[:,6*dim_n*dim_k:7*dim_n*dim_k] # first node on right side where to apply BC5
    floch_u2dx = tf.concat([floch_u2dx,u_pred[:,13*dim_n*dim_k:14*dim_n*dim_k]],axis=1) # second node on right side where to apply BC5
    floch_u2dx = tf.concat([floch_u2dx,u_pred[:,20*dim_n*dim_k:21*dim_n*dim_k]],axis=1) # third node on right side where to apply BC5
    BC5 = tf.expand_dims((1/floch_u2dx.get_shape().as_list()[1])*tf.math.abs(floch_u2sx - tf.reduce_sum(floch_u2dx,axis=1)),1) # BC5 transformed to real value through modulus of complex number. Output is [N_batch,1] tensor
    # BC 6
    floch_v2sx = v_pred[:,0:dim_n*dim_k] # first node on left side where to apply BC6
    floch_v2sx = tf.concat([floch_v2sx,v_pred[:,7*dim_n*dim_k:8*dim_n*dim_k]],axis=1) # second node on left side where to apply BC6, stack beside the previous one in column direction
    floch_v2sx = tf.concat([floch_v2sx,v_pred[:,14*dim_n*dim_k:15*dim_n*dim_k]],axis=1) # third node on left side where to apply BC6, stack beside the previous one in column direction. Resulting dimensions [N_batch rows, 3*n*k columns]
    floch_v2sx = tf.matmul(floch_v2sx,expk,transpose_b=True) # Floquet_Bloch BC reconstructed as a matrix multiplication. reduce_sum across number of coordinates is intrinsic in matmul
    floch_v2sx = tf.linalg.tensor_diag_part(floch_v2sx) # only diagonal terms of the resulting tensor are meaningful

    floch_v2dx = v_pred[:,6*dim_n*dim_k:7*dim_n*dim_k] # first node on right side where to apply BC6
    floch_v2dx = tf.concat([floch_v2dx,v_pred[:,13*dim_n*dim_k:14*dim_n*dim_k]],axis=1) # second node on right side where to apply BC6
    floch_v2dx = tf.concat([floch_v2dx,v_pred[:,20*dim_n*dim_k:21*dim_n*dim_k]],axis=1)  # third node on right side where to apply BC6
    BC6 = tf.expand_dims((1/floch_v2dx.get_shape().as_list()[1])*tf.math.abs(floch_v2sx - tf.reduce_sum(floch_v2dx,axis=1)),1) # BC6 transformed to real value through modulus of complex number. Output is [N_batch,1] tensor
    
    #Robin BCs - Interface compliance and equilibrium
    # BC 7
    BC7 = uRe[:,15*dim_n*dim_k:20*dim_n*dim_k]+uIm[:,15*dim_n*dim_k:20*dim_n*dim_k]-uRe[:,15*dim_n*dim_k:20*dim_n*dim_k]-uIm[:,15*dim_n*dim_k:20*dim_n*dim_k]+sigma_y_ALR_re[:,15*dim_n*dim_k:20*dim_n*dim_k]+sigma_y_ALR_im[:,15*dim_n*dim_k:20*dim_n*dim_k]-2.5*sigma_y_HS_re[:,15*dim_n*dim_k:20*dim_n*dim_k]+2.5*sigma_y_HS_im[:,15*dim_n*dim_k:20*dim_n*dim_k] # u displacements and normal stresses equal at the interface. 2.5 = w_plate/w_LRM
    # BC 8
    BC8 = vRe[:,15*dim_n*dim_k:20*dim_n*dim_k]+vIm[:,15*dim_n*dim_k:20*dim_n*dim_k]-vRe[:,15*dim_n*dim_k:20*dim_n*dim_k]-vIm[:,15*dim_n*dim_k:20*dim_n*dim_k]+sigma_y_ALR_re[:,15*dim_n*dim_k:20*dim_n*dim_k]+sigma_y_ALR_im[:,15*dim_n*dim_k:20*dim_n*dim_k]-2.5*sigma_y_HS_re[:,15*dim_n*dim_k:20*dim_n*dim_k]+2.5*sigma_y_HS_im[:,15*dim_n*dim_k:20*dim_n*dim_k] # v displacements and normal stresses equal at the interface. 2.5 = w_plate/w_LRM
    
    # Calculate losses
    BC1_loss = tf.reduce_mean(keras.losses.MeanSquaredError().call(tf.zeros_like(BC1,dtype=tf.float32),BC1)) # MSE across batch dimension
    BC2_loss = tf.reduce_mean(keras.losses.MeanSquaredError().call(tf.zeros_like(BC2,dtype=tf.float32),BC2)) # MSE across batch dimension
    BC3_loss = tf.reduce_mean(keras.losses.MeanSquaredError().call(tf.zeros_like(BC3,dtype=tf.float32),BC3)) # MSE across batch dimension
    BC4_loss = tf.reduce_mean(keras.losses.MeanSquaredError().call(tf.zeros_like(BC4,dtype=tf.float32),BC4)) # MSE across batch dimension
    BC5_loss = tf.reduce_mean(keras.losses.MeanSquaredError().call(tf.zeros_like(BC5,dtype=tf.float32),BC5)) # MSE across batch dimension (output is a scalar)
    BC6_loss = tf.reduce_mean(keras.losses.MeanSquaredError().call(tf.zeros_like(BC6,dtype=tf.float32),BC6)) # MSE across batch dimension (output is a scalar)
    BC7_loss = tf.reduce_mean(keras.losses.MeanSquaredError().call(tf.zeros_like(BC7,dtype=tf.float32),BC7)) # MSE across batch dimension
    BC8_loss = tf.reduce_mean(keras.losses.MeanSquaredError().call(tf.zeros_like(BC8,dtype=tf.float32),BC8)) # MSE across batch dimension
    
    BC_loss = BC1_loss + BC2_loss + BC3_loss + BC4_loss + BC5_loss + BC6_loss + BC7_loss + BC8_loss # total BC loss is the sum of the losses due to each BC
    
    data_loss = tf.reduce_mean(keras.losses.mean_squared_error(tf.concat([tf.math.real(u_FEM),tf.math.real(v_FEM),tf.math.imag(u_FEM),tf.math.imag(v_FEM)],1),tf.concat([uRe,vRe,uIm,vIm],1)))
           
    total_loss = PDE_weight*PDE_loss+BC_weight*BC_loss+data_weight*data_loss # weighted total loss
     
    return BC_loss, PDE_loss, data_loss, total_loss

class LossTracking: # class to keep track of losses

    def __init__(self): # initialize loss trackers
        self.mean_comp_loss = keras.metrics.Mean() # compliance constraints loss
        self.mean_geom_loss = keras.metrics.Mean() # geometry loss
        self.mean_IEPStotal_loss = keras.metrics.Mean() # IEPS total loss
        self.mean_BC_loss = keras.metrics.Mean() # BC loss
        self.mean_PDE_loss = keras.metrics.Mean() # PDE loss
        self.mean_data_loss = keras.metrics.Mean() # data loss
        self.mean_WEStotal_loss = keras.metrics.Mean() # WES total loss
        self.loss_history = defaultdict(list)  # set loss_history as a list

    def update(self, comp_loss, geom_loss, IEPStotal_loss, BC_loss, PDE_loss, data_loss, WEStotal_loss):
        self.mean_comp_loss(comp_loss) # add compliance constraints loss to its loss history
        self.mean_geom_loss(geom_loss) # add geometry loss to its loss history
        self.mean_IEPStotal_loss(IEPStotal_loss) # add IEPS total loss to its loss history
        self.mean_BC_loss(BC_loss) # add BC loss to its loss history
        self.mean_PDE_loss(PDE_loss) # add PDE loss to its loss history
        self.mean_data_loss(data_loss) # add data loss to its loss history
        self.mean_WEStotal_loss(WEStotal_loss) # add PDE total loss to its loss history

    def reset(self): # reset states of the different loss trackers
        self.mean_comp_loss.reset_states()
        self.mean_geom_loss.reset_states()
        self.mean_IEPStotal_loss.reset_states()
        self.mean_BC_loss.reset_states()
        self.mean_PDE_loss.reset_states()
        self.mean_data_loss.reset_states()
        self.mean_WEStotal_loss.reset_states()

    def print(self): # print losses
        print(f"-------------------------------------------------------------------------------\nLosses:\nCompliance constraints={self.mean_comp_loss.result().numpy():.4e}\nIEPS data loss={self.mean_geom_loss.result().numpy():.4e}\nIEPS total loss={self.mean_IEPStotal_loss.result().numpy():.4e}\nBC loss={self.mean_BC_loss.result().numpy():.4e}\nPDE loss={self.mean_PDE_loss.result().numpy():.4e}\nWES data loss={self.mean_data_loss.result().numpy():.4e}\nWES total loss={self.mean_WEStotal_loss.result().numpy():.4e}")
        
    def history(self): # append loss results to their corresponding list
        self.loss_history['comp_loss'].append(self.mean_comp_loss.result().numpy())
        self.loss_history['geom_loss'].append(self.mean_geom_loss.result().numpy())
        self.loss_history['IEPStotal_loss'].append(self.mean_IEPStotal_loss.result().numpy())
        self.loss_history['BC_loss'].append(self.mean_BC_loss.result().numpy())
        self.loss_history['PDE_loss'].append(self.mean_PDE_loss.result().numpy())
        self.loss_history['Data_loss'].append(self.mean_data_loss.result().numpy())
        self.loss_history['WEStotal_loss'].append(self.mean_WEStotal_loss.result().numpy())

def displForMetrics(xDenorm,yDenorm,u_True,dim_n,dim_k,model,mu_B_Coord,sigma2_B_Coord,size,size_val):
    '''Processes inputs and outputs of PDES model to have displacements ready for metrics evaluation
    
    Args:
    ----
    xDenorm: x-coordinates (n x 51 x 1, real)
    yDenorm: y-coordinates (n x 51 x 1, real)
    u_True: tensor of true displacements (n x 51 x 800, complex)
    dim_n: number of eigenfrequencies
    dim_k: number of wave number range discrete points
    model: the WES model
    mu_B_Coord: mean of the model input across whole dataset
    sigma2_B_Coord: variance of the model input across whole dataset
    size: size of the considered dataset
    size_val: size of the validation dataset
    
    Outputs:
    -------
    displPred: predicted displacement field from model
    displTrue: ground truth of displacement field
    '''
    # Test or Validation dataset
    # if size of the dataset is equal to validation dataset, function is applied to IT_val, else to IT_test
    if size==size_val:
        size1 = 0
    else:
        size1 = size_val
        
    # Normalize input
    e = tf.constant(1e-7,dtype=tf.float32) # standardization bias
    CoordNorm = (tf.concat([xDenorm,yDenorm],axis=2) - mu_B_Coord[size1:size1+size,:,:])/tf.sqrt(sigma2_B_Coord[size1:size1+size,:,:] + e) # standard gaussian normalization
    x,y = CoordNorm[:,:,0:1],CoordNorm[:,:,1:] # normalized coordinates
    X = tf.Variable(tf.zeros([x.get_shape().as_list()[0],2*x.get_shape().as_list()[1],x.get_shape().as_list()[2]])) # prepare coordinates to host twice their number for manipulation (102)
    Y = tf.Variable(tf.zeros([y.get_shape().as_list()[0],2*y.get_shape().as_list()[1],y.get_shape().as_list()[2]])) # prepare coordinates to host twice their number for manipulation (102)
    for i  in range(x.get_shape().as_list()[1]):
        x_i = tf.tile(tf.expand_dims(x[:,i,:],1),[1,2,1]) # repeat x twice near each other (u and v are subsequent)
        X[:,i:i+2,:].assign(x_i) # assign to X
        y_i = tf.tile(tf.expand_dims(y[:,i,:],1),[1,2,1]) # repeat y twice near each other (u and v are subsequent)
        Y[:,i:i+2,:].assign(y_i) # assign to Y
    X_new = tf.tile(X,[1,dim_n*dim_k*2,1]) # repeat the new tensor n*k*2 times (eigenfrequencies, wave number range and real + imaginary part)
    Y_new = tf.tile(Y,[1,dim_n*dim_k*2,1]) # repeat the new tensor n*k*2 times (eigenfrequencies, wave number range and real + imaginary part)
    In = tf.concat([X_new,Y_new],2) # stack the new X and Y coordinates next to each other [N_batch,81600,2] to feed to the PDES model as input

    displPred = model(In) # generate u,v predictions for metrics [N_batches,81600]

    uTVect = tf.reshape(u_True,[-1,x.get_shape().as_list()[1]*dim_n*dim_k*2,1]) # reshape ground truth displacements as a vector
    uTRe = tf.math.real(uTVect) # extract real part from ground truth displacements
    uTIm = tf.math.imag(uTVect) # extract imaginary part from ground truth displacements
    displTrue = tf.concat([uTRe,uTIm],1) # concatenate real and imaginary part for metrics evaluation
    displTrue = tf.squeeze(displTrue) # reduce 1 dimension

    return displPred, displTrue

#Problem initialization and solution
#-----------------------------------------------------------------------------------------------------------------------------------
# Import dataset
InputTensor = tf.convert_to_tensor(mat73.loadmat('TrainingTensorNN.mat')['TrainingTensor'],dtype=tf.complex64) # Tensor to store values to be used during training, imported from matlab file; for -v7.3 .mat files
InputTensor = tf.transpose(InputTensor,perm=[2,0,1]) # reorder batch dimension to let it be the first one

DispCurve = InputTensor[:,:10,3:43] # [rad2/s2] needed for instantiation of IEPS model
dim_n = tf.shape(DispCurve)[1] # number of eigenfrequencies considered (10)
dim_k = tf.shape(DispCurve)[2] # number of values in the wave number range considered (40)
coord = tf.tile(InputTensor[:,:,46:48],[1,dim_k*dim_n*2*2,1]) # XY coordinates needed for instantiation of PDE model [N_batches,81600,1]
E_mat = 1.215e9 #[Pa] Young's Modulus of the bulk material used for the ALR
rho_mat = 1010 #[kg/m3] density of the bulk material used for the ALR

# Set up training configurations
batch_size = 1000 # dimension of the training data set
sub_dataset = batch_size # dimension of data set used to train in a single epoch (equal to batch size)
N_mini_batches = 25 # number of mini-batches (integer divider of batch_size)
mini_batch_size = int(sub_dataset/N_mini_batches) # dimension of training data set for mini-batch computation

val_size = 501 # dimension of validation data set: MUST BE DIFFERENT FROM test_size
test_size = 499 # dimension of testing data set: MUST BE DIFFERENT FROM val_size

n_epochs = 40 # number of epochs	

clip_th = 1 # norm clipping threshold to apply to IEPS and WES gradients to avoid gradient exploding

comp_weight= tf.constant(0.0, dtype=tf.float32) # (IEPS) weight of the compliance constraints loss function: suggested to switch off compliance constraints loss in training to avoid convergence issues
geom_weight= tf.constant(1.0, dtype=tf.float32) # (IEPS) weight of the geometry loss function
cross_train_weight= tf.constant(1e8, dtype=tf.float32) # (IEPS) weight to apply to IEPS_total_loss component for the cross-training of the IEPS itself
BC_weight= tf.constant(1e-12, dtype=tf.float32) # (WES) weight for boundary condition loss function
PDE_weight= tf.constant(1e-14, dtype=tf.float32) # (WES) weight for partial derivative equation loss function
data_weight= tf.constant(1e12, dtype=tf.float32) # (WES) weight for data loss function    

loss_tracker = LossTracking() # initialize loss tracker
val_lossIEPS_hist = [] # initialize validation loss history of IEPS
val_lossWES_hist = [] # initialize validation loss history of WES
test_lossIEPS_hist = [] # initialize test loss history of IEPS
test_lossWES_hist = [] # initialize test loss history of WES
time_hist=[] # initialize time recording

# Set up optimizer
optimizerIEPS = keras.optimizers.legacy.Adam(learning_rate=1e-3) # IEPS optimizer settings: Adam, initial learning rate 1e-3
optimizerWES = keras.optimizers.legacy.Adam(learning_rate=1e-2) # WES optimizer settings: Adam, initial learning rate 1e-2

# Initialize Training data set
InputTraining = InputTensor[val_size+test_size:val_size+test_size+batch_size,:,:] # extract training dataset
IT_train = InputTraining

with tf.device("CPU:0"): # training run on CPU
    
    # Instantiate DeepF-fNet
    IEPS = create_IEPS_model(DispCurve[0,:,:], verbose=False) # instantiate IEPS model (batch dimension = None)
    IEPS.compile(optimizer=optimizerIEPS) # compile IEPS model
    WES = create_WES_model(coord[0,:,:], Nsamples=51, verbose=False) # instantiate WES model (batch dimension = None)
    WES.compile(optimizer=optimizerWES) # compile WES model
    
    # Create validation dataset
    IT_val = InputTensor[:val_size,:,:] # validation dataset is the first in batch dimension
    # Create testing dataset
    IT_test = InputTensor[val_size:val_size+test_size,:,:] # testing dataset is the second one in batch dimension

    # Configure callbacks
    _callbacks1 = [keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=1, min_delta=1e-10),
                   tf.keras.callbacks.ModelCheckpoint('IEPS_model.h5', monitor='val_loss', save_best_only=True)]
    callbacks1 = tf.keras.callbacks.CallbackList(
                    _callbacks1, add_history=False, model=IEPS)
    _callbacks2 = [keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=1, min_delta=1e-16),
                   tf.keras.callbacks.ModelCheckpoint('WES_model.h5', monitor='val_loss', save_best_only=True)]
    callbacks2 = tf.keras.callbacks.CallbackList(
                    _callbacks2, add_history=False, model=WES)
    
    Xa = None # global coordinate to call in WES train step
    Ya = None # global coordinate to call in WES train step
    new_tensor = tf.Variable(tf.zeros_like(tf.math.real(IT_train[:mini_batch_size,0,0])), dtype=tf.float32,trainable=False) # tf.Variable to update in WES train step
    new_tensor = tf.expand_dims(new_tensor,1) # expand dimension 1 to prepare for tensor concatenation in WES train step
    mask2b = tf.Variable(tf.fill([mini_batch_size,51*dim_n*dim_k],False),trainable=False) # create boolean variable to update in WES train step
    mask3b = tf.Variable(tf.fill([mini_batch_size,51*dim_n*dim_k],False),trainable=False) # create boolean variable to update in WES train step

    # Store normalization factors: mean and variance
    # Dispersion curve
    mu_B_omega2 = tf.tile(tf.expand_dims(tf.math.reduce_mean(tf.math.real(InputTensor[:,:10,3:43]),axis=0),0),[InputTensor[:,:10,3:43].get_shape().as_list()[0],1,1]) # mean across batch dimension + batch dimension restored for broadcasting
    sigma2_B_omega2 = tf.tile(tf.expand_dims(tf.math.reduce_variance(tf.math.real(InputTensor[:,:10,3:43]),axis=0),0),[InputTensor[:,:10,3:43].get_shape().as_list()[0],1,1]) # variance across batch dimension + batch dimension restored for broadcasting
    # Ground truth geometric parameters (FEM)
    mu_B_geomFEM = tf.tile(tf.expand_dims(tf.math.reduce_mean(tf.math.real(InputTensor[:,1,:3]),axis=0),0),[InputTensor[:,1,:3].get_shape().as_list()[0],1]) # mean across batch dimension + batch dimension restored for broadcasting
    sigma2_B_geomFEM = tf.tile(tf.expand_dims(tf.math.reduce_variance(tf.math.real(InputTensor[:,1,:3]),axis=0),0),[InputTensor[:,1,:3].get_shape().as_list()[0],1]) # variance across batch dimension + batch dimension restored for broadcasting
    # Coordinates
    mu_B_Coord = tf.tile(tf.expand_dims(tf.math.reduce_mean(tf.math.real(InputTensor[:,:,46:48]),axis=0),0),[InputTensor[:,:,46:48].get_shape().as_list()[0],1,1]) # mean across batch dimension + batch dimension restored for broadcasting
    sigma2_B_Coord = tf.tile(tf.expand_dims(tf.math.reduce_variance(tf.math.real(InputTensor[:,:,46:48]),axis=0),0),[InputTensor[:,:,46:48].get_shape().as_list()[0],1,1]) # variance across batch dimension + batch dimension restored for broadcasting
    # Ground truth displacement fields (FEM)
    uTVect = tf.reshape(InputTensor[:,:,48:],[-1,InputTensor[:,:,46].get_shape().as_list()[1]*dim_n*dim_k*2,1]) # reshape ground truth displacements as a vector
    uTRe = tf.math.real(uTVect) # extract real part from ground truth displacements
    uTIm = tf.math.imag(uTVect) # extract imaginary part from ground truth displacements
    displTrue = tf.squeeze(tf.concat([uTRe,uTIm],1)) # concatenate real and imaginary part for standardization. Reduce 1 dimension
    mu_B_uFEM = tf.tile(tf.expand_dims(tf.math.reduce_mean(displTrue,axis=0),0),[displTrue[:,:].get_shape().as_list()[0],1]) # mean across batch dimension + batch dimension restored for broadcasting
    sigma2_B_uFEM = tf.tile(tf.expand_dims(tf.math.reduce_variance(displTrue,axis=0),0),[displTrue[:,:].get_shape().as_list()[0],1]) # variance across batch dimension + batch dimension restored for broadcasting

    # Start training process
    t_start=time() # record training starting time 
    for epoch in range(1, n_epochs + 1): # for loop to change epoch
        t_start_epoch = time() # record epoch starting time
        print(f"_______________________________________________________________________________\nEpoch {epoch}:")

        # Update training dataset
        IT_train = InputTraining[:sub_dataset,:,:]

        for batch_i in range(0,sub_dataset,mini_batch_size): # for loop to change mini-batch
            with tf.GradientTape(persistent=True) as tape:
                tape.reset() # restart recording
                tape.watch([IEPS.trainable_variables,WES.trainable_variables]) # record IEPS and WES trainable variables for gradient computation
                
                #Trained Network: IEPS
                
                omega2 = tf.math.real(IT_train[batch_i:batch_i+mini_batch_size,:10,3:43]) # extract dispersion function from training data set
                geomFEM = tf.math.real(IT_train[batch_i:batch_i+mini_batch_size,1,:3]) # extract FEM geometry from training tensor
                # Train step
                comp_lossIEPS, geom_lossIEPS, IEPStotal_loss, geom_new = IEPS_train_step(omega2, geomFEM, comp_weight, geom_weight, IEPS, mu_B_omega2[val_size+test_size+batch_i:val_size+test_size+batch_i+mini_batch_size,:], sigma2_B_omega2[val_size+test_size+batch_i:val_size+test_size+batch_i+mini_batch_size,:], mu_B_geomFEM[val_size+test_size+batch_i:val_size+test_size+batch_i+mini_batch_size,:], sigma2_B_geomFEM[val_size+test_size+batch_i:val_size+test_size+batch_i+mini_batch_size,:], mini_batch_size) # calculate losses of IEPS model
                # Update of training tensor
                r = geom_new[:,0] # geometry parameter r
                L = geom_new[:,1] # geometry parameter L
                s = geom_new[:,2] # geometry parameter s
                theta = tf.math.atan(2*r/L) # dependent geometric parameter
                nu_s = tf.sqrt(3.0)*(tf.math.sin((tf.constant(m.pi)/6)-theta)*tf.math.cos((tf.constant(m.pi)/6)-theta)+tf.math.sin((tf.constant(m.pi)/6)+theta)*tf.math.cos((tf.constant(m.pi)/6)+theta))/((tf.math.cos((tf.constant(m.pi)/6)-theta))**2+4*(tf.math.sin(theta))**2+(tf.math.cos((tf.constant(m.pi)/6)+theta))**2) # computation of the effective Poisson's coefficient of the metamaterial
                E_s = E_mat*(4.0/tf.sqrt(3.0))*((s/L)**3.0)*((3.0/2.0)/((tf.math.cos((tf.constant(m.pi)/6)-theta))**2+4*(tf.math.sin(theta))**2+(tf.math.cos((tf.constant(m.pi)/6)+theta))**2)) # computation of effective Young's Modulus of the metamaterial
                rho_s = rho_mat*(2.0/tf.sqrt(3.0))*(s/L)*((1.0+(4.0*tf.constant(m.pi)/3.0)*(r/L))/(1.0+4.0*(r/L)**2.0)) # computation of effective density of the metamaterial
                n_discr_k = 40 # number of discretizations of the wave number range
                n_V = 9 # number of hexagonal cells (equal to 6 unitary cells) in height in the ALR
                n_H = 5 # number of hexagonal cells (equal to 6 unitary cells) in width in the ALR
                h_ALR = tf.sqrt(3.0)*L*n_V/tf.math.cos(theta) # height of the ALR, estimated from geometric considerations
                w_ALR = 3*L*n_H/tf.math.cos(theta) # width of the ALR, estimated from geometric considerations
                h_HS = tf.constant([0.8e-3],dtype=tf.float32) #[m] height of the HS
                h_HS = tf.tile(h_HS, [mini_batch_size]) # prepare for Coord creation (h_HS equal for every batch)
                w_HS = 2.5*w_ALR # width of the HS (optimal for vibration attenuation)
                lambda_x = w_HS # wave length of the 1st Brillouin zone, corresponding to the width of the HS
                kx = tf.linspace(-tf.constant(m.pi)/lambda_x,tf.constant(m.pi)/lambda_x,n_discr_k) # wave number range
                kx = tf.transpose(kx) # adjust dimensions of kx
                h_HS = tf.expand_dims(h_HS,1) # prepare h_HS for Coord creation
                h_ALR = tf.expand_dims(h_ALR,1) # prepare h_ALR for Coord creation
                w_HS = tf.expand_dims(w_HS,1) # prepare w_HS for Coord creation
                w_ALR = tf.expand_dims(w_ALR,1) # prepare w_ALR for Coord creation
                Coord = tf.concat([tf.expand_dims(tf.concat([-w_HS/2,tf.zeros([mini_batch_size,1],dtype=tf.float32)],1),1), # creation of the 51 physical nodes of the LRM
                                    tf.expand_dims(tf.concat([-w_ALR/2,tf.zeros([mini_batch_size,1],dtype=tf.float32)],1),1),
                                    tf.expand_dims(tf.concat([-w_ALR/4,tf.zeros([mini_batch_size,1],dtype=tf.float32)],1),1),
                                    tf.expand_dims(tf.concat([tf.zeros([mini_batch_size,1],dtype=tf.float32),tf.zeros([mini_batch_size,1],dtype=tf.float32)],1),1),
                                    tf.expand_dims(tf.concat([w_ALR/4,tf.zeros([mini_batch_size,1],dtype=tf.float32)],1),1),
                                    tf.expand_dims(tf.concat([w_ALR/2,tf.zeros([mini_batch_size,1],dtype=tf.float32)],1),1),
                                    tf.expand_dims(tf.concat([w_HS/2,tf.zeros([mini_batch_size,1],dtype=tf.float32)],1),1),
                                    tf.expand_dims(tf.concat([-w_HS/2,h_HS/2],1),1),
                                    tf.expand_dims(tf.concat([-w_ALR/2,h_HS/2],1),1),
                                    tf.expand_dims(tf.concat([-w_ALR/4,h_HS/2],1),1),
                                    tf.expand_dims(tf.concat([tf.zeros([mini_batch_size,1],dtype=tf.float32),h_HS/2],1),1),
                                    tf.expand_dims(tf.concat([w_ALR/4,h_HS/2],1),1),
                                    tf.expand_dims(tf.concat([w_ALR/2,h_HS/2],1),1),
                                    tf.expand_dims(tf.concat([w_HS/2,h_HS/2],1),1),
                                    tf.expand_dims(tf.concat([-w_HS/2,h_HS],1),1),
                                    tf.expand_dims(tf.concat([-w_ALR/2,h_HS],1),1),
                                    tf.expand_dims(tf.concat([-w_ALR/4,h_HS],1),1),
                                    tf.expand_dims(tf.concat([tf.zeros([mini_batch_size,1],dtype=tf.float32),h_HS],1),1),
                                    tf.expand_dims(tf.concat([w_ALR/4,h_HS],1),1),
                                    tf.expand_dims(tf.concat([w_ALR/2,h_HS],1),1),
                                    tf.expand_dims(tf.concat([w_HS/2,h_HS],1),1),
                                    tf.expand_dims(tf.concat([-w_ALR/2,h_HS+h_ALR/6],1),1),
                                    tf.expand_dims(tf.concat([-w_ALR/4,h_HS+h_ALR/6],1),1),
                                    tf.expand_dims(tf.concat([tf.zeros([mini_batch_size,1],dtype=tf.float32),h_HS+h_ALR/6],1),1),
                                    tf.expand_dims(tf.concat([w_ALR/4,h_HS+h_ALR/6],1),1),
                                    tf.expand_dims(tf.concat([w_ALR/2,h_HS+h_ALR/6],1),1),
                                    tf.expand_dims(tf.concat([-w_ALR/2,h_HS+h_ALR/3],1),1),
                                    tf.expand_dims(tf.concat([-w_ALR/4,h_HS+h_ALR/3],1),1),
                                    tf.expand_dims(tf.concat([tf.zeros([mini_batch_size,1],dtype=tf.float32),h_HS+h_ALR/3],1),1),
                                    tf.expand_dims(tf.concat([w_ALR/4,h_HS+h_ALR/3],1),1),
                                    tf.expand_dims(tf.concat([w_ALR/2,h_HS+h_ALR/3],1),1),
                                    tf.expand_dims(tf.concat([-w_ALR/2,h_HS+h_ALR/2],1),1),
                                    tf.expand_dims(tf.concat([-w_ALR/4,h_HS+h_ALR/2],1),1),
                                    tf.expand_dims(tf.concat([tf.zeros([mini_batch_size,1],dtype=tf.float32),h_HS+h_ALR/2],1),1),
                                    tf.expand_dims(tf.concat([w_ALR/4,h_HS+h_ALR/2],1),1),
                                    tf.expand_dims(tf.concat([w_ALR/2,h_HS+h_ALR/2],1),1),
                                    tf.expand_dims(tf.concat([-w_ALR/2,h_HS+2*h_ALR/3],1),1),
                                    tf.expand_dims(tf.concat([-w_ALR/4,h_HS+2*h_ALR/3],1),1),
                                    tf.expand_dims(tf.concat([tf.zeros([mini_batch_size,1],dtype=tf.float32),h_HS+2*h_ALR/3],1),1),
                                    tf.expand_dims(tf.concat([w_ALR/4,h_HS+2*h_ALR/3],1),1),
                                    tf.expand_dims(tf.concat([w_ALR/2,h_HS+2*h_ALR/3],1),1),
                                    tf.expand_dims(tf.concat([-w_ALR/2,h_HS+5*h_ALR/6],1),1),
                                    tf.expand_dims(tf.concat([-w_ALR/4,h_HS+5*h_ALR/6],1),1),
                                    tf.expand_dims(tf.concat([tf.zeros([mini_batch_size,1],dtype=tf.float32),h_HS+5*h_ALR/6],1),1),
                                    tf.expand_dims(tf.concat([w_ALR/4,h_HS+5*h_ALR/6],1),1),
                                    tf.expand_dims(tf.concat([w_ALR/2,h_HS+5*h_ALR/6],1),1),
                                    tf.expand_dims(tf.concat([-w_ALR/2,h_HS+h_ALR],1),1),
                                    tf.expand_dims(tf.concat([-w_ALR/4,h_HS+h_ALR],1),1),
                                    tf.expand_dims(tf.concat([tf.zeros([mini_batch_size,1],dtype=tf.float32),h_HS+h_ALR],1),1),
                                    tf.expand_dims(tf.concat([w_ALR/4,h_HS+h_ALR],1),1),
                                    tf.expand_dims(tf.concat([w_ALR/2,h_HS+h_ALR],1),1)],axis=1)
                w_HS = tf.reduce_mean(w_HS,1) # adjust dimension for processing
                
                #Trained Network: WES
                
                uFEM = IT_train[batch_i:batch_i+mini_batch_size,:,48:] # extract FEM displacements
                # Train step
                # Ignore the complex64-to-float32 casting warning: both inputs and outputs are real, so the complex gradients are unconnected and therefore equal to 0 
                tf.get_logger().setLevel('ERROR') # switch off warnings
                BC_loss, PDE_loss, data_loss, WEStotal_loss = WES_train_step(nu_s, rho_s, E_s, omega2, w_HS, kx, Coord, uFEM, PDE_weight, BC_weight, data_weight, WES, new_tensor,mask2b,mask3b, mu_B_Coord[val_size+test_size+batch_i:val_size+test_size+batch_i+mini_batch_size,:], sigma2_B_Coord[val_size+test_size+batch_i:val_size+test_size+batch_i+mini_batch_size,:], mu_B_uFEM[val_size+test_size+batch_i:val_size+test_size+batch_i+mini_batch_size,:], sigma2_B_uFEM[val_size+test_size+batch_i:val_size+test_size+batch_i+mini_batch_size,:], mini_batch_size) # calculate losses of WES
                IEPloss4grad = cross_train_weight*IEPStotal_loss + PDE_weight*PDE_loss + BC_weight*BC_loss # add the two losses to cross-train IEPS
                tf.get_logger().setLevel('WARNING') # switch on warning
                mask2b.assign(tf.fill([mask2b.get_shape().as_list()[0],mask2b.get_shape().as_list()[1]],False)) # re-initialize boolean mask to false
                mask3b.assign(tf.fill([mask3b.get_shape().as_list()[0],mask3b.get_shape().as_list()[1]],False)) # re-initailize boolean mask to false
                
            # Back-propagate losses to adjust weights and biases of IEPS and WES
            # Ignore the complex64-to-float32 casting warning: both inputs and outputs are real, so the complex gradients are unconnected and therefore equal to 0 
            tf.get_logger().setLevel('ERROR') # switch off warnings
            gradientsIEPS = tape.gradient(IEPloss4grad, IEPS.trainable_variables) # compute gradients to apply during backpropagation
            gradientsWES = tape.gradient(WEStotal_loss, WES.trainable_variables) # compute gradients to apply during backpropagation
            tf.get_logger().setLevel('WARNING') # switch on warning
            # Gradient descent
            gradientsIEPS = [tf.clip_by_norm(g, clip_th) for g in gradientsIEPS] # apply norm clipping to IEPS gradients
            gradientsWES = [tf.clip_by_norm(g, clip_th) for g in gradientsWES] # apply norm clipping to WES gradients
            WES.optimizer.apply_gradients(zip(gradientsWES, WES.trainable_variables))
            IEPS.optimizer.apply_gradients(zip(gradientsIEPS, IEPS.trainable_variables))
        
        # Loss tracking
        loss_tracker.update(comp_lossIEPS, geom_lossIEPS, IEPStotal_loss, BC_loss, PDE_loss, data_loss, WEStotal_loss) # update loss tracker

        # Loss summary
        loss_tracker.history()
        loss_tracker.print()
        loss_tracker.reset()

        # Validation
        e = tf.constant(1e-7,dtype=tf.float32) # standardization bias
        omega2_validNorm = (tf.math.real(IT_val[:,:10,3:43])- mu_B_omega2[:val_size,:])/tf.sqrt(sigma2_B_omega2[:val_size,:] + e) # standard gaussian normalization
        geom_validNorm = IEPS(omega2_validNorm) # prediction for validation of IEPS
        geom_valid = tf.sqrt(sigma2_B_geomFEM[:val_size,:] + e)*geom_validNorm + mu_B_geomFEM[:val_size,:] # inverted standard normalization
        val_lossIEPS = tf.reduce_mean(keras.losses.mean_squared_error(tf.math.real(IT_val[:,1,:3]), geom_valid)) # MSE across batches, mean across geometric parameters
        print(f"-------------------------------------------------------------------------------\nval_loss (IEPS): {val_lossIEPS:.4e}, lr (IEPS): {IEPS.optimizer.lr.numpy():.2e}")
    
        displ_validNorm, displ_true = displForMetrics(tf.math.real(tf.expand_dims(IT_val[:,:,46],2)),tf.math.real(tf.expand_dims(IT_val[:,:,47],2)),IT_val[:,:,48:],dim_n,dim_k,WES,mu_B_Coord,sigma2_B_Coord,val_size,val_size) # predict and post-process quantities for validation
        displ_valid = tf.sqrt(sigma2_B_uFEM[:val_size,:] + e)*displ_validNorm + mu_B_uFEM[:val_size,:] # inverted standard normalization
        val_lossWES = tf.reduce_mean(keras.losses.mean_squared_error(displ_true,displ_valid)) #MSE across batches, mean across u,v displacements
        print(f"val_loss (WES): {val_lossWES:.4e}, lr (WES): {WES.optimizer.lr.numpy():.2e}")

        # Callback at the end of epoch
        callbacks1.on_epoch_end(epoch, logs={'val_loss': val_lossIEPS})
        callbacks2.on_epoch_end(epoch, logs={'val_loss': val_lossWES})
        val_lossIEPS_hist.append(val_lossIEPS)
        val_lossWES_hist.append(val_lossWES)

        # Test dataset
        omega2_testNorm = (tf.math.real(IT_test[:,:10,3:43])- mu_B_omega2[val_size:val_size+test_size,:])/tf.sqrt(sigma2_B_omega2[val_size:val_size+test_size,:] + e) # standard gaussian normalization
        geom_testNorm = IEPS(omega2_testNorm) # MSE across batches, mean across geometric parameters
        geom_test = tf.sqrt(sigma2_B_geomFEM[val_size:val_size+test_size,:] + e)*geom_testNorm + mu_B_geomFEM[val_size:val_size+test_size,:] # inverted standard normalization
        test_lossIEPS = tf.reduce_mean(keras.losses.mean_squared_error(tf.math.real(IT_test[:,1,:3]), geom_test))

        displ_TestNorm, displTrue = displForMetrics(tf.math.real(tf.expand_dims(IT_test[:,:,46],2)),tf.math.real(tf.expand_dims(IT_test[:,:,47],2)),IT_test[:,:,48:],dim_n,dim_k,WES,mu_B_Coord,sigma2_B_Coord,test_size,val_size) # predict and post-process quantities for validation
        displ_Test = tf.sqrt(sigma2_B_uFEM[val_size:val_size+test_size,:] + e)*displ_TestNorm + mu_B_uFEM[val_size:val_size+test_size,:] # inverted standard normalization
        test_lossWES = tf.reduce_mean(keras.losses.mean_squared_error(displTrue, displ_Test))
        print(f"-------------------------------------------------------------------------------\ntest_loss (IEPS): {test_lossIEPS:.4e}\ntest_loss (WES): {test_lossWES:.4e}")  #MSE across batches, mean across geometry and u,v displacements
        test_lossIEPS_hist.append(test_lossIEPS)
        test_lossWES_hist.append(test_lossWES)

        # Shuffle data set randomly for next epoch
        InputTraining = tf.random.shuffle(InputTraining)

        # Print current run time at the end of each epoch
        t_curr=(time()-t_start_epoch)/60
        print(f"-------------------------------------------------------------------------------\nRun time: {round(t_curr,1)} min")

    # Print total run time at the end of training
    print(f"\nTotal run time: {round((time()-t_start)/60,1)} min")

# Post-Processing
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
# History for losses

font = {'size'   : 16}
mp.rc('font', **font)
fig, ax = plt.subplots(3, 3, figsize=(16, 9))
fig.subplots_adjust(hspace=0.5)
fig.subplots_adjust(wspace=0.5)

ax[0,0].plot(range(n_epochs), loss_tracker.loss_history['comp_loss'], color='b',lw=3)
ax[0,1].plot(range(n_epochs), loss_tracker.loss_history['geom_loss'], color='b',lw=3)
ax[1,2].plot(range(n_epochs), loss_tracker.loss_history['IEPStotal_loss'], color='r',lw=3)
ax[1,0].plot(range(n_epochs), loss_tracker.loss_history['BC_loss'], color='b',lw=3)
ax[1,1].plot(range(n_epochs), loss_tracker.loss_history['Data_loss'], color='b',lw=3)
ax[0,2].plot(range(n_epochs), loss_tracker.loss_history['PDE_loss'], color='b',lw=3)
ax[2,2].plot(range(n_epochs), loss_tracker.loss_history['WEStotal_loss'], color='r',lw=3)
ax[0,0].set_title('Compliance Constraints')
ax[0,1].set_title('Geometry Loss')
ax[1,2].set_title('IEPS Total Loss')
ax[1,0].set_title('BC Loss')
ax[1,1].set_title('Data Loss')
ax[0,2].set_title('PDE Loss')
ax[2,2].set_title('WES Total Loss')
for row in ax:
    for axs in row:
        axs.set_yscale('log')
        axs.tick_params(axis='both', which='major', labelsize=12)
        axs.grid(True)
ax[0,0].set_yscale('linear')
        
plt.savefig('loss.png', bbox_inches='tight', pad_inches=0, transparent=False)

# Test and Validation losses comparison

fig2, ax = plt.subplots(1, 2, figsize=(16, 9))
fig2.subplots_adjust(wspace=0.3)

ax[0].plot(range(n_epochs), loss_tracker.loss_history['geom_loss'], color='r',lw=3, label='Training')
ax[0].plot(range(n_epochs), val_lossIEPS_hist, color='b',lw=3, label='Validation')
ax[0].plot(range(n_epochs), test_lossIEPS_hist, color='g',lw=3, label='Test')
ax[0].legend()
ax[1].plot(range(n_epochs), loss_tracker.loss_history['Data_loss'], color='r',lw=3, label= 'Training')
ax[1].plot(range(n_epochs), val_lossWES_hist, color='b',lw=3, label='Validation')
ax[1].plot(range(n_epochs), test_lossWES_hist, color='g',lw=3, label='Test')
ax[1].legend()
ax[0].set_title('IEPS Net')
ax[1].set_title('WES Net')

for axs in ax:
    axs.set_yscale('log')
    axs.tick_params(axis='both', which='major', labelsize=12)
    axs.grid(True)
        
plt.savefig('validation.png', bbox_inches='tight', pad_inches=0, transparent=False)

# Export data to Matlab for post processing

sio.savemat('TrainingResults.mat', dict(manufacturing_loss=loss_tracker.loss_history['comp_loss'], IEPS_data_loss=loss_tracker.loss_history['geom_loss'],
                                             IEPS_total_loss=loss_tracker.loss_history['IEPStotal_loss'], BC_loss=loss_tracker.loss_history['BC_loss'],
                                             WES_data_loss=loss_tracker.loss_history['Data_loss'], PDE_loss=loss_tracker.loss_history['PDE_loss'],
                                             WES_total_loss=loss_tracker.loss_history['WEStotal_loss'], validation_IEPS=val_lossIEPS_hist,
                                             test_IEPS=test_lossIEPS_hist, validation_WES=val_lossWES_hist, test_WES=test_lossWES_hist))
