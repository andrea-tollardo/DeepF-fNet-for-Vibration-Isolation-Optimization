# Importation of the modules
import numpy as np

#==========================================#
#                                          #
#   DeepF-fNet: tri-chiral honeycomb LRM   #
#                                          #
#      Bandgaps extraction for SICE4       #
#                                          #
#         Author: Andrea Tollardo          #
#                                          #
#==========================================#

# Import dataset and extract the dispersion curves
DispCurve_np = np.load('omega2_database.npy') # load dispersion curve from database_extraction.py

# Extract max and min of each eigenfrequency
DC_max = np.max(DispCurve_np, axis=2) # extract higher boundaries of each dispersion curve
DC_min = np.min(DispCurve_np, axis=2) # extract lower boundaries of each dispersion curve

# Extract bandgap between two consecutive frequencies
DC_max_new = DC_max[:,:-1] # reduce the range of frequencies to the useful ones
DC_min_new = DC_min[:,1:] # reduce the range of frequencies to the useful ones
bg = DC_min_new - DC_max_new # compute bandgap
bg[bg < 0] = 0.0 # put negative bandgaps to zero: these will be penalized in the subsequent MSE calculation
f_bg = DC_min_new - bg/2 # compute the average frequencies of the bandgaps

# Save the bandgaps and the average bandgap frequencies
BG = [bg,f_bg] # list storing the relevant quantities
np.save('Bandgaps',BG) # save bandgaps
