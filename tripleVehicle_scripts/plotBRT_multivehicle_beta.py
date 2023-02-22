# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, training, loss_functions, modules

import torch
import numpy as np
import scipy
import scipy.io as spio
import scipy.ndimage
import math
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import json

model_name = 'tripleBeta_test1'
checkpoint_toload=157000

# Load the dataset
opt_path = os.path.join('./logs',model_name, 'commandline_args.txt')


parser = ArgumentParser()
opt = parser.parse_args()
with open(opt_path, 'r') as f:
    opt.__dict__ = json.load(f)


dataset = dataio.ReachabilityMultiVehicleCollisionBeta(numpoints=65000, velocity=opt.velocity, 
                                                          omega_max=opt.omega_max, pretrain=opt.pretrain, tMax=opt.tMax, tMin=opt.tMin,
                                                          counter_start=opt.counter_start, counter_end=opt.counter_end, 
                                                          numEvaders=opt.numEvaders, pretrain_iters=opt.pretrain_iters, 
                                                          angle_alpha=opt.angle_alpha, time_alpha=opt.time_alpha, 
                                                          num_src_samples=opt.num_src_samples,diffModel=opt.diffModel)


model = modules.SingleBVPNet(in_features=13, out_features=1, type=opt.model, mode=opt.mode,
                                final_layer_factor=1., hidden_features=512, num_hidden_layers=opt.num_hl)
model.cuda()
root_path = os.path.join('./logs', model_name)
ckpt_dir = os.path.join(root_path, 'checkpoints')
ckpt_path = os.path.join(ckpt_dir, 'model_epoch_%04d.pth' % checkpoint_toload)
checkpoint = torch.load(ckpt_path)
try:
  model_weights = checkpoint['model']
except:
  model_weights = checkpoint
model.load_state_dict(model_weights)
model.eval()

# Define the loss
level = 0.001 
angle_alpha=opt.angle_alpha
poss = {}
thetas = {}
# Position and theta slices to be plotted for the 1st Evader
poss['1E'] = [(-0.4, 0.0)]
thetas['1E'] = [0.0*math.pi]
# Position and theta slices to be plotted for the 2nd Evader
poss['2E'] = [(0.43, 0.33)]
thetas['2E'] = [-2.44]
# Theta of the ego vehicle
ego_vehicle_theta = [-2.54]
# betas for vehicle pairs
vehicle_betas_01 = [-1, -1/3, 0.5]
vehicle_betas_02 = [-1, -1/3, 0.5]
vehicle_betas_12 = [-1, -1/3, 0.5]

# Time at which the sets should be plotted
time = 1.0
# Number of slices to plot
num_slices = 3
# Save the value function arrays
val_functions = {}
val_functions['pairwise'] = []
val_functions['full'] = []



# Time values at which the function needs to be plotted
times = [0., 0.25, 0.5]
num_times = len(times)

# Create a figure
fig = plt.figure(figsize=(5*num_slices, 5*num_times))
#fig_error = plt.figure(figsize=(5*num_slices, 5))
fig_valfunc = plt.figure(figsize=(5*num_slices, 5*num_times))
# Get the meshgrid in the (x, y) coordinate
sidelen = 200
mgrid_coords = dataio.get_mgrid(sidelen)

for k in range (num_times):
  # Time coordinates
  time_coords = torch.ones(mgrid_coords.shape[0], 1) * times[k]
  # Start plotting the results
  for i in range(num_slices):
    coords = torch.cat((time_coords, mgrid_coords), dim=1) 
    #pairwise_coords = {}

    # Setup the X-Y coordinates
    for j in range(2):
      evader_key = '%i' %(j+1) + 'E'
      # X-Y coordinates of the evaders for the full game
      xcoords = torch.ones(mgrid_coords.shape[0], 1) * poss[evader_key][0][0]#first 0 hardcoded
      ycoords = torch.ones(mgrid_coords.shape[0], 1) * poss[evader_key][0][1]#first 0 hardcoded
      coords = torch.cat((coords, xcoords, ycoords), dim=1) 

      # X-Y coordinates of the evaders for the pairwise game
      #pairwise_coords[evader_key] = torch.cat((time_coords, xcoords, ycoords, mgrid_coords), dim=1)

    # Setup the theta coordinates
    coords_ego_theta = ego_vehicle_theta[0] * torch.ones(mgrid_coords.shape[0], 1)/(math.pi * angle_alpha)  #first 0 hardcoded
    coords = torch.cat((coords, coords_ego_theta), dim=1)
    for j in range(2):
      evader_key = '%i' %(j+1) + 'E'
      # Theta coordinates of the evaders for the full game
      tcoords = torch.ones(mgrid_coords.shape[0], 1) * thetas[evader_key][0]/(math.pi * angle_alpha) #thetas 0 hardcoded
      coords = torch.cat((coords, tcoords), dim=1)

      # Theta coordinates of the evaders for the pairwise game
      #pairwise_coords[evader_key] = torch.cat((pairwise_coords[evader_key], tcoords, coords_ego_theta), dim=1)

    # Setup the beta coordinates
    coords_beta01 = vehicle_betas_01[i] * torch.ones(mgrid_coords.shape[0], 1)
    coords_beta02 = vehicle_betas_02[i] * torch.ones(mgrid_coords.shape[0], 1)
    coords_beta12 = vehicle_betas_12[i] * torch.ones(mgrid_coords.shape[0], 1)
    coords = torch.cat((coords, coords_beta01, coords_beta02, coords_beta12), dim=1)

    model_in = {'coords': coords[:, None, :].cuda()}
    model_out = model(model_in)

    # Detatch model ouput and reshape
    model_out = model_out['model_out'].detach().cpu().numpy()
    model_out = model_out.reshape((sidelen, sidelen))

    # Unnormalize the value function
    model_out = (model_out*dataset.var/dataset.norm_to) + dataset.mean 

    if opt.diffModel:
      lx = dataset.compute_lx(coords)
      lx = lx.detach().cpu().numpy()
      lx = lx.reshape((sidelen, sidelen))
      model_out = model_out + lx - dataset.mean

    # Plot the zero level sets
    valfunc = model_out*1.
    model_out = (model_out <= level)*1.


    # Plot the actual data and small aircrafts
    ax = fig.add_subplot(num_times, num_slices, (i+1) + k*num_slices)
    ax_valfunc = fig_valfunc.add_subplot(num_times, num_slices, (i+1) + k*num_slices)
    #ax_error = fig_error.add_subplot(1, num_slices, i+1)
    aircraft_size = 0.2
    sA = {}

    for j in range(2):
      evader_key = '%i' %(j+1) + 'E'
      aircraft_image = scipy.ndimage.rotate(plt.imread('resources/ego_aircraft.png'), 180.0*thetas[evader_key][0]/math.pi)# thetas 0 hardcoded
      sA[evader_key] = ax.imshow(aircraft_image, extent=(poss[evader_key][0][0]-aircraft_size, poss[evader_key][0][0]+aircraft_size, poss[evader_key][0][1]-aircraft_size, poss[evader_key][0][1]+aircraft_size))# all poss first 0 hardcoded
      ax.plot(poss[evader_key][0][0], poss[evader_key][0][1], "o")# poss 0 hardcoded
    s = ax.imshow(model_out.T, cmap='bwr_r', alpha=0.5, origin='lower', vmin=-1., vmax=1., extent=(-1., 1., -1., 1.))
    sV1 = ax_valfunc.imshow(valfunc.T, cmap='bwr_r', alpha=0.8, origin='lower', vmin=-0.2, vmax=0.2, extent=(-1., 1., -1., 1.))
    sV2 = ax_valfunc.contour(valfunc.T, cmap='bwr_r', alpha=0.5, origin='lower', vmin=-0.2, vmax=0.2, levels=30, extent=(-1., 1., -1., 1.))
    plt.clabel(sV2, levels=30, colors='k')
    fig_valfunc.colorbar(sV1) 
    val_functions['full'].append(valfunc)

    fig.suptitle('t0=%.2f, t1=%.2f, t2=%.2f ' % (times[0],times[1],times[2]), fontsize=26)
    fig.suptitle('t0=%.2f, t1=%.2f, t2=%.2f ' % (times[0],times[1],times[2]), fontsize=26)
    fig_valfunc.suptitle('t0=%.2f, t1=%.2f, t2=%.2f ' % (times[0],times[1],times[2]), fontsize=26)
    


  fig.savefig(os.path.join(root_path, 'BRS_load_%04d.png' % checkpoint_toload))
  fig_valfunc.savefig(os.path.join(root_path, 'VAL_load_%04d.png' % checkpoint_toload))





