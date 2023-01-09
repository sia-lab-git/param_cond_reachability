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
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-5, help='learning rate. default=2e-5')
p.add_argument('--num_epochs', type=int, default=100000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model', type=str, default='sine', required=False, choices=['sine', 'tanh', 'sigmoid', 'relu'],
               help='Type of model to evaluate, default is sine.')
p.add_argument('--mode', type=str, default='mlp', required=False, choices=['mlp', 'rbf', 'pinn'],
               help='Whether to use uniform velocity parameter')
p.add_argument('--tMin', type=float, default=0.0, required=False, help='Start time of the simulation')
p.add_argument('--tMax', type=float, default=0.5, required=False, help='End time of simulation')
p.add_argument('--num_hl', type=int, default=3, required=False, help='The number of hidden layers')
p.add_argument('--pretrain_iters', type=int, default=2000, required=False, help='Number of pretrain iterations')

p.add_argument('--counter_start', type=int, default=-1, required=False, help='Defines the initial time for the curriculul training')
p.add_argument('--counter_end', type=int, default=-1, required=False, help='Defines the linear step for curriculum training starting from the initial time')
p.add_argument('--num_src_samples', type=int, default=1000, required=False, help='Number of source samples at each time step')

p.add_argument('--velocity', type=float, default=0.6, required=False, help='Speed of the dubins car')
p.add_argument('--omega_max', type=float, default=1.1, required=False, help='Turn rate of the car')
p.add_argument('--angle_alpha', type=float, default=1.0, required=False, help='Angle alpha coefficient.')
p.add_argument('--time_alpha', type=float, default=1.0, required=False, help='Time alpha coefficient.')
#p.add_argument('--collisionR', type=float, default=0.25, required=False, help='Collision radisu between vehicles')
p.add_argument('--numEvaders', type=int, default=1, required=False, help='Number of evaders that the ego vehicle need to avoid')
p.add_argument('--minWith', type=str, default='none', required=False, choices=['none', 'zero', 'target'], help='BRS vs BRT computation')

p.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
p.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')
p.add_argument('--pretrain', action='store_true', default=False, required=False, help='Pretrain dirichlet conditions')
p.add_argument('--adjust_relative_grads', action='store_true', default=False, required=False, help='Adjust the relative gradient values.')
p.add_argument('--diffModel', action='store_true', default=False, required=False, help='Should we train the difference model instead.')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--checkpoint_toload', type=int, default=0, help='Checkpoint from which to restart the training.')
opt = p.parse_args()

# Set the source coordinates for the target set and the obstacle sets
#source_coords = [0., 0., 0.]
if opt.counter_start == -1:
  opt.counter_start = opt.checkpoint_toload

if opt.counter_end == -1:
  opt.counter_end = opt.num_epochs

dataset = dataio.ReachabilityMultiVehicleCollisionBeta(numpoints=65000, velocity=opt.velocity, 
                                                          omega_max=opt.omega_max, pretrain=opt.pretrain, tMax=opt.tMax, tMin=opt.tMin,
                                                          counter_start=opt.counter_start, counter_end=opt.counter_end, 
                                                          numEvaders=opt.numEvaders, pretrain_iters=opt.pretrain_iters, 
                                                          angle_alpha=opt.angle_alpha, time_alpha=opt.time_alpha, 
                                                          num_src_samples=opt.num_src_samples,diffModel=opt.diffModel)

dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

model = modules.SingleBVPNet(in_features=13, out_features=1, type=opt.model, mode=opt.mode,
                                final_layer_factor=1., hidden_features=512, num_hidden_layers=opt.num_hl)

model.cuda()

# Define the loss
loss_fn = loss_functions.initialize_hji_MultiVehicleCollisionBeta(dataset, opt.minWith, opt.diffModel)

root_path = os.path.join(opt.logging_root, opt.experiment_name)

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


def val_fn(model, ckpt_dir, epoch):
  # Time values at which the function needs to be plotted
  times = [0., 0.5, 1.0]
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

    fig.savefig(os.path.join(ckpt_dir, 'BRS_epoch_%04d.png' % epoch))
    fig_valfunc.savefig(os.path.join(ckpt_dir, 'VAL_epoch_%04d.png' % epoch))
  return



training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, clip_grad=opt.clip_grad,
               use_lbfgs=opt.use_lbfgs, validation_fn=val_fn, start_epoch=opt.checkpoint_toload, 
               adjust_relative_grads=opt.adjust_relative_grads, args=opt.__dict__)