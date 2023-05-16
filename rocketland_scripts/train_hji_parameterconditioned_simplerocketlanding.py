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
p.add_argument('--num_epochs', type=int, default=200000,
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
p.add_argument('--tMax', type=float, default=1.0, required=False, help='End time of the simulation')
p.add_argument('--num_hl', type=int, default=3, required=False, help='The number of hidden layers')
p.add_argument('--num_nl', type=int, default=512, required=False, help='Number of neurons per hidden layer.')
p.add_argument('--pretrain_iters', type=int, default=100000, required=False, help='Number of pretrain iterations')
p.add_argument('--counter_start', type=int, default=0, required=False, help='Defines the initial time for the curriculul training')
p.add_argument('--counter_end', type=int, default=100000, required=False, help='Defines the linear step for curriculum training starting from the initial time')
p.add_argument('--num_src_samples', type=int, default=20000, required=False, help='Number of source samples at each time step')
p.add_argument('--num_target_samples', type=int, default=10000, required=False, help='Number of samples inside the target set')

p.add_argument('--minWith', type=str, default='target', required=False, choices=['none', 'zero', 'target'], help='BRS vs BRT computation')
#p.add_argument('--lxType', type=str, default='unit_normalized_max', required=False, help='Different definitions of the l(x) function.')
p.add_argument('--lxType', type=str, default='reducedlx_max_normalized', required=False, help='Different definitions of the l(x) function.')
p.add_argument('--diffModel', action='store_true', default=False, required=False, help='Should we train the difference model instead.')
p.add_argument('--dirichlet_loss_factor', default=750.0, required=False, type=float, help='The relative loss factor or dirichlet loss.')

p.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
p.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')
p.add_argument('--adjust_relative_grads', action='store_true', default=False, help='Adjust the relative weights between different losses')
p.add_argument('--pretrain', action='store_true', default=False, required=False, help='Pretrain dirichlet conditions')

p.add_argument('--seed', type=int, default=0, required=False, help='Seed for the simulation.')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--checkpoint_toload', type=int, default=0, help='Checkpoint from which to restart the training.')
opt = p.parse_args()

# Set the source coordinates for the target set and the obstacle sets
source_coords = [0., 0., 0.]
if opt.counter_start == -1:
  opt.counter_start = opt.checkpoint_toload

if opt.counter_end == -1:
  opt.counter_end = opt.num_epochs

dataset = dataio.ReachabilityParameterConditionedSimpleRocketLandingSource(numpoints=65000, pretrain=opt.pretrain, tMin=opt.tMin,
                                                                           tMax=opt.tMax, counter_start=opt.counter_start, counter_end=opt.counter_end,
                                                                           pretrain_iters=opt.pretrain_iters, num_src_samples=opt.num_src_samples,
                                                                          diffModel=opt.diffModel, num_target_samples=opt.num_target_samples, lxType=opt.lxType)

dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

model = modules.SingleBVPNet(in_features=8, out_features=1, type=opt.model, mode=opt.mode,
                             final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
model.cuda()

# Define the loss function
loss_fn = loss_functions.initialize_hji_rocketlanding(dataset, opt.minWith, opt.diffModel, opt.dirichlet_loss_factor)

root_path = os.path.join(opt.logging_root, opt.experiment_name)

def val_fn(model, ckpt_dir, epoch):
  # Normalization coefficients
  alpha = dataset.alpha
  beta = dataset.beta

  # Time values at which the function needs to be plotted
  times = np.array([0., 0.05, 0.1, 0.2, 0.4, 0.75, 1.0])#*opt.tMax
  num_times = np.shape(times)[0]

  # Slices to be plotted 
  slices_toplot = [{'th' : 0.0, 'y_dot' : 0.0, 'z_dot' : 0.0, 'th_dot' : 0.0, 'param' : -15.0},
                   {'th' : 0.0, 'y_dot' : 0.0, 'z_dot' : 0.0, 'th_dot' : 0.0, 'param' : 0.0},
                   {'th' : 0.0, 'y_dot' : 0.0, 'z_dot' : 0.0, 'th_dot' : 0.0, 'param' : 15.0},
                   {'th' : 1.0*math.pi, 'y_dot' : 0.0, 'z_dot' : -10.0, 'th_dot' : 0.0, 'param' : -15.0},
                   {'th' : 1.0*math.pi, 'y_dot' : 0.0, 'z_dot' : -10.0, 'th_dot' : 0.0, 'param' : 0.0},
                   {'th' : 1.0*math.pi, 'y_dot' : 0.0, 'z_dot' : -10.0, 'th_dot' : 0.0, 'param' : 15.0}]
  num_slices = len(slices_toplot)

  # Create a figure
  fig = plt.figure(figsize=(5*num_times, 5*num_slices))

  # Get the meshgrid in the (y, z) coordinate
  sidelen = 200
  mgrid_coords = dataio.get_mgrid(sidelen, dim=2)

  # Start plotting the results
  for i in range(num_times):
    time_coords = torch.ones(mgrid_coords.shape[0], 1) * times[i]

    for j in range(num_slices):
      th = torch.ones(mgrid_coords.shape[0], 1) * (slices_toplot[j]['th'] - beta['th']) / alpha['th']
      y_dot = torch.ones(mgrid_coords.shape[0], 1) * (slices_toplot[j]['y_dot'] - beta['y_dot']) / alpha['y_dot']
      z_dot = torch.ones(mgrid_coords.shape[0], 1) * (slices_toplot[j]['z_dot'] - beta['z_dot']) / alpha['z_dot']
      th_dot = torch.ones(mgrid_coords.shape[0], 1) * (slices_toplot[j]['th_dot'] - beta['th_dot']) / alpha['th_dot']
      param_values = torch.ones(mgrid_coords.shape[0], 1) * (slices_toplot[j]['param'] - beta['param']) / alpha['param']
      coords = torch.cat((time_coords, mgrid_coords, th, y_dot, z_dot, th_dot, param_values), dim=1) 
      model_in = {'coords': coords.cuda()}
      model_out = model(model_in)['model_out']

      # Detatch model ouput and reshape
      model_out = model_out.detach().cpu().numpy()
      model_out = model_out.reshape((sidelen, sidelen))

      # Unnormalize the value function 
      model_out = (model_out*dataset.var/dataset.norm_to) + dataset.mean 

      if opt.diffModel:
        lx = dataset.compute_lx(coords[..., 1:])
        lx = lx.detach().cpu().numpy()
        lx = lx.reshape((sidelen, sidelen))
        model_out = model_out + lx - dataset.mean

      # Plot the zero level sets
      model_out = (model_out <= 0.001)*1.

      # Plot the actual data
      ax = fig.add_subplot(num_times, num_slices, (j+1) + i*num_slices)
      ax.set_title('t = %0.2f, th = %0.2f, zd = %0.2f, p = %0.2f' % (times[i], slices_toplot[j]['th'], slices_toplot[j]['z_dot'], slices_toplot[j]['param']))
      s = ax.imshow(model_out.T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.)) #, vmin=-1.0, vmax=1.0)
      fig.colorbar(s) 

  fig.savefig(os.path.join(ckpt_dir, 'BRS_validation_plot_epoch_%04d.png' % epoch))

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, clip_grad=opt.clip_grad,
               use_lbfgs=opt.use_lbfgs, validation_fn=val_fn, start_epoch=opt.checkpoint_toload,
               adjust_relative_grads=opt.adjust_relative_grads, args=opt.__dict__)
