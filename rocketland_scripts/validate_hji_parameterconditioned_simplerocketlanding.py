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
from argparse import ArgumentParser
import json

model_name = 'rocket1'
checkpoint_toload=199000

# Load the dataset
opt_path = os.path.join('./logs',model_name, 'commandline_args.txt')
parser = ArgumentParser()
opt = parser.parse_args()
with open(opt_path, 'r') as f:
    opt.__dict__ = json.load(f)

dataset = dataio.ReachabilityParameterConditionedSimpleRocketLandingSource(numpoints=65000, pretrain=opt.pretrain, tMin=opt.tMin,
                                                                           tMax=opt.tMax, counter_start=opt.counter_start, counter_end=opt.counter_end,
                                                                           pretrain_iters=opt.pretrain_iters, num_src_samples=opt.num_src_samples,
                                                                          diffModel=opt.diffModel, num_target_samples=opt.num_target_samples, lxType=opt.lxType)

dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

model = modules.SingleBVPNet(in_features=8, out_features=1, type=opt.model, mode=opt.mode,
                             final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
model.cuda()

root_path = os.path.join(opt.logging_root, opt.experiment_name)
ckpt_dir = os.path.join(root_path, 'checkpoints')
ckpt_path = os.path.join(ckpt_dir, 'model_epoch_%04d.pth' % checkpoint_toload)
checkpoint = torch.load(ckpt_path)
try:
  model_weights = checkpoint['model']
except:
  model_weights = checkpoint
model.load_state_dict(model_weights)
model.eval()

# Normalization coefficients
alpha = dataset.alpha
beta = dataset.beta

# Time values at which the function needs to be plotted
times = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])*opt.tMax
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
  time_coords = torch.ones(mgrid_coords.shape[0], 1) * times[i]/alpha['time']

  for j in range(num_slices):
    th = torch.ones(mgrid_coords.shape[0], 1) * (slices_toplot[j]['th'] - beta['th']) / alpha['th']
    y_dot = torch.ones(mgrid_coords.shape[0], 1) * (slices_toplot[j]['y_dot'] - beta['y_dot']) / alpha['y_dot']
    z_dot = torch.ones(mgrid_coords.shape[0], 1) * (slices_toplot[j]['z_dot'] - beta['z_dot']) / alpha['z_dot']
    th_dot = torch.ones(mgrid_coords.shape[0], 1) * (slices_toplot[j]['th_dot'] - beta['th_dot']) / alpha['th_dot']
    param_values = torch.ones(mgrid_coords.shape[0], 1) * (slices_toplot[j]['param'] - beta['param']) / alpha['param']
    coords = torch.cat((time_coords, mgrid_coords, th, y_dot, z_dot, th_dot, param_values), dim=1) 

    model_in = {'coords': coords.cuda()}
    model_out = model(model_in)['model_out']

    # Unnormalize the value function 
    model_out = (model_out*dataset.var/dataset.norm_to) + dataset.mean 

    # Detatch model ouput and reshape
    model_out = model_out.detach().cpu().numpy()
    model_out = model_out.reshape((sidelen, sidelen))

    if opt.diffModel:
      lx = dataset.compute_lx(coords[..., 1:])
      lx = lx.detach().cpu().numpy()
      lx = lx.reshape((sidelen, sidelen))
      model_out = model_out + lx

    # Plot the zero level sets
    # model_out = (model_out <= 0.001)*1.

    # import ipdb; ipdb.set_trace()

    # Plot the actual data
    ax = fig.add_subplot(num_times, num_slices, (j+1) + i*num_slices)
    ax.set_title('t = %0.2f, th = %0.2f, zd = %0.2f, p = %0.2f' % (times[i], slices_toplot[j]['th'], slices_toplot[j]['z_dot'], slices_toplot[j]['param']))
    s = ax.imshow(model_out.T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
    fig.colorbar(s) 

directory = os.path.join(root_path, 'validation')
if not os.path.exists(directory):
    os.makedirs(directory)

fig.savefig(os.path.join(directory, 'BRS_validation_plot_epoch_%04d.png' % opt.checkpoint_toload))
