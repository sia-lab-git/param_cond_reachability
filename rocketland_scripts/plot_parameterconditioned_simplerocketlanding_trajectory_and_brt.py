# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, training, loss_functions, modules, diff_operators


import torch
import numpy as np
import scipy
from scipy import linalg
import math
from torch.utils.data import DataLoader
import configargparse
import scipy.io as spio

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

# Scenarios to run for each experiment
xinit = np.array([30.0, 60.0, 0.0, 0.0, -10.0, 0.0])
# xinit = np.array([100.0, 25.0, 0.0, 0.0, 0.0, 0.0])

# tMax for the BRT computation
tMax_BRT = 1.0

# Time horizon for simulation
tMax = 1.0  # Absolute time coordinates
dt = 0.0025 # Absolute time coordinates

# Time vector
tau = np.arange(0., tMax, dt)
num_timesteps = np.shape(tau)[0]

# Load the dataset
dataset = dataio.ReachabilityParameterConditionedSimpleRocketLandingSource(numpoints=65000, pretrain=opt.pretrain, tMin=opt.tMin,
                                                                           tMax=opt.tMax, counter_start=opt.counter_start, counter_end=opt.counter_end,
                                                                           pretrain_iters=opt.pretrain_iters, num_src_samples=opt.num_src_samples,
                                                                          diffModel=opt.diffModel, num_target_samples=opt.num_target_samples, lxType=opt.lxType)

# Initialize the model
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

# Traj colors 
traj_colors = ['r', 'k', 'b']

# Create the directory to save the plots
directory = os.path.join(root_path, 'Traj_plots')
if not os.path.exists(directory):
    os.makedirs(directory)

def angle_normalize(x):
  return (((x + np.pi) % (2 * np.pi)) - np.pi)

def unnormalize_valfunc(valfunc):
  # Unnormalize the value function
  norm_to = dataset.norm_to
  mean = dataset.mean
  var = dataset.var
  return (valfunc*var/norm_to) + mean 

def propagate_state(state, control):
  state_next = state + dt*np.array([state[3], state[4], state[5], \
                                    control[0]*np.cos(state[2]) - control[1]*np.sin(state[2]), \
                                    control[0]*np.sin(state[2]) + control[1]*np.cos(state[2]) - dataset.dynSys['g'], \
                                    dataset.dynSys['alpha'] * control[0]])
  state_next[2] = angle_normalize(state_next[2])
  return state_next

def compute_brt_slice(coords):
  # Compute the brt slice in y-param space, given the values of other states.
  sidelen = 200
  mgrid_coords = dataio.get_mgrid(sidelen, dim=2)
  ts = torch.ones(mgrid_coords.shape[0], 1) * coords[0, 0]
  zs = torch.ones(mgrid_coords.shape[0], 1) * (coords[0, 2] - dataset.beta['z']) / dataset.alpha['z']
  th = torch.ones(mgrid_coords.shape[0], 1) * (coords[0, 3] - dataset.beta['th']) / dataset.alpha['th']
  y_dot = torch.ones(mgrid_coords.shape[0], 1) * (coords[0, 4] - dataset.beta['y_dot']) / dataset.alpha['y_dot']
  z_dot = torch.ones(mgrid_coords.shape[0], 1) * (coords[0, 5] - dataset.beta['z_dot']) / dataset.alpha['z_dot']
  th_dot = torch.ones(mgrid_coords.shape[0], 1) * (coords[0, 6] - dataset.beta['th_dot']) / dataset.alpha['th_dot']

  coords = torch.cat((ts, mgrid_coords[:, 0:1], zs, th, y_dot, z_dot, th_dot, mgrid_coords[:, 1:2]), dim=1)

  model_in = {'coords': coords.cuda()}
  model_out = model(model_in)['model_out']

  # Unnormalize the value function 
  model_out = (model_out*dataset.var/dataset.norm_to) + dataset.mean 

  # Detatch model ouput and reshape
  model_out = model_out.detach().cpu().numpy()

  if opt.diffModel:
    lx = dataset.compute_lx(coords[..., 1:])
    lx = lx.detach().cpu().numpy()
    model_out = model_out + lx - dataset.mean

  model_out = model_out.reshape((sidelen, sidelen))
  return model_out


# Create a moving pattern for the landing pad (hyperbolic tangent pattern)
T = tMax
tau_pad = np.arange(0., T, dt)
pad_movement = -20.0 * (np.tanh(7*(tau_pad - 0.5)) + 1) # Low-frequency component
pad_movement = pad_movement + 0.5 * np.sin(20*math.pi*tau_pad/T) # High-frequency component
landing_pad_pos = pad_movement

# Setup the state and control arrays
states_fixedpad = np.zeros((7, num_timesteps))
controls_fixedpad = np.zeros((2, num_timesteps-1))
states_movingpad = np.zeros((7, num_timesteps))
controls_movingpad = np.zeros((2, num_timesteps-1))

# Store the values
values_fixedpad = np.zeros((1, num_timesteps-1))
values_movingpad = np.zeros((1, num_timesteps-1))
lx_fixedpad = np.zeros((1, num_timesteps-1))
lx_movingpad = np.zeros((1, num_timesteps-1))

# Store the BRT slices
brt_fixedpad = []
brt_movingpad = []

##### Fixed Pad Case ##### 
# Initialize the actual trajectories
states_fixedpad[:6, 0] = xinit

# Start the trajectory iteration
for k in range(num_timesteps-1):

  # Setup the input vector
  coords = torch.ones(1, 8)
  if (tMax - tau[k]) > tMax_BRT:
    query_time = tMax_BRT
  else:
    query_time = tMax - tau[k]
  coords[:, 0] = coords[:, 0] * query_time
  coords[:, 1] = coords[:, 1] * (states_fixedpad[0, k] - dataset.beta['y']) / dataset.alpha['y']
  coords[:, 2] = coords[:, 2] * (states_fixedpad[1, k] - dataset.beta['z']) / dataset.alpha['z']
  coords[:, 3] = coords[:, 3] * (states_fixedpad[2, k] - dataset.beta['th']) / dataset.alpha['th']
  coords[:, 4] = coords[:, 4] * (states_fixedpad[3, k] - dataset.beta['y_dot']) / dataset.alpha['y_dot']
  coords[:, 5] = coords[:, 5] * (states_fixedpad[4, k] - dataset.beta['z_dot']) / dataset.alpha['z_dot']
  coords[:, 6] = coords[:, 6] * (states_fixedpad[5, k] - dataset.beta['th_dot']) / dataset.alpha['th_dot']
  coords[:, 7] = coords[:, 7] * (states_fixedpad[6, k] - dataset.beta['param']) / dataset.alpha['param']

  coords_unnormalized = torch.tensor(states_fixedpad[:, k:k+1]).cuda()

  model_in = {'coords': coords.cuda()}
  model_out = model(model_in)

  # Compute the spatial derivative
  du, status = diff_operators.jacobian(model_out['model_out'], model_out['model_in'])
  dudx_normalized = du[0, :, 0, 1:].detach().cpu().numpy()

  # Store the values
  value = model_out['model_out'].detach().cpu().numpy()
  value = unnormalize_valfunc(value)[0, 0]

  # Account for the diff model
  if opt.diffModel:
    coords_var = torch.tensor(coords.clone(), requires_grad=True)
    lx = dataset.compute_lx(coords_var[:, 1:])
    lx_normalized = (lx - dataset.mean)*dataset.norm_to/dataset.var
    lx_grads = diff_operators.gradient(lx_normalized, coords_var)[..., 1:]

    # Add l(x) to the value function
    lx = lx.detach().cpu().numpy()
    value = value + lx[0, 0] - dataset.mean

    # Add l(x) gradients to the dudx
    # import ipdb; ipdb.set_trace()
    lx_grads = lx_grads.detach().cpu().numpy()
    dudx_normalized = dudx_normalized + lx_grads

  # Compute the true l(x) value
  coords_true = torch.ones(1, 8)
  coords_true[:, :6] = coords[:, :6]
  coords_true[:, 7] = coords_true[:, 7] * (landing_pad_pos[k] - dataset.beta['param']) / dataset.alpha['param']
  lx_true = dataset.compute_lx(coords_true[:, 1:]).detach().cpu().numpy()

  values_fixedpad[0, k] = value
  lx_fixedpad[0, k] = lx_true[0, 0]
  dudx_unnormalized = dataset.unnormalize_dVdX(dudx_normalized)
  dudx_unnormalized = dudx_unnormalized[0]

  print('Time step %i of %i. Value: %0.2f. l(x): %0.2f' %(k+1, num_timesteps, value, lx_true[0, 0]))

  # Compute the BRT slice for the current time step
  brt_fixedpad.append(compute_brt_slice(coords_true))

  ## Propagate the state
  # Optimal control computation
  u1_coeff = dudx_unnormalized[3] * torch.cos(coords_unnormalized[2, 0]) + dudx_unnormalized[4] * torch.sin(coords_unnormalized[2, 0]) + dataset.dynSys['alpha'] * dudx_unnormalized[5]
  u2_coeff = -dudx_unnormalized[3] * torch.sin(coords_unnormalized[2, 0]) + dudx_unnormalized[4] * torch.cos(coords_unnormalized[2, 0])
  opt_angle = torch.atan2(u2_coeff, u1_coeff) + math.pi
  opt_angle = opt_angle.detach().cpu().numpy()
  controls_fixedpad[:, k] = np.array([dataset.dynSys['thrustMax'] * np.cos(opt_angle), dataset.dynSys['thrustMax'] * np.sin(opt_angle)]) 

  # Dynamics propagation
  states_fixedpad[:6, k+1] = propagate_state(states_fixedpad[:6, k], controls_fixedpad[:, k])


##### Moving Pad Case ##### 
# Initialize the actual trajectories
states_movingpad[:6, 0] = xinit
states_movingpad[6, :] = landing_pad_pos

# Start the trajectory iteration
for k in range(num_timesteps-1):

  # Setup the input vector
  coords = torch.ones(1, 8)
  if (tMax - tau[k]) > tMax_BRT:
    query_time = tMax_BRT
  else:
    # import ipdb; ipdb.set_trace()
    query_time = tMax - tau[k]
  coords[:, 0] = coords[:, 0] * query_time
  coords[:, 1] = coords[:, 1] * (states_movingpad[0, k] - dataset.beta['y']) / dataset.alpha['y']
  coords[:, 2] = coords[:, 2] * (states_movingpad[1, k] - dataset.beta['z']) / dataset.alpha['z']
  coords[:, 3] = coords[:, 3] * (states_movingpad[2, k] - dataset.beta['th']) / dataset.alpha['th']
  coords[:, 4] = coords[:, 4] * (states_movingpad[3, k] - dataset.beta['y_dot']) / dataset.alpha['y_dot']
  coords[:, 5] = coords[:, 5] * (states_movingpad[4, k] - dataset.beta['z_dot']) / dataset.alpha['z_dot']
  coords[:, 6] = coords[:, 6] * (states_movingpad[5, k] - dataset.beta['th_dot']) / dataset.alpha['th_dot']
  coords[:, 7] = coords[:, 7] * (states_movingpad[6, k] - dataset.beta['param']) / dataset.alpha['param']

  coords_unnormalized = torch.tensor(states_movingpad[:, k:k+1]).cuda()

  model_in = {'coords': coords.cuda()}
  model_out = model(model_in)

  # Compute the spatial derivative
  du, status = diff_operators.jacobian(model_out['model_out'], model_out['model_in'])
  dudx_normalized = du[0, :, 0, 1:].detach().cpu().numpy()

  # Store the values
  value = model_out['model_out'].detach().cpu().numpy()
  value = unnormalize_valfunc(value)[0, 0]

  # Account for the diff model
  if opt.diffModel:
    coords_var = torch.tensor(coords.clone(), requires_grad=True)
    lx = dataset.compute_lx(coords_var[:, 1:])
    lx_normalized = (lx - dataset.mean)*dataset.norm_to/dataset.var
    lx_grads = diff_operators.gradient(lx_normalized, coords_var)[..., 1:]

    # Add l(x) to the value function
    lx = lx.detach().cpu().numpy()
    value = value + lx[0, 0] - dataset.mean

    # Add l(x) gradients to the dudx
    lx_grads = lx_grads.detach().cpu().numpy()
    dudx_normalized = dudx_normalized + lx_grads

  values_movingpad[0, k] = value
  lx_movingpad[0, k] = lx[0, 0]
  dudx_unnormalized = dataset.unnormalize_dVdX(dudx_normalized)
  dudx_unnormalized = dudx_unnormalized[0]

  print('Time step %i of %i. Value: %0.2f. l(x): %0.2f' %(k+1, num_timesteps, value, lx[0, 0]))

  # Compute the BRT slice for the current time step
  brt_movingpad.append(compute_brt_slice(coords))

  ## Propagate the state
  # Optimal control computation
  u1_coeff = dudx_unnormalized[3] * torch.cos(coords_unnormalized[2, 0]) + dudx_unnormalized[4] * torch.sin(coords_unnormalized[2, 0]) + dataset.dynSys['alpha'] * dudx_unnormalized[5]
  u2_coeff = -dudx_unnormalized[3] * torch.sin(coords_unnormalized[2, 0]) + dudx_unnormalized[4] * torch.cos(coords_unnormalized[2, 0])
  opt_angle = torch.atan2(u2_coeff, u1_coeff) + math.pi
  opt_angle = opt_angle.detach().cpu().numpy()
  controls_movingpad[:, k] = np.array([dataset.dynSys['thrustMax'] * np.cos(opt_angle), dataset.dynSys['thrustMax'] * np.sin(opt_angle)]) 

  # Dynamics propagation
  states_movingpad[:6, k+1] = propagate_state(states_movingpad[:6, k], controls_movingpad[:, k])

# Stack the BRT slices
brt_fixedpad = np.stack(brt_fixedpad, axis=2)
brt_movingpad = np.stack(brt_movingpad, axis=2)

##### Compute the BRT slices for visualization #####
sidelen = 200
points = sidelen * sidelen
mgrid_coords = dataio.get_mgrid(sidelen, dim=2)
ts = torch.ones(mgrid_coords.shape[0], 1) * tMax_BRT
ys = torch.ones(mgrid_coords.shape[0], 1) * (xinit[0] - dataset.beta['y']) / dataset.alpha['y']
zs = torch.ones(mgrid_coords.shape[0], 1) * (xinit[1] - dataset.beta['z']) / dataset.alpha['z']
th = torch.ones(mgrid_coords.shape[0], 1) * (xinit[2] - dataset.beta['th']) / dataset.alpha['th']
y_dot = torch.ones(mgrid_coords.shape[0], 1) * (xinit[3] - dataset.beta['y_dot']) / dataset.alpha['y_dot']
z_dot = torch.ones(mgrid_coords.shape[0], 1) * (xinit[4] - dataset.beta['z_dot']) / dataset.alpha['z_dot']
th_dot = torch.ones(mgrid_coords.shape[0], 1) * (xinit[5] - dataset.beta['th_dot']) / dataset.alpha['th_dot']
params = torch.ones(mgrid_coords.shape[0], 1) * (0 - dataset.beta['param']) / dataset.alpha['param']

coords_yparam = torch.cat((ts, mgrid_coords[:, 0:1], zs, th, y_dot, z_dot, th_dot, mgrid_coords[:, 1:2]), dim=1)
coords_zparam = torch.cat((ts, ys, mgrid_coords[:, 0:1], th, y_dot, z_dot, th_dot, mgrid_coords[:, 1:2]), dim=1)
coords_yz = torch.cat((ts, mgrid_coords, th, y_dot, z_dot, th_dot, params), dim=1)
coords = torch.cat((coords_yparam, coords_zparam, coords_yz), dim=0)

model_in = {'coords': coords.cuda()}
model_out = model(model_in)['model_out']

# Unnormalize the value function 
model_out = (model_out*dataset.var/dataset.norm_to) + dataset.mean 

# Detatch model ouput and reshape
model_out = model_out.detach().cpu().numpy()

if opt.diffModel:
  lx = dataset.compute_lx(coords[..., 1:])
  lx = lx.detach().cpu().numpy()
  model_out = model_out + lx - dataset.mean

# Plot the zero level sets
model_out = (model_out <= -0.001)*1. 

model_out_yparam = model_out[:points].reshape((sidelen, sidelen))
model_out_zparam = model_out[points:2*points].reshape((sidelen, sidelen))
model_out_yz = model_out[2*points:].reshape((sidelen, sidelen))

##### Plot results #####
# Create a figure
num_figures = 10
fig = plt.figure(figsize=(5, 5*num_figures))

axis_num = 1

# Plot the landing pad position
ax = fig.add_subplot(num_figures, 1, axis_num)
s1 = ax.plot(tau, landing_pad_pos, 'r')
ax.set_title('Landing pad position')
axis_num += 1

# Plot the l(x) values over time
ax = fig.add_subplot(num_figures, 1, axis_num)
s1 = ax.plot(tau[-num_timesteps:-1], lx_fixedpad[0, :], 'r', label="FixedPad")
s2 = ax.plot(tau[-num_timesteps:-1], lx_movingpad[0, :], 'r--', label="MovingPad")
# ax.set_ylim(-0.2, 0.2)
plt.legend(loc="upper left")
ax.set_title('l(x) values')
axis_num += 1

# Plot the xy trajectory
ax = fig.add_subplot(num_figures, 1, axis_num)
s1 = ax.plot(states_fixedpad[0, :], states_fixedpad[1, :], 'r-', label="FixedPad")
s2 = ax.plot(states_movingpad[0, :], states_movingpad[1, :], 'r--', label="MovingPad")
plt.legend(loc="upper left")
ax.set_title('YZ trajectory')
axis_num += 1

# Plot the theta trajectory
ax = fig.add_subplot(num_figures, 1, axis_num)
s1 = ax.plot(tau, states_fixedpad[2, :], 'r-', label="FixedPad")
s2 = ax.plot(tau, states_movingpad[2, :], 'r--', label="MovingPad")
plt.legend(loc="upper left")
ax.set_title('Theta trajectory')
axis_num += 1

# Plot the velocity profiles
ax = fig.add_subplot(num_figures, 1, axis_num)
s1 = ax.plot(tau, states_fixedpad[3, :], 'r', label="FixedPad")
s3 = ax.plot(tau, states_movingpad[3, :], 'r--', label="MovingPad")
plt.legend(loc="upper left")
ax.set_title('Y velocity')
axis_num += 1

ax = fig.add_subplot(num_figures, 1, axis_num)
s2 = ax.plot(tau, states_fixedpad[4, :], 'b', label="zd FixedPad")
s4 = ax.plot(tau, states_movingpad[4, :], 'b--', label="zd MovingPad")
plt.legend(loc="upper left")
ax.set_title('Z velocity')
axis_num += 1

ax = fig.add_subplot(num_figures, 1, axis_num)
s2 = ax.plot(tau, states_fixedpad[5, :], 'k', label="thd FixedPad")
s4 = ax.plot(tau, states_movingpad[5, :], 'k--', label="thd MovingPad")
plt.legend(loc="upper left")
ax.set_title('Theta velocity')
axis_num += 1

# Plot the control u1
ax = fig.add_subplot(num_figures, 1, axis_num)
s1 = ax.plot(tau[-num_timesteps:-1], controls_fixedpad[0, :], 'r', label="FixedPad")
s2 = ax.plot(tau[-num_timesteps:-1], controls_movingpad[0, :], 'r--', label="MovingPad")
plt.legend(loc="upper left")
ax.set_title('U1 control')
axis_num += 1

# Plot the control u2
ax = fig.add_subplot(num_figures, 1, axis_num)
s1 = ax.plot(tau[-num_timesteps:-1], controls_fixedpad[1, :], 'b', label="FixedPad")
s2 = ax.plot(tau[-num_timesteps:-1], controls_movingpad[1, :], 'b--', label="MovingPad")
plt.legend(loc="upper left")
ax.set_title('U2 control')
axis_num += 1

# Plot the BRT slices 
yMin = dataset.beta['y'] - dataset.alpha['y']
yMax = dataset.beta['y'] + dataset.alpha['y']
zMin = dataset.beta['z'] - dataset.alpha['z']
zMax = dataset.beta['z'] + dataset.alpha['z']
paramMin = dataset.beta['param'] - dataset.alpha['param']
paramMax = dataset.beta['param'] + dataset.alpha['param']

ax = fig.add_subplot(num_figures, 1, axis_num)
s1 = ax.imshow(model_out_yz.T, cmap='bwr', origin='lower', extent=(yMin, yMax, zMin, zMax), aspect='auto')
s2 = ax.plot(states_fixedpad[0, :], states_fixedpad[1, :], 'k-', label="FixedPad")
s3 = ax.plot(states_movingpad[0, :], states_movingpad[1, :], 'k--', label="MovingPad")
plt.legend(loc="upper left")
ax.set_title('YZ Trajectory')
axis_num += 1


fig.savefig(os.path.join(directory, 'Traj_plot_fixed_landing_pad.png'))

# Save everything to matlab
filename = os.path.join(directory, 'traj_and_brt_data_deepreach.mat')
datadict = {}
datadict['tau'] = tau
datadict['landing_pad_pos'] = landing_pad_pos

datadict['brt_fixedpad'] = brt_fixedpad
datadict['brt_movingpad'] = brt_movingpad
datadict['lx_fixedpad'] = lx_fixedpad
datadict['lx_movingpad'] = lx_movingpad
datadict['values_fixedpad'] = values_fixedpad
datadict['values_movingpad'] = values_movingpad

datadict['states_fixedpad'] = states_fixedpad
datadict['states_movingpad'] = states_movingpad
datadict['controls_fixedpad'] = controls_fixedpad
datadict['controls_movingpad'] = controls_movingpad

scipy.io.savemat(filename, datadict)
