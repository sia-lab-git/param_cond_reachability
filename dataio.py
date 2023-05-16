import csv
import glob
import math
import os

import matplotlib.colors as colors
import numpy as np
import scipy.io as spio
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import diff_operators

import utils
import pickle


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords

def to_uint8(x):
    return (255. * x).astype(np.uint8)

def to_numpy(x):
    return x.detach().cpu().numpy()

def gaussian(x, mu=[0, 0], sigma=1e-4, d=2):
    x = x.numpy()
    if isinstance(mu, torch.Tensor):
        mu = mu.numpy()

    q = -0.5 * ((x - mu) ** 2).sum(1)
    return torch.from_numpy(1 / np.sqrt(sigma ** d * (2 * np.pi) ** d) * np.exp(q / sigma)).float()


class ReachabilityAir3DSource(Dataset):
    def __init__(self, numpoints, 
        collisionR=0.25, velocity=0.6, omega_max=1.1, 
        pretrain=False, tMin=0.0, tMax=0.5, counter_start=0, counter_end=100e3, 
        pretrain_iters=2000, angle_alpha=1.0, num_src_samples=1000, seed=0):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        
        self.velocity = velocity
        self.omega_max = omega_max
        self.collisionR = collisionR

        self.alpha_angle = angle_alpha * math.pi

        self.num_states = 3

        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # set up the initial value function
        boundary_values = torch.norm(coords[:, 1:3], dim=1, keepdim=True) - self.collisionR

        # normalize the value function
        norm_to = 0.02
        mean = 0.25
        var = 0.5

        boundary_values = (boundary_values - mean)*norm_to/var
        
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}

class ReachabilityAir3Dp1DSource(Dataset):
    def __init__(self, numpoints, 
        collisionR=0.25, velocity=0.75, omega_max=3.0, 
        pretrain=False, tMin=0.0, tMax=0.5, counter_start=0, counter_end=100e3, 
        pretrain_iters=2000, angle_alpha=1.0, num_src_samples=1000, seed=0):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        
        self.velocity = velocity
        self.omega_max = omega_max
        self.collisionR = collisionR

        self.alpha_angle = angle_alpha * math.pi

        self.num_states = 4

        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # set up the initial value function
        boundary_values = torch.norm(coords[:, 1:(2+1)], dim=1, keepdim=True) - self.collisionR

        # normalize the value function
        norm_to = 0.02
        mean = 0.25
        var = 0.5

        boundary_values = (boundary_values - mean)*norm_to/var
        
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}

class ReachabilityMultiVehicleCollisionSourceNE(Dataset):
    def __init__(self, numpoints,
     collisionR=0.25, velocity=0.6, omega_max=1.1,
     pretrain=False, tMin=0.0, tMax=0.5, counter_start=0, counter_end=100e3, 
     numEvaders=1, pretrain_iters=2000, angle_alpha=1.0, time_alpha=1.0,
     num_src_samples=1000, diffModel=False):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        self.diffModel = diffModel
        
        self.velocity = velocity
        self.omega_max = omega_max
        self.collisionR = collisionR

        self.alpha_angle = angle_alpha * math.pi
        self.alpha_time = time_alpha

        self.norm_to = 0.02
        self.mean = 0.25
        self.var = 0.5

        self.numEvaders = numEvaders
        self.num_states_per_vehicle = 3
        self.num_states = self.num_states_per_vehicle * (numEvaders + 1)
        self.num_pos_states = 2 * (numEvaders + 1)
        # The state sequence will be as follows
        # [x-y position of vehicle 1, x-y position of vehicle 2, ...., x-y position of vehicle N, heading of vehicle 1, heading of vehicle 2, ...., heading of vehicle N]

        self.tMin = tMin
        self.tMax = tMax

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

    def __len__(self):
        return 1

    def compute_lx(self,coords):# set up the initial value function
        # Collision cost between the pursuer0 and the evader1
        boundary_values = torch.norm(coords[:, 1:3] - coords[:, 3:5], dim=1, keepdim=True) - self.collisionR
        # Collision cost between the pursuer0 and the evader2
        boundary_values_current = torch.norm(coords[:, 1:3] - coords[:, 5:7], dim=1, keepdim=True) - self.collisionR
        boundary_values = torch.min(boundary_values, boundary_values_current)
        # Collision cost between the evader1 and evader2        
        boundary_values_current = torch.norm(coords[:, 3:5] - coords[:, 5:7], dim=1, keepdim=True) - self.collisionR
        boundary_values = torch.min(boundary_values, boundary_values_current)

        return boundary_values 

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

        if self.pretrain:
            # only sample in time around the initial condition
            # time = torch.zeros(self.numpoints, 1).uniform_(start_time - 0.001, start_time + 0.001)
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = tMin and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
        
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        if self.diffModel:
            #coords_var = torch.tensor(coords.clone(), requires_grad=True)
            coords_var = coords.clone().detach().requires_grad_(True)
            boundary_values = self.compute_lx(coords_var)
            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
            # Compute the gradients of the value function
            lx_grads = diff_operators.gradient(boundary_values, coords_var)[..., 1:]
        else:
            boundary_values = self.compute_lx(coords)
            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
        
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        if self.diffModel:
            return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask, 'lx_grads': lx_grads}
        else:
            return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}
            
class ReachabilityMultiVehicleCollisionBeta(Dataset):
    def __init__(self, numpoints, velocity,
                  omega_max, pretrain, tMax, tMin,
                  counter_start, counter_end, 
                  numEvaders, pretrain_iters, 
                  angle_alpha, time_alpha, 
                  num_src_samples,diffModel):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        self.diffModel = diffModel
        
        self.velocity = velocity
        self.omega_max = omega_max
        #self.collisionR = collisionR

        self.alpha_angle = angle_alpha * math.pi
        self.alpha_time = time_alpha

        self.norm_to = 0.02
        self.mean = 0.25
        self.var = 0.5

        self.numEvaders = numEvaders
        self.num_states_per_vehicle = 3
        self.num_betas = 3
        self.num_states = 12
        self.num_pos_states = 2 * (numEvaders + 1)
        # The state sequence will be as follows
        # [x-y position of vehicle 1, x-y position of vehicle 2, ...., x-y position of vehicle N, heading of vehicle 1, heading of vehicle 2, ...., heading of vehicle N]

        self.tMin = tMin
        self.tMax = tMax

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

    def __len__(self):
        return 1

    def compute_lx(self,coords):# set up the initial value function
        #size uncertain parameters
        b01 = coords[:, 10:11]
        b02 = coords[:, 11:12]
        b12 = coords[:, 12:13]

        # import ipdb; ipdb.set_trace()
        #collision radius R(b_)=m*b_+c
        m=3/16
        c=5/16
        r01=m*b01+c
        r02=m*b02+c
        r12=m*b12+c
                 
        # Collision cost between the pursuer0 and the evader1
        boundary_values = torch.norm(coords[:, 1:3] - coords[:, 3:5], dim=1, keepdim=True) - r01
        # Collision cost between the pursuer0 and the evader2
        boundary_values_current = torch.norm(coords[:, 1:3] - coords[:, 5:7], dim=1, keepdim=True) - r02
        boundary_values = torch.min(boundary_values, boundary_values_current)
        # Collision cost between the evader1 and evader2        
        boundary_values_current = torch.norm(coords[:, 3:5] - coords[:, 5:7], dim=1, keepdim=True) - r12
        boundary_values = torch.min(boundary_values, boundary_values_current)

        return boundary_values 

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

        if self.pretrain:
            # only sample in time around the initial condition
            # time = torch.zeros(self.numpoints, 1).uniform_(start_time - 0.001, start_time + 0.001)
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = tMin and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
        
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        if self.diffModel:
            #coords_var = torch.tensor(coords.clone(), requires_grad=True)
            coords_var = coords.clone().detach().requires_grad_(True)
            boundary_values = self.compute_lx(coords_var)
            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
            # Compute the gradients of the value function
            # import ipdb; ipdb.set_trace()
            lx_grads = diff_operators.gradient(boundary_values, coords_var)[..., 1:]
        else:
            boundary_values = self.compute_lx(coords)
            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
        
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        if self.diffModel:
            return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask, 'lx_grads': lx_grads}
        else:
            return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}

class ReachabilityDrone2DSource(Dataset):
    def __init__(self, numpoints, 
        collisionR=0.2, velocity=0.1, disturbance_mag=0.1, theta_range= math.pi/6,
        pretrain=False, tMin=0.0, tMax=3.0, counter_start=0, counter_end=90000, 
        pretrain_iters=10000, position_alpha=1.0, num_src_samples=10000, seed=0):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        
        self.velocity = velocity
        self.collisionR = collisionR
        self.disturbance_mag = disturbance_mag
        self.theta_range = theta_range

        self.position_alpha = position_alpha

        self.num_states = 2

        self.tMax = tMax
        self.tMin = tMin

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

        # Set the seed
        torch.manual_seed(seed)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)  #nx(#states) tensor filled with uniformly distributed values

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time
            # this currently assumes start_time = 0 and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1) #concatenate more columns

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # set up the initial value function
        boundary_values = torch.norm(coords[:, 1:(self.num_states+1)], dim=1, keepdim=True) - self.collisionR

        # normalize the value function (-1,1)x(-1,1) grid target radius .2
        norm_to = 0.02
        mean = 0.5
        var = 0.7

        boundary_values = (boundary_values - mean)*norm_to/var
        
        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}

class ReachabilityParameterConditionedSimpleRocketLandingSource(Dataset):
    def __init__(self, numpoints, pretrain=False, tMin=0.0, tMax=1.0, counter_start=0, counter_end=100e3, 
                 pretrain_iters=10000, num_src_samples=10000, num_target_samples=10000, diffModel=False,
                lxType='unit_normalized_max'):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.diffModel = diffModel
        self.lxType = lxType

        self.numpoints = numpoints

        self.num_states = 7

        # Normalization coeffs        
        self.alpha = {}
        self.beta = {}

        #time is between [0 1]
        self.alpha['time'] = tMax
        self.tMax = 1
        self.tMin = tMin

        # Define state alphas and betas so that all coordinates are from [-1, 1]. 
        # The conversion rule is state' = (state - beta)/alpha. state' is in [-1, 1].
        # [state sequence: y, z, th, psi, y_dot, z_dot, th_dot, psi_dot, m]
        self.alpha['y'] = 150.0 # y in [-150, 150]m
        self.alpha['z'] = 70.0 # z in [10, 150]m
        self.alpha['th'] = 1.2 * math.pi

        self.alpha['y_dot'] = 200.0 # in [-200, 200] range
        self.alpha['z_dot'] = 200.0 # in [-200, 200] range
        self.alpha['th_dot'] = 10.0 # in [-10, 10] range

        self.alpha['param'] = 20.0 # in [-20, 20]m range

        self.beta['y'] = 0.0
        self.beta['z'] = 80.0
        self.beta['th'] = 0.0

        self.beta['y_dot'] = 0.0
        self.beta['z_dot'] = 0.0
        self.beta['th_dot'] = 0.0

        self.beta['param'] = 0.0

        # Define system parameters
        self.dynSys = {}
        self.dynSys['alpha'] = 0.3
        self.dynSys['thrustMax'] = 250.0

        # distances
        self.dynSys['L'] = 10.0  #  Distance from COM of rocket to bottom of rocket  (Rocket length is 2*L)
        self.dynSys['r'] = 2.0  #  Radius of rocket

        # mass
        self.dynSys['m_nofuel'] = 25.0 # Mass without fuel
        self.dynSys['max_m_fuel'] = 6*self.dynSys['m_nofuel']  # Max fuel mass
        
        # inertia
        self.dynSys['J'] = 1/12*(self.dynSys['m_nofuel']+self.dynSys['max_m_fuel'])*(2*self.dynSys['L'])**2  #  Inertia of the rocket in body frame
        self.dynSys['JT'] = (83/320 + 1)*0.05*self.dynSys['m_nofuel']*self.dynSys['r']**2 #  Inertial of the gimballed thruster
        
        # other constants
        self.dynSys['g'] = 9.81 #  Acceleration due to gravity
        self.dynSys['gamma'] = 1000.0 # Output velocity of the gas

        # Max control parameters
        self.dynSys['max_tau'] = 10.0 # Max thrust vectoring gimbal torque
        self.dynSys['max_fT'] = 25.0 # Max fuel burn rate - with gamma=1000, this corresponds to 14g

        # Target parameters
        self.dynSys['max_y'] = 20.0 #  Max y (horizontal position) at landing - landing pad goes from -y to +y
        self.dynSys['max_z'] = self.dynSys['L']
        self.dynSys['max_th'] = math.pi/6.0 #  Max theta at landing (radians)
        self.dynSys['max_speed'] = 5.0 # Max speed at landing
        self.dynSys['max_dth'] = 1.0 # Max rotational speed at landing

        # Value function normalization parameters
        if self.lxType in ['reducedlx_max_normalized', 'reducedlx2_max_normalized']:
            self.norm_to = 0.02
            self.mean = 0.0
            self.var = 1.0
        else: 
            raise NotImplementedError

        self.N_src_samples = num_src_samples
        self.N_target_coords = num_target_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end

    def compute_lx(self, state_coords_normalized):
        # Compute the target boundary condition given the unnormalized state coordinates.
        state_coords_unnormalized = self.unnormalize_states(state_coords_normalized)

        if self.lxType == 'reducedlx_max_normalized':
            # Only target set in the yz direction
            # Target set position in y direction
            dist_y = torch.abs(state_coords_unnormalized[:, 0:1] - state_coords_unnormalized[:, 6:7]) - self.dynSys['max_y'] #[-20, 150] range

            # Target set position in z direction
            dist_z = state_coords_unnormalized[:, 1:2] - self.dynSys['L'] - self.dynSys['max_z']  #[-10, 130] range

            # First compute the l(x) as you normally would but then normalize it later.
            lx = torch.max(dist_y, dist_z)
            lx = torch.where((lx >= 0), lx/150.0, lx/10.0)
        elif self.lxType == 'reducedlx2_max_normalized':
            # Only target set in the yz direction
            # Target set position in y direction
            dist_y = torch.abs(state_coords_unnormalized[:, 0:1] - state_coords_unnormalized[:, 6:7]) - self.dynSys['max_y'] #[-20, 150] range

            # Target set position in z direction
            dist_z = state_coords_unnormalized[:, 1:2] - self.dynSys['L'] - self.dynSys['max_z']  #[-10, 130] range

            # Target set in velocity direction
            dist_v = torch.norm(state_coords_unnormalized[:, 3:5], dim=1, keepdim=True) - self.dynSys['max_speed'] #[-5, 200] range

            # First compute the l(x) as you normally would but then normalize it later.
            lx = torch.max(torch.max(dist_y, dist_z), dist_v)
            lx = torch.where((lx >= 0), lx/280.0, lx/5.0)
        else:
            raise NotImplementedError

        return lx

    def sample_inside_target_set(self, num_samples):
        # Sample coordinates that are inside the target set.
        target_coords = torch.zeros(num_samples, self.num_states).uniform_(-1, 1)

        if self.lxType == 'reducedlx_max_normalized':
            # Landing pad position
            pad_position = target_coords[:, 6] * self.alpha['param'] + self.beta['param']

            # y position should be between [pad_psoition-20, pad_psoition + 20]
            target_coords[:, 0] = 1.5 * self.dynSys['max_y'] * target_coords[:, 0] + pad_position
            target_coords[:, 0] = (target_coords[:, 0] - self.beta['y'])/ self.alpha['y'] 

            # z position should be between [10, 20]
            target_coords[:, 1] = 0.5 * 1.5 * self.dynSys['max_z'] * target_coords[:, 1] + 0.5*(2*self.dynSys['L'] + 1.5*self.dynSys['max_z'])
            target_coords[:, 1] = (target_coords[:, 1] - self.beta['z'])/ self.alpha['z']

        elif self.lxType == 'reducedlx2_max_normalized':
            # Landing pad position
            pad_position = target_coords[:, 6] * self.alpha['param'] + self.beta['param']

            # y position should be between [pad_psoition-20, pad_psoition + 20]
            target_coords[:, 0] = 1.5 * self.dynSys['max_y'] * target_coords[:, 0] + pad_position
            target_coords[:, 0] = (target_coords[:, 0] - self.beta['y'])/ self.alpha['y'] 

            # z position should be between [10, 20]
            target_coords[:, 1] = 0.5 * 1.5 * self.dynSys['max_z'] * target_coords[:, 1] + 0.5*(2*self.dynSys['L'] + 1.5*self.dynSys['max_z'])
            target_coords[:, 1] = (target_coords[:, 1] - self.beta['z'])/ self.alpha['z']

            # The maximum translatin speed should be 5.0
            length = 1.5 * self.dynSys['max_speed'] * torch.sqrt(0.5*target_coords[:, 3] + 0.5)
            angle = target_coords[:, 4] * math.pi
            target_coords[:, 3] = length * torch.cos(angle)
            target_coords[:, 4] = length * torch.sin(angle)
            target_coords[:, 3] = (target_coords[:, 3] - self.beta['y_dot'])/ self.alpha['y_dot']
            target_coords[:, 4] = (target_coords[:, 4] - self.beta['z_dot'])/ self.alpha['z_dot']
        else:
            raise NotImplementedError

        return target_coords


    def normalize_states(self, unnormalized_states):
        # Normalize the states given the normalized state coordinates
        state_coords_normalized = unnormalized_states * 1.0
        state_coords_normalized[..., 0] = (state_coords_normalized[..., 0] - self.beta['y'])/ self.alpha['y'] 
        state_coords_normalized[..., 1] = (state_coords_normalized[..., 1] - self.beta['z'])/ self.alpha['z']
        state_coords_normalized[..., 2] = (state_coords_normalized[..., 2] - self.beta['th'])/ self.alpha['th']
        state_coords_normalized[..., 3] = (state_coords_normalized[..., 3] - self.beta['y_dot'])/ self.alpha['y_dot']
        state_coords_normalized[..., 4] = (state_coords_normalized[..., 4] - self.beta['z_dot'])/ self.alpha['z_dot']
        state_coords_normalized[..., 5] = (state_coords_normalized[..., 5] - self.beta['th_dot'])/ self.alpha['th_dot']
        state_coords_normalized[..., 6] = (state_coords_normalized[..., 6] - self.beta['param'])/ self.alpha['param']
        return state_coords_normalized

    def unnormalize_states(self, normalized_states):
        # Unnormalize the states given the normalized state coordinates
        state_coords_unnormalized = normalized_states * 1.0
        state_coords_unnormalized[..., 0] = state_coords_unnormalized[..., 0] * self.alpha['y'] + self.beta['y']
        state_coords_unnormalized[..., 1] = state_coords_unnormalized[..., 1] * self.alpha['z'] + self.beta['z']
        state_coords_unnormalized[..., 2] = state_coords_unnormalized[..., 2] * self.alpha['th'] + self.beta['th']
        state_coords_unnormalized[..., 3] = state_coords_unnormalized[..., 3] * self.alpha['y_dot'] + self.beta['y_dot']
        state_coords_unnormalized[..., 4] = state_coords_unnormalized[..., 4] * self.alpha['z_dot'] + self.beta['z_dot']
        state_coords_unnormalized[..., 5] = state_coords_unnormalized[..., 5] * self.alpha['th_dot'] + self.beta['th_dot']
        state_coords_unnormalized[..., 6] = state_coords_unnormalized[..., 6] * self.alpha['param'] + self.beta['param']
        return state_coords_unnormalized

    def unnormalize_dVdX(self, normalized_dVdX):
        # Unnormalize dVdX given the normalized value function spatial gradient.
        alpha = self.alpha

        # Scale the costates appropriately.
        unnormalized_dVdX = normalized_dVdX * 1.0
        unnormalized_dVdX[..., 0] = unnormalized_dVdX[..., 0] / alpha['y']
        unnormalized_dVdX[..., 1] = unnormalized_dVdX[..., 1] / alpha['z']
        unnormalized_dVdX[..., 2] = unnormalized_dVdX[..., 2] / alpha['th']
        unnormalized_dVdX[..., 3] = unnormalized_dVdX[..., 3] / alpha['y_dot']
        unnormalized_dVdX[..., 4] = unnormalized_dVdX[..., 4] / alpha['z_dot']
        unnormalized_dVdX[..., 5] = unnormalized_dVdX[..., 5] / alpha['th_dot']
        unnormalized_dVdX[..., 6] = unnormalized_dVdX[..., 6] / alpha['param']
        return unnormalized_dVdX

    def compute_overall_ham(self, x, dudx, compute_xdot=False):
        # Scale the costates appropriately.
        dudx_unnormalized = self.unnormalize_dVdX(dudx)

        # Scale the states appropriately.
        x_unnormalized = self.unnormalize_states(x)

        ## Compute the Hamiltonian
        # Control Hamiltonian
        u1_coeff = dudx_unnormalized[..., 3] * torch.cos(x_unnormalized[..., 2]) + dudx_unnormalized[..., 4] * torch.sin(x_unnormalized[..., 2]) + self.dynSys['alpha'] * dudx_unnormalized[..., 5]
        u2_coeff = -dudx_unnormalized[..., 3] * torch.sin(x_unnormalized[..., 2]) + dudx_unnormalized[..., 4] * torch.cos(x_unnormalized[..., 2])
        ham_ctrl = -self.dynSys['thrustMax'] * torch.sqrt(u1_coeff * u1_coeff + u2_coeff * u2_coeff)

        # Constant Hamiltonian
        ham_constant = dudx_unnormalized[..., 0] * x_unnormalized[..., 3] + dudx_unnormalized[..., 1] * x_unnormalized[..., 4] + \
                      dudx_unnormalized[..., 2] * x_unnormalized[..., 5]  - dudx_unnormalized[..., 4] * self.dynSys['g']

        # Compute the Hamiltonian
        ham_vehicle = ham_ctrl + ham_constant

        temp_value = torch.abs(ham_vehicle).sum()
        if torch.isnan(temp_value):
            import ipdb; ipdb.set_trace()
            print('Need to debug this')
      
        return ham_vehicle        

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
            # Add some samples that are inside the target set
            coords[-self.N_target_coords:, 1:] = self.sample_inside_target_set(self.N_target_coords)
        else:
            # slowly grow time values from start time; this currently assumes t \in [tMin, tMax]
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

            # Add some samples that are inside the target set
            coords[-self.N_target_coords:, 1:] = self.sample_inside_target_set(self.N_target_coords)

        # Compute the initial value function
        if self.diffModel:
            coords_var = torch.tensor(coords.clone(), requires_grad=True)
            boundary_values = self.compute_lx(coords_var[..., 1:])
            
            # Normalize the value function
            # print('Min and max value before normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var

            # Compute the gradients of the value function wrt state
            lx_grads = diff_operators.gradient(boundary_values, coords_var)[..., 1:]
        else:
            boundary_values = self.compute_lx(coords[..., 1:])

            # Normalize the value function
            # print('Min and max value before normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
            boundary_values = (boundary_values - self.mean)*self.norm_to/self.var

        if self.pretrain:
            dirichlet_mask = torch.ones(coords.shape[0], 1) > 0
        else:
            # only enforce initial conditions around start_time
            dirichlet_mask = (coords[:, 0, None] == start_time)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.full_count:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        if self.diffModel:
            return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask, 'lx_grads': lx_grads}
        else:
            return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask}