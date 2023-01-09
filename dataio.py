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


#experimental models
class ReachabilityDrone3Dp1DSource(Dataset):
    def __init__(self, numpoints, velocity=8.0,
        omega_max=80.0, pretrain=False, tMin=0.0,
        tMax=1.0, counter_start=0, counter_end=100e3, 
        pretrain_iters=2000, angle_alpha=1.0, num_src_samples=1000, seed=0):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        
        self.velocity = velocity
        self.omega_max = omega_max       

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

    def compute_lx(self, state_coords): #target set hardcoded to circle centered in (15,25) with r=4
        # Compute the target boundary condition given the normalized state coordinates.
        goal_tensor = torch.tensor([-.25, .25]).type(torch.FloatTensor)[None]
        dist = torch.norm(state_coords[:, 0:2] - goal_tensor, dim=1, keepdim=True) - 0.2
        return dist

    def compute_gx(self, state_coords):
        # Compute the obstacle boundary condition given the state coordinates. Negative inside the obstacle positive outside.
        # signed distance except on diag
        dist_obs1 = torch.max(torch.abs(state_coords[:, 0] - (-.75)) - .25, torch.abs(state_coords[:, 1] - (-.25)) - .25)
        dist_obs2 = torch.max(torch.abs(state_coords[:, 0] - (-.15)) - .15, torch.abs(state_coords[:, 1] - (-.25)) - .25)
        dist = torch.min(dist_obs1,dist_obs2)
        dist = torch.unsqueeze(dist, 1)
        return dist

    def compute_IC(self, state_coords):
        lx = self.compute_lx(state_coords)
        gx_factor = 6.0
        gx = self.compute_gx(state_coords) * gx_factor
        hx = -gx       
        vx = torch.max(lx, hx)
        return lx, hx, vx    


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
        lx, hx, boundary_values = self.compute_IC(coords[:, 1:])
        # normalize the value function
        norm_to = 0.02
        mean = 0.7
        var = 0.9
        
        boundary_values = (boundary_values - mean)*norm_to/var
        lx = (lx - mean)*norm_to/var
        hx = (hx - mean)*norm_to/var
                
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

        return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask,'lx': lx, 'hx': hx}

class ReachabilityDrone3DSource(Dataset):
    def __init__(self, numpoints, velocity=2.0, dbar=0.8,
        omega_max=1.0, pretrain=False, tMin=0.0,
        tMax=1.0, state_setting='big',
        counter_start=0, counter_end=100e3,
        pretrain_iters=2000, num_src_samples=1000,
        seed=0):
     
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        
        self.velocity = velocity
        self.omega_max = omega_max
        self.dbar = dbar

        self.num_states = 3
      
        self.tMin = tMin
        self.tMax = tMax

        self.alpha = {}
        self.beta = {}

        if state_setting == 'big':
            self.alpha['x'] = 20.0
            self.alpha['y'] = 20.0
            self.alpha['th'] = 1.2*math.pi           
            self.alpha['time'] = 80.0

            self.beta['x'] = 20.0
            self.beta['y'] = 20.0
            self.beta['th'] = 0.0

            # Target positions
            self.goalX = np.array([15.0])
            self.goalY = np.array([25.0])
            self.L = 4.0            
        else:
            raise NotImplementedError

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

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
            # this currently assumes start_time = tMin and max time value is tMax
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
        
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # set up the initial value function
        state_coords_unnormalized = coords[..., 1:] * 1.0
        state_coords_unnormalized[:, 0] = state_coords_unnormalized[:, 0] * self.alpha['x'] + self.beta['x']
        state_coords_unnormalized[:, 1] = state_coords_unnormalized[:, 1] * self.alpha['y'] + self.beta['y']
        state_coords_unnormalized[:, 2] = state_coords_unnormalized[:, 2] * self.alpha['th'] + self.beta['th']

        goal_tensor = torch.tensor([self.goalX[0], self.goalY[0]]).type(torch.FloatTensor)[None]
        boundary_values = torch.norm(state_coords_unnormalized[:, 0:2] - goal_tensor, dim=1, keepdim=True) - self.L
        # goal_tensor = torch.tensor([-.25, .25]).type(torch.FloatTensor)[None]
        # boundary_values = torch.norm(coords[:, 1:3] - goal_tensor, dim=1, keepdim=True) - 0.2
        
        # normalize the value function
        norm_to = 0.02
        mean = 0.7 * self.alpha['x']
        var = 0.9 * self.alpha['x']
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

class ReachabilityNarrow8DSource(Dataset):
    def __init__(self, numpoints, pretrain, tMin, tMax, mu, counter_start, counter_end, pretrain_iters, norm_scheme, clip_value_gradients,
                 gx_factor, speed_setting, sampling_bias_ratio, env_setting, ham_version, target_setting, collision_setting, 
                 curriculum_version, HJIVI_smoothing_setting, smoothing_exponent, num_src_samples):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        
        self.num_vehicles = 2
        self.num_states = 4 * self.num_vehicles

        self.tMax = tMax
        self.tMin = tMin
        self.mu = mu

        # Define state alphas and betas so that all coordinates are from [-1, 1]. 
        # The conversion rule is state' = (state - beta)/alpha. state' is in [-1, 1].
        # [state sequence: x_Ri, y_Ri, th_Ri, v_Ri, phi_Ri, ...]. Ri is the ith vehicle.
        self.alpha = {}
        self.beta = {}
        
        if speed_setting == 'medium_v2':
            self.alpha['x'] = 8.0
            self.alpha['y'] = 3.8
            self.alpha['th'] = 1.2*math.pi
            self.alpha['v'] = 4.0
            #self.alpha['phi'] = 1.2*0.3*math.pi
            self.alpha['time'] = 10.0/self.tMax

            self.beta['x'] = 0.0
            self.beta['y'] = 0.0
            self.beta['th'] = 0.0
            self.beta['v'] = 3.0
            self.beta['phi'] = 0.0

            # Target positions
            self.goalX = np.array([6.0, -6.0])
            self.goalY = np.array([-1.4, 1.4])

            # State bounds
            self.vMin = 0.001
            self.vMax = 6.50
            self.phiMin = -0.3*math.pi + 0.001
            self.phiMax = 0.3*math.pi - 0.001

            # Control bounds
            self.aMin = -4.0
            self.aMax = 2.0
            self.psiMin = -3.0*math.pi
            self.psiMax = 3.0*math.pi
        else:
            raise NotImplementedError

        # How to weigh the obstacles
        self.gx_factor = gx_factor

        if env_setting == 'v2':
            # Vehicle diameter/length
            self.L = 2.0

            # Lower and upper curb positions (in the y direction)
            self.curb_positions = np.array([-2.8, 2.8])

            # Stranded car position
            self.stranded_car_pos = np.array([0.0, -1.8])
        else:
            raise NotImplementedError

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

        self.normalization_scheme = norm_scheme
        self.sampling_bias_ratio = sampling_bias_ratio
        self.clip_value_gradients = clip_value_gradients

        self.ham_version = ham_version
        self.target_setting = target_setting
        self.collision_setting = collision_setting
        self.curriculum_version = curriculum_version
        self.HJIVI_smoothing_setting = HJIVI_smoothing_setting
        self.smoothing_exponent = smoothing_exponent
        if self.normalization_scheme == 'hack1':
            self.norm_to = 0.02
            self.mean = 0.25 * self.alpha['x']
            self.var = 0.5 * self.alpha['x']
        else:
            raise NotImplementedError

    def compute_lx(self, state_coords_unnormalized):
        # Compute the target boundary condition given the unnormalized state coordinates.
        if self.target_setting in ['v1']:
            # Vehicle 1
            goal_tensor_R1 = torch.tensor([self.goalX[0], self.goalY[0]]).type(torch.FloatTensor)[None]
            dist_R1 = torch.norm(state_coords_unnormalized[:, 0:2] - goal_tensor_R1, dim=1, keepdim=True) - self.L
            # Vehicle 2
            goal_tensor_R2 = torch.tensor([self.goalX[1], self.goalY[1]]).type(torch.FloatTensor)[None]
            dist_R2 = torch.norm(state_coords_unnormalized[:, 4:6] - goal_tensor_R2, dim=1, keepdim=True) - self.L
            if self.target_setting == 'v1':
                return torch.max(dist_R1, dist_R2)
        else:
            raise NotImplementedError

    def compute_gx(self, state_coords_unnormalized):
        # Compute the obstacle boundary condition given the unnormalized state coordinates. Negative inside the obstacle positive outside.
        # Distance from the lower curb
        dist_lc_R1 = state_coords_unnormalized[:, 1:2] - self.curb_positions[0] - 0.5*self.L
        dist_lc_R2 = state_coords_unnormalized[:, 5:6] - self.curb_positions[0] - 0.5*self.L
        dist_lc = torch.min(dist_lc_R1, dist_lc_R2)
        # dist_lc = torch.min(state_coords_unnormalized[:, 1:2] - self.curb_positions[0] - 0.5*self.L, state_coords_unnormalized[:, 6:7] - self.curb_positions[0] - 0.5*self.L)
        
        # Distance from the upper curb
        dist_uc_R1 = self.curb_positions[1] - state_coords_unnormalized[:, 1:2] - 0.5*self.L
        dist_uc_R2 = self.curb_positions[1] - state_coords_unnormalized[:, 5:6] - 0.5*self.L
        dist_uc = torch.min(dist_uc_R1, dist_uc_R2)
        # dist_uc = torch.min(self.curb_positions[1] - state_coords_unnormalized[:, 1:2] - 0.5*self.L, self.curb_positions[1] - state_coords_unnormalized[:, 6:7] - 0.5*self.L)
        
        # Distance from the stranded car
        stranded_car_pos = torch.tensor(self.stranded_car_pos*1.0).type(torch.FloatTensor)
        if self.mu == -10.0:  #no mu scenario
          dist_stranded_R1 = torch.norm(state_coords_unnormalized[:, 0:2] - stranded_car_pos, dim=1, keepdim=True) - self.L
          dist_stranded_R2 = torch.norm(state_coords_unnormalized[:, 4:6] - stranded_car_pos, dim=1, keepdim=True) - self.L
        else:
          le = 2 + self.mu               #radius*
          wi = 0.75 + 0.25 * self.mu     #radius*
          le_inflated = le + 0.5*self.L
          wi_inflated = wi + 0.5*self.L
          state_centered_R1 = state_coords_unnormalized[:, 0:2] - stranded_car_pos
          state_centered_R1[:, 1] = state_centered_R1[:, 1] * (le_inflated/wi_inflated)#y component scaled
          state_centered_R2 = state_coords_unnormalized[:, 4:6] - stranded_car_pos
          state_centered_R2[:, 1] = state_centered_R2[:, 1] * (le_inflated/wi_inflated)#y component scaled
          dist_stranded_R1 = torch.norm(state_centered_R1, dim=1, keepdim=True) - le_inflated
          dist_stranded_R2 = torch.norm(state_centered_R2, dim=1, keepdim=True) - le_inflated          

        dist_stranded = torch.min(dist_stranded_R1, dist_stranded_R2)


        # Distance between the vehicles themselves
        dist_R1R2 = torch.norm(state_coords_unnormalized[:, 0:2] - state_coords_unnormalized[:, 4:6], dim=1, keepdim=True) - self.L

        if self.collision_setting == 'v1':
            return torch.min(torch.min(torch.min(dist_lc, dist_uc), dist_stranded), dist_R1R2) * self.gx_factor
        else:
            raise NotImplementedError

    def compute_IC(self, state_coords):
        # Compute the boundary condition given the normalized state coordinates.
        state_coords_unnormalized = state_coords * 1.0
        state_coords_unnormalized[:, 0] = state_coords_unnormalized[:, 0] * self.alpha['x'] + self.beta['x']
        state_coords_unnormalized[:, 1] = state_coords_unnormalized[:, 1] * self.alpha['y'] + self.beta['y']
        state_coords_unnormalized[:, 2] = state_coords_unnormalized[:, 2] * self.alpha['th'] + self.beta['th']
        state_coords_unnormalized[:, 3] = state_coords_unnormalized[:, 3] * self.alpha['v'] + self.beta['v']
        #state_coords_unnormalized[:, 4] = state_coords_unnormalized[:, 4] * self.alpha['phi'] + self.beta['phi']

        state_coords_unnormalized[:, 4] = state_coords_unnormalized[:, 4] * self.alpha['x'] + self.beta['x']
        state_coords_unnormalized[:, 5] = state_coords_unnormalized[:, 5] * self.alpha['y'] + self.beta['y']
        state_coords_unnormalized[:, 6] = state_coords_unnormalized[:, 6] * self.alpha['th'] + self.beta['th']
        state_coords_unnormalized[:, 7] = state_coords_unnormalized[:, 7] * self.alpha['v'] + self.beta['v']
        #state_coords_unnormalized[:, 9] = state_coords_unnormalized[:, 9] * self.alpha['phi'] + self.beta['phi']

        lx = self.compute_lx(state_coords_unnormalized)
        gx = self.compute_gx(state_coords_unnormalized)
        hx = -gx
        vx = torch.max(lx, hx)
        # return lx, gx, vx
        return lx, hx, vx

    def compute_vehicle_ham(self, x, dudx, return_opt_ctrl=False, Rindex=None):
        # Limit acceleration bounds based on the speed
        zero_tensor = torch.Tensor([0]).cuda()
        aMin = torch.ones_like(x[..., 3]) * self.aMin
        aMin = torch.where((x[..., 3] <= self.vMin), zero_tensor, aMin)
        aMax = torch.ones_like(x[..., 3]) * self.aMax
        aMax = torch.where((x[..., 3] >= self.vMax), zero_tensor, aMax)

        # Limit steering bounds based on the speed
        # psiMin = torch.ones_like(x[..., 4]) * self.psiMin
        # psiMin = torch.where((x[..., 4] <= self.phiMin), zero_tensor, psiMin)
        # psiMax = torch.ones_like(x[..., 4]) * self.psiMax
        # psiMax = torch.where((x[..., 4] >= self.phiMax), zero_tensor, psiMax)
        phiMin = torch.ones_like(x[..., 2]) * self.phiMin
        phiMax = torch.ones_like(x[..., 2]) * self.phiMax

        # Compute optimal control
        opt_acc = torch.where((dudx[..., 3] > 0), aMin, aMax)
        opt_phi = torch.where((dudx[..., 2] > 0), phiMin, phiMax)

        if (self.curriculum_version in ['v2', 'v3']) and (Rindex == 1):
            # Velocity can't change
            opt_acc = 0.0*opt_acc

        # Compute Hamiltonian
        ham_vehicle = x[..., 3] * torch.cos(x[..., 2]) * dudx[..., 0] + \
                      x[..., 3] * torch.sin(x[..., 2]) * dudx[..., 1] + \
                      x[..., 3] * torch.tan(opt_phi) * dudx[..., 2] / self.L + \
                      opt_acc * dudx[..., 3]

        # Freeze the Hamiltonian if required
        if self.ham_version == 'v2':
            # Check if vehicle is within the target point and if so, freeze the Hamiltonian selectively
            goal_tensor = torch.tensor([self.goalX[Rindex], self.goalY[Rindex]]).type(torch.FloatTensor)[None, None].cuda()
            dist_to_goal = torch.norm(x[..., 0:2] - goal_tensor, dim=-1) - 0.5*self.L
            ham_vehicle = torch.where((dist_to_goal <= 0), zero_tensor, ham_vehicle)
        
        if return_opt_ctrl:
            opt_ctrl = torch.cat((opt_acc, opt_phi), dim=1)
            return ham_vehicle, opt_ctrl
        else:
            return ham_vehicle

    def compute_overall_ham(self, x, dudx, return_components=False):
        alpha = self.alpha
        beta = self.beta

        # Scale the costates appropriately.
        dudx[..., 0] = dudx[..., 0] / alpha['x']
        dudx[..., 1] = dudx[..., 1] / alpha['y']
        dudx[..., 2] = dudx[..., 2] / alpha['th']
        dudx[..., 3] = dudx[..., 3] / alpha['v']
        #dudx[..., 4] = dudx[..., 4] / alpha['phi']

        dudx[..., 4] = dudx[..., 4] / alpha['x']
        dudx[..., 5] = dudx[..., 5] / alpha['y']
        dudx[..., 6] = dudx[..., 6] / alpha['th']
        dudx[..., 7] = dudx[..., 7] / alpha['v']
        #dudx[..., 9] = dudx[..., 9] / alpha['phi']

        # Scale for output normalization
        norm_to = 0.02
        mean = 0.25 * alpha['x']
        var = 0.5 * alpha['x']
        dudx = dudx * var/norm_to

        # Scale the states appropriately.
        x_unnormalized = x * 1.0
        x_unnormalized[..., 0] = x_unnormalized[..., 0] * alpha['x'] + beta['x']
        x_unnormalized[..., 1] = x_unnormalized[..., 1] * alpha['y'] + beta['y']
        x_unnormalized[..., 2] = x_unnormalized[..., 2] * alpha['th'] + beta['th']
        x_unnormalized[..., 3] = x_unnormalized[..., 3] * alpha['v'] + beta['v']
        #x_unnormalized[..., 4] = x_unnormalized[..., 4] * alpha['phi'] + beta['phi']

        x_unnormalized[..., 4] = x_unnormalized[..., 4] * alpha['x'] + beta['x']
        x_unnormalized[..., 5] = x_unnormalized[..., 5] * alpha['y'] + beta['y']
        x_unnormalized[..., 6] = x_unnormalized[..., 6] * alpha['th'] + beta['th']
        x_unnormalized[..., 7] = x_unnormalized[..., 7] * alpha['v'] + beta['v']
        #x_unnormalized[..., 9] = x_unnormalized[..., 9] * alpha['phi'] + beta['phi']

        # Compute the hamiltonian
        ham_R1 = self.compute_vehicle_ham(x_unnormalized[..., 0:4], dudx[..., 0:4], Rindex=0) 

        if self.curriculum_version == 'v4':
            ham_R2 = 0.0
        else:
            ham_R2 = self.compute_vehicle_ham(x_unnormalized[..., 4:], dudx[..., 4:], Rindex=1)

        ## Total Hamiltonian (take care of normalization again)
        ham_R1 = ham_R1 / (var/norm_to)
        ham_R2 = ham_R2 / (var/norm_to)
        ham_total = ham_R1 + ham_R2

        if return_components:
            return ham_total, ham_R1, ham_R2
        else:
            return ham_total

    def propagate_state(self, x, u, dt):
        x_next = torch.zeros_like(x)

        x_next[0] = x[3] * torch.cos(x[2])        #x1
        x_next[1] = x[3] * torch.sin(x[2])        #y1
        x_next[2] = x[3] * np.tan(u[1]) / self.L  #th1
        x_next[3] = u[0]                          #v1

        x_next[4] = x[7] * torch.cos(x[6])        #x2
        x_next[5] = x[7] * torch.sin(x[6])        #y2
        x_next[6] = x[7] * np.tan(u[3]) / self.L  #th2
        x_next[7] = u[2]                          #v2

        return x + dt*x_next          

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

        if self.sampling_bias_ratio > 0.0:
            valid_upper_boundary =  (self.curb_positions[1] - 0.5*self.L - self.beta['y'])/self.alpha['y']
            num_samples = int(self.numpoints * self.sampling_bias_ratio)
            coords[-num_samples:, 1] = coords[-num_samples:, 1] * valid_upper_boundary
            coords[-num_samples:, 5] = coords[-num_samples:, 5] * valid_upper_boundary

        if self.curriculum_version in ['v2', 'v4']:
            # Set velocity to zero, only sample x and y around the goal state
            speed_value = -self.beta['v']/self.alpha['v']
            x_value_upper = (self.goalX[1] + 1.0 - self.beta['x'])/self.alpha['x']
            x_value_lower = (self.goalX[1] - 1.0 - self.beta['x'])/self.alpha['x']
            y_value_upper = (self.goalY[1] + 0.2 - self.beta['y'])/self.alpha['y']
            y_value_lower = (self.goalY[1] - 0.2 - self.beta['y'])/self.alpha['y']
            # coords[:, 5] = torch.zeros(self.numpoints).uniform_(x_value_lower, x_value_upper)???????????
            # coords[:, 6] = torch.zeros(self.numpoints).uniform_(y_value_lower, y_value_upper)
            # coords[:, 8] = torch.ones(self.numpoints) * speed_value
        elif self.curriculum_version == 'v3':
            # Set velocity to zero, sample x and y anywhere
            speed_value = -self.beta['v']/self.alpha['v']
            coords[:, 7] = torch.ones(self.numpoints) * speed_value

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time; this currently assumes t \in [tMin, tMax]
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # Compute the initial value function
        # lx, gx, boundary_values = self.compute_IC(coords[:, 1:])
        lx, hx, boundary_values = self.compute_IC(coords[:, 1:])

        # print('Min and max value before normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
        # print('Min and max l(x) before normalization are %0.4f and %0.4f' %(min(lx), max(lx)))
        # print('Min and max g(x) before normalization are %0.4f and %0.4f' %(min(gx), max(gx)))
        # print('Min and max h(x) before normalization are %0.4f and %0.4f' %(min(hx), max(hx)))
        boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
        lx = (lx - self.mean)*self.norm_to/self.var
        # gx = (gx - mean)*norm_to/var
        hx = (hx - self.mean)*self.norm_to/self.var
        # print('Min and max value after normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
        # print('Min and max l(x) after normalization are %0.4f and %0.4f' %(min(lx), max(lx)))
        # print('Min and max g(x) after normalization are %0.4f and %0.4f' %(min(gx), max(gx)))
        # print('Min and max h(x) after normalization are %0.4f and %0.4f' %(min(hx), max(hx)))

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

        # return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask, 'lx': lx, 'gx': gx}
        return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask, 'lx': lx, 'hx': hx}

class ReachabilityNarrowAvoidSource(Dataset):
    def __init__(self, numpoints, pretrain, tMin, tMax, mu, counter_start, counter_end, pretrain_iters, norm_scheme, clip_value_gradients,
                 speed_setting, sampling_bias_ratio, env_setting, ham_version, target_setting, collision_setting, 
                 curriculum_version, HJIVI_smoothing_setting, smoothing_exponent, num_src_samples):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        
        self.num_vehicles = 2
        self.num_states = 5 * self.num_vehicles

        self.tMax = tMax
        self.tMin = tMin
        self.mu = mu

        # Define state alphas and betas so that all coordinates are from [-1, 1]. 
        # The conversion rule is state' = (state - beta)/alpha. state' is in [-1, 1].
        # [state sequence: x_Ri, y_Ri, th_Ri, v_Ri, phi_Ri, ...]. Ri is the ith vehicle.
        self.alpha = {}
        self.beta = {}
        if speed_setting == 'medium_v2':
            self.alpha['x'] = 8.0
            self.alpha['y'] = 3.8
            self.alpha['th'] = 1.2*math.pi
            self.alpha['v'] = 4.0
            self.alpha['phi'] = 1.2*0.3*math.pi
            self.alpha['time'] = 10.0/self.tMax

            self.beta['x'] = 0.0
            self.beta['y'] = 0.0
            self.beta['th'] = 0.0
            self.beta['v'] = 3.0
            self.beta['phi'] = 0.0

            # Target positions
            self.goalX = np.array([6.0, -6.0])
            self.goalY = np.array([-1.4, 1.4])

            # State bounds
            self.vMin = 0.001
            self.vMax = 6.50
            self.phiMin = -0.3*math.pi + 0.001
            self.phiMax = 0.3*math.pi - 0.001

            # Control bounds
            self.aMin = -4.0
            self.aMax = 2.0
            self.psiMin = -3.0*math.pi
            self.psiMax = 3.0*math.pi
        else:
            raise NotImplementedError

        # How to weigh the obstacles
        #self.gx_factor = gx_factor

        if env_setting == 'v2':
            # Vehicle diameter/length
            self.L = 2.0

            # Lower and upper curb positions (in the y direction)
            self.curb_positions = np.array([-2.8, 2.8])

            # Stranded car position
            self.stranded_car_pos = np.array([0.0, -1.8])
        else:
            raise NotImplementedError

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

        self.normalization_scheme = norm_scheme
        self.sampling_bias_ratio = sampling_bias_ratio
        self.clip_value_gradients = clip_value_gradients

        self.ham_version = ham_version
        self.target_setting = target_setting
        self.collision_setting = collision_setting
        self.curriculum_version = curriculum_version
        self.HJIVI_smoothing_setting = HJIVI_smoothing_setting
        self.smoothing_exponent = smoothing_exponent
        if self.normalization_scheme == 'hack1':
            self.norm_to = 0.02
            self.mean = 0.25 * self.alpha['x']
            self.var = 0.5 * self.alpha['x']
        else:
            raise NotImplementedError

    def compute_lx(self, state_coords_unnormalized):
        # Compute the obstacle boundary condition given the unnormalized state coordinates. Negative inside the obstacle positive outside.
        # Distance from the lower curb
        dist_lc_R1 = state_coords_unnormalized[:, 1:2] - self.curb_positions[0] - 0.5*self.L
        dist_lc_R2 = state_coords_unnormalized[:, 6:7] - self.curb_positions[0] - 0.5*self.L
        dist_lc = torch.min(dist_lc_R1, dist_lc_R2)
        # dist_lc = torch.min(state_coords_unnormalized[:, 1:2] - self.curb_positions[0] - 0.5*self.L, state_coords_unnormalized[:, 6:7] - self.curb_positions[0] - 0.5*self.L)
        
        # Distance from the upper curb
        dist_uc_R1 = self.curb_positions[1] - state_coords_unnormalized[:, 1:2] - 0.5*self.L
        dist_uc_R2 = self.curb_positions[1] - state_coords_unnormalized[:, 6:7] - 0.5*self.L
        dist_uc = torch.min(dist_uc_R1, dist_uc_R2)
        # dist_uc = torch.min(self.curb_positions[1] - state_coords_unnormalized[:, 1:2] - 0.5*self.L, self.curb_positions[1] - state_coords_unnormalized[:, 6:7] - 0.5*self.L)
        
        # Distance from the stranded car
        stranded_car_pos = torch.tensor(self.stranded_car_pos*1.0).type(torch.FloatTensor)

        #dist_stranded_R1 = torch.norm(state_coords_unnormalized[:, 0:2] - stranded_car_pos, dim=1, keepdim=True) - self.L
        #dist_stranded_R2 = torch.norm(state_coords_unnormalized[:, 5:7] - stranded_car_pos, dim=1, keepdim=True) - self.L
                
        le = 2 + self.mu               #radius*
        wi = 0.75 + 0.25 * self.mu     #radius*
        le_inflated = le + 0.5*self.L
        wi_inflated = wi + 0.5*self.L
        state_centered_R1 = state_coords_unnormalized[:, 0:2] - stranded_car_pos
        state_centered_R1[:, 1] = state_centered_R1[:, 1] * (le_inflated/wi_inflated)#y component scaled
        state_centered_R2 = state_coords_unnormalized[:, 5:7] - stranded_car_pos
        state_centered_R2[:, 1] = state_centered_R2[:, 1] * (le_inflated/wi_inflated)#y component scaled
        dist_stranded_R1 = torch.norm(state_centered_R1, dim=1, keepdim=True) - le_inflated
        dist_stranded_R2 = torch.norm(state_centered_R2, dim=1, keepdim=True) - le_inflated

        dist_stranded = torch.min(dist_stranded_R1, dist_stranded_R2)

        # Distance between the vehicles themselves
        dist_R1R2 = torch.norm(state_coords_unnormalized[:, 0:2] - state_coords_unnormalized[:, 5:7], dim=1, keepdim=True) - self.L

        if self.collision_setting == 'v1':
            return torch.min(torch.min(torch.min(dist_lc, dist_uc), dist_stranded), dist_R1R2) 
        elif self.collision_setting == 'v2':
            return torch.min(torch.min(dist_lc, dist_uc), dist_stranded_R1) 
        elif self.collision_setting == 'v3':
            return torch.min(dist_stranded, dist_R1R2)
        elif self.collision_setting == 'v4':
            return torch.min(torch.min(torch.min(dist_lc_R1, dist_uc_R1), dist_stranded_R1), dist_R1R2)
        else:
            raise NotImplementedError

    def compute_IC(self, state_coords):
        # Compute the boundary condition given the normalized state coordinates.
        state_coords_unnormalized = state_coords * 1.0
        state_coords_unnormalized[:, 0] = state_coords_unnormalized[:, 0] * self.alpha['x'] + self.beta['x']
        state_coords_unnormalized[:, 1] = state_coords_unnormalized[:, 1] * self.alpha['y'] + self.beta['y']
        state_coords_unnormalized[:, 2] = state_coords_unnormalized[:, 2] * self.alpha['th'] + self.beta['th']
        state_coords_unnormalized[:, 3] = state_coords_unnormalized[:, 3] * self.alpha['v'] + self.beta['v']
        state_coords_unnormalized[:, 4] = state_coords_unnormalized[:, 4] * self.alpha['phi'] + self.beta['phi']

        state_coords_unnormalized[:, 5] = state_coords_unnormalized[:, 5] * self.alpha['x'] + self.beta['x']
        state_coords_unnormalized[:, 6] = state_coords_unnormalized[:, 6] * self.alpha['y'] + self.beta['y']
        state_coords_unnormalized[:, 7] = state_coords_unnormalized[:, 7] * self.alpha['th'] + self.beta['th']
        state_coords_unnormalized[:, 8] = state_coords_unnormalized[:, 8] * self.alpha['v'] + self.beta['v']
        state_coords_unnormalized[:, 9] = state_coords_unnormalized[:, 9] * self.alpha['phi'] + self.beta['phi']

        lx = self.compute_lx(state_coords_unnormalized)
        return lx

    def compute_vehicle_ham(self, x, dudx, return_opt_ctrl=False, Rindex=None):
        # Limit acceleration bounds based on the speed
        zero_tensor = torch.Tensor([0]).cuda()
        aMin = torch.ones_like(x[..., 3]) * self.aMin
        aMin = torch.where((x[..., 3] <= self.vMin), zero_tensor, aMin)
        aMax = torch.ones_like(x[..., 3]) * self.aMax
        aMax = torch.where((x[..., 3] >= self.vMax), zero_tensor, aMax)

        # Limit steering bounds based on the speed
        psiMin = torch.ones_like(x[..., 4]) * self.psiMin
        psiMin = torch.where((x[..., 4] <= self.phiMin), zero_tensor, psiMin)
        psiMax = torch.ones_like(x[..., 4]) * self.psiMax
        psiMax = torch.where((x[..., 4] >= self.phiMax), zero_tensor, psiMax)

        # Compute optimal control
        opt_acc = torch.where((dudx[..., 3] < 0), aMin, aMax)######
        opt_psi = torch.where((dudx[..., 4] < 0), psiMin, psiMax)######

        if (self.curriculum_version in ['v2', 'v3']) and (Rindex == 1):
            # Velocity can't change
            opt_acc = 0.0*opt_acc

        # Compute Hamiltonian
        ham_vehicle = x[..., 3] * torch.cos(x[..., 2]) * dudx[..., 0] + \
                      x[..., 3] * torch.sin(x[..., 2]) * dudx[..., 1] + \
                      x[..., 3] * torch.tan(x[..., 4]) * dudx[..., 2] / self.L + \
                      opt_acc * dudx[..., 3] + \
                      opt_psi * dudx[..., 4]

        # Freeze the Hamiltonian if required
        if self.ham_version == 'v2':
            # Check if vehicle is within the target point and if so, freeze the Hamiltonian selectively
            goal_tensor = torch.tensor([self.goalX[Rindex], self.goalY[Rindex]]).type(torch.FloatTensor)[None, None].cuda()
            dist_to_goal = torch.norm(x[..., 0:2] - goal_tensor, dim=-1) - 0.5*self.L
            ham_vehicle = torch.where((dist_to_goal <= 0), zero_tensor, ham_vehicle)
        
        if return_opt_ctrl:
            opt_ctrl = torch.cat((opt_acc, opt_psi), dim=1)
            return ham_vehicle, opt_ctrl
        else:
            return ham_vehicle

    def compute_overall_ham(self, x, dudx, return_components=False):
        alpha = self.alpha
        beta = self.beta

        # Scale the costates appropriately.
        dudx[..., 0] = dudx[..., 0] / alpha['x']
        dudx[..., 1] = dudx[..., 1] / alpha['y']
        dudx[..., 2] = dudx[..., 2] / alpha['th']
        dudx[..., 3] = dudx[..., 3] / alpha['v']
        dudx[..., 4] = dudx[..., 4] / alpha['phi']

        dudx[..., 5] = dudx[..., 5] / alpha['x']
        dudx[..., 6] = dudx[..., 6] / alpha['y']
        dudx[..., 7] = dudx[..., 7] / alpha['th']
        dudx[..., 8] = dudx[..., 8] / alpha['v']
        dudx[..., 9] = dudx[..., 9] / alpha['phi']

        # Scale for output normalization
        norm_to = 0.02
        mean = 0.25 * alpha['x']
        var = 0.5 * alpha['x']
        dudx = dudx * var/norm_to

        # Scale the states appropriately.
        x_unnormalized = x * 1.0
        x_unnormalized[..., 0] = x_unnormalized[..., 0] * alpha['x'] + beta['x']
        x_unnormalized[..., 1] = x_unnormalized[..., 1] * alpha['y'] + beta['y']
        x_unnormalized[..., 2] = x_unnormalized[..., 2] * alpha['th'] + beta['th']
        x_unnormalized[..., 3] = x_unnormalized[..., 3] * alpha['v'] + beta['v']
        x_unnormalized[..., 4] = x_unnormalized[..., 4] * alpha['phi'] + beta['phi']

        x_unnormalized[..., 5] = x_unnormalized[..., 5] * alpha['x'] + beta['x']
        x_unnormalized[..., 6] = x_unnormalized[..., 6] * alpha['y'] + beta['y']
        x_unnormalized[..., 7] = x_unnormalized[..., 7] * alpha['th'] + beta['th']
        x_unnormalized[..., 8] = x_unnormalized[..., 8] * alpha['v'] + beta['v']
        x_unnormalized[..., 9] = x_unnormalized[..., 9] * alpha['phi'] + beta['phi']

        # Compute the hamiltonian
        ham_R1 = self.compute_vehicle_ham(x_unnormalized[..., 0:5], dudx[..., 0:5], Rindex=0) 

        if self.curriculum_version == 'v4':
            ham_R2 = 0.0
        else:
            ham_R2 = self.compute_vehicle_ham(x_unnormalized[..., 5:], dudx[..., 5:], Rindex=1)

        ## Total Hamiltonian (take care of normalization again)
        ham_R1 = ham_R1 / (var/norm_to)
        ham_R2 = ham_R2 / (var/norm_to)
        ham_total = ham_R1 + ham_R2

        if return_components:
            return ham_total, ham_R1, ham_R2
        else:
            return ham_total

    def propagate_state(self, x, u, dt):
        alpha = self.alpha
        beta = self.beta

        x_next = torch.zeros_like(x)
        x_next[0] = x[3] * torch.cos(x[2])
        x_next[1] = x[3] * torch.sin(x[2])
        x_next[2] = x[3] * torch.tan(x[4]) / self.L
        x_next[3] = u[0]
        x_next[4] = u[1]

        x_next[5] = x[8] * torch.cos(x[7])
        x_next[6] = x[8] * torch.sin(x[7])
        x_next[7] = x[8] * torch.tan(x[9]) / self.L
        x_next[8] = u[2]
        x_next[9] = u[3]

        return x + dt*x_next          

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

        if self.sampling_bias_ratio > 0.0:
            valid_upper_boundary =  (self.curb_positions[1] - 0.5*self.L - self.beta['y'])/self.alpha['y']
            num_samples = int(self.numpoints * self.sampling_bias_ratio)
            coords[-num_samples:, 1] = coords[-num_samples:, 1] * valid_upper_boundary
            coords[-num_samples:, 6] = coords[-num_samples:, 6] * valid_upper_boundary

        if self.curriculum_version in ['v2', 'v4']:
            # Set velocity to zero, only sample x and y around the goal state
            speed_value = -self.beta['v']/self.alpha['v']
            x_value_upper = (self.goalX[1] + 1.0 - self.beta['x'])/self.alpha['x']
            x_value_lower = (self.goalX[1] - 1.0 - self.beta['x'])/self.alpha['x']
            y_value_upper = (self.goalY[1] + 0.2 - self.beta['y'])/self.alpha['y']
            y_value_lower = (self.goalY[1] - 0.2 - self.beta['y'])/self.alpha['y']
            coords[:, 5] = torch.zeros(self.numpoints).uniform_(x_value_lower, x_value_upper)
            coords[:, 6] = torch.zeros(self.numpoints).uniform_(y_value_lower, y_value_upper)
            coords[:, 8] = torch.ones(self.numpoints) * speed_value
        elif self.curriculum_version == 'v3':
            # Set velocity to zero, sample x and y anywhere
            speed_value = -self.beta['v']/self.alpha['v']
            coords[:, 8] = torch.ones(self.numpoints) * speed_value

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time; this currently assumes t \in [tMin, tMax]
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # Compute the initial value function
        # lx, gx, boundary_values = self.compute_IC(coords[:, 1:])
        lx = self.compute_IC(coords[:, 1:])

        # print('Min and max value before normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
        # print('Min and max l(x) before normalization are %0.4f and %0.4f' %(min(lx), max(lx)))
        # print('Min and max g(x) before normalization are %0.4f and %0.4f' %(min(gx), max(gx)))
        # print('Min and max h(x) before normalization are %0.4f and %0.4f' %(min(hx), max(hx)))
        lx = (lx - self.mean)*self.norm_to/self.var
        ##lx = (lx - self.mean)*self.norm_to/self.var
        # gx = (gx - mean)*norm_to/var
        ##hx = (hx - self.mean)*self.norm_to/self.var
        # print('Min and max value after normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
        # print('Min and max l(x) after normalization are %0.4f and %0.4f' %(min(lx), max(lx)))
        # print('Min and max g(x) after normalization are %0.4f and %0.4f' %(min(gx), max(gx)))
        # print('Min and max h(x) after normalization are %0.4f and %0.4f' %(min(hx), max(hx)))

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

        # return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask, 'lx': lx, 'gx': gx}
        return {'coords': coords}, {'lx': lx, 'dirichlet_mask': dirichlet_mask}

class ReachabilityNarrowRefSource(Dataset):
    def __init__(self, numpoints, pretrain, tMin, tMax, mu, counter_start, counter_end, pretrain_iters, norm_scheme, clip_value_gradients,
                 gx_factor, speed_setting, sampling_bias_ratio, env_setting, ham_version, target_setting, collision_setting, 
                 curriculum_version, HJIVI_smoothing_setting, smoothing_exponent, num_src_samples):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        
        self.num_vehicles = 2
        self.num_states = 5 * self.num_vehicles

        self.tMax = tMax
        self.tMin = tMin
        self.mu = mu

        # Define state alphas and betas so that all coordinates are from [-1, 1]. 
        # The conversion rule is state' = (state - beta)/alpha. state' is in [-1, 1].
        # [state sequence: x_Ri, y_Ri, th_Ri, v_Ri, phi_Ri, ...]. Ri is the ith vehicle.
        self.alpha = {}
        self.beta = {}

        if speed_setting == 'high':
            self.alpha['x'] = 60.0
            self.alpha['y'] = 3.8
            self.alpha['th'] = 1.2*math.pi
            self.alpha['v'] = 7.0
            self.alpha['phi'] = 1.2*0.1*math.pi
            self.alpha['time'] = 10.0/self.tMax

            self.beta['x'] = 0.0
            self.beta['y'] = 0.0
            self.beta['th'] = 0.0
            self.beta['v'] = 6.0
            self.beta['phi'] = 0.0

            # Target positions
            self.goalX = np.array([36.0, -36.0])
            self.goalY = np.array([-1.4, 1.4])

            # State bounds
            self.vMin = 0.001
            self.vMax = 11.999
            self.phiMin = -0.1*math.pi + 0.001
            self.phiMax = 0.1*math.pi - 0.001

            # Control bounds
            self.aMin = -4.0
            self.aMax = 2.0
            self.psiMin = -0.1*math.pi
            self.psiMax = 0.1*math.pi

        elif speed_setting == 'low':
            self.alpha['x'] = 8.0
            self.alpha['y'] = 3.8
            self.alpha['th'] = 1.2*math.pi
            self.alpha['v'] = 3.0
            self.alpha['phi'] = 1.2*0.1*math.pi
            self.alpha['time'] = 6.0/self.tMax

            self.beta['x'] = 0.0
            self.beta['y'] = 0.0
            self.beta['th'] = 0.0
            self.beta['v'] = 2.0
            self.beta['phi'] = 0.0

            # Target positions
            self.goalX = np.array([6.0, -6.0])
            self.goalY = np.array([-1.4, 1.4])

            # State bounds
            self.vMin = 0.001
            self.vMax = 4.50
            self.phiMin = -0.1*math.pi + 0.001
            self.phiMax = 0.1*math.pi - 0.001

            # Control bounds
            self.aMin = -4.0
            self.aMax = 2.0
            self.psiMin = -0.1*math.pi
            self.psiMax = 0.1*math.pi

        elif speed_setting == 'medium':
            self.alpha['x'] = 8.0
            self.alpha['y'] = 3.8
            self.alpha['th'] = 1.2*math.pi
            self.alpha['v'] = 4.0
            self.alpha['phi'] = 1.2*0.3*math.pi
            self.alpha['time'] = 10.0/self.tMax

            self.beta['x'] = 0.0
            self.beta['y'] = 0.0
            self.beta['th'] = 0.0
            self.beta['v'] = 3.0
            self.beta['phi'] = 0.0

            # Target positions
            self.goalX = np.array([6.0, -6.0])
            self.goalY = np.array([-1.4, 1.4])

            # State bounds
            self.vMin = 0.001
            self.vMax = 6.50
            self.phiMin = -0.3*math.pi + 0.001
            self.phiMax = 0.3*math.pi - 0.001

            # Control bounds
            self.aMin = -4.0
            self.aMax = 2.0
            self.psiMin = -0.3*math.pi
            self.psiMax = 0.3*math.pi

        elif speed_setting == 'medium_v2':
            self.alpha['x'] = 8.0
            self.alpha['y'] = 3.8
            self.alpha['th'] = 1.2*math.pi
            self.alpha['v'] = 4.0
            self.alpha['phi'] = 1.2*0.3*math.pi
            self.alpha['time'] = 10.0/self.tMax

            self.beta['x'] = 0.0
            self.beta['y'] = 0.0
            self.beta['th'] = 0.0
            self.beta['v'] = 3.0
            self.beta['phi'] = 0.0

            # Target positions
            self.goalX = np.array([6.0, -6.0])
            self.goalY = np.array([-1.4, 1.4])

            # State bounds
            self.vMin = 0.001
            self.vMax = 6.50
            self.phiMin = -0.3*math.pi + 0.001
            self.phiMax = 0.3*math.pi - 0.001

            # Control bounds
            self.aMin = -4.0
            self.aMax = 2.0
            self.psiMin = -3.0*math.pi
            self.psiMax = 3.0*math.pi

        elif speed_setting == 'medium_v3':
            self.alpha['x'] = 8.0
            self.alpha['y'] = 4.0
            self.alpha['th'] = 1.2*math.pi
            self.alpha['v'] = 4.0
            self.alpha['phi'] = 1.2*0.3*math.pi
            self.alpha['time'] = 10.0/self.tMax

            self.beta['x'] = 0.0
            self.beta['y'] = 0.0
            self.beta['th'] = 0.0
            self.beta['v'] = 3.0
            self.beta['phi'] = 0.0

            # Target positions
            self.goalX = np.array([6.0, -6.0])
            self.goalY = np.array([-2.0, 2.0])

            # State bounds
            self.vMin = 0.001
            self.vMax = 6.50
            self.phiMin = -0.3*math.pi + 0.001
            self.phiMax = 0.3*math.pi - 0.001

            # Control bounds
            self.aMin = -4.0
            self.aMax = 2.0
            self.psiMin = -3.0*math.pi
            self.psiMax = 3.0*math.pi

        else:
            raise NotImplementedError

        # How to weigh the obstacles
        self.gx_factor = gx_factor

        if env_setting == 'v1':
            # Vehicle diameter/length
            self.L = 2.0

            # Lower and upper curb positions (in the y direction)
            self.curb_positions = np.array([-2.8, 2.8])

            # Stranded car position
            self.stranded_car_pos = np.array([0.0, -1.4])

        elif env_setting == 'v2':
            # Vehicle diameter/length
            self.L = 2.0

            # Lower and upper curb positions (in the y direction)
            self.curb_positions = np.array([-2.8, 2.8])

            # Stranded car position
            self.stranded_car_pos = np.array([0.0, -1.8])

        elif env_setting == 'v3':
            # Vehicle diameter/length
            self.L = 2.0

            # Lower and upper curb positions (in the y direction)
            self.curb_positions = np.array([-4.0, 4.0])

            # Stranded car position
            self.stranded_car_pos = np.array([0.0, -2.0])

        else:
            raise NotImplementedError

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

        self.normalization_scheme = norm_scheme
        self.sampling_bias_ratio = sampling_bias_ratio
        self.clip_value_gradients = clip_value_gradients

        self.ham_version = ham_version
        self.target_setting = target_setting
        self.collision_setting = collision_setting
        self.curriculum_version = curriculum_version
        self.HJIVI_smoothing_setting = HJIVI_smoothing_setting
        self.smoothing_exponent = smoothing_exponent
        if self.normalization_scheme == 'hack1':
            self.norm_to = 0.02
            self.mean = 0.25 * self.alpha['x']
            self.var = 0.5 * self.alpha['x']
        else:
            raise NotImplementedError

    def compute_lx(self, state_coords_unnormalized):
        # Compute the target boundary condition given the unnormalized state coordinates.
        if self.target_setting in ['v1', 'v2', 'v4']:
            # Vehicle 1
            goal_tensor_R1 = torch.tensor([self.goalX[0], self.goalY[0]]).type(torch.FloatTensor)[None]
            dist_R1 = torch.norm(state_coords_unnormalized[:, 0:2] - goal_tensor_R1, dim=1, keepdim=True) - self.L
            # Vehicle 2
            goal_tensor_R2 = torch.tensor([self.goalX[1], self.goalY[1]]).type(torch.FloatTensor)[None]
            dist_R2 = torch.norm(state_coords_unnormalized[:, 5:7] - goal_tensor_R2, dim=1, keepdim=True) - self.L
            if self.target_setting == 'v1':
                return torch.max(dist_R1, dist_R2)
            elif self.target_setting == 'v2':
                return dist_R1
            elif self.target_setting == 'v4':
                sum_tensor = 0.5*(dist_R1 + dist_R2)
                max_tensor = 0.5*torch.max(dist_R1, dist_R2)
                sign_tensor = torch.sign(dist_R1 * dist_R2)
                return torch.where(sign_tensor < 0, max_tensor, sum_tensor)
        
        elif self.target_setting in ['v3']:
            # Have an infinitely extended target set above and below the center lane
            dist_R1 = torch.max((self.goalX[0] - 0.5*self.L) - state_coords_unnormalized[..., 0:1], state_coords_unnormalized[..., 1:2])
            dist_R2 = torch.max(state_coords_unnormalized[..., 5:6] - (self.goalX[1] + 0.5*self.L), -state_coords_unnormalized[..., 6:7])
            return torch.max(dist_R1, dist_R2)

        else:
            raise NotImplementedError

    def compute_gx(self, state_coords_unnormalized):
        # Compute the obstacle boundary condition given the unnormalized state coordinates. Negative inside the obstacle positive outside.
        # Distance from the lower curb
        dist_lc_R1 = state_coords_unnormalized[:, 1:2] - self.curb_positions[0] - 0.5*self.L
        dist_lc_R2 = state_coords_unnormalized[:, 6:7] - self.curb_positions[0] - 0.5*self.L
        dist_lc = torch.min(dist_lc_R1, dist_lc_R2)
        # dist_lc = torch.min(state_coords_unnormalized[:, 1:2] - self.curb_positions[0] - 0.5*self.L, state_coords_unnormalized[:, 6:7] - self.curb_positions[0] - 0.5*self.L)
        
        # Distance from the upper curb
        dist_uc_R1 = self.curb_positions[1] - state_coords_unnormalized[:, 1:2] - 0.5*self.L
        dist_uc_R2 = self.curb_positions[1] - state_coords_unnormalized[:, 6:7] - 0.5*self.L
        dist_uc = torch.min(dist_uc_R1, dist_uc_R2)
        # dist_uc = torch.min(self.curb_positions[1] - state_coords_unnormalized[:, 1:2] - 0.5*self.L, self.curb_positions[1] - state_coords_unnormalized[:, 6:7] - 0.5*self.L)
        
        # Distance from the stranded car
        stranded_car_pos = torch.tensor(self.stranded_car_pos*1.0).type(torch.FloatTensor)

        #dist_stranded_R1 = torch.norm(state_coords_unnormalized[:, 0:2] - stranded_car_pos, dim=1, keepdim=True) - self.L
        #dist_stranded_R2 = torch.norm(state_coords_unnormalized[:, 5:7] - stranded_car_pos, dim=1, keepdim=True) - self.L
                
        le = 2 + self.mu               #radius*
        wi = 0.75 + 0.25 * self.mu     #radius*
        le_inflated = le + 0.5*self.L
        wi_inflated = wi + 0.5*self.L
        state_centered_R1 = state_coords_unnormalized[:, 0:2] - stranded_car_pos
        state_centered_R1[:, 1] = state_centered_R1[:, 1] * (le_inflated/wi_inflated)#y component scaled
        state_centered_R2 = state_coords_unnormalized[:, 5:7] - stranded_car_pos
        state_centered_R2[:, 1] = state_centered_R2[:, 1] * (le_inflated/wi_inflated)#y component scaled
        dist_stranded_R1 = torch.norm(state_centered_R1, dim=1, keepdim=True) - le_inflated
        dist_stranded_R2 = torch.norm(state_centered_R2, dim=1, keepdim=True) - le_inflated

        dist_stranded = torch.min(dist_stranded_R1, dist_stranded_R2)


        # Distance between the vehicles themselves
        dist_R1R2 = torch.norm(state_coords_unnormalized[:, 0:2] - state_coords_unnormalized[:, 5:7], dim=1, keepdim=True) - self.L

        if self.collision_setting == 'v1':
            return torch.min(torch.min(torch.min(dist_lc, dist_uc), dist_stranded), dist_R1R2) * self.gx_factor
        elif self.collision_setting == 'v2':
            return torch.min(torch.min(dist_lc, dist_uc), dist_stranded_R1) * self.gx_factor
        elif self.collision_setting == 'v3':
            return torch.min(dist_stranded, dist_R1R2) * self.gx_factor
        elif self.collision_setting == 'v4':
            return torch.min(torch.min(torch.min(dist_lc_R1, dist_uc_R1), dist_stranded_R1), dist_R1R2) * self.gx_factor
        else:
            raise NotImplementedError

    def compute_IC(self, state_coords):
        # Compute the boundary condition given the normalized state coordinates.
        state_coords_unnormalized = state_coords * 1.0
        state_coords_unnormalized[:, 0] = state_coords_unnormalized[:, 0] * self.alpha['x'] + self.beta['x']
        state_coords_unnormalized[:, 1] = state_coords_unnormalized[:, 1] * self.alpha['y'] + self.beta['y']
        state_coords_unnormalized[:, 2] = state_coords_unnormalized[:, 2] * self.alpha['th'] + self.beta['th']
        state_coords_unnormalized[:, 3] = state_coords_unnormalized[:, 3] * self.alpha['v'] + self.beta['v']
        state_coords_unnormalized[:, 4] = state_coords_unnormalized[:, 4] * self.alpha['phi'] + self.beta['phi']

        state_coords_unnormalized[:, 5] = state_coords_unnormalized[:, 5] * self.alpha['x'] + self.beta['x']
        state_coords_unnormalized[:, 6] = state_coords_unnormalized[:, 6] * self.alpha['y'] + self.beta['y']
        state_coords_unnormalized[:, 7] = state_coords_unnormalized[:, 7] * self.alpha['th'] + self.beta['th']
        state_coords_unnormalized[:, 8] = state_coords_unnormalized[:, 8] * self.alpha['v'] + self.beta['v']
        state_coords_unnormalized[:, 9] = state_coords_unnormalized[:, 9] * self.alpha['phi'] + self.beta['phi']

        lx = self.compute_lx(state_coords_unnormalized)
        gx = self.compute_gx(state_coords_unnormalized)
        hx = -gx
        vx = torch.max(lx, hx)
        # return lx, gx, vx
        return lx, hx, vx

    def compute_vehicle_ham(self, x, dudx, return_opt_ctrl=False, Rindex=None):
        # Limit acceleration bounds based on the speed
        zero_tensor = torch.Tensor([0]).cuda()
        aMin = torch.ones_like(x[..., 3]) * self.aMin
        aMin = torch.where((x[..., 3] <= self.vMin), zero_tensor, aMin)
        aMax = torch.ones_like(x[..., 3]) * self.aMax
        aMax = torch.where((x[..., 3] >= self.vMax), zero_tensor, aMax)

        # Limit steering bounds based on the speed
        psiMin = torch.ones_like(x[..., 4]) * self.psiMin
        psiMin = torch.where((x[..., 4] <= self.phiMin), zero_tensor, psiMin)
        psiMax = torch.ones_like(x[..., 4]) * self.psiMax
        psiMax = torch.where((x[..., 4] >= self.phiMax), zero_tensor, psiMax)

        # Compute optimal control
        opt_acc = torch.where((dudx[..., 3] > 0), aMin, aMax)
        opt_psi = torch.where((dudx[..., 4] > 0), psiMin, psiMax)

        if (self.curriculum_version in ['v2', 'v3']) and (Rindex == 1):
            # Velocity can't change
            opt_acc = 0.0*opt_acc

        # Compute Hamiltonian
        ham_vehicle = x[..., 3] * torch.cos(x[..., 2]) * dudx[..., 0] + \
                      x[..., 3] * torch.sin(x[..., 2]) * dudx[..., 1] + \
                      x[..., 3] * torch.tan(x[..., 4]) * dudx[..., 2] / self.L + \
                      opt_acc * dudx[..., 3] + \
                      opt_psi * dudx[..., 4]

        # Freeze the Hamiltonian if required
        if self.ham_version == 'v2':
            # Check if vehicle is within the target point and if so, freeze the Hamiltonian selectively
            goal_tensor = torch.tensor([self.goalX[Rindex], self.goalY[Rindex]]).type(torch.FloatTensor)[None, None].cuda()
            dist_to_goal = torch.norm(x[..., 0:2] - goal_tensor, dim=-1) - 0.5*self.L
            ham_vehicle = torch.where((dist_to_goal <= 0), zero_tensor, ham_vehicle)
        
        if return_opt_ctrl:
            opt_ctrl = torch.cat((opt_acc, opt_psi), dim=1)
            return ham_vehicle, opt_ctrl
        else:
            return ham_vehicle

    def compute_overall_ham(self, x, dudx, return_components=False):
        alpha = self.alpha
        beta = self.beta

        # Scale the costates appropriately.
        dudx[..., 0] = dudx[..., 0] / alpha['x']
        dudx[..., 1] = dudx[..., 1] / alpha['y']
        dudx[..., 2] = dudx[..., 2] / alpha['th']
        dudx[..., 3] = dudx[..., 3] / alpha['v']
        dudx[..., 4] = dudx[..., 4] / alpha['phi']

        dudx[..., 5] = dudx[..., 5] / alpha['x']
        dudx[..., 6] = dudx[..., 6] / alpha['y']
        dudx[..., 7] = dudx[..., 7] / alpha['th']
        dudx[..., 8] = dudx[..., 8] / alpha['v']
        dudx[..., 9] = dudx[..., 9] / alpha['phi']

        # Scale for output normalization
        norm_to = 0.02
        mean = 0.25 * alpha['x']
        var = 0.5 * alpha['x']
        dudx = dudx * var/norm_to

        # Scale the states appropriately.
        x_unnormalized = x * 1.0
        x_unnormalized[..., 0] = x_unnormalized[..., 0] * alpha['x'] + beta['x']
        x_unnormalized[..., 1] = x_unnormalized[..., 1] * alpha['y'] + beta['y']
        x_unnormalized[..., 2] = x_unnormalized[..., 2] * alpha['th'] + beta['th']
        x_unnormalized[..., 3] = x_unnormalized[..., 3] * alpha['v'] + beta['v']
        x_unnormalized[..., 4] = x_unnormalized[..., 4] * alpha['phi'] + beta['phi']

        x_unnormalized[..., 5] = x_unnormalized[..., 5] * alpha['x'] + beta['x']
        x_unnormalized[..., 6] = x_unnormalized[..., 6] * alpha['y'] + beta['y']
        x_unnormalized[..., 7] = x_unnormalized[..., 7] * alpha['th'] + beta['th']
        x_unnormalized[..., 8] = x_unnormalized[..., 8] * alpha['v'] + beta['v']
        x_unnormalized[..., 9] = x_unnormalized[..., 9] * alpha['phi'] + beta['phi']

        # Compute the hamiltonian
        ham_R1 = self.compute_vehicle_ham(x_unnormalized[..., 0:5], dudx[..., 0:5], Rindex=0) 

        if self.curriculum_version == 'v4':
            ham_R2 = 0.0
        else:
            ham_R2 = self.compute_vehicle_ham(x_unnormalized[..., 5:], dudx[..., 5:], Rindex=1)

        ## Total Hamiltonian (take care of normalization again)
        ham_R1 = ham_R1 / (var/norm_to)
        ham_R2 = ham_R2 / (var/norm_to)
        ham_total = ham_R1 + ham_R2

        if return_components:
            return ham_total, ham_R1, ham_R2
        else:
            return ham_total

    def propagate_state(self, x, u, dt):
        alpha = self.alpha
        beta = self.beta

        x_next = torch.zeros_like(x)
        x_next[0] = x[3] * torch.cos(x[2])
        x_next[1] = x[3] * torch.sin(x[2])
        x_next[2] = x[3] * torch.tan(x[4]) / self.L
        x_next[3] = u[0]
        x_next[4] = u[1]

        x_next[5] = x[8] * torch.cos(x[7])
        x_next[6] = x[8] * torch.sin(x[7])
        x_next[7] = x[8] * torch.tan(x[9]) / self.L
        x_next[8] = u[2]
        x_next[9] = u[3]

        return x + dt*x_next          

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

        if self.sampling_bias_ratio > 0.0:
            valid_upper_boundary =  (self.curb_positions[1] - 0.5*self.L - self.beta['y'])/self.alpha['y']
            num_samples = int(self.numpoints * self.sampling_bias_ratio)
            coords[-num_samples:, 1] = coords[-num_samples:, 1] * valid_upper_boundary
            coords[-num_samples:, 6] = coords[-num_samples:, 6] * valid_upper_boundary

        if self.curriculum_version in ['v2', 'v4']:
            # Set velocity to zero, only sample x and y around the goal state
            speed_value = -self.beta['v']/self.alpha['v']
            x_value_upper = (self.goalX[1] + 1.0 - self.beta['x'])/self.alpha['x']
            x_value_lower = (self.goalX[1] - 1.0 - self.beta['x'])/self.alpha['x']
            y_value_upper = (self.goalY[1] + 0.2 - self.beta['y'])/self.alpha['y']
            y_value_lower = (self.goalY[1] - 0.2 - self.beta['y'])/self.alpha['y']
            coords[:, 5] = torch.zeros(self.numpoints).uniform_(x_value_lower, x_value_upper)
            coords[:, 6] = torch.zeros(self.numpoints).uniform_(y_value_lower, y_value_upper)
            coords[:, 8] = torch.ones(self.numpoints) * speed_value
        elif self.curriculum_version == 'v3':
            # Set velocity to zero, sample x and y anywhere
            speed_value = -self.beta['v']/self.alpha['v']
            coords[:, 8] = torch.ones(self.numpoints) * speed_value

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time; this currently assumes t \in [tMin, tMax]
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # Compute the initial value function
        # lx, gx, boundary_values = self.compute_IC(coords[:, 1:])
        lx, hx, boundary_values = self.compute_IC(coords[:, 1:])

        # print('Min and max value before normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
        # print('Min and max l(x) before normalization are %0.4f and %0.4f' %(min(lx), max(lx)))
        # print('Min and max g(x) before normalization are %0.4f and %0.4f' %(min(gx), max(gx)))
        # print('Min and max h(x) before normalization are %0.4f and %0.4f' %(min(hx), max(hx)))
        boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
        lx = (lx - self.mean)*self.norm_to/self.var
        # gx = (gx - mean)*norm_to/var
        hx = (hx - self.mean)*self.norm_to/self.var
        # print('Min and max value after normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
        # print('Min and max l(x) after normalization are %0.4f and %0.4f' %(min(lx), max(lx)))
        # print('Min and max g(x) after normalization are %0.4f and %0.4f' %(min(gx), max(gx)))
        # print('Min and max h(x) after normalization are %0.4f and %0.4f' %(min(hx), max(hx)))

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

        # return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask, 'lx': lx, 'gx': gx}
        return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask, 'lx': lx, 'hx': hx}

class ReachabilityNarrowMuSource(Dataset):
    def __init__(self, numpoints, pretrain, tMin, tMax, counter_start, counter_end, pretrain_iters, norm_scheme, clip_value_gradients,
                 gx_factor, speed_setting, sampling_bias_ratio, env_setting, ham_version, target_setting, collision_setting, 
                 curriculum_version, HJIVI_smoothing_setting, smoothing_exponent, num_src_samples):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        
        self.num_vehicles = 2
        self.num_states = 5 * self.num_vehicles + 1

        self.tMax = tMax
        self.tMin = tMin
        #self.mu = mu

        # Define state alphas and betas so that all coordinates are from [-1, 1]. 
        # The conversion rule is state' = (state - beta)/alpha. state' is in [-1, 1].
        # [state sequence: x_Ri, y_Ri, th_Ri, v_Ri, phi_Ri, ...]. Ri is the ith vehicle.
        self.alpha = {}
        self.beta = {}

        if speed_setting == 'high':
            self.alpha['x'] = 60.0
            self.alpha['y'] = 3.8
            self.alpha['th'] = 1.2*math.pi
            self.alpha['v'] = 7.0
            self.alpha['phi'] = 1.2*0.1*math.pi
            self.alpha['time'] = 10.0/self.tMax

            self.beta['x'] = 0.0
            self.beta['y'] = 0.0
            self.beta['th'] = 0.0
            self.beta['v'] = 6.0
            self.beta['phi'] = 0.0

            # Target positions
            self.goalX = np.array([36.0, -36.0])
            self.goalY = np.array([-1.4, 1.4])

            # State bounds
            self.vMin = 0.001
            self.vMax = 11.999
            self.phiMin = -0.1*math.pi + 0.001
            self.phiMax = 0.1*math.pi - 0.001

            # Control bounds
            self.aMin = -4.0
            self.aMax = 2.0
            self.psiMin = -0.1*math.pi
            self.psiMax = 0.1*math.pi

        elif speed_setting == 'low':
            self.alpha['x'] = 8.0
            self.alpha['y'] = 3.8
            self.alpha['th'] = 1.2*math.pi
            self.alpha['v'] = 3.0
            self.alpha['phi'] = 1.2*0.1*math.pi
            self.alpha['time'] = 6.0/self.tMax

            self.beta['x'] = 0.0
            self.beta['y'] = 0.0
            self.beta['th'] = 0.0
            self.beta['v'] = 2.0
            self.beta['phi'] = 0.0

            # Target positions
            self.goalX = np.array([6.0, -6.0])
            self.goalY = np.array([-1.4, 1.4])

            # State bounds
            self.vMin = 0.001
            self.vMax = 4.50
            self.phiMin = -0.1*math.pi + 0.001
            self.phiMax = 0.1*math.pi - 0.001

            # Control bounds
            self.aMin = -4.0
            self.aMax = 2.0
            self.psiMin = -0.1*math.pi
            self.psiMax = 0.1*math.pi

        elif speed_setting == 'medium':
            self.alpha['x'] = 8.0
            self.alpha['y'] = 3.8
            self.alpha['th'] = 1.2*math.pi
            self.alpha['v'] = 4.0
            self.alpha['phi'] = 1.2*0.3*math.pi
            self.alpha['time'] = 10.0/self.tMax   #####

            self.beta['x'] = 0.0
            self.beta['y'] = 0.0
            self.beta['th'] = 0.0
            self.beta['v'] = 3.0
            self.beta['phi'] = 0.0

            # Target positions
            self.goalX = np.array([6.0, -6.0])
            self.goalY = np.array([-1.4, 1.4])

            # State bounds
            self.vMin = 0.001
            self.vMax = 6.50
            self.phiMin = -0.3*math.pi + 0.001
            self.phiMax = 0.3*math.pi - 0.001

            # Control bounds
            self.aMin = -4.0
            self.aMax = 2.0
            self.psiMin = -0.3*math.pi
            self.psiMax = 0.3*math.pi

        elif speed_setting == 'medium_v2':
            self.alpha['x'] = 8.0
            self.alpha['y'] = 3.8
            self.alpha['th'] = 1.2*math.pi
            self.alpha['v'] = 4.0
            self.alpha['phi'] = 1.2*0.3*math.pi
            self.alpha['time'] = 10.0/self.tMax

            self.beta['x'] = 0.0
            self.beta['y'] = 0.0
            self.beta['th'] = 0.0
            self.beta['v'] = 3.0
            self.beta['phi'] = 0.0

            # Target positions
            self.goalX = np.array([6.0, -6.0])
            self.goalY = np.array([-1.4, 1.4])

            # State bounds
            self.vMin = 0.001
            self.vMax = 6.50
            self.phiMin = -0.3*math.pi + 0.001
            self.phiMax = 0.3*math.pi - 0.001

            # Control bounds
            self.aMin = -4.0
            self.aMax = 2.0
            self.psiMin = -3.0*math.pi
            self.psiMax = 3.0*math.pi

        elif speed_setting == 'medium_v3':
            self.alpha['x'] = 8.0
            self.alpha['y'] = 4.0
            self.alpha['th'] = 1.2*math.pi
            self.alpha['v'] = 4.0
            self.alpha['phi'] = 1.2*0.3*math.pi
            self.alpha['time'] = 10.0/self.tMax

            self.beta['x'] = 0.0
            self.beta['y'] = 0.0
            self.beta['th'] = 0.0
            self.beta['v'] = 3.0
            self.beta['phi'] = 0.0

            # Target positions
            self.goalX = np.array([6.0, -6.0])
            self.goalY = np.array([-2.0, 2.0])

            # State bounds
            self.vMin = 0.001
            self.vMax = 6.50
            self.phiMin = -0.3*math.pi + 0.001
            self.phiMax = 0.3*math.pi - 0.001

            # Control bounds
            self.aMin = -4.0
            self.aMax = 2.0
            self.psiMin = -3.0*math.pi
            self.psiMax = 3.0*math.pi

        else:
            raise NotImplementedError

        # How to weigh the obstacles
        self.gx_factor = gx_factor

        if env_setting == 'v1':
            # Vehicle diameter/length
            self.L = 2.0

            # Lower and upper curb positions (in the y direction)
            self.curb_positions = np.array([-2.8, 2.8])

            # Stranded car position
            self.stranded_car_pos = np.array([0.0, -1.4])

        elif env_setting == 'v2':
            # Vehicle diameter/length
            self.L = 2.0

            # Lower and upper curb positions (in the y direction)
            self.curb_positions = np.array([-2.8, 2.8])

            # Stranded car position
            self.stranded_car_pos = np.array([0.0, -1.8])

        elif env_setting == 'v3':
            # Vehicle diameter/length
            self.L = 2.0

            # Lower and upper curb positions (in the y direction)
            self.curb_positions = np.array([-4.0, 4.0])

            # Stranded car position
            self.stranded_car_pos = np.array([0.0, -2.0])

        else:
            raise NotImplementedError

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

        self.normalization_scheme = norm_scheme
        self.sampling_bias_ratio = sampling_bias_ratio
        self.clip_value_gradients = clip_value_gradients

        self.ham_version = ham_version
        self.target_setting = target_setting
        self.collision_setting = collision_setting
        self.curriculum_version = curriculum_version
        self.HJIVI_smoothing_setting = HJIVI_smoothing_setting
        self.smoothing_exponent = smoothing_exponent
        if self.normalization_scheme == 'hack1':
            self.norm_to = 0.02
            self.mean = 0.25 * self.alpha['x']
            self.var = 0.5 * self.alpha['x']
        else:
            raise NotImplementedError

    def compute_lx(self, state_coords_unnormalized):
        # Compute the target boundary condition given the unnormalized state coordinates.
        if self.target_setting in ['v1', 'v2', 'v4']:
            # Vehicle 1
            goal_tensor_R1 = torch.tensor([self.goalX[0], self.goalY[0]]).type(torch.FloatTensor)[None]
            dist_R1 = torch.norm(state_coords_unnormalized[:, 0:2] - goal_tensor_R1, dim=1, keepdim=True) - self.L
            # Vehicle 2
            goal_tensor_R2 = torch.tensor([self.goalX[1], self.goalY[1]]).type(torch.FloatTensor)[None]
            dist_R2 = torch.norm(state_coords_unnormalized[:, 5:7] - goal_tensor_R2, dim=1, keepdim=True) - self.L
            if self.target_setting == 'v1':
                return torch.max(dist_R1, dist_R2)
            elif self.target_setting == 'v2':
                return dist_R1
            elif self.target_setting == 'v4':
                sum_tensor = 0.5*(dist_R1 + dist_R2)
                max_tensor = 0.5*torch.max(dist_R1, dist_R2)
                sign_tensor = torch.sign(dist_R1 * dist_R2)
                return torch.where(sign_tensor < 0, max_tensor, sum_tensor)
        
        elif self.target_setting in ['v3']:
            # Have an infinitely extended target set above and below the center lane
            dist_R1 = torch.max((self.goalX[0] - 0.5*self.L) - state_coords_unnormalized[..., 0:1], state_coords_unnormalized[..., 1:2])
            dist_R2 = torch.max(state_coords_unnormalized[..., 5:6] - (self.goalX[1] + 0.5*self.L), -state_coords_unnormalized[..., 6:7])
            return torch.max(dist_R1, dist_R2)

        else:
            raise NotImplementedError

    def compute_gx(self, state_coords_unnormalized):
        # Compute the obstacle boundary condition given the unnormalized state coordinates. Negative inside the obstacle positive outside.
        # Distance from the lower curb
        dist_lc_R1 = state_coords_unnormalized[:, 1:2] - self.curb_positions[0] - 0.5*self.L
        dist_lc_R2 = state_coords_unnormalized[:, 6:7] - self.curb_positions[0] - 0.5*self.L
        dist_lc = torch.min(dist_lc_R1, dist_lc_R2)
        # dist_lc = torch.min(state_coords_unnormalized[:, 1:2] - self.curb_positions[0] - 0.5*self.L, state_coords_unnormalized[:, 6:7] - self.curb_positions[0] - 0.5*self.L)
        
        # Distance from the upper curb
        dist_uc_R1 = self.curb_positions[1] - state_coords_unnormalized[:, 1:2] - 0.5*self.L
        dist_uc_R2 = self.curb_positions[1] - state_coords_unnormalized[:, 6:7] - 0.5*self.L
        dist_uc = torch.min(dist_uc_R1, dist_uc_R2)
        # dist_uc = torch.min(self.curb_positions[1] - state_coords_unnormalized[:, 1:2] - 0.5*self.L, self.curb_positions[1] - state_coords_unnormalized[:, 6:7] - 0.5*self.L)
        
        # Distance from the stranded car
        stranded_car_pos = torch.tensor(self.stranded_car_pos*1.0).type(torch.FloatTensor)

        #dist_stranded_R1 = torch.norm(state_coords_unnormalized[:, 0:2] - stranded_car_pos, dim=1, keepdim=True) - self.L
        #dist_stranded_R2 = torch.norm(state_coords_unnormalized[:, 5:7] - stranded_car_pos, dim=1, keepdim=True) - self.L
        mu = state_coords_unnormalized[:, 10]
        le = 2 + mu               #radius*
        wi = 0.75 + 0.25 * mu     #radius*
        le_inflated = le + 0.5*self.L
        wi_inflated = wi + 0.5*self.L

        state_centered_R1 = state_coords_unnormalized[:, 0:2] - stranded_car_pos
        state_centered_R2 = state_coords_unnormalized[:, 5:7] - stranded_car_pos

        scale = le_inflated/wi_inflated
        dont_scale=torch.ones_like(scale)
        scaling_mat=torch.t(torch.stack([dont_scale,scale],dim=0))

        state_centered_R1=state_centered_R1*scaling_mat
        state_centered_R2=state_centered_R1*scaling_mat        

        le_inflated=torch.unsqueeze(le_inflated,0)   #le as column to be added with norm
        le_inflated=torch.t(le_inflated)  
        dist_stranded_R1 = torch.norm(state_centered_R1, dim=1, keepdim=True) - le_inflated
        dist_stranded_R2 = torch.norm(state_centered_R2, dim=1, keepdim=True) - le_inflated

        dist_stranded = torch.min(dist_stranded_R1, dist_stranded_R2)


        # Distance between the vehicles themselves
        dist_R1R2 = torch.norm(state_coords_unnormalized[:, 0:2] - state_coords_unnormalized[:, 5:7], dim=1, keepdim=True) - self.L

        if self.collision_setting == 'v1':
            return torch.min(torch.min(torch.min(dist_lc, dist_uc), dist_stranded), dist_R1R2) * self.gx_factor
        elif self.collision_setting == 'v2':
            return torch.min(torch.min(dist_lc, dist_uc), dist_stranded_R1) * self.gx_factor
        elif self.collision_setting == 'v3':
            return torch.min(dist_stranded, dist_R1R2) * self.gx_factor
        elif self.collision_setting == 'v4':
            return torch.min(torch.min(torch.min(dist_lc_R1, dist_uc_R1), dist_stranded_R1), dist_R1R2) * self.gx_factor
        else:
            raise NotImplementedError

    def compute_IC(self, state_coords):
        # Compute the boundary condition given the normalized state coordinates.
        state_coords_unnormalized = state_coords * 1.0
        state_coords_unnormalized[:, 0] = state_coords_unnormalized[:, 0] * self.alpha['x'] + self.beta['x']
        state_coords_unnormalized[:, 1] = state_coords_unnormalized[:, 1] * self.alpha['y'] + self.beta['y']
        state_coords_unnormalized[:, 2] = state_coords_unnormalized[:, 2] * self.alpha['th'] + self.beta['th']
        state_coords_unnormalized[:, 3] = state_coords_unnormalized[:, 3] * self.alpha['v'] + self.beta['v']
        state_coords_unnormalized[:, 4] = state_coords_unnormalized[:, 4] * self.alpha['phi'] + self.beta['phi']

        state_coords_unnormalized[:, 5] = state_coords_unnormalized[:, 5] * self.alpha['x'] + self.beta['x']
        state_coords_unnormalized[:, 6] = state_coords_unnormalized[:, 6] * self.alpha['y'] + self.beta['y']
        state_coords_unnormalized[:, 7] = state_coords_unnormalized[:, 7] * self.alpha['th'] + self.beta['th']
        state_coords_unnormalized[:, 8] = state_coords_unnormalized[:, 8] * self.alpha['v'] + self.beta['v']
        state_coords_unnormalized[:, 9] = state_coords_unnormalized[:, 9] * self.alpha['phi'] + self.beta['phi']

        state_coords_unnormalized[..., 10] = state_coords_unnormalized[..., 10] *1.0

        lx = self.compute_lx(state_coords_unnormalized)
        gx = self.compute_gx(state_coords_unnormalized)
        hx = -gx
        vx = torch.max(lx, hx)
        # return lx, gx, vx
        return lx, hx, vx

    def compute_vehicle_ham(self, x, dudx, return_opt_ctrl=False, Rindex=None):
        # Limit acceleration bounds based on the speed
        zero_tensor = torch.Tensor([0]).cuda()
        aMin = torch.ones_like(x[..., 3]) * self.aMin
        aMin = torch.where((x[..., 3] <= self.vMin), zero_tensor, aMin)
        aMax = torch.ones_like(x[..., 3]) * self.aMax
        aMax = torch.where((x[..., 3] >= self.vMax), zero_tensor, aMax)

        # Limit steering bounds based on the speed
        psiMin = torch.ones_like(x[..., 4]) * self.psiMin
        psiMin = torch.where((x[..., 4] <= self.phiMin), zero_tensor, psiMin)
        psiMax = torch.ones_like(x[..., 4]) * self.psiMax
        psiMax = torch.where((x[..., 4] >= self.phiMax), zero_tensor, psiMax)

        # Compute optimal control
        opt_acc = torch.where((dudx[..., 3] > 0), aMin, aMax)
        opt_psi = torch.where((dudx[..., 4] > 0), psiMin, psiMax)

        if (self.curriculum_version in ['v2', 'v3']) and (Rindex == 1):
            # Velocity can't change
            opt_acc = 0.0*opt_acc

        # Compute Hamiltonian
        ham_vehicle = x[..., 3] * torch.cos(x[..., 2]) * dudx[..., 0] + \
                      x[..., 3] * torch.sin(x[..., 2]) * dudx[..., 1] + \
                      x[..., 3] * torch.tan(x[..., 4]) * dudx[..., 2] / self.L + \
                      opt_acc * dudx[..., 3] + \
                      opt_psi * dudx[..., 4]

        # Freeze the Hamiltonian if required
        if self.ham_version == 'v2':
            # Check if vehicle is within the target point and if so, freeze the Hamiltonian selectively
            goal_tensor = torch.tensor([self.goalX[Rindex], self.goalY[Rindex]]).type(torch.FloatTensor)[None, None].cuda()
            dist_to_goal = torch.norm(x[..., 0:2] - goal_tensor, dim=-1) - 0.5*self.L
            ham_vehicle = torch.where((dist_to_goal <= 0), zero_tensor, ham_vehicle)
        
        if return_opt_ctrl:
            opt_ctrl = torch.cat((opt_acc, opt_psi), dim=1)
            return ham_vehicle, opt_ctrl
        else:
            return ham_vehicle

    def compute_overall_ham(self, x, dudx, return_components=False):
        alpha = self.alpha
        beta = self.beta

        # Scale the costates appropriately.
        dudx[..., 0] = dudx[..., 0] / alpha['x']
        dudx[..., 1] = dudx[..., 1] / alpha['y']
        dudx[..., 2] = dudx[..., 2] / alpha['th']
        dudx[..., 3] = dudx[..., 3] / alpha['v']
        dudx[..., 4] = dudx[..., 4] / alpha['phi']

        dudx[..., 5] = dudx[..., 5] / alpha['x']
        dudx[..., 6] = dudx[..., 6] / alpha['y']
        dudx[..., 7] = dudx[..., 7] / alpha['th']
        dudx[..., 8] = dudx[..., 8] / alpha['v']
        dudx[..., 9] = dudx[..., 9] / alpha['phi']

        dudx[..., 10] = dudx[..., 10] * 1.0

        # Scale for output normalization
        norm_to = 0.02
        mean = 0.25 * alpha['x']
        var = 0.5 * alpha['x']
        dudx = dudx * var/norm_to

        # Scale the states appropriately.
        x_unnormalized = x * 1.0
        x_unnormalized[..., 0] = x_unnormalized[..., 0] * alpha['x'] + beta['x']
        x_unnormalized[..., 1] = x_unnormalized[..., 1] * alpha['y'] + beta['y']
        x_unnormalized[..., 2] = x_unnormalized[..., 2] * alpha['th'] + beta['th']
        x_unnormalized[..., 3] = x_unnormalized[..., 3] * alpha['v'] + beta['v']
        x_unnormalized[..., 4] = x_unnormalized[..., 4] * alpha['phi'] + beta['phi']

        x_unnormalized[..., 5] = x_unnormalized[..., 5] * alpha['x'] + beta['x']
        x_unnormalized[..., 6] = x_unnormalized[..., 6] * alpha['y'] + beta['y']
        x_unnormalized[..., 7] = x_unnormalized[..., 7] * alpha['th'] + beta['th']
        x_unnormalized[..., 8] = x_unnormalized[..., 8] * alpha['v'] + beta['v']
        x_unnormalized[..., 9] = x_unnormalized[..., 9] * alpha['phi'] + beta['phi']

        x_unnormalized[..., 10] = x_unnormalized[..., 10] * 1.0

        # Compute the hamiltonian
        ham_R1 = self.compute_vehicle_ham(x_unnormalized[..., 0:5], dudx[..., 0:5], Rindex=0) 

        if self.curriculum_version == 'v4':
            ham_R2 = 0.0
        else:
            ham_R2 = self.compute_vehicle_ham(x_unnormalized[..., 5:], dudx[..., 5:], Rindex=1)

        ## Total Hamiltonian (take care of normalization again)
        ham_R1 = ham_R1 / (var/norm_to)
        ham_R2 = ham_R2 / (var/norm_to)
        ham_total = ham_R1 + ham_R2

        if return_components:
            return ham_total, ham_R1, ham_R2
        else:
            return ham_total

    def propagate_state(self, x, u, dt):
        alpha = self.alpha
        beta = self.beta

        x_next = torch.zeros_like(x)
        x_next[0] = x[3] * torch.cos(x[2])
        x_next[1] = x[3] * torch.sin(x[2])
        x_next[2] = x[3] * torch.tan(x[4]) / self.L
        x_next[3] = u[0]
        x_next[4] = u[1]

        x_next[5] = x[8] * torch.cos(x[7])
        x_next[6] = x[8] * torch.sin(x[7])
        x_next[7] = x[8] * torch.tan(x[9]) / self.L
        x_next[8] = u[2]
        x_next[9] = u[3]

        x_next[10] = 0

        return x + dt*x_next          

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

        if self.sampling_bias_ratio > 0.0:
            valid_upper_boundary =  (self.curb_positions[1] - 0.5*self.L - self.beta['y'])/self.alpha['y']
            num_samples = int(self.numpoints * self.sampling_bias_ratio)
            coords[-num_samples:, 1] = coords[-num_samples:, 1] * valid_upper_boundary
            coords[-num_samples:, 6] = coords[-num_samples:, 6] * valid_upper_boundary

        if self.curriculum_version in ['v2', 'v4']:
            # Set velocity to zero, only sample x and y around the goal state
            speed_value = -self.beta['v']/self.alpha['v']
            x_value_upper = (self.goalX[1] + 1.0 - self.beta['x'])/self.alpha['x']
            x_value_lower = (self.goalX[1] - 1.0 - self.beta['x'])/self.alpha['x']
            y_value_upper = (self.goalY[1] + 0.2 - self.beta['y'])/self.alpha['y']
            y_value_lower = (self.goalY[1] - 0.2 - self.beta['y'])/self.alpha['y']
            coords[:, 5] = torch.zeros(self.numpoints).uniform_(x_value_lower, x_value_upper)
            coords[:, 6] = torch.zeros(self.numpoints).uniform_(y_value_lower, y_value_upper)
            coords[:, 8] = torch.ones(self.numpoints) * speed_value
        elif self.curriculum_version == 'v3':
            # Set velocity to zero, sample x and y anywhere
            speed_value = -self.beta['v']/self.alpha['v']
            coords[:, 8] = torch.ones(self.numpoints) * speed_value

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time; this currently assumes t \in [tMin, tMax]
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # Compute the initial value function
        # lx, gx, boundary_values = self.compute_IC(coords[:, 1:])
        lx, hx, boundary_values = self.compute_IC(coords[:, 1:])

        # print('Min and max value before normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
        # print('Min and max l(x) before normalization are %0.4f and %0.4f' %(min(lx), max(lx)))
        # print('Min and max g(x) before normalization are %0.4f and %0.4f' %(min(gx), max(gx)))
        # print('Min and max h(x) before normalization are %0.4f and %0.4f' %(min(hx), max(hx)))
        boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
        lx = (lx - self.mean)*self.norm_to/self.var
        # gx = (gx - mean)*norm_to/var
        hx = (hx - self.mean)*self.norm_to/self.var
        # print('Min and max value after normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
        # print('Min and max l(x) after normalization are %0.4f and %0.4f' %(min(lx), max(lx)))
        # print('Min and max g(x) after normalization are %0.4f and %0.4f' %(min(gx), max(gx)))
        # print('Min and max h(x) after normalization are %0.4f and %0.4f' %(min(hx), max(hx)))

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

        # return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask, 'lx': lx, 'gx': gx}
        return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask, 'lx': lx, 'hx': hx}

class ReachabilityNarrowMuBugSource(Dataset): 
    def __init__(self, numpoints, pretrain=False, tMin=0.0, tMax=0.5, counter_start=0, counter_end=100e3, pretrain_iters=2000, norm_scheme='hack1', clip_value_gradients=False,
                 gx_factor=1.0, speed_setting='low', sampling_bias_ratio=0.0, env_setting='v1', ham_version='v1', target_setting='v1', collision_setting='v1', 
                 curriculum_version='v1', HJIVI_smoothing_setting='v1', smoothing_exponent=2.0, num_src_samples=1000, diffModel=False):
        super().__init__()
        torch.manual_seed(0)

        self.pretrain = pretrain
        self.numpoints = numpoints
        self.diffModel = diffModel
        
        self.num_vehicles = 2
        self.num_states = 5 * self.num_vehicles + 1  #+mu

        self.tMax = tMax
        self.tMin = tMin

        # Define state alphas and betas so that all coordinates are from [-1, 1]. 
        # The conversion rule is state' = (state - beta)/alpha. state' is in [-1, 1].
        # [state sequence: x_Ri, y_Ri, th_Ri, v_Ri, phi_Ri, ...]. Ri is the ith vehicle.
        self.alpha = {}
        self.beta = {}

        # if speed_setting == 'high':
        #     self.alpha['x'] = 60.0
        #     self.alpha['y'] = 3.8
        #     self.alpha['th'] = 1.2*math.pi
        #     self.alpha['v'] = 7.0
        #     self.alpha['phi'] = 1.2*0.1*math.pi
        #     self.alpha['time'] = 10.0/self.tMax

        #     self.beta['x'] = 0.0
        #     self.beta['y'] = 0.0
        #     self.beta['th'] = 0.0
        #     self.beta['v'] = 6.0
        #     self.beta['phi'] = 0.0

        #     # Target positions
        #     self.goalX = np.array([36.0, -36.0])
        #     self.goalY = np.array([-1.4, 1.4])

        #     # State bounds
        #     self.vMin = 0.001
        #     self.vMax = 11.999
        #     self.phiMin = -0.1*math.pi + 0.001
        #     self.phiMax = 0.1*math.pi - 0.001

        #     # Control bounds
        #     self.aMin = -4.0
        #     self.aMax = 2.0
        #     self.psiMin = -0.1*math.pi
        #     self.psiMax = 0.1*math.pi
        # elif speed_setting == 'low':
        #     self.alpha['x'] = 8.0
        #     self.alpha['y'] = 3.8
        #     self.alpha['th'] = 1.2*math.pi
        #     self.alpha['v'] = 3.0
        #     self.alpha['phi'] = 1.2*0.1*math.pi
        #     self.alpha['time'] = 6.0/self.tMax

        #     self.beta['x'] = 0.0
        #     self.beta['y'] = 0.0
        #     self.beta['th'] = 0.0
        #     self.beta['v'] = 2.0
        #     self.beta['phi'] = 0.0

        #     # Target positions
        #     self.goalX = np.array([6.0, -6.0])
        #     self.goalY = np.array([-1.4, 1.4])

        #     # State bounds
        #     self.vMin = 0.001
        #     self.vMax = 4.50
        #     self.phiMin = -0.1*math.pi + 0.001
        #     self.phiMax = 0.1*math.pi - 0.001

        #     # Control bounds
        #     self.aMin = -4.0
        #     self.aMax = 2.0
        #     self.psiMin = -0.1*math.pi
        #     self.psiMax = 0.1*math.pi
        # elif speed_setting == 'medium':
        #     self.alpha['x'] = 8.0
        #     self.alpha['y'] = 3.8
        #     self.alpha['th'] = 1.2*math.pi
        #     self.alpha['v'] = 4.0
        #     self.alpha['phi'] = 1.2*0.3*math.pi
        #     self.alpha['time'] = 10.0/self.tMax

        #     self.beta['x'] = 0.0
        #     self.beta['y'] = 0.0
        #     self.beta['th'] = 0.0
        #     self.beta['v'] = 3.0
        #     self.beta['phi'] = 0.0

        #     # Target positions
        #     self.goalX = np.array([6.0, -6.0])
        #     self.goalY = np.array([-1.4, 1.4])

        #     # State bounds
        #     self.vMin = 0.001
        #     self.vMax = 6.50
        #     self.phiMin = -0.3*math.pi + 0.001
        #     self.phiMax = 0.3*math.pi - 0.001

        #     # Control bounds
        #     self.aMin = -4.0
        #     self.aMax = 2.0
        #     self.psiMin = -0.3*math.pi
        #     self.psiMax = 0.3*math.pi
        if speed_setting == 'medium_v2':
            self.alpha['x'] = 8.0
            self.alpha['y'] = 3.8
            self.alpha['th'] = 1.2*math.pi
            self.alpha['v'] = 4.0
            self.alpha['phi'] = 1.2*0.3*math.pi
            self.alpha['time'] = 1.0

            self.beta['x'] = 0.0
            self.beta['y'] = 0.0
            self.beta['th'] = 0.0
            self.beta['v'] = 3.0
            self.beta['phi'] = 0.0

            # Target positions
            self.goalX = np.array([6.0, -6.0])
            self.goalY = np.array([-1.4, 1.4])

            # State bounds
            self.vMin = 0.001
            self.vMax = 6.50
            self.phiMin = -0.3*math.pi + 0.001 #0.95
            self.phiMax = 0.3*math.pi - 0.001

            # Control bounds
            self.aMin = -4.0
            self.aMax = 2.0
            self.psiMin = -3.0*math.pi
            self.psiMax = 3.0*math.pi
        # elif speed_setting == 'medium_v3':
        #     self.alpha['x'] = 8.0
        #     self.alpha['y'] = 4.0
        #     self.alpha['th'] = 1.2*math.pi
        #     self.alpha['v'] = 4.0
        #     self.alpha['phi'] = 1.2*0.3*math.pi
        #     self.alpha['time'] = 10.0/self.tMax

        #     self.beta['x'] = 0.0
        #     self.beta['y'] = 0.0
        #     self.beta['th'] = 0.0
        #     self.beta['v'] = 3.0
        #     self.beta['phi'] = 0.0

        #     # Target positions
        #     self.goalX = np.array([6.0, -6.0])
        #     self.goalY = np.array([-2.0, 2.0])

        #     # State bounds
        #     self.vMin = 0.001
        #     self.vMax = 6.50
        #     self.phiMin = -0.3*math.pi + 0.001
        #     self.phiMax = 0.3*math.pi - 0.001

        #     # Control bounds
        #     self.aMin = -4.0
        #     self.aMax = 2.0
        #     self.psiMin = -3.0*math.pi
        #     self.psiMax = 3.0*math.pi
        else:
            raise NotImplementedError

        # How to weigh the obstacles
        self.gx_factor = gx_factor

        # if env_setting == 'v1':
        #     # Vehicle diameter/length
        #     self.L = 2.0
        #     self.le = 3.0
        #     self.wi = 1.0

        #     # Lower and upper curb positions (in the y direction)
        #     self.curb_positions = np.array([-2.8, 2.8])

        #     # Stranded car position
        #     self.stranded_car_pos = np.array([0.0, -1.4])
        if env_setting == 'v2':
            # Vehicle diameter/length
            self.L = 2.0
            
            # Lower and upper curb positions (in the y direction)
            self.curb_positions = np.array([-2.8, 2.8])

            # Stranded car position
            self.stranded_car_pos = np.array([0.0, -1.8])
        # elif env_setting == 'v3':
        #     # Vehicle diameter/length
        #     self.L = 2.0

        #     # Lower and upper curb positions (in the y direction)
        #     self.curb_positions = np.array([-4.0, 4.0])

        #     # Stranded car position
        #     self.stranded_car_pos = np.array([0.0, -2.0])
        else:
            raise NotImplementedError

        self.N_src_samples = num_src_samples

        self.pretrain_counter = 0
        self.counter = counter_start
        self.pretrain_iters = pretrain_iters
        self.full_count = counter_end 

        self.normalization_scheme = norm_scheme
        self.sampling_bias_ratio = sampling_bias_ratio
        self.clip_value_gradients = clip_value_gradients

        self.ham_version = ham_version
        self.target_setting = target_setting
        self.collision_setting = collision_setting
        self.curriculum_version = curriculum_version
        self.HJIVI_smoothing_setting = HJIVI_smoothing_setting
        self.smoothing_exponent = smoothing_exponent

        if self.normalization_scheme == 'hack1':
            self.norm_to = 0.02
            self.mean = 0.25 * self.alpha['x']
            self.var = 0.5 * self.alpha['x']
        else:
            raise NotImplementedError

    def compute_lx(self, state_coords_unnormalized):
        # Compute the target boundary condition given the unnormalized state coordinates.
        if self.target_setting in ['v1', 'v2', 'v4']:
            # Vehicle 1
            goal_tensor_R1 = torch.tensor([self.goalX[0], self.goalY[0]]).type(torch.FloatTensor)[None]
            dist_R1 = torch.norm(state_coords_unnormalized[:, 0:2] - goal_tensor_R1, dim=1, keepdim=True) - self.L
            # Vehicle 2
            goal_tensor_R2 = torch.tensor([self.goalX[1], self.goalY[1]]).type(torch.FloatTensor)[None]
            dist_R2 = torch.norm(state_coords_unnormalized[:, 5:7] - goal_tensor_R2, dim=1, keepdim=True) - self.L
            if self.target_setting == 'v1':
                return torch.max(dist_R1, dist_R2)
            elif self.target_setting == 'v2':
                return dist_R1
            elif self.target_setting == 'v4':
                sum_tensor = 0.5*(dist_R1 + dist_R2)
                max_tensor = 0.5*torch.max(dist_R1, dist_R2)
                sign_tensor = torch.sign(dist_R1 * dist_R2)
                return torch.where(sign_tensor < 0, max_tensor, sum_tensor)
        
        elif self.target_setting in ['v3']:
            # Have an infinitely extended target set above and below the center lane
            dist_R1 = torch.max((self.goalX[0] - 0.5*self.L) - state_coords_unnormalized[..., 0:1], state_coords_unnormalized[..., 1:2])
            dist_R2 = torch.max(state_coords_unnormalized[..., 5:6] - (self.goalX[1] + 0.5*self.L), -state_coords_unnormalized[..., 6:7])
            return torch.max(dist_R1, dist_R2)

        else:
            raise NotImplementedError

    def compute_gx(self, state_coords_unnormalized):
        # Compute the obstacle boundary condition given the unnormalized state coordinates. Negative inside the obstacle positive outside.
        # Distance from the lower curb
        dist_lc_R1 = state_coords_unnormalized[:, 1:2] - self.curb_positions[0] - 0.5*self.L
        dist_lc_R2 = state_coords_unnormalized[:, 6:7] - self.curb_positions[0] - 0.5*self.L
        dist_lc = torch.min(dist_lc_R1, dist_lc_R2)
        # dist_lc = torch.min(state_coords_unnormalized[:, 1:2] - self.curb_positions[0] - 0.5*self.L, state_coords_unnormalized[:, 6:7] - self.curb_positions[0] - 0.5*self.L)
        
        # Distance from the upper curb
        dist_uc_R1 = self.curb_positions[1] - state_coords_unnormalized[:, 1:2] - 0.5*self.L
        dist_uc_R2 = self.curb_positions[1] - state_coords_unnormalized[:, 6:7] - 0.5*self.L
        dist_uc = torch.min(dist_uc_R1, dist_uc_R2)
        # dist_uc = torch.min(self.curb_positions[1] - state_coords_unnormalized[:, 1:2] - 0.5*self.L, self.curb_positions[1] - state_coords_unnormalized[:, 6:7] - 0.5*self.L)
        
        # Distance from the stranded car
        stranded_car_pos = torch.tensor(self.stranded_car_pos*1.0).type(torch.FloatTensor)
        # dist_stranded_R1 = torch.norm(state_coords_unnormalized[:, 0:2] - stranded_car_pos, dim=1, keepdim=True) - self.L
        # dist_stranded_R2 = torch.norm(state_coords_unnormalized[:, 5:7] - stranded_car_pos, dim=1, keepdim=True) - self.L
        mu = state_coords_unnormalized[:, 10]
        le = 2 + mu               #radius*
        wi = 0.75 + 0.25 * mu     #radius*
        le_inflated = le + 0.5*self.L
        wi_inflated = wi + 0.5*self.L

        state_centered_R1 = state_coords_unnormalized[:, 0:2] - stranded_car_pos
        state_centered_R2 = state_coords_unnormalized[:, 5:7] - stranded_car_pos

        scale = le_inflated/wi_inflated
        dont_scale=torch.ones_like(scale)
        scaling_mat=torch.t(torch.stack([dont_scale,scale],dim=0))

        state_centered_R1=state_centered_R1*scaling_mat
        state_centered_R2=state_centered_R1*scaling_mat        

        le_inflated=torch.unsqueeze(le_inflated,0)   #le as column to be added with norm
        le_inflated=torch.t(le_inflated)  
        dist_stranded_R1 = torch.norm(state_centered_R1, dim=1, keepdim=True) - le_inflated
        dist_stranded_R2 = torch.norm(state_centered_R2, dim=1, keepdim=True) - le_inflated

        dist_stranded = torch.min(dist_stranded_R1, dist_stranded_R2)
        
        # Distance between the vehicles themselves
        dist_R1R2 = torch.norm(state_coords_unnormalized[:, 0:2] - state_coords_unnormalized[:, 5:7], dim=1, keepdim=True) - self.L

        if self.collision_setting == 'v1':
            return torch.min(torch.min(torch.min(dist_lc, dist_uc), dist_stranded), dist_R1R2) * self.gx_factor
        elif self.collision_setting == 'v2':
            return torch.min(torch.min(dist_lc, dist_uc), dist_stranded_R1) * self.gx_factor
        elif self.collision_setting == 'v3':
            return torch.min(dist_stranded, dist_R1R2) * self.gx_factor
        elif self.collision_setting == 'v4':
            return torch.min(torch.min(torch.min(dist_lc_R1, dist_uc_R1), dist_stranded_R1), dist_R1R2) * self.gx_factor
        else:
            raise NotImplementedError

    def compute_IC(self, state_coords):
        # Compute the boundary condition given the normalized state coordinates.
        state_coords_unnormalized = state_coords * 1.0
        state_coords_unnormalized[:, 0] = state_coords_unnormalized[:, 0] * self.alpha['x'] + self.beta['x']
        state_coords_unnormalized[:, 1] = state_coords_unnormalized[:, 1] * self.alpha['y'] + self.beta['y']
        state_coords_unnormalized[:, 2] = state_coords_unnormalized[:, 2] * self.alpha['th'] + self.beta['th']
        state_coords_unnormalized[:, 3] = state_coords_unnormalized[:, 3] * self.alpha['v'] + self.beta['v']
        state_coords_unnormalized[:, 4] = state_coords_unnormalized[:, 4] * self.alpha['phi'] + self.beta['phi']

        state_coords_unnormalized[:, 5] = state_coords_unnormalized[:, 5] * self.alpha['x'] + self.beta['x']
        state_coords_unnormalized[:, 6] = state_coords_unnormalized[:, 6] * self.alpha['y'] + self.beta['y']
        state_coords_unnormalized[:, 7] = state_coords_unnormalized[:, 7] * self.alpha['th'] + self.beta['th']
        state_coords_unnormalized[:, 8] = state_coords_unnormalized[:, 8] * self.alpha['v'] + self.beta['v']
        state_coords_unnormalized[:, 9] = state_coords_unnormalized[:, 9] * self.alpha['phi'] + self.beta['phi']

        state_coords_unnormalized[..., 10] = state_coords_unnormalized[..., 10] *1.0

        lx = self.compute_lx(state_coords_unnormalized)
        gx = self.compute_gx(state_coords_unnormalized)
        hx = gx * (-1)

        # combined = torch.cat((lx.unsqueeze(1), hx.unsqueeze(1)), dim=1)
        # vx=torch.max(combined, dim=1)[0]
        vx = torch.max(lx, hx)
        # return lx, gx, vx
        return lx, gx, vx

    def compute_vehicle_ham(self, x, dudx, return_opt_ctrl=False, Rindex=None):
        # Limit acceleration bounds based on the speed
        zero_tensor = torch.Tensor([0]).cuda()
        aMin = torch.ones_like(x[..., 3]) * self.aMin
        aMin = torch.where((x[..., 3] <= self.vMin), zero_tensor, aMin)
        aMax = torch.ones_like(x[..., 3]) * self.aMax
        aMax = torch.where((x[..., 3] >= self.vMax), zero_tensor, aMax)

        # Limit steering bounds based on the speed
        psiMin = torch.ones_like(x[..., 4]) * self.psiMin
        psiMin = torch.where((x[..., 4] <= self.phiMin), zero_tensor, psiMin)
        psiMax = torch.ones_like(x[..., 4]) * self.psiMax
        psiMax = torch.where((x[..., 4] >= self.phiMax), zero_tensor, psiMax)

        # Compute optimal control
        opt_acc = torch.where((dudx[..., 3] > 0), aMin, aMax)
        opt_psi = torch.where((dudx[..., 4] > 0), psiMin, psiMax)

        if (self.curriculum_version in ['v2', 'v3']) and (Rindex == 1):
            # Velocity can't change
            opt_acc = 0.0*opt_acc

        # Compute Hamiltonian
        ham_vehicle = x[..., 3] * torch.cos(x[..., 2]) * dudx[..., 0] + \
                      x[..., 3] * torch.sin(x[..., 2]) * dudx[..., 1] + \
                      x[..., 3] * torch.tan(x[..., 4]) * dudx[..., 2] / self.L + \
                      opt_acc * dudx[..., 3] + \
                      opt_psi * dudx[..., 4]

        # Freeze the Hamiltonian if required
        if self.ham_version == 'v2':
            # Check if vehicle is within the target point and if so, freeze the Hamiltonian selectively
            goal_tensor = torch.tensor([self.goalX[Rindex], self.goalY[Rindex]]).type(torch.FloatTensor)[None, None].cuda()
            dist_to_goal = torch.norm(x[..., 0:2] - goal_tensor, dim=-1) - 0.5*self.L
            ham_vehicle = torch.where((dist_to_goal <= 0), zero_tensor, ham_vehicle)
        
        if return_opt_ctrl:
            opt_ctrl = torch.cat((opt_acc, opt_psi), dim=1)
            return ham_vehicle, opt_ctrl
        else:
            return ham_vehicle

    def compute_overall_ham(self, x, dudx, return_components=False):
        alpha = self.alpha
        beta = self.beta

        # Scale the costates appropriately.
        dudx[..., 0] = dudx[..., 0] / alpha['x']
        dudx[..., 1] = dudx[..., 1] / alpha['y']
        dudx[..., 2] = dudx[..., 2] / alpha['th']
        dudx[..., 3] = dudx[..., 3] / alpha['v']
        dudx[..., 4] = dudx[..., 4] / alpha['phi']

        dudx[..., 5] = dudx[..., 5] / alpha['x']
        dudx[..., 6] = dudx[..., 6] / alpha['y']
        dudx[..., 7] = dudx[..., 7] / alpha['th']
        dudx[..., 8] = dudx[..., 8] / alpha['v']
        dudx[..., 9] = dudx[..., 9] / alpha['phi']

        dudx[..., 10] = dudx[..., 10] * 1.0

        # Scale for output normalization
        norm_to = 0.02
        mean = 0.25 * alpha['x']
        var = 0.5 * alpha['x']
        dudx = dudx * var/norm_to

        # Scale the states appropriately.
        x_unnormalized = x * 1.0
        x_unnormalized[..., 0] = x_unnormalized[..., 0] * alpha['x'] + beta['x']
        x_unnormalized[..., 1] = x_unnormalized[..., 1] * alpha['y'] + beta['y']
        x_unnormalized[..., 2] = x_unnormalized[..., 2] * alpha['th'] + beta['th']
        x_unnormalized[..., 3] = x_unnormalized[..., 3] * alpha['v'] + beta['v']
        x_unnormalized[..., 4] = x_unnormalized[..., 4] * alpha['phi'] + beta['phi']

        x_unnormalized[..., 5] = x_unnormalized[..., 5] * alpha['x'] + beta['x']
        x_unnormalized[..., 6] = x_unnormalized[..., 6] * alpha['y'] + beta['y']
        x_unnormalized[..., 7] = x_unnormalized[..., 7] * alpha['th'] + beta['th']
        x_unnormalized[..., 8] = x_unnormalized[..., 8] * alpha['v'] + beta['v']
        x_unnormalized[..., 9] = x_unnormalized[..., 9] * alpha['phi'] + beta['phi']
        x_unnormalized[..., 10] = x_unnormalized[..., 10] * 1.0

        # Compute the hamiltonian
        ham_R1 = self.compute_vehicle_ham(x_unnormalized[..., 0:5], dudx[..., 0:5], Rindex=0) 

        if self.curriculum_version == 'v4':
            ham_R2 = 0.0
        else:
            ham_R2 = self.compute_vehicle_ham(x_unnormalized[..., 5:], dudx[..., 5:], Rindex=1)

        ## Total Hamiltonian (take care of normalization again)
        ham_R1 = ham_R1 / (var/norm_to)
        ham_R2 = ham_R2 / (var/norm_to)
        ham_total = ham_R1 + ham_R2

        if return_components:
            return ham_total, ham_R1, ham_R2
        else:
            return ham_total

    def propagate_state(self, x, u, dt):
        alpha = self.alpha
        beta = self.beta

        x_next = torch.zeros_like(x)
        x_next[0] = x[3] * torch.cos(x[2])
        x_next[1] = x[3] * torch.sin(x[2])
        x_next[2] = x[3] * torch.tan(x[4]) / self.L
        x_next[3] = u[0]
        x_next[4] = u[1]

        x_next[5] = x[8] * torch.cos(x[7])
        x_next[6] = x[8] * torch.sin(x[7])
        x_next[7] = x[8] * torch.tan(x[9]) / self.L
        x_next[8] = u[2]
        x_next[9] = u[3]

        x_next[10] = 0 
        return x + dt*x_next          

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        start_time = 0.  # time to apply  initial conditions

        # uniformly sample domain and include coordinates where source is non-zero 
        coords = torch.zeros(self.numpoints, self.num_states).uniform_(-1, 1)

        if self.sampling_bias_ratio > 0.0:
            valid_upper_boundary =  (self.curb_positions[1] - 0.5*self.L - self.beta['y'])/self.alpha['y']
            num_samples = int(self.numpoints * self.sampling_bias_ratio)
            coords[-num_samples:, 1] = coords[-num_samples:, 1] * valid_upper_boundary
            coords[-num_samples:, 6] = coords[-num_samples:, 6] * valid_upper_boundary

        if self.curriculum_version in ['v2', 'v4']:
            # Set velocity to zero, only sample x and y around the goal state
            speed_value = -self.beta['v']/self.alpha['v']
            x_value_upper = (self.goalX[1] + 1.0 - self.beta['x'])/self.alpha['x']
            x_value_lower = (self.goalX[1] - 1.0 - self.beta['x'])/self.alpha['x']
            y_value_upper = (self.goalY[1] + 0.2 - self.beta['y'])/self.alpha['y']
            y_value_lower = (self.goalY[1] - 0.2 - self.beta['y'])/self.alpha['y']
            coords[:, 5] = torch.zeros(self.numpoints).uniform_(x_value_lower, x_value_upper)
            coords[:, 6] = torch.zeros(self.numpoints).uniform_(y_value_lower, y_value_upper)
            coords[:, 8] = torch.ones(self.numpoints) * speed_value
        elif self.curriculum_version == 'v3':
            # Set velocity to zero, sample x and y anywhere
            speed_value = -self.beta['v']/self.alpha['v']
            coords[:, 8] = torch.ones(self.numpoints) * speed_value

        if self.pretrain:
            # only sample in time around the initial condition
            time = torch.ones(self.numpoints, 1) * start_time
            coords = torch.cat((time, coords), dim=1)
        else:
            # slowly grow time values from start time; this currently assumes t \in [tMin, tMax]
            time = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.full_count))
            coords = torch.cat((time, coords), dim=1)

            # make sure we always have training samples at the initial time
            coords[-self.N_src_samples:, 0] = start_time

        # Compute the initial value function
        if self.diffModel:
            coords_var = coords.clone().detach().requires_grad_(True)
            #coords_var = torch.tensor(coords.clone(), requires_grad=True)
            lx, hx, boundary_values = self.compute_IC(coords_var[:, 1:])
        else:
            # lx, gx, boundary_values = self.compute_IC(coords[:, 1:])
            lx, hx, boundary_values = self.compute_IC(coords[:, 1:])

        # Normalize the value function
        # print('Min and max value before normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
        # print('Min and max l(x) before normalization are %0.4f and %0.4f' %(min(lx), max(lx)))
        # print('Min and max g(x) before normalization are %0.4f and %0.4f' %(min(gx), max(gx)))
        # print('Min and max h(x) before normalization are %0.4f and %0.4f' %(min(hx), max(hx)))
        boundary_values = (boundary_values - self.mean)*self.norm_to/self.var
        lx = (lx - self.mean)*self.norm_to/self.var
        # gx = (gx - self.mean)*self.norm_to/self.var
        hx = (hx - self.mean)*self.norm_to/self.var
        # print('Min and max value after normalization are %0.4f and %0.4f' %(min(boundary_values), max(boundary_values)))
        # print('Min and max l(x) after normalization are %0.4f and %0.4f' %(min(lx), max(lx)))
        # print('Min and max g(x) after normalization are %0.4f and %0.4f' %(min(gx), max(gx)))
        # print('Min and max h(x) after normalization are %0.4f and %0.4f' %(min(hx), max(hx)))

        # Compute the boundary value fuction gradient if required
        if self.diffModel:
            boundary_valfunc_grads = diff_operators.gradient(boundary_values, coords_var)[..., 1:]
        #kek replaces undefined diff_operators
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
            return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask, 'boundary_valfunc_grads': boundary_valfunc_grads, 'lx': lx, 'hx': hx}
        else:
            # return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask, 'lx': lx, 'gx': gx}
            return {'coords': coords}, {'source_boundary_values': boundary_values, 'dirichlet_mask': dirichlet_mask, 'lx': lx, 'hx': hx}

