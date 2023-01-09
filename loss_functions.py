import torch
import torch.nn.functional as F

import diff_operators
import modules, utils

import math
import numpy as np


def initialize_hji_air3D(dataset, minWith):
    # Initialize the loss function for the air3D problem
    # The dynamics parameters
    velocity = dataset.velocity
    omega_max = dataset.omega_max
    alpha_angle = dataset.alpha_angle

    def hji_air3D(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 4)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        du, status = diff_operators.jacobian(y, x)
        dudt = du[..., 0, 0]
        dudx = du[..., 0, 1:]

        x_theta = x[..., 3] * 1.0

        # Scale the costate for theta appropriately to align with the range of [-pi, pi]
        dudx[..., 2] = dudx[..., 2] / alpha_angle
        # Scale the coordinates
        x_theta = alpha_angle * x_theta

        # Air3D dynamics
        # \dot x    = -v_a + v_b \cos \psi + a y
        # \dot y    = v_b \sin \psi - a x
        # \dot \psi = b - a

        # Compute the hamiltonian for the ego vehicle
        ham = omega_max * torch.abs(dudx[..., 0] * x[..., 2] - dudx[..., 1] * x[..., 1] - dudx[..., 2])  # Control component
        ham = ham - omega_max * torch.abs(dudx[..., 2])  # Disturbance component
        ham = ham + (velocity * (torch.cos(x_theta) - 1.0) * dudx[..., 0]) + (velocity * torch.sin(x_theta) * dudx[..., 1])  # Constant component

        # If we are computing BRT then take min with zero
        if minWith == 'zero':
            ham = torch.clamp(ham, max=0.0)

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            diff_constraint_hom = dudt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], y - source_boundary_values)

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 15e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_air3D

def initialize_hji_air3Dp1D(dataset, minWith):
    # Initialize the loss function for the air3D problem
    # The dynamics parameters
    velocity = dataset.velocity
    omega_max = dataset.omega_max
    alpha_angle = dataset.alpha_angle

    def hji_air3Dp1D(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 3+1+1)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        du, status = diff_operators.jacobian(y, x)
        dudt = du[..., 0, 0]
        dudx = du[..., 0, 1:]

        x_theta = x[..., 3] * 1.0
        x_omega_a = x[..., 4] * 1.0

        # Scale the costate for theta appropriately to align with the range of [-pi, pi]
        dudx[..., 2] = dudx[..., 2] / alpha_angle
        # Scale the coordinates
        x_theta = alpha_angle * x_theta
        x_omega_a = (x_omega_a * 1.5) + 4  #(2.5 to 5.5)

        # Air3Dp1D dynamics
        # \dot x    = -v_a + v_b \cos \psi + a y
        # \dot y    = v_b \sin \psi - a x
        # \dot \psi = b - a
        # \dot \omega_a  = 0

        # Compute the hamiltonian               #p1*y-p2*x-p3
        ham = x_omega_a * torch.abs(dudx[..., 0] * x[..., 2] - dudx[..., 1] * x[..., 1] - dudx[..., 2])  # Control component
        ham = ham - omega_max * torch.abs(dudx[..., 2])  # Disturbance component
        ham = ham + (velocity * (torch.cos(x_theta) - 1.0) * dudx[..., 0]) + (velocity * torch.sin(x_theta) * dudx[..., 1])  # Constant component

        # If we are computing BRT then take min with zero
        if minWith == 'zero':
            ham = torch.clamp(ham, max=0.0)

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            diff_constraint_hom = dudt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], y - source_boundary_values)

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask] 

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 15e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_air3Dp1D

def initialize_hji_MultiVehicleCollisionNE(dataset, minWith,diffModel):
    # Initialize the loss function for the multi-vehicle collision avoidance problem
    # The dynamics parameters
    velocity = dataset.velocity
    omega_max = dataset.omega_max
    numEvaders = dataset.numEvaders
    num_pos_states = dataset.num_pos_states
    alpha_angle = dataset.alpha_angle
    alpha_time = dataset.alpha_time

    def hji_MultiVehicleCollision(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 4)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            du, status = diff_operators.jacobian(y, x)
            dudt = du[..., 0, 0]
            dudx = du[..., 0, 1:]

            if diffModel:
                # Compute the spatial gradient of lx
                dudx = dudx + gt['lx_grads']
                diff_from_lx = y
            else:
                diff_from_lx = y - source_boundary_values

            # Scale the costate for theta appropriately to align with the range of [-pi, pi]
            dudx[..., num_pos_states:] = dudx[..., num_pos_states:] / alpha_angle

            # Compute the hamiltonian for the ego vehicle
            ham = velocity*(torch.cos(alpha_angle*x[..., num_pos_states+1]) * dudx[..., 0] + torch.sin(alpha_angle*x[..., num_pos_states+1]) * dudx[..., 1]) - omega_max * torch.abs(dudx[..., num_pos_states])

            # Hamiltonian effect due to other vehicles
            for i in range(numEvaders):
                theta_index = num_pos_states+1+i+1
                xcostate_index = 2*(i+1)
                ycostate_index = 2*(i+1) + 1
                thetacostate_index = num_pos_states+1+i
                ham_local = velocity*(torch.cos(alpha_angle*x[..., theta_index]) * dudx[..., xcostate_index] + torch.sin(alpha_angle*x[..., theta_index]) * dudx[..., ycostate_index]) + omega_max * torch.abs(dudx[..., thetacostate_index])
                ham = ham + ham_local

            # Effect of time factor
            ham = ham * alpha_time

            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dudt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], diff_from_lx)

        # Boundary loss
        if diffModel:
            dirichlet = y[dirichlet_mask]
        else:
            dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 15e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_MultiVehicleCollision

def initialize_hji_MultiVehicleCollisionBeta(dataset, minWith, diffModel):
    # Initialize the loss function for the multi-vehicle collision avoidance problem
    # The dynamics parameters
    velocity = dataset.velocity
    omega_max = dataset.omega_max
    numEvaders = dataset.numEvaders
    num_pos_states = dataset.num_pos_states
    alpha_angle = dataset.alpha_angle
    alpha_time = dataset.alpha_time

    def hji_MultiVehicleCollisionBeta(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 4)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            du, status = diff_operators.jacobian(y, x)
            dudt = du[..., 0, 0]
            dudx = du[..., 0, 1:]

            if diffModel:
                # Compute the spatial gradient of lx
                dudx = dudx + gt['lx_grads']
                diff_from_lx = y
            else:
                diff_from_lx = y - source_boundary_values

            # Scale the costate for theta appropriately to align with the range of [-pi, pi]
            dudx[..., num_pos_states:] = dudx[..., num_pos_states:] / alpha_angle

            # Compute the hamiltonian for the ego vehicle
            ham = velocity*(torch.cos(alpha_angle*x[..., num_pos_states+1]) * dudx[..., 0] + torch.sin(alpha_angle*x[..., num_pos_states+1]) * dudx[..., 1]) - omega_max * torch.abs(dudx[..., num_pos_states])

            # Hamiltonian effect due to other vehicles
            for i in range(numEvaders):
                theta_index = num_pos_states+1+i+1
                xcostate_index = 2*(i+1)
                ycostate_index = 2*(i+1) + 1
                thetacostate_index = num_pos_states+1+i
                ham_local = velocity*(torch.cos(alpha_angle*x[..., theta_index]) * dudx[..., xcostate_index] + torch.sin(alpha_angle*x[..., theta_index]) * dudx[..., ycostate_index]) + omega_max * torch.abs(dudx[..., thetacostate_index])
                ham = ham + ham_local

            # Effect of time factor
            ham = ham * alpha_time

            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dudt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], diff_from_lx)

        # Boundary loss
        if diffModel:
            dirichlet = y[dirichlet_mask]
        else:
            dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 15e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_MultiVehicleCollisionBeta

def initialize_hji_drone2D(dataset, minWith):
    # Initialize the loss function for the drone2D problem
    # The dynamics parameters
    velocity = dataset.velocity
    theta_range = dataset.theta_range
    disturbance_mag = dataset.disturbance_mag
    position_alpha = dataset.position_alpha


    def hji_drone2D(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 4)????
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)????
        dirichlet_mask = gt['dirichlet_mask']
        
        batch_size = x.shape[1]   #size on the second dimention
    
        # Drone2D dynamics
        # \dot x    = v*cos(u) + d_x
        # \dot y    = v*sin(u) + d_y
        # abs(u)<theta_range
        # norm(d_x,d_y)=disturbance_mag

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            du, status = diff_operators.jacobian(y, x)
            dudt = du[..., 0, 0]
            dudx = du[..., 0, 1:]

            x_x = x[..., 1] * 1.0
            x_y = x[..., 2] * 1.0

            a=torch.atan2(dudx[..., 1], dudx[..., 0])
            # Compute the hamiltonian for the Drone2D
            ham = velocity*torch.norm(dudx, dim=2)*torch.where((torch.abs(a)<=theta_range), torch.tensor([1.0]), torch.max(torch.cos(a-theta_range),torch.cos(a+theta_range)))
            ham = ham - disturbance_mag*torch.norm(dudx,dim=2)

            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dudt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], y - source_boundary_values)

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 15e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_drone2D



#experimental models

def initialize_hji_drone3Dp1D(dataset, minWith):
    # Initialize the loss function for the air3D problem
    # The dynamics parameters
    velocity = dataset.velocity
    omega_max = dataset.omega_max
    alpha_angle = dataset.alpha_angle

    def hji_drone3Dp1D(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        lx = gt['lx']        
        hx = gt['hx']  
        x = model_output['model_in']  # (meta_batch_size, num_points, 3+1+1)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        du, status = diff_operators.jacobian(y, x)
        dudt = du[..., 0, 0]
        dudx = du[..., 0, 1:]

        x_theta = x[..., 3] * 1.0
        x_dbar = x[..., 4] * 1.0

        # Scale the costate for theta appropriately to align with the range of [-pi, pi]
        dudx[..., 2] = dudx[..., 2] / alpha_angle
        # Scale the coordinates
        x_theta = alpha_angle * x_theta
        x_dbar = (x_dbar * 3.2) + 3.2  #(0 to 6.4)

        # Drone3Dp1D dynamics
        # \dot x    = v*cos(th) + dx
        # \dot y    = v*sin(th) + dy
        # \dot th   = w = u_control
        # \dot dbar = 0

        # Compute the hamiltonian #dudx[..., 0]=b1  x[..., 1]=x_pos 
        ham = -omega_max * torch.abs(dudx[..., 2])  # Control component
        ham = ham + x_dbar * torch.norm(dudx[..., 0:2], dim=2)  # Disturbance component
        ham = ham + velocity * dudx[..., 0] * torch.cos(x_theta) + velocity * dudx[..., 1] * torch.sin(x_theta)  # Constant component

        # If we are computing BRT then take min with zero
        if minWith == 'zero':
            ham = torch.clamp(ham, max=0.0)

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            diff_constraint_hom = dudt - ham
            if minWith == 'target':
                #diff_constraint_hom = torch.min(torch.max(diff_constraint_hom[:, :, None], y - lx), (y - hx))#BRAT form

                source_boundary_values= lx #BRT form
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], y - lx)

                # smoothing_exponent=8.0  #smooth BRAT form
                # HJIVI_inner = torch.max(diff_constraint_hom[:, :, None], y - lx)
                # soft_min_num = HJIVI_inner * torch.exp(-smoothing_exponent * HJIVI_inner) + (y-hx) * torch.exp(-smoothing_exponent * (y-hx))
                # soft_min_den = torch.exp(-smoothing_exponent * HJIVI_inner) + torch.exp(-smoothing_exponent * (y-hx))
                # diff_constraint_hom = soft_min_num/soft_min_den

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]  
        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 15e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_drone3Dp1D

def initialize_hji_drone3D(dataset, minWith):
# Initialize the loss function for the Drone3D problem
# The dynamics parameters
    velocity = dataset.velocity
    omega_max = dataset.omega_max
    dbar = dataset.dbar

    alpha = dataset.alpha
    beta  = dataset.beta

    def hji_Drone3D(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        x = model_output['model_in']  # (meta_batch_size, num_points, 4)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            du, status = diff_operators.jacobian(y, x)
            dudt = du[..., 0, 0]
            dudx = du[..., 0, 1:]

            x_x = x[..., 1] * 1.0
            x_y = x[..., 2] * 1.0
            x_theta = x[..., 3] * 1.0

            x_x = x_x * alpha['x'] + beta['x']
            x_y = x_y * alpha['y'] + beta['y']
            x_theta =  x_theta * alpha['th'] + beta['th']
            # Scale the costate for theta appropriately to align with the range of [-pi, pi]
            dudx[..., 0] = dudx[..., 0] / alpha['x']
            dudx[..., 1] = dudx[..., 1] / alpha['y']
            dudx[..., 2] = dudx[..., 2] / alpha['th']
            dudt = dudt / alpha['time']            

            # Compute the hamiltonian #dudx[..., 0]=b1   
            ham = -omega_max * torch.abs(dudx[..., 2])  # Control component
            ham = ham + dbar * torch.norm(dudx[..., 0:2], dim=2)  # Disturbance component
            ham = ham + velocity * dudx[..., 0] * torch.cos(x_theta) + velocity * dudx[..., 1] * torch.sin(x_theta)  # Constant component
            
            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dudt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], y - source_boundary_values)

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 15e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum() * 8.0}

    return hji_Drone3D

def initialize_hji_narrow8D(dataset, minWith):
    # Initialize the loss function for the narrow passage problem
    # Normalization co-efficients
    alpha = dataset.alpha 
    beta = dataset.beta
    compute_overall_ham = dataset.compute_overall_ham
    clip_value_gradients = dataset.clip_value_gradients
    HJIVI_smoothing_setting = dataset.HJIVI_smoothing_setting
    smoothing_exponent = dataset.smoothing_exponent    

    def hji_narrow8D(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        lx = gt['lx']
        # gx = gt['gx']
        hx = gt['hx']
        x = model_output['model_in']  # (meta_batch_size, num_points, 8)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            du, status = diff_operators.jacobian(y, x)
            if clip_value_gradients:
                mean_tensor = torch.mean(du, dim=1, keepdim=True) 
                std_tensor = torch.std(du, dim=1, keepdim=True) 
                interval = 2.0
                normalized_du = (du - mean_tensor) / std_tensor
                du = mean_tensor + std_tensor * torch.clamp(normalized_du, min=-interval, max=interval)

            dudt = du[..., 0, 0]
            dudx = du[..., 0, 1:]

            # Compute Hamiltonian
            ham = compute_overall_ham(x[..., 1:], dudx)
            
            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            # diff_constraint_hom = dudt - ham * alpha['time']
            diff_constraint_hom = (dudt / alpha['time']) - ham
            # import ipdb; ipdb.set_trace()
            if minWith == 'target':
                if HJIVI_smoothing_setting in ['v2']:
                    soft_max_num =  diff_constraint_hom[:, :, None] * torch.exp(smoothing_exponent * diff_constraint_hom[:, :, None]) + (y-lx) * torch.exp(smoothing_exponent * (y-lx))
                    soft_max_den = torch.exp(smoothing_exponent * diff_constraint_hom[:, :, None]) + torch.exp(smoothing_exponent * (y-lx))
                    HJIVI_inner = soft_max_num/soft_max_den

                    soft_min_num = HJIVI_inner * torch.exp(-smoothing_exponent * HJIVI_inner) + (y-hx) * torch.exp(-smoothing_exponent * (y-hx))
                    soft_min_den = torch.exp(-smoothing_exponent * HJIVI_inner) + torch.exp(-smoothing_exponent * (y-hx))
                    diff_constraint_hom = soft_min_num/soft_min_den
                elif HJIVI_smoothing_setting in ['v3']:
                    HJIVI_inner = torch.max(diff_constraint_hom[:, :, None], y - lx)
                    soft_min_num = HJIVI_inner * torch.exp(-smoothing_exponent * HJIVI_inner) + (y-hx) * torch.exp(-smoothing_exponent * (y-hx))
                    soft_min_den = torch.exp(-smoothing_exponent * HJIVI_inner) + torch.exp(-smoothing_exponent * (y-hx))
                    # import ipdb; ipdb.set_trace()
                    diff_constraint_hom = soft_min_num/soft_min_den
                else:
                    # diff_constraint_hom = torch.min(torch.max(diff_constraint_hom[:, :, None], y - lx), (y + gx))
                    diff_constraint_hom = torch.min(torch.max(diff_constraint_hom[:, :, None], y - lx), (y - hx))

                    # # Limit the gradients
                    # mean_diff = torch.mean(diff_constraint_hom)[None]
                    # std_diff = torch.std(diff_constraint_hom)[None]
                    # upper_bar = (mean_diff + 2*std_diff).detach().cpu().numpy()
                    # lower_bar = (mean_diff - 2*std_diff).detach().cpu().numpy()
                    # diff_constraint_hom = torch.clamp(diff_constraint_hom, max=upper_bar[0], min=lower_bar[0])
                    # print('Warning! Currently clipping the PDE loss.')

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 15e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_narrow8D

def initialize_hji_narrowAvoid(dataset, minWith):
    # Initialize the loss function for the narrow passage problem
    # Normalization co-efficients
    alpha = dataset.alpha 
    beta = dataset.beta
    compute_overall_ham = dataset.compute_overall_ham
    clip_value_gradients = dataset.clip_value_gradients
    HJIVI_smoothing_setting = dataset.HJIVI_smoothing_setting
    smoothing_exponent = dataset.smoothing_exponent    

    def hji_narrowAvoid(model_output, gt):
        lx = gt['lx']
        #lx = gt['lx']
        # gx = gt['gx']
        #hx = gt['hx']
        x = model_output['model_in']  # (meta_batch_size, num_points, 10)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            du, status = diff_operators.jacobian(y, x)
            if clip_value_gradients:
                mean_tensor = torch.mean(du, dim=1, keepdim=True) 
                std_tensor = torch.std(du, dim=1, keepdim=True) 
                interval = 2.0
                normalized_du = (du - mean_tensor) / std_tensor
                du = mean_tensor + std_tensor * torch.clamp(normalized_du, min=-interval, max=interval)

            dudt = du[..., 0, 0]
            dudx = du[..., 0, 1:]

            # Compute Hamiltonian
            ham = compute_overall_ham(x[..., 1:], dudx)
            
            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            # diff_constraint_hom = dudt - ham * alpha['time']
            diff_constraint_hom = (dudt / alpha['time']) - ham
            # import ipdb; ipdb.set_trace()
            if minWith == 'target':
                if HJIVI_smoothing_setting in ['v2']:
                    soft_max_num =  diff_constraint_hom[:, :, None] * torch.exp(smoothing_exponent * diff_constraint_hom[:, :, None]) + (y-lx) * torch.exp(smoothing_exponent * (y-lx))
                    soft_max_den = torch.exp(smoothing_exponent * diff_constraint_hom[:, :, None]) + torch.exp(smoothing_exponent * (y-lx))
                    diff_constraint_hom = soft_max_num/soft_max_den
                else:
                    # diff_constraint_hom = torch.min(torch.max(diff_constraint_hom[:, :, None], y - lx), (y + gx))
                    diff_constraint_hom = torch.max(diff_constraint_hom[:, :, None], y - lx)

        dirichlet = y[dirichlet_mask] - lx[dirichlet_mask]

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 15e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_narrowAvoid

def initialize_hji_narrowRef(dataset, minWith):
    # Initialize the loss function for the narrow passage problem
    # Normalization co-efficients
    alpha = dataset.alpha 
    beta = dataset.beta
    compute_overall_ham = dataset.compute_overall_ham
    clip_value_gradients = dataset.clip_value_gradients
    HJIVI_smoothing_setting = dataset.HJIVI_smoothing_setting
    smoothing_exponent = dataset.smoothing_exponent    

    def hji_narrowRef(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        lx = gt['lx']
        # gx = gt['gx']
        hx = gt['hx']
        x = model_output['model_in']  # (meta_batch_size, num_points, 8)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            du, status = diff_operators.jacobian(y, x)
            if clip_value_gradients:
                mean_tensor = torch.mean(du, dim=1, keepdim=True) 
                std_tensor = torch.std(du, dim=1, keepdim=True) 
                interval = 2.0
                normalized_du = (du - mean_tensor) / std_tensor
                du = mean_tensor + std_tensor * torch.clamp(normalized_du, min=-interval, max=interval)

            dudt = du[..., 0, 0]
            dudx = du[..., 0, 1:]

            # Compute Hamiltonian
            ham = compute_overall_ham(x[..., 1:], dudx)
            
            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            # diff_constraint_hom = dudt - ham * alpha['time']
            diff_constraint_hom = (dudt / alpha['time']) - ham
            # import ipdb; ipdb.set_trace()
            if minWith == 'target':
                if HJIVI_smoothing_setting in ['v2']:
                    soft_max_num =  diff_constraint_hom[:, :, None] * torch.exp(smoothing_exponent * diff_constraint_hom[:, :, None]) + (y-lx) * torch.exp(smoothing_exponent * (y-lx))
                    soft_max_den = torch.exp(smoothing_exponent * diff_constraint_hom[:, :, None]) + torch.exp(smoothing_exponent * (y-lx))
                    HJIVI_inner = soft_max_num/soft_max_den

                    soft_min_num = HJIVI_inner * torch.exp(-smoothing_exponent * HJIVI_inner) + (y-hx) * torch.exp(-smoothing_exponent * (y-hx))
                    soft_min_den = torch.exp(-smoothing_exponent * HJIVI_inner) + torch.exp(-smoothing_exponent * (y-hx))
                    diff_constraint_hom = soft_min_num/soft_min_den
                elif HJIVI_smoothing_setting in ['v3']:
                    HJIVI_inner = torch.max(diff_constraint_hom[:, :, None], y - lx)
                    soft_min_num = HJIVI_inner * torch.exp(-smoothing_exponent * HJIVI_inner) + (y-hx) * torch.exp(-smoothing_exponent * (y-hx))
                    soft_min_den = torch.exp(-smoothing_exponent * HJIVI_inner) + torch.exp(-smoothing_exponent * (y-hx))
                    # import ipdb; ipdb.set_trace()
                    diff_constraint_hom = soft_min_num/soft_min_den
                else:
                    # diff_constraint_hom = torch.min(torch.max(diff_constraint_hom[:, :, None], y - lx), (y + gx))
                    diff_constraint_hom = torch.min(torch.max(diff_constraint_hom[:, :, None], y - lx), (y - hx))

                    # # Limit the gradients
                    # mean_diff = torch.mean(diff_constraint_hom)[None]
                    # std_diff = torch.std(diff_constraint_hom)[None]
                    # upper_bar = (mean_diff + 2*std_diff).detach().cpu().numpy()
                    # lower_bar = (mean_diff - 2*std_diff).detach().cpu().numpy()
                    # diff_constraint_hom = torch.clamp(diff_constraint_hom, max=upper_bar[0], min=lower_bar[0])
                    # print('Warning! Currently clipping the PDE loss.')

        dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 15e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_narrowRef

def initialize_hji_narrowMu(dataset, minWith):
    # Initialize the loss function for the narrow passage problem
    # Normalization co-efficients
    alpha = dataset.alpha 
    beta = dataset.beta
    compute_overall_ham = dataset.compute_overall_ham
    clip_value_gradients = dataset.clip_value_gradients
    HJIVI_smoothing_setting = dataset.HJIVI_smoothing_setting
    smoothing_exponent = dataset.smoothing_exponent   
    
    def hji_narrowMu(model_output, gt):
        source_boundary_values = gt['source_boundary_values']
        lx = gt['lx']
        # gx = gt['gx']
        hx = gt['hx']
        x = model_output['model_in']  # (meta_batch_size, num_points, 8)
        y = model_output['model_out']  # (meta_batch_size, num_points, 1)
        dirichlet_mask = gt['dirichlet_mask']
        batch_size = x.shape[1]

        if torch.all(dirichlet_mask):
            diff_constraint_hom = torch.Tensor([0])
        else:
            du, status = diff_operators.jacobian(y, x)
            if clip_value_gradients:
                mean_tensor = torch.mean(du, dim=1, keepdim=True) 
                std_tensor = torch.std(du, dim=1, keepdim=True) 
                interval = 2.0
                normalized_du = (du - mean_tensor) / std_tensor
                du = mean_tensor + std_tensor * torch.clamp(normalized_du, min=-interval, max=interval)

            dudt = du[..., 0, 0]
            dudx = du[..., 0, 1:]


            diff_from_lx = y - lx
            diff_from_hx = y - hx

            # Compute Hamiltonian
            ham = compute_overall_ham(x[..., 1:], dudx)
            
            # If we are computing BRT then take min with zero
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            # diff_constraint_hom = dudt - ham * alpha['time']
            diff_constraint_hom = (dudt / alpha['time']) - ham
            # import ipdb; ipdb.set_trace()
            if minWith == 'target':
                if HJIVI_smoothing_setting in ['v2']:
                    soft_max_num =  diff_constraint_hom[:, :, None] * torch.exp(smoothing_exponent * diff_constraint_hom[:, :, None]) + diff_from_lx * torch.exp(smoothing_exponent * diff_from_lx)
                    soft_max_den = torch.exp(smoothing_exponent * diff_constraint_hom[:, :, None]) + torch.exp(smoothing_exponent * diff_from_lx)
                    HJIVI_inner = soft_max_num/soft_max_den

                    soft_min_num = HJIVI_inner * torch.exp(-smoothing_exponent * HJIVI_inner) + diff_from_hx * torch.exp(-smoothing_exponent * diff_from_hx)
                    soft_min_den = torch.exp(-smoothing_exponent * HJIVI_inner) + torch.exp(-smoothing_exponent * diff_from_hx)
                    diff_constraint_hom = soft_min_num/soft_min_den
                elif HJIVI_smoothing_setting in ['v3']:
                    HJIVI_inner = torch.max(diff_constraint_hom[:, :, None], diff_from_lx)
                    soft_min_num = HJIVI_inner * torch.exp(-smoothing_exponent * HJIVI_inner) + diff_from_hx * torch.exp(-smoothing_exponent * diff_from_hx)
                    soft_min_den = torch.exp(-smoothing_exponent * HJIVI_inner) + torch.exp(-smoothing_exponent * diff_from_hx)
                    # import ipdb; ipdb.set_trace()
                    diff_constraint_hom = soft_min_num/soft_min_den
                else:
                    # diff_constraint_hom = torch.min(torch.max(diff_constraint_hom[:, :, None], y - lx), (y + gx))
                    diff_constraint_hom = torch.min(torch.max(diff_constraint_hom[:, :, None], diff_from_lx), diff_from_hx)

                    # # Limit the gradients
                    # mean_diff = torch.mean(diff_constraint_hom)[None]
                    # std_diff = torch.std(diff_constraint_hom)[None]
                    # upper_bar = (mean_diff + 2*std_diff).detach().cpu().numpy()
                    # lower_bar = (mean_diff - 2*std_diff).detach().cpu().numpy()
                    # diff_constraint_hom = torch.clamp(diff_constraint_hom, max=upper_bar[0], min=lower_bar[0])
                    # print('Warning! Currently clipping the PDE loss.')


            dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]

        # A factor of 15e2 to make loss roughly equal
        return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 15e2,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return hji_narrowMu