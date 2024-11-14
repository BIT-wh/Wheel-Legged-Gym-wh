import torch.nn as nn
import torch
import numpy as np
from .utils import get_activation, check_cnnoutput
from torch.distributions import Normal
from torch.nn import functional
from torchsummary import summary

class estimator(nn.Module):
    def __init__(
        self,
        num_obs,
        output_dim=3,
        activation="elu",
        estimator_hidden_dims=[512, 256,256],
        device = "cuda"
    ):
        super().__init__()
        self.device = device
        self.num_obs = num_obs

        self.mlp_est = MLP(
            num_obs,
            output_dim,
            activation,
            estimator_hidden_dims,
        )
        print(self.mlp_est)


    def forward(self, obs_history):
        output= self.mlp_est(obs_history)
        return output

    def loss_fn(self, obs_history, vel):
        vel_est = self.forward(obs_history)
        # print("+++++++++++",vel_est.shape)
        # print("+++++++++++",vel.shape)
        # Body velocity estimation loss
        vel_loss = functional.mse_loss(vel_est, vel, reduction="none").mean(-1)
        v_v = vel_est
        v_vel = vel
        v_Difference = v_v.cpu().detach().numpy() - v_vel.cpu().detach().numpy()
        v_avg_diff_x ,v_avg_diff_y,v_avg_diff_z= np.mean(np.abs(v_Difference),axis=0)


        loss = vel_loss
        return {
            "loss": loss,
            "v_avg_diff_x": v_avg_diff_x,
            "v_avg_diff_y": v_avg_diff_y,
            "v_avg_diff_z": v_avg_diff_z,
        }


class MLP(nn.Module):
    def __init__(self, input_size, num_history, output_size, hidden_dims, activation):
        super().__init__()
        self.input_size = input_size * num_history
        self.output_size = output_size
        self.activation = activation

        module = []
        module.append(nn.Linear(self.input_size, hidden_dims[0]))
        module.append(self.activation)
        for i in range(len(hidden_dims) - 1):
            module.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            module.append(self.activation)
        module.append(nn.Linear(hidden_dims[-1], self.output_size))
        self.encoder = nn.Sequential(*module)

    def forward(self, obs_history):
        RS_obs_history = obs_history.reshape(obs_history.shape[0],-1)
        return self.encoder(RS_obs_history)

class TCNHistoryEncoder(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_history,
                 num_latent,
                 activation = 'elu',):
        super(TCNHistoryEncoder, self).__init__()
        self.num_obs = num_obs
        self.num_history = num_history  
        self.num_latent = num_latent    

        activation_fn = get_activation(activation)
        self.tsteps = tsteps = num_history
        input_size = num_obs
        output_size = num_latent
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            activation_fn,
            nn.Linear(128, 32),
        )
        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 8, stride = 4), nn.LeakyReLU(),
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.LeakyReLU(),
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.LeakyReLU(), nn.Flatten())
            last_dim = 32 * 3
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.LeakyReLU(), 
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 2, stride = 1), nn.LeakyReLU(), 
                nn.Flatten())
            last_dim = 32 * 3
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 6, stride = 2), nn.LeakyReLU(), 
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.LeakyReLU(), 
                nn.Flatten())
            last_dim = 32 * 3
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.LeakyReLU(), 
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 2, stride = 1), nn.LeakyReLU(), 
                nn.Flatten())
            last_dim = check_cnnoutput(input_size = (tsteps,32), list_modules = [self.conv_layers])

        self.linear_output = nn.Sequential(
            nn.Linear(last_dim, output_size)
            )

    def forward(self, obs_history):
        """
        obs_history.shape = (bz, T , obs_dim)
        """
        bs = obs_history.shape[0]
        T = self.tsteps
        projection = self.encoder(obs_history) # (bz, T , 32) -> (bz, 32, T) bz, channel_dim, Temporal_dim
        output = self.conv_layers(projection.permute(0, 2, 1)) # (bz, last_dim)
        output = self.linear_output(output)
        return output
    
