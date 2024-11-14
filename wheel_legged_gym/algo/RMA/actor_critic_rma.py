import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from .state_estimator import estimator, MLP
class ActorCritic_RMA(nn.Module):
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        num_latent=0,
                        frame_stack=1,
                        c_frame_stack=1,
                        actor_input_stack = 1,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        expert_hidden_dims=[256,128],
                        adaptation_hidden_dims=[256,128],
                        init_noise_std=1.0,
                        activation = nn.ELU(),
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic_RMA, self).__init__()
        # 初始化student与teacher
        self.expert_encoder = MLP(num_critic_obs, 1, num_latent, expert_hidden_dims, activation)
        self.adaptation_encoder = MLP(num_actor_obs, frame_stack, num_latent, adaptation_hidden_dims, activation)

        # actor和critic的输入维度
        mlp_input_dim_a = num_actor_obs * actor_input_stack + num_latent
        mlp_input_dim_c = num_critic_obs * c_frame_stack
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"expert MLP: {self.expert_encoder}")
        print(f"adaptation MLP: {self.adaptation_encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    # def act(self, observations, **kwargs):
    #     velocity = self.estimator(observations)
    #     # observations = torch.cat((observations, velocity), dim=-1)
    #     self.pre_vel = velocity.detach()
    #     self.update_distribution(observations)
    #     return self.distribution.sample()
    
    # obs_input 可以对应多帧数的 
    def act_student(self, obs_input, obs_history):
        latent = self.adaptation_encoder.forward(obs_history)
        action_mean = self.actor.forward(torch.cat([obs_input, latent], dim=-1))
        self.distribution = Normal(action_mean, action_mean * 0. + self.std)
        return self.distribution.sample()
    
    def act_expert(self, obs_input, privileged_obs):
        # obs_dict: obs, obs_history, privileged_obs
        latent = self.expert_encoder.forward(privileged_obs)
        action_mean = self.actor.forward(torch.cat([obs_input, latent], dim=-1))
        self.distribution = Normal(action_mean, action_mean * 0. + self.std)
        return self.distribution.sample()

    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_input, obs_history):
        latent = self.adaptation_encoder.forward(obs_history)
        action_mean = self.actor.forward(torch.cat([obs_input, latent], dim=-1))
        self.distribution = Normal(action_mean, action_mean * 0. + self.std)
        return self.distribution.sample()

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    def get_adaptation_latent(self, obs_history):
        return self.adaptation_encoder.forward(obs_history)
    def get_expert_latent(self, privileged_obs):
        return self.expert_encoder.forward(privileged_obs)
    def _update_with_latent(self,obs,latent):
        return self.actor.forward(obs, latent)
        
