
from diablo.diablo_config import DiabloCfg, DiabloCfgPPO


class DiabloCfg_EST(DiabloCfg):
    pass

class DiabloCfgPPO_EST(DiabloCfgPPO):
    runner_class_name = 'OnPolicyRunnerEstimator'
    
    class policy:
        num_encoder_obs = (
                DiabloCfg.env.obs_history_length * DiabloCfg.env.num_observations
        )
        latent_dim = 4  # at least 3 to estimate base linear velocity, 4:5 is estimate feet height
        encoder_hidden_dims = [128, 64]

    class algorithm(DiabloCfgPPO.algorithm):
        # estimator para
        extra_learning_rate = 5e-4        
        num_adaptation_module_substeps = 3
        
    class runner:
        policy_class_name = 'ActorCritic_Estimator'
        algorithm_class_name = 'PPO_Estimator'
        num_steps_per_env = 48  # per iteration
        max_iterations = 10001  # number of policy updates        #  xxw

        # logging
        save_interval = 50  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'Cowa_ppo_Estimator'
        run_name = 'v4'
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None