from envs.diablo.diablo_config import DiabloCfg, DiabloCfgPPO


class DiabloCfg_RMA(DiabloCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(DiabloCfg.env):
        # rma para
        # xxx frame_stack用于控制存储的历史大小，用于输入给adaptation encoder
        frame_stack = 15

        # xxx 这个actor_input_stack用于控制输入给actor的观测数量，应该小于frame_stack(总的历史)
        actor_input_stack = 5
        # xxx c_frame_stack用于控制存储的历史大小,用于输入给critic
        c_frame_stack = 3

        num_latent = 12

class DiabloCfgPPO_RMA(DiabloCfgPPO):
    seed = 10
    runner_class_name = 'OnPolicyRunnerRMA'
    

    class policy:
        # rma para
        expert_hidden_dims=[256, 128]
        adaptation_hidden_dims=[256, 128]

    class algorithm(DiabloCfgPPO.algorithm):
        # rma para
        adaptation_module_learning_rate=5e-4
        num_adaptation_module_substeps = 3
        
    class runner:
        policy_class_name = 'ActorCritic_RMA'
        algorithm_class_name = 'PPO_RMA'
        num_steps_per_env = 60  # per iteration
        max_iterations = 10001  # number of policy updates        #  xxw

        # logging
        save_interval = 50  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'Diablo_rma'
        run_name = 'v4'
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = '/home/zhengde.ma/humanoid-gym-main/logs/Diablo_ppo/success/9-4/4-融合了unitree&humanoid'  # updated from load_run and chkpt