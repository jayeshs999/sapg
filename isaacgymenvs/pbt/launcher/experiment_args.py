from datetime import datetime
import sys

from isaacgymenvs.pbt.launcher.run_description import Experiment, ParamGrid, RunDescription

def get_name_prefix_suffix(args):
    name_prefix = f'{args.env}_{args.task}' if args.env.startswith('allegro_kuka') else f'{args.env}'
    name_suffix = (f"{args.experiment_prefix}_" if args.experiment_prefix else  "") + f'{args.num_envs}envs'
    name_suffix += f'{"_pbt"  if args.pbt else ""}_{args.expl_type}_{args.use_others_experience}_{args.num_policies}p'
    name_suffix += f'_mgpu' if args.multi_gpu else ''
    name_suffix += f'_{datetime.now().strftime("%d_%m_%Hh%Mm%Ss")}' if args.time_str is None else f'_{args.time_str}'
    return name_prefix, name_suffix

def add_experiment_args(parser):
    # parser.add_argument(
    #     "--experiment",
    #     default=None,
    #     type=str,
    #     help="Name of the python module that describes the experiment, e.g. sf_examples.vizdoom.experiments.paper_doom_all_basic_envs.py "
    #     "Experiment module must be importable in your Python environment. It must define a global variable EXPERIMENT_DESCRIPTION (see existing experiment modules for examples).",
    # )
    
    parser.add_argument("--env", default="allegro_kuka", choices=["ant","allegro_kuka", "allegro_kuka_two_arms","franka_cube_stack","franka_cube_push", "shadow_hand", "allegro_hand", "trifinger", "franka_cabinet", "humanoid", "cont_mountain_car"], type=str, help="Environment to run")
    parser.add_argument("--task" , default="reorientation", type=str, help="Task to run")
    parser.add_argument("--sparse-reward", action='store_true', help="Whether to use sparse reward")
    parser.add_argument("--episode-length", default=None, type=int, help="Episode length")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    
    parser.add_argument("--num-envs", default=8192, type=int, help="Number of parallel environments")
    parser.add_argument("--num-policies", default=8, type=int, help="Number of policies to train")
    parser.add_argument("--lstm", action='store_true', help="Whether to use LSTM")
    parser.add_argument("--max-frames", default=None, type=int, help="Number of frames to train")
    parser.add_argument("--minibatch-size", default=None, type=int, help="Minibatch size")
    parser.add_argument("--multi-gpu", action='store_true', help="Whether to use multi gpu")
    parser.add_argument("--max-iterations", default=None, type=int, help="Max iterations")

    parser.add_argument("--wandb-activate", default=True, type=bool, help="Whether to activate wandb")
    parser.add_argument("--wandb-entity", default="", type=str, help="Wandb entity")
    parser.add_argument("--wandb-project", default="sapg", type=str, help="Wandb project")
    parser.add_argument("--wandb-tags", default='[]', type=str, help="Wandb tags formatted as [t1,t2]")
    parser.add_argument("--wandb-notes", default="", type=str, help="Wandb notes")

    parser.add_argument("--pbt", action='store_true', help="PBT config")
    
    parser.add_argument("--good-reset-boundary", default=0, type=int, help="Good reset boundary")
    
    parser.add_argument("--expl-type", choices=['none', 'mixed_expl_disjoint', 'mixed_expl_learn_param'], default='none',help="Type of exploration")
    parser.add_argument("--num-expl-coef-blocks", default=None, type=int, help="Number of exploration coefficient blocks")
    parser.add_argument("--ir-embd-sz", default=None, type=int, help="Intrisic reward coefficient embedding size")
    parser.add_argument("--ir-coef-scale", default=1.0, type=float, help="Intrisic reward coefficient scale")
    parser.add_argument("--ir-type", default='none', type=str, choices=['none','rnd', 'rnd_ensemble', 'entropy', 'diayn'], help="Intrinsic reward type")
    
    parser.add_argument("--use-others-experience", default='none', choices=['none', 'all', 'lf'], help="Use others experience")
    parser.add_argument("--off-policy-ratio", default=1.0, type=float, help="Off policy ratio")
    
    parser.add_argument("--sigma", default="fixed", type=str, choices=["fixed", "obs_cond", "coef_cond"], help="Sigma type")
    parser.add_argument("--checkpoint", default=None, type=str, help="Checkpoint path")
    parser.add_argument("--time-str", default=None, type=str, help="Time string")
    parser.add_argument("--sapg", action='store_true', help="SAPG")
    parser.add_argument("--extra-args", nargs='*', help="Extra args") # Of the form --key a1=value1 --key a2=value2


    partial_cfg, _ = parser.parse_known_args(sys.argv[1:])

    if partial_cfg.pbt:
        parser.add_argument("--initial-delay", default=200000000, type=int, help="Initial delay")
        parser.add_argument("--pbt-interval-steps", default=20000000, type=int, help="PBT interval steps")
        parser.add_argument("--pbt-start-after", default=50000000, type=int, help="PBT start after")
        parser.add_argument("--replace-fraction-worst", default=0.3, type=float, help="Replace fraction worst")
        parser.add_argument("--mutation-rate", default=0.2, type=float, help="Replace fraction worst")

    return parser

def get_experiment_run_description(args):
    if args.sapg:
        args.expl_type = 'mixed_expl_learn_param'
        args.use_others_experience = 'lf'
        args.sigma = 'coef_cond'

    name_prefix, name_suffix = get_name_prefix_suffix(args)

    env_dict = {
        "ant": "Ant",
        "allegro_kuka": "AllegroKuka",
        "allegro_kuka_two_arms": "AllegroKukaTwoArms",
        "franka_cube_stack": "FrankaCubeStack",
        "franka_cube_push": "FrankaCubePush",
        "shadow_hand": "ShadowHand",
        "allegro_hand": "AllegroHand",
        "trifinger": "Trifinger",
        "franka_cabinet" : "FrankaCabinet",
        "humanoid" : "Humanoid",
        "cont_mountain_car" : "ContinuousMountainCar"
    }

    mutation_dict = {
        "ant": "ant_mutation",
        "allegro_kuka": "allegro_kuka_mutation",
        "allegro_kuka_two_arms": "allegro_kuka_mutation",
        "franka_cube_stack": "franka_cube_stack_mutation",
        "franka_cube_push": "franka_cube_push_mutation",
        "shadow_hand": "shadow_hand_mutation",
        "allegro_hand": "allegro_hand_mutation",
        "trifinger": "default_mutation",
        "franka_cabinet" : "default_mutation",
        "humanoid" : "default_mutation",
        "cont_mountain_car" : "default_mutation"
    }

    if args.minibatch_size is None:
        args.minibatch_size = 4*args.num_envs
    
    cli = (f'python -m isaacgymenvs.train '
           + f'task={env_dict[args.env]}{"LSTM" if args.lstm else ""} ' + (f'task/env={args.task} ' if 'allegro_kuka' in args.env else '')
           + f'++task.env.useSparseReward={args.sparse_reward} '
           + (f'pbt=pbt_default pbt.workspace=workspace ' if args.pbt else '')
           + f'headless=True '
           + (f'max_iterations={args.max_iterations} train.params.config.save_frequency={min(2000, args.max_iterations/10)} ' if args.max_iterations else '')
           + (f'task.env.episodeLength={args.episode_length} ' if args.episode_length else '')
           + f'task.env.numEnvs={args.num_envs} train.params.config.minibatch_size={args.minibatch_size} multi_gpu={args.multi_gpu} '
           + (f'train.params.config.max_frames={args.max_frames} ' if args.max_frames else '')
           + f'train.params.config.good_reset_boundary={args.good_reset_boundary} task.env.goodResetBoundary={args.good_reset_boundary} '
           + f'train.params.config.use_others_experience={args.use_others_experience} train.params.config.off_policy_ratio={args.off_policy_ratio} '
           + (f'train.params.config.expl_type={args.expl_type} train.params.config.expl_reward_type={args.ir_type} train.params.config.expl_coef_block_size={args.num_envs // (args.num_expl_coef_blocks or args.num_policies)} train.params.config.expl_reward_coef_scale={args.ir_coef_scale} ')
           + (f'train.params.config.expl_reward_coef_embd_size={args.ir_embd_sz} ' if args.ir_embd_sz else '')
           + (f'train.params.network.space.continuous.fixed_sigma={args.sigma} ')
           + (f'checkpoint={args.checkpoint} ' if args.checkpoint else '')
           + (" ".join(args.extra_args) + " " if args.extra_args else "")
           + f'wandb_project={args.wandb_project}_{name_prefix} wandb_entity={args.wandb_entity} wandb_activate={args.wandb_activate} wandb_group={name_suffix} wandb_tags={args.wandb_tags} ++wandb_notes=\'{args.wandb_notes}\'')
    
    if args.pbt:
        # don't set seed for PBT
        cli += f' seed=-1 pbt.num_policies={args.num_policies} pbt.initial_delay={args.initial_delay} pbt.interval_steps={args.pbt_interval_steps} pbt.start_after={args.pbt_start_after} pbt.replace_fraction_worst={args.replace_fraction_worst} pbt/mutation={mutation_dict[args.env]}'
        params = ParamGrid([
            ('pbt.policy_idx', list(range(args.num_policies))),
        ])
    else:
        params = ParamGrid([
            ('seed', [args.seed + i for i in range(args.num_policies)]),
        ])
    

    return RunDescription(
        name_prefix,
        experiments=[Experiment(name_suffix, cli, params.generate_params(randomize=False))],
        experiment_arg_name='experiment', experiment_dir_arg_name='hydra.run.dir',
        param_prefix='', customize_experiment_name=False,
    )