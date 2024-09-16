import os
import argparse

def visualize(checkpoint, vis_type, num_envs):

    extra_opts = {
        "save_info": ["headless=True", "++train.params.config.player.save_obs=True", "++train.params.config.player.games_num=25"],
        "render": ["headless=False", "++train.params.config.player.print_stats=True", "++train.params.config.player.games_num=25"]
    }

    checkpoint_dir = os.path.abspath(os.path.dirname(checkpoint) + "/../../..")
    base_name = os.path.basename(checkpoint_dir)
    experiment_dir = os.path.join(checkpoint_dir, 'runs', f"{0:02d}_{base_name}")
    cmd_path = os.path.join(experiment_dir, "cmd.txt")
    if not os.path.exists(cmd_path):
        print("No cmd.txt found")
        exit(1)

    with open(cmd_path, "r") as f:
        cmd = f.read()
        cmd = cmd.split(" ")
        cmd += [f"checkpoint={checkpoint}", 'test=True', f"task.env.numEnvs={num_envs}"] + extra_opts[vis_type]
        cmd = " ".join(["python", "-m", "isaacgymenvs.train"] + cmd[1:])
        
    # Run cmd
    print(cmd)
    ret = os.system("DISPLAY=:99 " + cmd)
    print("Return value:", ret)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--checkpoint", type=str, help="Checkpoint directory")
    parser.add_argument("-v", "--vis_type", type=str, default='render', choices=['render', 'save_info'], help="Visualization type")
    parser.add_argument("-e", "--num_envs", type=int, default=6, help="Number of environments to run")

    args = parser.parse_args()
    visualize(args.checkpoint, args.vis_type, args.num_envs)