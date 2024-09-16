import os


def generate_experiments_from_checkpoint(checkpoint_dir, multi_gpu=False):
    checkpoint_dir = os.path.abspath(checkpoint_dir).rstrip("/")
    base_name = os.path.basename(checkpoint_dir)

    base_dir_name = os.path.basename(os.path.dirname(checkpoint_dir))

    for i in range(100): # Won't run more than 100 experiments at a time
        experiment_dir = os.path.join(checkpoint_dir, 'runs', f"{i:02d}_{base_name}")
        model_dir = os.path.join(checkpoint_dir, 'runs', f'00_{base_name}' if multi_gpu else f"{i:02d}_{base_name}")
        cmd_path = os.path.join(experiment_dir, "cmd.txt")
        if not os.path.exists(cmd_path):
            break

        with open(cmd_path, "r") as f:
            cmd = f.read()
            cmd = cmd.split(" ")
            cmd = " ".join(["python", "-m", "isaacgymenvs.train"] + cmd[1:])
        
        if os.path.exists(os.path.join(model_dir,'last', 'model.pth')):
            cmd += f" checkpoint={os.path.join(model_dir,'last', 'model.pth')}"
            cmd += f" checkpoint_yaml={os.path.join(model_dir,'last', 'pbt_config.yaml')}"
        
        yield cmd, f"{i:02d}_{base_name}", os.path.join(base_dir_name, base_name), None


    