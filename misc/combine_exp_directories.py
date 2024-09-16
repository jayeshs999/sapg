# Usage python combine_exp_directories.py <path_to_train_dir1> <path_to_train_dir2> <path_to_train_dir3> ... <destination_train_dir>

import os

def combine_exp_directories(train_dirs, destination_train_dir):
    for train_dir in train_dirs:
        for env in os.listdir(train_dir):
            env_dir = os.path.join(train_dir, env)
            os.makedirs(os.path.join(destination_train_dir, env), exist_ok=True)
            for exp in os.listdir(env_dir):
                exp_dir = os.path.join(env_dir, exp)
                dest_exp_dir = os.path.join(destination_train_dir, env, exp)
                if not os.path.exists(dest_exp_dir):
                    # recursively copy contents of exp_dir to destination_train_dir/env/exp
                    os.system(f'scp -r {exp_dir} {dest_exp_dir}')
                else:
                    print(f"{exp_dir} as it already exists in {destination_train_dir}")
                    
def main():
    import sys
    train_dirs = sys.argv[1:-1]
    destination_train_dir = sys.argv[-1]
    combine_exp_directories(train_dirs, destination_train_dir)
    
if __name__ == "__main__":
    main()
                
                
                    