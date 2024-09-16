# Usage: python clean_old_checkpoints.py <path_to_train_dir>

def find_nn_subdirectories(directory_path):
    nn_subdirectories = []

    try:
        # Walk through all directories and subdirectories
        for root, dirs, files in os.walk(directory_path):
            # Check if 'nn' is in the list of directories
            if 'nn' in dirs:
                nn_subdirectories.append(os.path.join(root, 'nn'))

        return nn_subdirectories

    except FileNotFoundError:
        print(f"Directory not found: {directory_path}")
        return []

def find_workspace_dirs(directory_path):
    workspace_dirs = []
    
    try:
        # Walk through all directories and subdirectories
        for root, dirs, files in os.walk(directory_path):
            # Check if 'workspace' is in the list of directories
            if os.path.basename(root) == 'workspace':
                workspace_dirs.extend([os.path.join(root, dir) for dir in dirs])

        return workspace_dirs
    except FileNotFoundError:
        print(f"Directory not found: {directory_path}")
        return []
    

import os
import glob
import sys

def keep_top_n_files(directory_path, pattern, keyer, n=2):
    try:
        # Find all files matching the specified pattern
        files = glob.glob(os.path.join(directory_path, pattern))

        # Extract epoch numbers and rewards from the file names
        file_info = [(file, keyer(file)) for file in files]

        # Sort files based on epoch numbers
        sorted_files = sorted(file_info, key=lambda x: x[1], reverse=True)

        # Keep only the top n files
        files_to_keep = [file[0] for file in sorted_files[:n]]

        # Delete files that are not in the top n
        for file in files:
            if file not in files_to_keep:
                os.remove(file)
                print(f"Deleted: {file}")

    except FileNotFoundError:
        print(f"Directory not found: {directory_path}")

if __name__ == '__main__':
    train_dir = sys.argv[1]    
    file_pattern = '*_ep_*_rew_*.pth'
    splitter = lambda x: int(x.split('_ep_')[-1].split('_rew_')[0])
    for directory in find_nn_subdirectories(train_dir):
        keep_top_n_files(directory, file_pattern, splitter, n=3)
    
    file_pattern = '*.pth'
    splitter = lambda x: x
    for directory in find_workspace_dirs(train_dir):
        keep_top_n_files(directory, file_pattern, splitter, n=3)