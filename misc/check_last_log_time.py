# Usage: python check_last_log_time.py <train_dir_base>

import glob
import os
import sys
import tensorflow as tf
from datetime import datetime, timedelta

def latest_log_time_run(files):
    latest_log_time = datetime.min
    for file in files:
        events = tf.compat.v1.train.summary_iterator(file)

        try:
            for event in events:
                wall_time = datetime.fromtimestamp(event.wall_time)
                if latest_log_time is None or wall_time > latest_log_time:
                    latest_log_time = wall_time
        except:
            pass
    return latest_log_time

def get_latest_log_time(train_dir_base, env_name, exp_name):
    latest_log_time = datetime.min

    experiment_dir = os.path.join(train_dir_base, env_name, exp_name)
    for i in range(100):
        summaries_dir = os.path.join(experiment_dir, 'runs', f'{i:02d}_{exp_name}', 'summaries')
        if os.path.exists(summaries_dir):
            time = latest_log_time_run(glob.glob(f'{summaries_dir}/*.tfevents.*'))
            latest_log_time = max(latest_log_time, time)
    return latest_log_time

def main():
    current_time = datetime.now()
    threshold_time = current_time - timedelta(hours=5)

    for experiment_dir in glob.glob(f"{sys.argv[1]}/*/*/"):
        env, exp_name = experiment_dir.strip('/').split('/')[-2:]
        latest_log_time = get_latest_log_time(sys.argv[1], env, exp_name)

        if latest_log_time and latest_log_time < threshold_time:
            print(f"Experiment: ({env}, {exp_name}), Latest Log Time: {latest_log_time}")

if __name__ == "__main__":
    main()
