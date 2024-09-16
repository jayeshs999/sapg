if [ $# -lt 5 ]
then
    echo "Usage: ./train.sh <task> <experiment_prefix> <num_policies> <num_envs> <wandb_tags> <extra_args>"
    exit 1
fi

gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
(( exp_per_gpu=($3+$gpus-1)/$gpus ))

task=$1
experiment_prefix=$2
num_policies=$3
num_envs=$4
tags=$5
extra_args=${@:6}

python3 -m isaacgymenvs.pbt.launcher.run --experiments_per_gpu=$exp_per_gpu --num-policies=$num_policies --num-envs=$num_envs --env=$task --num_gpus=$gpus --wandb-tags=$tags --experiment_prefix=$experiment_prefix $extra_args
