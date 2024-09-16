if [ $# -lt 2 ]
then
    echo "Usage: ./train.sh <checkpoint_dir> <num-policies> [--multi-gpu]"
    exit 1
fi

GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
(( exp_per_gpu=($2+$GPUS-1)/$GPUS ))

extraArgs="${@:3}"

python3 -m isaacgymenvs.pbt.launcher.run --max_parallel=$2 --num-policies=$2 --experiments_per_gpu=$exp_per_gpu --num_gpus=$GPUS --checkpoint_dir=$1 $extraArgs
