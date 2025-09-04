#!/bin/bash
#SBATCH --account=hai_1065
#SBATCH --nodes=1
#SBATCH --output=logs/extract_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4

module load CUDA/12

for size in 0.5 1.5 3 7 14; do
    for order in 4 5; do
        srun --exclusive -n 1 --gres=gpu:1  --cpus-per-task=4 uv run src/playground.py --model_name "Qwen/Qwen2.5-${size}B" --markov_order $order --n_reps 2 --n_permute $order --n_permute_primitive $order --chunk_size $order --repeats 10 &
    done
done

wait