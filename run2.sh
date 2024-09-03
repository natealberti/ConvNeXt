#!/bin/bash
#SBATCH --job-name=DEBUG_ConvNeXt_MIMIC_train

#SBATCH -N 1
#SBATCH -G a100:1
#SBATCH -c 12
##SBATCH --exclusive
#SBATCH -p general
#SBATCH -t 12-00:00:00
#SBATCH -q public

#SBATCH -o /scratch/nralbert/CSE507/ConvNeXt/logs/%x_slurm_%j.out     
#SBATCH -e /scratch/nralbert/CSE507/ConvNeXt/logs/%xslurm_%j.err      

module load mamba/latest
source activate convnext

LOG_DIR=/scratch/nralbert/CSE507/ConvNeXt/logs/debug

GPUS=1

~/.conda/envs/convnext/bin/python -m torch.distributed.launch --nproc_per_node=$GPUS main.py --model convnext_base  \
	--finetune https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth --input_size 224 --drop_path 0.2 \
	--data_path /scratch/jliang12/data/MIMIC_jpeg/physionet.org/files/mimic-cxr-jpg/2.0.0 --data_set MIMIC \
	--log_dir $LOG_DIR --output_dir $LOG_DIR --batch_size 64

# --eval true