#!/bin/bash 
#SBATCH --job-name=exp_srdiff_train_backbone_joint_srdiff_lcc_maxvit_hr5_maxvit_sr4_multirescond_caf_focal_all
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100m40:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH --time=48:00:00
#SBATCH --mail-user=kanyamahanga@ipi.uni-hannover.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output logs/exp_srdiff_train_backbone_joint_srdiff_lcc_maxvit_hr5_maxvit_sr4_multirescond_caf_focal_all_%j.out
#SBATCH --error logs/exp_srdiff_train_backbone_joint_srdiff_lcc_maxvit_hr5_maxvit_sr4_multirescond_caf_focal_all_%j.err
source load_modules.sh
export CONDA_ENVS_PATH=$HOME/.conda/envs
export DATA_DIR=$BIGWORK
conda activate flair_venv
which python
cd $HOME/MISR_JOINT_SRDiff_LCC_MaxViT_HR5_MaxViT_SR4_MultiResCond_CAF_FOCAL_ALL
srun python trainer.py --config configs/diffsr_maxvit_ltae.yaml --config_file flair-config-server.yml --exp_name misr/srdiff_maxvit_ltae_ckpt --reset
