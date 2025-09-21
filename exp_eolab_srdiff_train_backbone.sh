#!/bin/bash 
# #SBATCH --job-name=exp_srdiff_train_backbone_joint_srdiff_lcc_maxvit_hr5_convformer_sr4_nir_caf_focal_all_eolab
# #SBATCH --partition=gpu
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --gres=gpu:a100m40:1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem-per-cpu=4G
# #SBATCH --time=48:00:00
# #SBATCH --mail-user=kanyamahanga@ipi.uni-hannover.de
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --output logs/exp_srdiff_train_backbone_joint_srdiff_lcc_maxvit_hr5_convformer_sr4_nir_caf_focal_all_eolab_%j.out
# #SBATCH --error logs/exp_srdiff_train_backbone_joint_srdiff_lcc_maxvit_hr5_convformer_sr4_nir_caf_focal_all_eolab_%j.err
# source load_modules.sh

export CONDA_ENVS_PATH=$HOME/.conda/envs
DATA_DIR="/mydata/"
export DATA_DIR
source /home/eouser/flair_venv/bin activate
which python
cd $HOME/exp_2025/MISR_JOINT_SRDiff_LCC_MaxViT_HR5_MaxViT_SR4_MultiResCond_CAF_FOCAL_ALL
python trainer.py --config configs/diffsr_highresnet_ltae.yaml --config_file flair-config-server-eolab.yml --exp_name misr/srdiff_highresnet_ltae_ckpt --hparams="cond_net_ckpt=/mydata/pretrain_weights/MISR_JOINT_SRDiff_HIGHRESNET_PRETRAINED/results/checkpoints/misr/highresnet_ltae_ckpt" --reset
