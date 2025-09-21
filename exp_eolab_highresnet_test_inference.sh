#!/bin/bash 
# #SBATCH --job-name=exp_highresnet_test_inference_joint_srdiff_lcc_maxvit_hr5_convformer_sr4_nir_caf_ce_dice_all
# #SBATCH --partition=tnt
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem-per-cpu=4G
# #SBATCH --time=5:00:00
# #SBATCH --mail-user=kanyamahanga@ipi.uni-hannover.de
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --output logs/exp_highresnet_test_inference_joint_srdiff_lcc_maxvit_hr5_convformer_sr4_nir_caf_ce_dice_all_%j.out
# #SBATCH --error logs/exp_highresnet_test_inference_joint_srdiff_lcc_maxvit_hr5_convformer_sr4_nir_caf_ce_dice_all_%j.err
# source load_modules.sh

export CONDA_ENVS_PATH=$HOME/.conda/envs
export DATA_DIR=$mydata/
conda activate flair_venv
which python
cd $HOME/MISR_JOINT_SRDiff_LCC_MaxViT_HR5_MaxViT_SR4_MultiResCond_CAF_FOCAL_ALL
srun python trainer.py --config configs/misr/highresnet_ltae.yaml --config_file flair-config-server-eolab.yml --exp_name misr/highresnet_ltae_ckpt --hparams="cond_net_ckpt=/mydata/Results/MISR_JOINT_SRDiff_LCC_MaxViT_HR5_MaxViT_SR4_MultiResCond_CAF_FOCAL_ALL/results/checkpoints/misr/highresnet_ltae_ckpt" --infer
