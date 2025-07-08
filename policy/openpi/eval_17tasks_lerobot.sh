DEBUG=False

# task_name='block_hammer_beat'
# task_name='dual_bottles_pick_easy'
head_camera_type='D435'
train_config_name='none'
model_name='pi0/lerobot'
checkpoint_num=90000
seed=1
gpu_id=0

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

source .venv/bin/activate
cd ../..
python ./script/eval_policy_pi_all_tasks.py $task_name $head_camera_type $train_config_name $model_name $checkpoint_num $seed --eval-video-log --use_lerobot_pi0