CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name carla_SAC --task_name driving_v73_new_old_rew_temp_0.2_tau_0.005_buffer_100k \
    --encoder_type pixel \
    --action_repeat 2 --num_eval_episodes 5 \
    --image_size 84 --replay_buffer_capacity 100000 --latent_dim 256 --encoder_feature_dim 256 --alpha_beta 0.1\
    --agent rad_sac --frame_stack 4 --data_augs no_aug --save_tb --save_video --save_model --init_steps 1000 --log_interval 10 --discount 0.99 --critic_tau 0.005 --init_temperature 0.1\
    --seed 23 --work_dir /home/ws/ujvhi/rad_carla/rad/results --critic_lr 3e-4 --actor_lr 3e-4 --eval_freq 1000 --batch_size 512 --num_train_steps 1000000
