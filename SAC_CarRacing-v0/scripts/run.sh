CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name CarRacing-v0 --task_name driving_test_v21_no_img_crop \
    --encoder_type pixel \
    --action_repeat 2 --num_eval_episodes 10 \
    --image_size 96 --replay_buffer_capacity 50000 --latent_dim 256\
    --agent rad_sac --frame_stack 4 --data_augs no_aug --save_tb --save_model --init_steps 1000 --save_video --log_interval 10 --discount 0.99 --critic_tau 0.02 \
    --seed 23 --work_dir /home/ws/ujvhi/rad/results/ --critic_lr 3e-4 --actor_lr 3e-4 --eval_freq 1000 --batch_size 256 --num_train_steps 1000000
