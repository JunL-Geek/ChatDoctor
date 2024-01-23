# WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=37860 train_lora.py \
#   --base_model './pretrained' \
#   --data_path 'HealthCareMagic-100k.json' \
#   --output_dir './lora_models/' \
#   --batch_size 32 \
#   --micro_batch_size 4 \
#   --num_epochs 1 \
#   --learning_rate 3e-5 \
#   --cutoff_len 256 \
#   --val_set_size 120 \
#   --adapter_name lora

CUDA_VISIBLE_DEVICES=0 python train_lora.py \
  --base_model './pretrained' \
  --data_path 'HealthCareMagic-100k.json' \
  --output_dir './lora_models/' \
  --batch_size 32 \
  --micro_batch_size 4 \
  --num_epochs 1 \
  --learning_rate 3e-5 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora