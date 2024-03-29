# export CUDA_VISIBLE_DEVICES=0,1,2,3
# torchrun --nproc_per_node=4 --master_port=37860 train.py \
#    --model_name_or_path <your_path_to_hf_converted_llama_ckpt_and_tokenizer> \
#    --data_path ./HealthCareMagic-100k.json \
#    --bf16 True \
#    --output_dir pretrained \
#    --num_train_epochs 1 \
#    --per_device_train_batch_size 4 \
#    --per_device_eval_batch_size 4 \
#    --gradient_accumulation_steps 8 \
#    --evaluation_strategy "no" \
#    --save_strategy "steps" \
#    --save_steps 2000 \
#    --save_total_limit 1 \
#    --learning_rate 2e-6 \
#    --weight_decay 0. \
#    --warmup_ratio 0.03 \
#    --lr_scheduler_type "cosine" \
#    --logging_steps 1 \
#    --fsdp "full_shard auto_wrap" \
#    --fsdp_transformer_layer_cls_to_wrap 'LLaMADecoderLayer' \
#    --tf32 True

python train.py \
--model_name_or_path ./llama-7b-hf \
--data_path ./HealthCareMagic-100k.json \
--bf16 True \
--output_dir pretrained \
--num_train_epochs 1 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 8 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2000 \
--save_total_limit 1 \
--learning_rate 2e-6 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--fsdp "full_shard auto_wrap" \
--fsdp_transformer_layer_cls_to_wrap 'LLaMADecoderLayer' \
--tf32 True