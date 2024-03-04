# !bin/bash
# Train the first stage of the attention model
python train_attention_ema.py \
    --dataroot /mnt/disk1/MLICdataset/COCO2014/ \
    --num_classes 80 --load_size 448 \
    --dataset_mode new_coco \
    --model coco_att_stage_one \
    --batch_size 32 --lr 0.0002 \
    --gpu_ids 0,1 --name coco_stage_one \
    --lr_policy cosine --niter_decay 25 --warm


python train_augmented_ema.py \
    --dataroot /mnt/disk1/MLICdataset/COCO2014/ \
    --load_size 448 --gpu_ids 0,1 --num_classes 80 \
    --dataset_mode con_coco --model coco_att_con_stage_two \
    --batch_size 24 --lr 0.01 --gpu_ids 0,1 \
    --name coco_stage_two --lr_policy step \
    --lr_decay_iters 20 \
    --pretrain_folder checkpoints/coco_stage_one/ \
    --epoch 15 --ema



# python test_new_ap.py --eval \
#     --dataroot /mnt/disk1/MLICdataset/COCO2014/ \
#     --num_classes 80 --load_size 448 \
#     --dataset_mode new_coco --model coco_test  \
#     --name coco_model --data_type val \
#     --epoch 15 --batch_size 64 --ema


# python test_new_ap.py --eval \
#     --dataroot /mnt/disk1/MLICdataset/COCO2014/ \
#     --num_classes 80 --load_size 448 \
#     --dataset_mode new_coco --model coco_test  \
#     --name coco_stage_one --data_type val \
#     --epoch 15 --batch_size 64 --ema