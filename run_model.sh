#!/bin/bash


gpu_id=0
dataset=sample_imagenet
data_dir=./experiment/data/image/imagenet10/
model=efficientnet_b0
CUDA_VISIBLE_DEVICES=$gpu_id 

for (( i = 0; i < 15; i++ )); do
	{	
		experiment_id=$i
		model_dir="experiment/model/efficientnet_imagenet10_"$experiment_id
		seed_id=`expr 666 + $experiment_id`
		log_dir="experiment/model/efficientnet_imagenet10_"$experiment_id/log.txt

		f="CUDA_VISIBLE_DEVICES=$gpu_id python train.py --color --verbose 1 --device gpu --dataset $dataset --data_dir $data_dir --model $model --epoch 50 --lr_scheduler --lr_scheduler_type StepLR --cutout --grad_clip 5.0 --save --model_dir $model_dir --seed $seed_id >$log_dir 2>&1 &"
		eval ${f}
	}
	wait
done

