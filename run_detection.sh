#!/bin/bash


gpu_id=1
dataset=sample_imagenet
data_dir=/data/xuxx/experiment_uap/experiments_trojanzoo/data/image/imagenet10/
model=efficientnet_b0
attack=badnet
mark_height=20
mark_width=20
defense_remask_epoch=40
defense=usb

x=(5 30 60 90 120 150)
y=(5 20 40 60 80 100 120 140 160 180)

# x=(0 6 12 18 24 30)
# y=(0 3 9 12 15 18 21 24 27 30)


for (( i = 0; i < 15; i++ )); do
	{	
		experiment_id=$i
		pos_y=`expr $i % 10`
		pos_x=`expr $i / 10`
		mark_height_offset=${y[pos_y]}
		mark_width_offset=${x[pos_x]}
		# model_dir="experiment/model/${model}_${dataset}_"$experiment_id
		# attack_dir="experiment/attack/${model}_${dataset}_"$experiment_id
		# defense_dir="experiment/defense/${model}_${dataset}_"$experiment_id
		model_dir="/data/xuxx/experiment_uap/experiments_trojanzoo/model/efficientnet_imagenet10_"$experiment_id
		attack_dir="/data/xuxx/experiment_uap/experiments_trojanzoo/attack/efficientnet_imagenet10_"$experiment_id
		defense_dir="/data/xuxx/experiment_uap/experiments_trojanzoo/defense/efficientnet_imagenet10_"$experiment_id

		seed_id=`expr 666 + $experiment_id`
		log_dir="experiment/defense/efficientnet_imagenet10_"$experiment_id/mark${mark_width}x${mark_height}_${defense}_log.txt

		f="CUDA_VISIBLE_DEVICES=$gpu_id python backdoor_defense.py --color --verbose 1 --defense $defense --batch_size 128 --defense_dir $defense_dir --device gpu --validate_interval 1 --lr 0.01 --defense_remask_lr 0.1 --defense_remask_epoch $defense_remask_epoch --dataset $dataset --data_dir $data_dir --model $model --attack $attack --attack_dir $attack_dir --mark_height $mark_height --mark_width $mark_height --mark_random_init --epoch 10 --lr_scheduler --lr_scheduler_type StepLR --save --model_dir $model_dir --seed $seed_id >$log_dir 2>&1 &"
		echo ${f}
	}
	wait
done
