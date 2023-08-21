#!/bin/bash


# imagenet
gpu_id=0
dataset=sample_imagenet
data_dir=./experiment/data/image/imagenet10/
model=efficientnet_b0
attack=badnet
mark_height=25
mark_width=25
num_epoch=50

# mark_height_offset=0
# mark_width_offset=0

# x=(5 30 60 90 120 150)
# y=(5 20 40 60 80 100 120 140 160 180)


# cifar10
# gpu_id=1
# dataset=cifar10
# data_dir=./experiment/data/image/cifar10/
# model=vgg16
# attack=input_aware_dynamic
# num_epoch=20
# mark_height=32
# mark_width=32

x=(0 6 12 18 24 27)
y=(0 3 9 12 15 18 21 24 26 28)


for (( i = 0; i < 15; i++ )); do
	{	
		experiment_id=$i
		pos_y=`expr $i % 10`
		pos_x=`expr $i / 10`
		mark_height_offset=${y[pos_y]}
		mark_width_offset=${x[pos_x]}
		model_dir="experiment/model/efficientnet_imagenet10_"$experiment_id
		attack_dir="experiment/attack/efficientnet_imagenet10_"$experiment_id

		seed_id=`expr 666 + $experiment_id`
		log_dir="experiment/attack/${model}_${dataset}_"$experiment_id/mark${mark_width}x${mark_height}_${attack}_log.txt

		f="CUDA_VISIBLE_DEVICES=$gpu_id python backdoor_attack.py --color --verbose 1 --device gpu --pretrained --validate_interval 10 --lr 0.01 --dataset $dataset --data_dir $data_dir --model $model  --attack $attack --attack_dir $attack_dir --mark_random_init --mark_height $mark_height --mark_width $mark_height --epoch $num_epoch --lr_scheduler --lr_scheduler_type StepLR --save --model_dir $model_dir --seed $seed_id >$log_dir 2>&1 &"
		eval ${f}
	}
	wait
done

