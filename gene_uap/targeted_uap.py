import os
import matplotlib.pyplot as plt
import adversarial_perturbation
import numpy as np
import argparse
import torch
from tqdm import tqdm
import time

import sys
sys.path.append('..')
import trojanvision


def main():
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    kwargs = parser.parse_args().__dict__

    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model = trojanvision.models.create(dataset=dataset, **kwargs)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)


    attcked_or_models = 'attack'
    mark='mark(20,20)'
    model_struc='efficientnet'
    imagenetdataset='imagenet10'
    model_file='square_white_tar0_alpha1.00_'+mark+'.pth'
    att_method='badnet'

    for i in range(0, 15):
        time_start=time.time()

        model_dir = '/data/xuxx/experiment_uap/experiments_trojanzoo/{model_type}/{model_st}_{dataset1}_{model_id}/image/{dataset}/{model_st}_b0/{att_method}/{model_file}'.format(
            model_type=attcked_or_models, model_file=model_file,dataset1=imagenetdataset, dataset=kwargs['dataset_name'], model_id=str(i), model_st=model_struc, att_method=att_method)
        if attcked_or_models=='model':
            model_dir = '/data/xuxx/experiment_uap/experiments_trojanzoo/{model_type}/{model_st}_{dataset1}_{model_id}/image/{dataset}/{model_st}.pth'.format(
                model_type=attcked_or_models, dataset1=imagenetdataset, dataset=kwargs['dataset_name'], model_id=str(i), model_st=model_struc)
        
        if not os.path.exists(model_dir):
            print('not exist: ', model_dir)
            continue

        print('load from: ', model_dir)
        model.load_state_dict(torch.load(model_dir))
        model._validate()

        

        for target_label in range(10):
            upa_dir = '/data/xuxx/experiment_uap/experiments_trojanzoo/{model_type}/{model_st}_{dataset1}_{model_id}/image/{dataset}/{model_st}_b0/{att_method}/uap_tar_{target_class}_{mark}.pth'.format(
                model_type=attcked_or_models, dataset1=imagenetdataset, dataset=kwargs['dataset_name'], model_id=str(i), target_class=target_label, mark=mark, model_st=model_struc, att_method=att_method)
            if attcked_or_models=='model':
                upa_dir = '/data/xuxx/experiment_uap/experiments_trojanzoo/{model_type}/{model_st}_{dataset1}_{model_id}/image/{dataset}/uap_tar_{target_class}.pth'.format(
                    model_type=attcked_or_models, dataset1=imagenetdataset, dataset=kwargs['dataset_name'], model_id=str(i), target_class=target_label, mark=mark, model_st=model_struc)

            v, fooling_rates, accuracies, total_iterations=adversarial_perturbation.generate(
                dataset.get_dataloader(mode='train'), dataset.get_dataloader(mode='valid'), 
                model, target_label=target_label, xi=20, delta=0.6, max_iter_uni=5)
            # torch.save(v.cpu(), upa_dir)

            time_end=time.time()
            print('model id:' + str(i) + '  target class: ' + str(target_label) + '  time cost',time_end-time_start,'s')



if __name__ == '__main__':
	main()