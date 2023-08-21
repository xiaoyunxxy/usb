#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python ./examples/backdoor_defense.py --color --verbose 1 --attack badnet --defense neural_cleanse --validate_interval 1 --epochs 50 --lr 1e-2
"""  # noqa: E501

import trojanvision
import argparse
import torch
from collections import OrderedDict

from trojanvision.attacks import BackdoorAttack
from trojanvision.defenses import BackdoorDefense

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)
    trojanvision.defenses.add_argument(parser)
    kwargs = parser.parse_args().__dict__

    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model = trojanvision.models.create(dataset=dataset, **kwargs)


    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)
    mark = trojanvision.marks.create(dataset=dataset, **kwargs)
    attack: BackdoorAttack = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **kwargs)
    defense: BackdoorDefense = trojanvision.defenses.create(dataset=dataset, model=model, attack=attack, **kwargs)

    # necessary when load mnist pretrained models.  and do not add --pretrained when load pretrained mnist models

    model._validate()
    
    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack, defense=defense)
    defense.detect(**trainer)

