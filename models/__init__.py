import os
import numpy as np
import torch
from .protonet import ProtoNet
from .deploy import ProtoNet_Finetune, ProtoNet_Auto_Finetune, ProtoNet_AdaTok, ProtoNet_AdaTok_EntMin
from collections import OrderedDict

def get_backbone(args):
    if args.arch == 'vit_base':
        from . import vision_transformer as vit

        model = vit.__dict__['vit_base'](patch_size=16, num_classes=0)
        if not args.no_pretrain:
            pretrain_model = torch.load('pretraind model',map_location='cpu')["model"]
            model.load_state_dict(pretrain_model, strict=False)
            print('Pretrained weights found at {}'.format(url))


    elif args.arch == 'vit_small_':
        from . import vision_transformer as vit

        model = vit.__dict__['vit_small'](patch_size=16, num_classes=0)

        if not args.no_pretrain:
            pretrain_model = torch.load('pretraind model',map_location='cpu')["model"]
            model.load_state_dict(pretrain_model, strict=False)
            print('Pretrained weights found at {}'.format(url))


        # fine_tune
        # pretrained_dict1 = torch.load('your model',map_location='cpu')["model"]
        # new_state_dict = OrderedDict()
        # for k, v in pretrained_dict1.items():
        #     name = k[9:]  # remove `module.`
        #     # print(name)
        #     if name!='':   new_state_dict[name] = v
        # model.load_state_dict(new_state_dict, strict = False)
        # for k, p in model.named_parameters():
        #     if "prompt1" not in k and "top_down_transform1" not in k and "prompt2" not in k and "top_down_transform2" not in k  and "blocks.10.attn.qkv.bias" not in k and "blocks.11.attn.qkv.bias" not in k:
        #         p.requires_grad = False

    elif args.arch == 'vit_tiny':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_tiny'](patch_size=16, num_classes=0)

        if not args.no_pretrain:
            pretrain_model = torch.load('pretraind model',map_location='cpu')["model"]
            model.load_state_dict(pretrain_model, strict=False)
            print('Pretrained weights found at {}'.format(url))

    else:
        raise ValueError(f'{args.arch} is not conisdered in the current code.')

    return model


def get_model(args):
    backbone = get_backbone(args)

    if args.deploy == 'vanilla':
        model = ProtoNet(backbone)
    elif args.deploy == 'finetune':
        model = ProtoNet_Finetune(backbone, args.ada_steps, args.ada_lr, args.aug_prob, args.aug_types)
    else:
        raise ValueError(f'deploy method {args.deploy} is not supported.')
    return model
