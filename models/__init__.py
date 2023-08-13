import os
import numpy as np
import torch
from .protonet import ProtoNet
from .deploy import ProtoNet_Finetune, ProtoNet_Auto_Finetune, ProtoNet_AdaTok, ProtoNet_AdaTok_EntMin
from collections import OrderedDict

def get_backbone(args):
    if args.arch == 'vit_base_patch16_224_in21k':
        from .vit_google import VisionTransformer, CONFIGS

        config = CONFIGS['ViT-B_16']
        model = VisionTransformer(config, 224)

        url = 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz'
        pretrained_weights = 'pretrained_ckpts/vit_base_patch16_224_in21k.npz'

        if not os.path.exists(pretrained_weights):
            try:
                import wget
                os.makedirs('pretrained_ckpts', exist_ok=True)
                wget.download(url, pretrained_weights)
            except:
                print(f'Cannot download pretrained weights from {url}. Check if `pip install wget` works.')

        model.load_from(np.load(pretrained_weights))
        print('Pretrained weights found at {}'.format(pretrained_weights))

    elif args.arch == 'dino_base_patch16':
        from . import vision_transformer as vit

        model = vit.__dict__['vit_base'](patch_size=16, num_classes=0)
        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)

        model.load_state_dict(state_dict, strict=True)
        print('Pretrained weights found at {}'.format(url))


    elif args.arch == 'dino_small_patch16':
        from . import vision_transformer as vit

        model = vit.__dict__['vit_small'](patch_size=16, num_classes=0)

        if not args.no_pretrain:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)

            # model.load_state_dict(state_dict, strict=True)
            model.load_state_dict(state_dict, strict=False)
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
