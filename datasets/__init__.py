import os
import random
import torch
import numpy as np
from functools import partial
from torch.utils.data import DataLoader

from .samplers import RASampler
from .episodic_dataset import EpisodeDataset, EpisodeJSONDataset


def get_sets(args):
    if args.dataset == 'cifar_fs':
        from .cifar_fs import dataset_setting
    elif args.dataset == 'mini_imagenet':
        from .mini_imagenet import dataset_setting
    else:
        raise ValueError(f'{dataset} is not supported.')

    # If not meta_dataset
    trainTransform, valTransform, inputW, inputH, \
    trainDir, valDir, testDir, episodeJson, nbCls = \
            dataset_setting(args.nSupport, args.img_size)

    trainSet = EpisodeDataset(imgDir = trainDir,
                              nCls = args.nClsEpisode,
                              nSupport = args.nSupport,
                              nQuery = args.nQuery,
                              transform = trainTransform,
                              inputW = inputW,
                              inputH = inputH,
                              nEpisode = args.nEpisode)

    valSet = EpisodeJSONDataset(episodeJson,
                                valDir,
                                inputW,
                                inputH,
                                valTransform)

    testSet = EpisodeDataset(imgDir = testDir,
                             nCls = args.nClsEpisode,
                             nSupport = args.nSupport,
                             nQuery = args.nQuery,
                             transform = valTransform,
                             inputW = inputW,
                             inputH = inputH,
                             nEpisode = args.nEpisode)

    return trainSet, valSet, testSet


def get_loaders(args, num_tasks, global_rank):
    # datasets
    if args.eval:
        _, _, dataset_vals = get_sets(args)
    else:
        dataset_train, dataset_vals, _ = get_sets(args)

    # Worker init function
    if 'meta_dataset' in args.dataset: # meta_dataset & meta_dataset_h5
        #worker_init_fn = partial(worker_init_fn_, seed=args.seed)
        #worker_init_fn = lambda _: np.random.seed()
        def worker_init_fn(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
    else:
        worker_init_fn = None

    # Val loader
    # NOTE: meta-dataset has separate val-set per domain
    if not isinstance(dataset_vals, dict):
        dataset_vals = {'single': dataset_vals}

    data_loader_val = {}

    for j, (source, dataset_val) in enumerate(dataset_vals.items()):
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        generator = torch.Generator()
        generator.manual_seed(args.seed + 10000 + j)

        data_loader = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=1,
            num_workers=3, # more workers can take too much CPU
            pin_memory=args.pin_mem,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            generator=generator
        )
        data_loader_val[source] = data_loader

    if 'single' in dataset_vals:
        data_loader_val = data_loader_val['single']

    if args.eval:
        return None, data_loader_val

    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator
    )

    return data_loader_train, data_loader_val


def get_bscd_loader(dataset="EuroSAT", test_n_way=5, n_shot=5, image_size=224):
    iter_num = 600
    n_query = 15
    few_shot_params = dict(n_way=test_n_way , n_support=n_shot)

    if dataset == "EuroSAT":
        from .cdfsl.EuroSAT_few_shot import SetDataManager
    elif dataset == "ISIC":
        from .cdfsl.ISIC_few_shot import SetDataManager
    elif dataset == "CropDisease":
        from .cdfsl.CropDisease_few_shot import SetDataManager
    elif dataset == "ChestX":
        from .cdfsl.ChestX_few_shot import SetDataManager
    else:
        raise ValueError(f'Datast {dataset} is not supported.')

    datamgr = SetDataManager(image_size, n_eposide=iter_num, n_query=n_query, **few_shot_params)
    novel_loader = datamgr.get_data_loader(aug =False)

    def _loader_wrap():
        for x, y in novel_loader:
            SupportTensor = x[:,:n_shot].contiguous().view(1, test_n_way*n_shot, *x.size()[2:])
            QryTensor = x[:, n_shot:].contiguous().view(1, test_n_way*n_query, *x.size()[2:])
            SupportLabel = torch.from_numpy( np.repeat(range( test_n_way ), n_shot) ).view(1, test_n_way*n_shot)
            QryLabel = torch.from_numpy( np.repeat(range( test_n_way ), n_query) ).view(1, test_n_way*n_query)

            yield SupportTensor, SupportLabel, QryTensor, QryLabel

    class _DummyGenerator:
        def manual_seed(self, seed):
            pass

    class _Loader(object):
        def __init__(self):
            self.iterable = _loader_wrap()
            # NOTE: the following are required by engine.py:_evaluate()
            self.dataset = self
            self.generator = _DummyGenerator()

        def __len__(self):
            return len(novel_loader)
        def __iter__(self):
            return self.iterable

    return _Loader()