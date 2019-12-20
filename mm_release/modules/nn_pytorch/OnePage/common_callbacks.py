from torchvision import transforms
from OnePage.core import OnePageEngine

from OnePage.engine_utils import find_class, create_dataset

from DGPT.DataLoader.Datasets import MixDatasetInfo
from torch.utils.data import DataLoader

from quicklab.common import init_weights
from quicklab.nettrainer import NetTrainer

engine = getattr(__import__('__main__'), 'engine')
assert(isinstance(engine, OnePageEngine))

print("COMMON CALLBACKS: engine ", engine)


@engine.autoreg('on_prepare')
def common_on_prepare(e):
    print('[COMMON] on_prepare')

    # prepare dataloader
    opt = engine.opt.datasets
    for k, v in vars(opt).items():
        dataset_class = find_class(None, v.type)
        print("find dataset class ", dataset_class)
        hook = f'on_create_dataset_{k}'
        dataset = engine.fire(hook, type=dataset_class, opt=v)
        if dataset is None:
            # sys.exit(f"Please implement dataset creation callback '{hook}'")
            dataset = engine.fire('on_create_dataset_default', type=dataset_class, opt=v)

        dataloader_class = None
        if hasattr(v, 'loader_type'):
            dataloader_class = find_class(None, v.loader_type)

        batch_size = engine.opt.common.batch_size
        if hasattr(v, 'batch_size'):
            batch_size = v.batch_size

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) \
            if dataloader_class is None else dataloader_class(dataset, batch_size=batch_size)

        data_iter = iter(dataloader)

        setattr(engine, f'dataset_{k}', dataset)
        setattr(engine, f'dataloader_{k}', dataloader)
        setattr(engine, f'dataiter_{k}', data_iter)

    # prepare preprocess
    engine.fire('on_prepare_preprocess')

    # prepare networks
    engine.fire('on_prepare_models')

    # prepare loss
    # prepare other config
    engine.fire('on_prepare_others')

@engine.autoreg('on_test')
def on_test(e):
    print('[COMMON] on_test', engine, e.source)


@engine.autoreg('on_finish')
def common_on_finish(e):
    print("finished ", engine.epoch)


@engine.autoreg('on_load_callback')
def on_common_load_callback(e):
    img = e.img
    transform = e.tf
    if transform is None:
        data = transforms.ToTensor()(img)
    else:
        data = transform(img)

    return data


@engine.autoreg('on_train_epoch')
def on_common_train_epoch(e):
    # has_next = True
    while True:
        data = engine.fire('on_get_next_data')
        if data is None:
        # if None in data:
            break

        engine.count += 1

        engine.fire('on_iter_start')

        engine.fire('on_train_iter', data=data)

        engine.fire('on_iter_end')


@engine.autoreg('on_create_dataset_default')
def create_dataset_default(e):
    dataset_class = e.type
    opt = e.opt

    # if dataset_class is MixedDataset:
    #     print("A")

    dirs = []
    for i in range(len(opt.path)):
        dirs.append(MixDatasetInfo(opt.path[i], e.opt.info_type[i], 0, 0))

    print("Init dataset: DIRs", dirs, " base size", e.opt.params.bs)
    params = dict(
        root_dirs=dirs,
        bs=opt.params.bs,
        testing=False,
        cb=engine.dataset_load_callback,
        crop_cb=engine.dataset_crop_callback,
        overexposure=True,
        # augments=[1] * 6
        )

    dataset = create_dataset(dataset_class, params)

    return dataset


@engine.autoreg('on_get_next_data')
def common_on_get_next_data(e):
    data = next(engine.dataiter_main, None)
    return data


@engine.autoreg('on_prepare_models')
def common_on_prepare_models(e):
    print("prepare networks")
    opt = engine.opt.models
    for k, v in vars(opt).items():
        model_class = find_class(None, v.type)
        print("find model class ", model_class)
        hook = f'on_create_model_{k}'
        model = engine.fire(hook, type=model_class, opt=v)
        if model is None:
            model = engine.fire('on_create_model_default', type=model_class, opt=v)

        optim_class = find_class(None, v.optimizer)

        hook = f'on_create_optim_{k}'
        optim_instance = engine.fire(hook, net=model, type=optim_class, opt=v)
        trainer = NetTrainer(
            name=v.name,
            net=model,
            optimizer=optim_class if optim_instance is None else optim_instance,
            lr=v.lr,
            save_interval=v.save_interval,
            save_fork=engine.opt.train.fork,
            save_path=v.save_path,
            save_last_only=v.save_last_only,
            device=v.device if hasattr(v, 'device') else engine.opt.device,
            loss_weight=v.loss_weight if hasattr(v, 'loss_weight') else 1.0,
            train_only=v.train_only if hasattr(v, 'train_only') else False,
            opt=vars(v.opt) if hasattr(v, 'opt') else dict(enable=True, use=True),
            optim_params=vars(v.optim_params) if hasattr(v, 'optim_params') else dict(),
        )

        trainer.start()

        setattr(engine, f'model_{k}', model)
        setattr(engine, f'trainer_{k}', trainer)


@engine.autoreg('on_create_model_default')
def common_on_create_model_default(e):
    net_class = e.type
    opt = e.opt
    if hasattr(opt, 'params'):
        net = net_class(**vars(opt.params))
    else:
        net = net_class()

    net.apply(init_weights)

    return net


def common_gan_iter_train(e):
    data = e.data

    input_data = engine.fire('on_preprocess', data=data)
    output_data = engine.fire('on_generate', input=input_data)

    engine.fire('on_d_start')
    engine.fire('on_d_train', output=output_data, input=input_data)
    engine.fire('on_d_end')

    if engine.count % engine.opt.train.G_iter_num == 0:
        engine.fire('on_g_start')
        engine.fire('on_g_train', output=output_data, input=input_data)
        engine.fire('on_g_end')


