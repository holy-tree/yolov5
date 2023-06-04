import os
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path
import torch
import torch.distributed as dist
from torch.optim import lr_scheduler


from models.yolo import Model
from utils.dataloaders import create_dataloader
from utils.autoanchor import check_anchors
from utils.downloads import attempt_download, is_url
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.loggers import Loggers
from utils.general import (LOGGER, check_amp, check_dataset, check_file, check_git_status, check_img_size,
                           check_requirements, check_suffix, check_yaml, colorstr, get_latest_run, increment_path,
                           init_seeds, intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                           one_cycle, print_args, print_mutation, strip_optimizer, yaml_save)
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))




class ATeacherTrainer():
    def __init__(self, hyp, opt, device, callbacks):
        save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

        # Directories
        w = save_dir / 'weights'  # weights dir
        (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
        last, best = w / 'last.pt', w / 'best.pt'

        # Hyperparameters
        if isinstance(hyp, str):
            with open(hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
        LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
        opt.hyp = hyp.copy()  # for saving hyps to checkpoints

        # Save run settings
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))


        # Loggers
        data_dict = None
        if RANK in {-1, 0}:
            loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance

            # Register actions
            for k in methods(loggers):
                callbacks.register_action(k, callback=getattr(loggers, k))

            # Process custom dataset artifact link
            data_dict = loggers.remote_dataset
            if resume:  # If resuming runs from remote artifact
                weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

        # Config
        plots = not evolve and not opt.noplots  # create plots
        cuda = device.type != 'cpu'
        init_seeds(opt.seed + 1 + RANK, deterministic=True)
        with torch_distributed_zero_first(LOCAL_RANK):
            data_dict = data_dict or check_dataset(data)  # check if None

        train_path, val_path = data_dict['train'], data_dict['val']
        nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
        names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
        is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

        # Model 
        check_suffix(weights, '.pt')  # check weights
        pretrained = weights.endswith('.pt')
        if pretrained:
            with torch_distributed_zero_first(LOCAL_RANK):
                weights = attempt_download(weights)  # download if not found locally
            ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
            model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
            exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
            model.load_state_dict(csd, strict=False)  # load
            LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
        else:
            model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        amp = check_amp(model)  # check AMP


        # Freeze 冻结模型部分参数
        freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze):
                LOGGER.info(f'freezing {k}')
                v.requires_grad = False

        # Image size
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

        # Batch size
        if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
            batch_size = check_train_batch_size(model, imgsz, amp)
            loggers.on_params_update({"batch_size": batch_size})
        
        # Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
        hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
        optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

        # Scheduler
        if opt.cos_lr:
            lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
        else:
            lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

        # EMA
        ema = ModelEMA(model) if RANK in {-1, 0} else None

        # Resume
        best_fitness, start_epoch = 0.0, 0
        if pretrained:
            if resume:
                best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
            del ckpt, csd

        # DP mode
        if cuda and RANK == -1 and torch.cuda.device_count() > 1:
            LOGGER.warning('WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
            model = torch.nn.DataParallel(model)

        # SyncBatchNorm
        if opt.sync_bn and cuda and RANK != -1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
            LOGGER.info('Using SyncBatchNorm()')

        # Trainloader
        train_loader, dataset = create_dataloader(train_path,
                                                imgsz,
                                                batch_size // WORLD_SIZE,
                                                gs,
                                                single_cls,
                                                hyp=hyp,
                                                augment=True,
                                                cache=None if opt.cache == 'val' else opt.cache,
                                                rect=opt.rect,
                                                rank=LOCAL_RANK,
                                                workers=workers,
                                                image_weights=opt.image_weights,
                                                quad=opt.quad,
                                                prefix=colorstr('train: '),
                                                shuffle=True)
        # labels.shape = (2912, 5)
        labels = np.concatenate(dataset.labels, 0)
        mlc = int(labels[:, 0].max())  # max label class
        assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

        # Process 0
        if RANK in {-1, 0}:
            val_loader = create_dataloader(val_path,
                                        imgsz,
                                        batch_size // WORLD_SIZE * 2,
                                        gs,
                                        single_cls,
                                        hyp=hyp,
                                        cache=None if noval else opt.cache,
                                        rect=True,
                                        rank=-1,
                                        workers=workers * 2,
                                        pad=0.5,
                                        prefix=colorstr('val: '))[0]

            if not resume:
                if not opt.noautoanchor:
                    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor
                model.half().float()  # pre-reduce anchor precision

            callbacks.run('on_pretrain_routine_end', labels, names)

        # DDP mode
        if cuda and RANK != -1:
            model = smart_DDP(model)





    def train():
        return None




def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'weights/TN.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=5, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    # 图像校正
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    # 是否从训练中断处继续训练
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    # 不保存模型
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    # 自动计算锚框
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    # 是否进行超参数优化
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    # 加载数据的进程数量
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements()
    
    # Resume (from specified or most recent last.pt)
    # not add

    opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
        check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    # if opt.evolve:
    #     if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
    #         opt.project = str(ROOT / 'runs/evolve')
    #     opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
    if opt.name == 'cfg':
        opt.name = Path(opt.cfg).stem  # use model.yaml as name
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    # device = select_device(opt.device, batch_size=opt.batch_size)
    # if LOCAL_RANK != -1:
    #     msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
    #     assert not opt.image_weights, f'--image-weights {msg}'
    #     assert not opt.evolve, f'--evolve {msg}'
    #     assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
    #     assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
    #     assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
    #     torch.cuda.set_device(LOCAL_RANK)
    #     device = torch.device('cuda', LOCAL_RANK)
    #     dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Train
    # if not opt.evolve:
    #     train(opt.hyp, opt, device, callbacks)
    
    Trainer = ATeacherTrainer(opt.hyp, opt, device, callbacks)
    Trainer.train()



# def run(**kwargs):
#     # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
#     opt = parse_opt(True)
#     for k, v in kwargs.items():
#         setattr(opt, k, v)
#     main(opt)
#     return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)