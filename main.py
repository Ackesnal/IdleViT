import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from torch.nn import parameter

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss, IdleViTTotalLoss
from samplers import RASampler
import utils
from functools import partial
import torch.nn as nn
from vit import IdleVisionTransformer, VisionTransformer, _cfg, checkpoint_filter_fn
from lvvit import IdleLVViT, LVViT
import math
import shutil 

def get_args_parser():
    parser = argparse.ArgumentParser('IdleViT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--arch', default='deit_small', type=str, help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--base_rate', type=float, default=0.7, help='Base keep rate for IdleViT')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    
    parser.add_argument('--checkpoint', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
                        
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    
    # * Finetuning params                    
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 

    # Distillation parameters
    parser.add_argument('--teacher-model', type=str, default='', help='Name of teacher model to train (default: the full-size model)')
    parser.add_argument('--teacher-path', type=str, default='', help='')
    parser.add_argument('--distill', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--distillw', type=float, default=5, help='distill weight (default: 5)')
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # Dataset parameters
    parser.add_argument('--data-path', default='', type=str, help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'], type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name', choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # token cut loss parameters
    parser.add_argument('--cutw', type=float, default=20.0, help='weight for token cut loss (default: 20.0)')
    parser.add_argument('--cutloss_type', default="both", help='cut loss type (default: both)', choices=['both', 'intra', 'inter', 'none'])
    
    # speed test
    parser.add_argument('--test_speed', action='store_true', help='whether to measure throughput of model')
    parser.add_argument('--only_test_speed', action='store_true', help='only measure throughput of model')

    return parser



def get_param_groups(model, weight_decay):
    decay = []
    no_decay = []
    predictor = []
    for name, param in model.named_parameters():
        if 'predictor' in name:
            predictor.append(param)
        elif not param.requires_grad:
            continue  # frozen weights
        elif 'cls_token' in name or 'pos_embed' in name:
            continue  # frozen weights
        elif len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': predictor, 'weight_decay': weight_decay, 'name': 'predictor'},
        {'params': no_decay, 'weight_decay': 0., 'name': 'base_no_decay'},
        {'params': decay, 'weight_decay': weight_decay, 'name': 'base_decay'}
        ]
        
def speed_test(model, ntest=100, batchsize=128, x=None, **kwargs):
    if x is None:
        x = torch.rand(batchsize, 3, 224, 224).cuda()
    else:
        batchsize = x.shape[0]
    model.eval()

    start = time.time()
    for i in range(ntest):
        model(x, **kwargs)
    torch.cuda.synchronize()
    end = time.time()

    elapse = end - start
    speed = batchsize * ntest / elapse

    return speed

def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    else:
        print('Attention: mixup/cutmix are not used')
    
    if args.arch == 'deit_small':
        # IdleViT configs
        idle_layers = [i for i in range(4,12)] # layers using the IdleViT module
        keep_ratios = [args.base_rate**(i+1) for i in range(3) for _ in range(3)] # the keep ratios for each layer
        print(f"Creating model: {args.arch}")
        print('Keep_ratio =', keep_ratios, 'at layer', idle_layers)
        
        # Load pre-trained model
        model = IdleVisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, 
                                      idle_layers=idle_layers, keep_ratios=keep_ratios, distill=args.distill)
        model_path = './deit_small_patch16_224-cd65a155.pth'
        checkpoint = torch.load(model_path, map_location="cpu")
        ckpt = checkpoint_filter_fn(checkpoint, model)
        model.default_cfg = _cfg()
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        print('# missing keys=', missing_keys)
        print('# unexpected keys=', unexpected_keys)
        print('Sucessfully loaded from pre-trained weights:', model_path)
        
        # Load full-size model as the teacher for distillation
        if args.distill:
            model_t = VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True)
            model_t.load_state_dict(ckpt, strict=True)
            model_t.to(device)
            print('Sucessfully loaded from pre-trained weights for the teach model')
            
    elif args.arch == 'deit_base':
        # IdleViT configs
        idle_layers = [i for i in range(4,12)] # layers using the IdleViT module
        keep_ratios = [args.base_rate**(i+1) for i in range(3) for _ in range(3)] # the keep ratios for each layer
        print(f"Creating model: {args.arch}")
        print('Keep_ratio =', keep_ratios, 'at layer', idle_layers)
        
        # Load pre-trained model
        model = IdleVisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, 
                                      idle_layers=idle_layers, keep_ratios=keep_ratios, distill=args.distill)
        model_path = './deit_base_patch16_224-b5f2ef4d.pth'
        checkpoint = torch.load(model_path, map_location="cpu")
        ckpt = checkpoint_filter_fn(checkpoint, model)
        model.default_cfg = _cfg()
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        print('# missing keys=', missing_keys)
        print('# unexpected keys=', unexpected_keys)
        print('Sucessfully loaded from pre-trained weights:', model_path)
        
        # Load full-size model as the teacher for distillation
        if args.distill:
            model_t = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True)
            model_t.load_state_dict(ckpt, strict=True)
            model_t.to(device)
            print('Sucessfully loaded from pre-trained weights for the teach model')

    elif args.arch == 'lvvit_s':
        # IdleViT configs
        idle_layers = [i for i in range(4,16)] # layers using the IdleViT module
        keep_ratios = [args.base_rate**(i+1) for i in range(3) for _ in range(4)] # the keep ratios for each layer
        print(f"Creating model: {args.arch}")
        print('Keep_ratio =', keep_ratios, 'at layer', idle_layers)
        
        # Load pre-trained model
        model = IdleLVViT(patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
                          p_emb='4_2',skip_lam=2., return_dense=True, mix_token=True,
                          idle_layers=idle_layers, keep_ratios=keep_ratios, distill=args.distill)
        model_path = './lvvit_s-224-83.3.pth.tar'
        checkpoint = torch.load(model_path, map_location="cpu")
        model.default_cfg = _cfg()
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        print('# missing keys=', missing_keys)
        print('# unexpected keys=', unexpected_keys)
        print('Sucessfully loaded from pre-trained weights:', model_path)
        
        # Load full-size model as the teacher for distillation
        if args.distill:
            model_t = LVViT(patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
                            p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True)
            model_t.load_state_dict(checkpoint, strict=True)
            model_t.to(device)
            print('Sucessfully loaded from pre-trained weights for the teach model')
            
    elif args.arch == 'lvvit_m':
        # IdleViT configs
        idle_layers = [i for i in range(5,20)] # layers using the IdleViT module
        keep_ratios = [args.base_rate**(i+1) for i in range(3) for _ in range(5)] # the keep ratios for each layer
        print(f"Creating model: {args.arch}")
        print('Keep_ratio =', keep_ratios, 'at layer', idle_layers)
        
        # Load pre-trained model
        model = IdleLVViT(patch_size=16, embed_dim=512, depth=20, num_heads=8, mlp_ratio=3.,
                      p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True,
                      idle_layers=idle_layers, keep_ratios=keep_ratios, distill=args.distill)
        model_path = './lvvit_m-56M-224-84.0.tar'
        checkpoint = torch.load(model_path, map_location="cpu")
        model.default_cfg = _cfg()
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        print('# missing keys=', missing_keys)
        print('# unexpected keys=', unexpected_keys)
        print('Sucessfully loaded from pre-trained weights:', model_path)
        
        # Load full-size model as the teacher for distillation
        if args.distill:
            model_t = LVViT(patch_size=16, embed_dim=512, depth=20, num_heads=8, mlp_ratio=3.,
                            p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True)
            model_t.load_state_dict(checkpoint, strict=True)
            model_t.to(device)
            print('Sucessfully loaded from pre-trained weights for the teach model')
            
    else:
        raise NotImplementedError


    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    if args.test_speed:
        # test model throughput for three times to ensure accuracy
        inference_speed = speed_test(model)
        print('inference_speed (inaccurate):', inference_speed, 'images/s')
        inference_speed = speed_test(model)
        print('inference_speed:', inference_speed, 'images/s')
        inference_speed = speed_test(model)
        print('inference_speed:', inference_speed, 'images/s')
    if args.only_test_speed:
        return

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr

    opt_args = dict(lr=args.lr, weight_decay=args.weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas
    
    parameter_group = get_param_groups(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(parameter_group, **opt_args)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'

    criterion = IdleViTTotalLoss(criterion, model_t, keep_ratios=keep_ratios, idle_layers=idle_layers,
                                 cutloss_type=args.cutloss_type, cut_weight = args.cutw,
                                 distill_loss=args.distill, distill_weight=args.distillw)
    
    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        
    if args.eval:
        test_stats = evaluate(data_loader_val, model_t, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        # criterion.adjust_cut_loss(args.cutw * (1 - math.cos((1-epoch/args.epochs)*math.pi/2)))
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune == '' # keep in eval mode during finetuning
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
        
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if max_accuracy == test_stats["acc1"]:
            checkpoint_paths = [output_dir / 'checkpoint_best.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('IdleViT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
