# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# iBOT: https://github.com/bytedance/ibot
# --------------------------------------------------------

import argparse
import os
import datetime
import time
import numpy as np
import utils
import models.vision_transformer_hssl as vision_transformer
from models.vision_transformer_hssl import HSSLHead, MultiCropWrapper
import jittor as jt
import jittor.nn as nn

import jittor.transform as transforms

from pathlib import Path
from PIL import Image
from loader import ImageFolderMask
from logging import getLogger
jt.flags.use_cuda = 1


logger = getLogger()


def get_args_parser():
    parser = argparse.ArgumentParser('HSSL', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str, 
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--auxiliary_depth', default=3, type=int, help="""The depth of the auxiliary head.""")
    parser.add_argument('--window_size', default=7, type=int, help="""Size of window - default 7.
        This config is only valid for Swin Transofmer and is ignoired for vanilla ViT architectures.""")
    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        output for [CLS] token.""")
    parser.add_argument('--patch_out_dim', default=8192, type=int, help="""Dimensionality of
        output for patch tokens.""")
    parser.add_argument('--shared_head', default=False, type=utils.bool_flag, help="""Wether to share 
        the same head for [CLS] token output and patch tokens output. When set to false, patch_out_dim
        is ignored and enforced to be same with out_dim. (Default: False)""")
    parser.add_argument('--shared_head_teacher', default=True, type=utils.bool_flag, help="""See above.
        Only works for teacher model. (Defeault: True)""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--norm_in_head', default=None,
        help="Whether to use batch normalizations in projection head (Default: None)")
    parser.add_argument('--act_in_head', default='gelu',
        help="Whether to use batch normalizations in projection head (Default: gelu)")
    parser.add_argument('--use_masked_im_modeling', default=True, type=utils.bool_flag,
        help="Whether to use masked image modeling (mim) in backbone (Default: True)")
    parser.add_argument('--pred_ratio', default=0.3, type=float, nargs='+', help="""Ratio of partial prediction.
        If a list of ratio is specified, one of them will be randomly choosed for each patch.""")
    parser.add_argument('--pred_ratio_var', default=0, type=float, nargs='+', help="""Variance of partial prediction
        ratio. Length should be indentical to the length of pred_ratio. 0 for disabling. """)
    parser.add_argument('--pred_shape', default='block', type=str, help="""Shape of partial prediction.""")
    parser.add_argument('--pred_start_epoch', default=0, type=int, help="""Start epoch to perform masked
        image prediction. We typically set this to 50 for swin transformer. (Default: 0)""")
    parser.add_argument('--lambda1', default=1.0, type=float, help="""loss weight for dino
        loss over [CLS] tokens (Default: 1.0)""")
    parser.add_argument('--lambda2', default=1.0, type=float, help="""loss weight for beit 
        loss over masked patch tokens (Default: 1.0)""")
        
    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_patch_temp', default=0.04, type=float, help="""See 
        `--warmup_teacher_temp`""")
    parser.add_argument('--teacher_patch_temp', default=0.07, type=float, help=""""See 
        `--teacher_temp`""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size', default=128, type=int,
        help='Batch-size : number of distinct images loaded on all GPUs.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--load_from', default=None, help="""Path to load checkpoints to resume training.""")
    parser.add_argument('--drop_path', type=float, default=0.1, help="""Drop path rate for student network.""")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Multi-crop parameters
    parser.add_argument('--global_crops_number', type=int, default=2, help="""Number of global
        views to generate. Default is to use two global crops. """)
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.14, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=40, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                        it is set automatically and should not be passed as argument""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def train_hssl(args):
    utils.fix_random_seeds(args.seed)
    logger, _ = utils.initialize_exp(args, "epoch", "loss")
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    # ============ preparing data ... ============
    transform = DataAugmentationHSSL(
        args.global_crops_scale,
        args.local_crops_scale,
        args.global_crops_number,
        args.local_crops_number,
    )
    pred_size = args.patch_size * 8 if 'swin' in args.arch else args.patch_size
    dataset = ImageFolderMask(
        args.data_path, 
        transform=transform,
        patch_size=pred_size,
        pred_ratio=args.pred_ratio,
        pred_ratio_var=args.pred_ratio_var,
        pred_aspect_ratio=(0.3, 1/0.3),
        pred_shape=args.pred_shape,
        pred_start_epoch=args.pred_start_epoch)
    data_loader = dataset.set_attrs(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True
    )

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    if args.arch in vision_transformer.__dict__.keys():
        student = vision_transformer.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling, 
            auxiliary_depth=args.auxiliary_depth,
        )
        teacher = vision_transformer.__dict__[args.arch](
            patch_size=args.patch_size,
            return_all_tokens=True, 
            auxiliary_depth=args.auxiliary_depth,
        )
        embed_dim = student.embed_dim
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = MultiCropWrapper(student, HSSLHead(
        embed_dim,
        args.out_dim,
        patch_out_dim=args.patch_out_dim,
        norm=args.norm_in_head,
        act=args.act_in_head,
        norm_last_layer=args.norm_last_layer,
        shared_head=args.shared_head,
    ))
    teacher = MultiCropWrapper(
        teacher,
        HSSLHead(
            embed_dim, 
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            shared_head=args.shared_head_teacher,
        ),
    )
    if jt.in_mpi:
        for n, p in teacher.named_parameters():
            p.assign(p.mpi_broadcast())
        for n, p in student.named_parameters():
            p.assign(p.mpi_broadcast())

    # teacher and student start with the same weights
    state_dict = {}
    state_dict_tmp = student.state_dict()
    for k in list(state_dict_tmp.keys()):
        state_dict[k] = state_dict_tmp[k].copy()
    teacher.load_state_dict(state_dict)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    logger.info(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    same_dim = args.shared_head or args.shared_head_teacher
    pretrain_loss = PretrainLoss(
        args.out_dim,
        args.out_dim if same_dim else args.patch_out_dim,
        args.global_crops_number,
        args.local_crops_number,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_patch_temp,
        args.teacher_patch_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        mim_start_epoch=args.pred_start_epoch,
        accum_iter=args.accum_iter
    ).cuda()
        
    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups_vit(student)
    if args.optimizer == "adamw":
        optimizer = jt.optim.AdamW(params_groups, lr=0)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = jt.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler


    # ============ init schedulers ... ============
    eff_batch_size = args.batch_size * args.accum_iter
    print("Effective batch size: %d" % eff_batch_size)
    print("Base lr: %.2e" % args.lr)
    print("Actual lr: %.2e" % (args.lr * (eff_batch_size) / 256.))
    lr_schedule = utils.cosine_scheduler(
        args.lr * (eff_batch_size) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                            args.epochs, len(data_loader))
                  
    print("Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    if args.load_from:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            pretrain_loss=pretrain_loss,
        )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting HSSL training!")
    for epoch in range(start_epoch, args.epochs):

        # ============ training one epoch of HSSL ... ============
        train_stats = train_one_epoch(student, teacher, pretrain_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'pretrain_loss': pretrain_loss.state_dict(),
        }
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth.tar'))
        if args.saveckp_freq and (epoch % args.saveckp_freq == 0) and epoch:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth.tar'))
        jt.sync_all()
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, pretrain_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    args):

    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    losses_patch1 = utils.AverageMeter()
    losses_cls1 = utils.AverageMeter()
    losses_patch2 = utils.AverageMeter()
    losses_cls2 = utils.AverageMeter()
    lrs = utils.AverageMeter()
    wds = utils.AverageMeter()
    accs = utils.AverageMeter()

    end = time.time()   

    # common params
    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in student.named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in teacher.named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
    params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]

    pred_labels, real_labels = [], []
    for it_current_epoch, (images, labels, masks) in enumerate(data_loader):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it_current_epoch  # global training iteration

        if it_current_epoch % args.accum_iter == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[it]      

        # get global views
        with jt.no_grad():
            teacher_output_cls1, teacher_output_patch1, teacher_output_cls2, teacher_output_patch2 = teacher(images[:args.global_crops_number])
        student_output_cls1, student_output_patch1, student_output_cls2, student_output_patch2 = student(images[:args.global_crops_number], mask=masks[:args.global_crops_number])

        # get local views
        student.backbone.masked_im_modeling = False
        if len(images) > args.global_crops_number:
            # assert False
            student_local_cls1, _, student_local_cls2, _ = student(images[args.global_crops_number:])
        else:
            student_local_cls1 = None
            student_local_cls2 = None
        student.backbone.masked_im_modeling = args.use_masked_im_modeling

        all_loss = pretrain_loss(
            (student_output_cls1, student_output_patch1), 
            (student_output_cls2, None), 
            (teacher_output_cls2, teacher_output_patch1), 
            (student_local_cls1, student_local_cls2), 
            masks, epoch, it_current_epoch)
        loss = all_loss.pop('loss')

        # log statistics
        probs1 = teacher_output_cls1.chunk(args.global_crops_number)
        probs2 = student_output_cls1.chunk(args.global_crops_number)
        pred1 = probs1[0].argmax(dim=1)[0]
        pred2 = probs2[1].argmax(dim=1)[0]
        acc = (pred1 == pred2).sum() / pred1.size(0)
        pred_labels.append(pred1)
        real_labels.append(labels)

        # student update
        optimizer.backward(loss / args.accum_iter)
        if (it_current_epoch + 1) % args.accum_iter == 0:
            if args.clip_grad:
                utils.clip_gradients(student, optimizer, args.clip_grad)
            utils.cancel_gradients_last_layer(
                epoch, student, args.freeze_last_layer)
            optimizer.step()

        # EMA update for the teacher
        if (it_current_epoch + 1) % args.accum_iter == 0:
            with jt.no_grad():
                m = momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(params_q, params_k):
                    param_k.assign(param_k.detach() * m + (1 - m) * param_q.detach())

        # logging
        losses.update(loss.item(), images[0].size(0))
        losses_patch1.update(all_loss['patch1'].item(), images[0].size(0))
        losses_cls1.update(all_loss['cls1'].item(), images[0].size(0))
        losses_patch2.update(all_loss['patch2'].item() if not isinstance(all_loss['patch2'], float) else all_loss['patch2'], images[0].size(0))
        losses_cls2.update(all_loss['cls2'].item() if not isinstance(all_loss['cls2'], float) else all_loss['cls2'], images[0].size(0))
        accs.update(acc)
        batch_time.update(time.time() - end)
        end = time.time()

        if jt.rank == 0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                "Patch1 {losses_patch1.val:.4f} ({losses_patch1.avg:.4f})\t"
                "Cls1 {losses_cls1.val:.4f} ({losses_cls1.avg:.4f})\t"
                "Patch2 {losses_patch2.val:.4f} ({losses_patch2.avg:.4f})\t"
                "Cls2 {losses_cls2.val:.4f} ({losses_cls2.avg:.4f})\t"
                "Acc {accs.val:.4f} ({accs.avg:.4f})\t"
                "Lr: {lr:.6f}\t Wd: {wd:.3f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    losses=losses,
                    losses_patch1=losses_patch1,
                    losses_cls1=losses_cls1,
                    losses_patch2=losses_patch2,
                    losses_cls2=losses_cls2,
                    accs=accs, 
                    lr=optimizer.param_groups[0]["lr"],
                    wd=optimizer.param_groups[0]["weight_decay"],
                )
            )
        jt.sync_all()
        jt.gc()

    pred_labels = jt.concat(pred_labels).detach().numpy()
    real_labels = jt.concat(real_labels).detach().numpy()

    nmi, ari, fscore, adjacc = utils.eval_pred(real_labels, pred_labels, calc_acc=False)
    print("NMI: {}, ARI: {}, F: {}, ACC: {}".format(nmi, ari, fscore, adjacc))


# Modified from iBOT: https://github.com/bytedance/ibot
class PretrainLoss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, ngcrops, nlcrops, warmup_teacher_temp, 
                 teacher_temp, warmup_teacher_temp2, teacher_temp2, 
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, 
                 center_momentum=0.9, center_momentum2=0.9,
                 lambda1=1.0, lambda2=1.0, mim_start_epoch=0, accum_iter=1):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.center = jt.zeros((1, out_dim))
        self.center2 = jt.zeros((1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.accum_iter = accum_iter

        self.center_accum = jt.zeros((1, out_dim))
        self.center2_accum = jt.zeros((1, 1, patch_out_dim))

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp2_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2
        )) if mim_start_epoch == 0 else np.concatenate((
            np.ones(mim_start_epoch) * warmup_teacher_temp2,
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch) * teacher_temp2
        ))

    def execute(self, student_output1, student_output2, teacher_output, student_local_cls, student_mask, epoch, it):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_cls1, student_patch1 = student_output1
        student_cls2, student_patch2 = student_output2
        teacher_cls, teacher_patch = teacher_output
        student_local_cls1, student_local_cls2 = student_local_cls
        
        if student_local_cls1 is not None:
            student_cls1 = jt.concat([student_cls1, student_local_cls1])
        if student_local_cls2 is not None:
            student_cls2 = jt.concat([student_cls2, student_local_cls2])

        # [CLS] and patch for global patches
        if student_cls1 is not None:
            student_cls1 = student_cls1 / self.student_temp
            student_cls1_c = student_cls1.chunk(self.ncrops)
        else:
            student_cls1 = student_cls1_c = None

        if student_cls2 is not None:
            student_cls2 = student_cls2 / self.student_temp
            student_cls2_c = student_cls2.chunk(self.ncrops)
        else:
            student_cls2 = student_cls2_c = None

        if student_patch1 is not None:
            student_patch1 = student_patch1 / self.student_temp
            student_patch1_c = student_patch1.chunk(self.ngcrops)
        else:
            student_patch1 = student_patch1_c = None

        if student_patch2 is not None:
            student_patch2 = student_patch2 / self.student_temp
            student_patch2_c = student_patch2.chunk(self.ngcrops)
        else:
            student_patch2 = student_patch2_c = None
        
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = jt.nn.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        teacher_patch_c = jt.nn.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1_1, n_loss_terms1_1 = 0, 0
        total_loss1_2, n_loss_terms1_2 = 0, 0
        total_loss2_1, n_loss_terms2_1 = 0, 0
        total_loss2_2, n_loss_terms2_2 = 0, 0
        for q in range(len(teacher_cls_c)):
            for v in range(self.ncrops):
                if v == q:
                    if student_patch1_c is not None:
                        loss2 = jt.sum(-teacher_patch_c[q] * jt.nn.log_softmax(student_patch1_c[v], dim=-1), dim=-1)
                        mask = student_mask[v].flatten(-2, -1)
                        loss2 = jt.sum(loss2 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min_v=1.0)
                        total_loss1_2 += loss2.mean()
                        n_loss_terms1_2 += 1
                else:
                    if student_cls1_c is not None:
                        loss1 = jt.sum(-teacher_cls_c[q] * jt.nn.log_softmax(student_cls1_c[v], dim=-1), dim=-1)
                        total_loss1_1 += loss1.mean()
                        n_loss_terms1_1 += 1
            
            for v in range(self.ncrops):
                if v == q:
                    if student_patch2_c is not None:
                        loss2 = jt.sum(-teacher_patch_c[q] * jt.nn.log_softmax(student_patch2_c[v], dim=-1), dim=-1)
                        mask = student_mask[v].flatten(-2, -1)
                        loss2 = jt.sum(loss2 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min_v=1.0)
                        total_loss2_2 += loss2.mean()
                        n_loss_terms2_2 += 1
                else:
                    if student_cls2_c is not None:
                        loss1 = jt.sum(-teacher_cls_c[q] * jt.nn.log_softmax(student_cls2_c[v], dim=-1), dim=-1)
                        total_loss2_1 += loss1.mean()
                        n_loss_terms2_1 += 1
            
        total_loss1_1 = total_loss1_1 / max(n_loss_terms1_1, 1) * self.lambda1
        total_loss1_2 = total_loss1_2 / max(n_loss_terms1_2, 1) * self.lambda2
        total_loss2_1 = total_loss2_1 / max(n_loss_terms2_1, 1) * self.lambda1
        total_loss2_2 = total_loss2_2 / max(n_loss_terms2_2, 1) * self.lambda2
        factor_loss_terms_patch = max(n_loss_terms1_2, n_loss_terms2_2) * 2 / (n_loss_terms1_2 + n_loss_terms2_2)
        factor_loss_terms_cls = max(n_loss_terms1_1, n_loss_terms2_1) * 2 / (n_loss_terms1_1 + n_loss_terms2_1)
        total_loss = dict(
            cls1=total_loss1_1, patch1=total_loss1_2, 
            cls2=total_loss2_1, patch2=total_loss2_2, 
            loss=(
            (total_loss1_1 + total_loss2_1) * factor_loss_terms_cls + \
                (total_loss1_2 + total_loss2_2) * factor_loss_terms_patch) * 0.5)
        self.update_center_accum(teacher_cls, teacher_patch)
        if (it + 1) % self.accum_iter == 0:
            self.update_center(teacher_cls, teacher_patch)
               
        return total_loss

    def update_center(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        with jt.no_grad():
            # cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
            # dist.all_reduce(cls_center)
            cls_center = self.center_accum
            cls_center = cls_center / (len(teacher_cls) * jt.world_size * self.accum_iter)
            self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)

            # patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
            # dist.all_reduce(patch_center)
            patch_center = self.center2_accum
            patch_center = patch_center / (len(teacher_patch) *jt.world_size * self.accum_iter)
            self.center2 = self.center2 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)

            self.center_accum.fill_(0)
            self.center2_accum.fill_(0)

    def update_center_accum(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        with jt.no_grad():
            cls_center = jt.sum(teacher_cls, dim=0, keepdim=True)
            if jt.in_mpi:
                cls_center = cls_center.mpi_all_reduce('add')
            self.center_accum = self.center_accum + cls_center

            patch_center = jt.sum(teacher_patch.mean(1), dim=0, keepdim=True)
            if jt.in_mpi:
                patch_center = patch_center.mpi_all_reduce('add')
            self.center2_accum = self.center2_accum + patch_center


# Copied from iBOT: https://github.com/bytedance/ibot
class DataAugmentationHSSL(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGray(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.ImageNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # transformation for the rest of global crops
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HSSL', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_hssl(args)
