import argparse
import copy
import torch
from torch.utils.data import DataLoader
import numpy as np
from resnet import resnet20, resnet32, resnet44, resnet56
import torch.nn as nn
import timm
from continuum import rehearsal
from utils import MetricLogger, SoftTarget, init_distributed_mode, build_dataset


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Class-Incremental Learning training and evaluation script', add_help=False)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_bases', default=50, type=int)
    parser.add_argument('--increment', default=10, type=int)
    parser.add_argument('--backbone', default="resnet32", type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--input_size', default=32, type=int)
    parser.add_argument('--color_jitter', default=0.4, type=float)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--herding_method', default="barycenter", type=str)
    parser.add_argument('--memory_size', default=2000, type=int)
    parser.add_argument('--fixed_memory', default=False, action="store_true")
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--num_epochs', default=140, type=int)
    parser.add_argument('--smooth', default=0.0, type=float)
    parser.add_argument('--eval_every_epoch', default=5, type=float)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--data_set', default='cifar')
    parser.add_argument('--data_path', default='/data/data/data/cifar100')
    parser.add_argument('--lambda_kd', default=0.5, type=float)
    parser.add_argument('--dynamic_lambda_kd', action="store_true")
    return parser


def init_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_parameters(m, requires_grad=False):
    if m is None:
        return

    if isinstance(m, nn.Parameter):
        m.requires_grad = requires_grad
    else:
        for p in m.parameters():
            p.requires_grad = requires_grad


def get_backbone(args):
    if args.backbone == "resnet32":
        backbone = resnet32()
    elif args.backbone == "resnet20":
        backbone = resnet20()
    elif args.backbone == "resnet44":
        backbone = resnet44()
    elif args.backbone == "resnet56":
        backbone = resnet56()
    else:
        raise NotImplementedError(f'Unknown backbone {args.model}')

    return backbone


class CilClassifier(nn.Module):
    def __init__(self, embed_dim, nb_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = nn.ModuleList([nn.Linear(embed_dim, nb_classes).cuda()])

    def __getitem__(self, index):
        return self.heads[index]

    def __len__(self):
        return len(self.heads)

    def forward(self, x):
        logits = torch.cat([head(x) for head in self.heads], dim=1)
        return logits

    def adaption(self, nb_classes):
        self.heads.append(nn.Linear(self.embed_dim, nb_classes).cuda())


class CilModel(nn.Module):
    def __init__(self, backbone):
        super(CilModel, self).__init__()
        self.backbone = get_backbone(backbone)
        self.fc = None

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
        out = self.fc(x)
        return out, x

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self, names=["all"]):
        freeze_parameters(self, requires_grad=True)
        self.train()
        for name in names:
            if name == 'fc':
                freeze_parameters(self.fc)
                self.fc.eval()
            elif name == 'backbone':
                freeze_parameters(self.backbone)
                self.backbone.eval()
            elif name == 'all':
                freeze_parameters(self)
                self.eval()
            else:
                raise NotImplementedError(
                    f'Unknown module name to freeze {name}')
        return self

    def prev_model_adaption(self, nb_classes):
        if self.fc is None:
            self.fc = CilClassifier(self.feature_dim, nb_classes).cuda()
        else:
            self.fc.adaption(nb_classes)

    def after_model_adaption(self, nb_classes, args):
        if args.task_id > 0:
            self.weight_align(nb_classes)

    @torch.no_grad()
    def weight_align(self, nb_new_classes):
        w = torch.cat([head.weight.data for head in self.fc], dim=0)
        norms = torch.norm(w, dim=1)

        norm_old = norms[:-nb_new_classes]
        norm_new = norms[-nb_new_classes:]

        gamma = torch.mean(norm_old) / torch.mean(norm_new)
        print(f"old norm / new norm ={gamma}")
        self.fc[-1].weight.data = gamma * w[-nb_new_classes:]


@torch.no_grad()
def eval(model, val_loader):
    metric_logger = MetricLogger(delimiter="  ")
    criterion = nn.CrossEntropyLoss()
    model.eval()
    for images, target, task_ids in val_loader:
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        logits, _ = model(images)
        loss = criterion(logits, target)
        acc1, acc5 = timm.utils.accuracy(
            logits, target, topk=(1, min(5, logits.shape[1])))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    metric_logger.synchronize_between_processes()
    print(' Acc@1 {top1.global_avg:.3f}  loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return metric_logger.acc1.global_avg


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        'Class-Incremental Learning training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    init_distributed_mode(args)

    init_seed(args)

    args.class_order = [68, 56, 78, 8,
                        23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50, 28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69, 36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33]
    scenario_train, args.nb_classes = build_dataset(is_train=True, args=args)
    scenario_val, _ = build_dataset(is_train=False, args=args)

    model = CilModel(args)
    model = model.cuda()
    model_without_ddp = model

    torch.distributed.barrier()

    memory = rehearsal.RehearsalMemory(
        memory_size=args.memory_size,
        herding_method=args.herding_method,
        fixed_memory=args.fixed_memory
    )
    teacher_model = None

    criterion = nn.CrossEntropyLoss(label_smoothing=args.smooth)

    kd_criterion = SoftTarget(T=2)
    args.increment_per_task = [args.num_bases] + \
        [args.increment for _ in range(len(scenario_train) - 1)]
    args.known_classes = 0
    acc1s = []
    for task_id, dataset_train in enumerate(scenario_train):
        args.task_id = task_id

        dataset_val = scenario_val[:task_id + 1]
        if task_id > 0:
            dataset_train.add_samples(*memory.get())
        train_sampler = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        val_sampler = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=args.world_size, rank=args.rank, shuffle=False)
        train_loader = DataLoader(dataset_train, batch_size=args.batch_size,
                                  sampler=train_sampler, num_workers=10, pin_memory=True)
        val_loader = DataLoader(
            dataset_val, batch_size=args.batch_size, sampler=val_sampler, num_workers=10)

        model_without_ddp.prev_model_adaption(args.increment_per_task[task_id])

        model = torch.nn.parallel.DistributedDataParallel(
            model_without_ddp, device_ids=[args.rank])

        optimizer = torch.optim.SGD(model_without_ddp.parameters(
        ), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs)

        for epoch in range(args.num_epochs):
            model.train()
            train_sampler.set_epoch(epoch)
            metric_logger = MetricLogger(delimiter="  ")
            for idx, (inputs, targets, task_ids) in enumerate(train_loader):
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
                logits, _ = model(inputs)
                loss_ce = criterion(logits, targets)
                if teacher_model is not None:
                    t_logits, _ = teacher_model(inputs)
                    loss_kd = args.lambda_kd * \
                        kd_criterion(logits[:, :args.known_classes], t_logits)
                else:
                    loss_kd = torch.tensor(0.).cuda(non_blocking=True)
                loss = loss_ce + loss_kd
                acc1, acc5 = timm.utils.accuracy(
                    logits, targets, topk=(1, min(5, logits.shape[1])))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.distributed.barrier()
                metric_logger.update(ce=loss_ce)
                metric_logger.update(kd=loss_kd)
                metric_logger.update(loss=loss)
                metric_logger.update(acc1=acc1)
            metric_logger.synchronize_between_processes()
            lr_scheduler.step()
            print(
                f"train states: epoch :[{epoch+1}/{args.num_epochs}] {metric_logger}")

            if (epoch+1) % args.eval_every_epoch == 0:
                eval(model, val_loader)

        model_without_ddp.after_model_adaption(
            args.increment_per_task[task_id], args)
        acc1 = eval(model, val_loader)
        acc1s.append(acc1)
        print(f"task id = {task_id}  @Acc1 = {acc1:.5f}, acc1s = {acc1s}")
        teacher_model = model_without_ddp.copy().freeze()

        unshuffle_train_loader = DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=False)
        features = []
        for i, (inputs, labels, task_ids) in enumerate(unshuffle_train_loader):
            inputs = inputs.cuda(non_blocking=True)
            features.append(model_without_ddp.extract_vector(
                inputs).detach().cpu().numpy())
        features = np.concatenate(features, axis=0)
        memory.add(
            *dataset_train.get_raw_samples(), features
        )
        args.known_classes += args.increment_per_task[task_id]

