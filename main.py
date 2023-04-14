import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
import numpy as np

import utils.misc as utils
from dataset import build_dataset
from utils.plot_utils import show
from utils.yolo_utils import get_anchors, get_classes
from models.yolo import YoloBody, weights_init
from loss.yolo_loss import YOLOLoss
from engine import train_one_epoch
from utils.misc import Animator


def main(args):
    device = torch.device(args.device)

    if args.dataset_file == 'sonar':
        classes_path = 'model_data/sonar_classes.txt'
        anchors_path = 'model_data/sonar_anchors.txt'
        model_path = 'model_data/yolo_sonar_weights.pth'
        input_shape = [512, 512]
    else:
        classes_path = 'model_data/voc_classes.txt'
        anchors_path = 'model_data/yolo_anchors.txt'

    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn,
                                   num_workers=args.num_workers)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 collate_fn=utils.collate_fn,
                                 drop_last=False,
                                 num_workers=args.num_workers)

    model = YoloBody(args.anchors_mask, num_classes, args.dataset_file, pretrained=args.pretrained)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if not args.pretrained:
        weights_init(model)

    if Path(model_path).exists():
        print('Load weights {}.'.format(model_path))
        pretrained_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(pretrained_dict)

    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, device, args.anchors_mask)

    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

    animator = Animator(xlabel='epoch', xlim=[1, args.epochs],
                        legend=['loss'])
    for epoch in range(args.epochs):
        train_one_epoch(model, yolo_loss, data_loader_train, optimizer, device, epoch, 0, animator)
        lr_scheduler.step()


def get_args_parser():
    parser = argparse.ArgumentParser(description='', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--train_annotation_path', default='train2023.txt')
    parser.add_argument('--val_annotation_path', default='train2023.txt')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--anchors_mask', default=[[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    parser.add_argument('--dataset_file', default='sonar')
    parser.add_argument('--coco_path')
    parser.add_argument('--sonar_path', default='/Users/zlr/Desktop/Object_Detection/MY_dataset')
    parser.add_argument('--device', default='cpu')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
