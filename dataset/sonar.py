from pathlib import Path
from torch.utils.data.dataset import Dataset
import torch
import numpy as np

import dataset.transforms as T


class SonarDetection(Dataset):
    def __init__(self, annotation_lines, transforms=None):
        super().__init__()
        self.annotation_lines = annotation_lines
        self.prepare = get_data_target
        self._transforms = transforms
        self.length = len(self.annotation_lines)

    def __getitem__(self, index):
        image, target = self.prepare(self.annotation_lines[index])
        if self._transforms is not None:
            image, target = self._transforms(image, target)
        return image, target

    def __len__(self):
        return self.length


def get_data_target(annotation_line):
    line = annotation_line.split()
    image = torch.load(line[0])
    h, w = image.shape[-2:]

    boxes = np.array([np.array(list(map(int, box.split(',')))[:4]) for box in line[1:]])
    boxes = torch.as_tensor(boxes, dtype=torch.float32)

    classes = np.array([np.array(list(map(int, box.split(',')))[-1]) for box in line[1:]])
    classes = torch.tensor(classes, dtype=torch.int64)

    target = {'boxes': boxes, 'labels': classes,
              'orig_size': torch.as_tensor([int(h), int(w)]),
              'size': torch.as_tensor([int(h), int(w)])}

    return image, target


def make_sonar_transforms(image_set):

    normalize = T.Compose([
        T.Normalize([0.485], [0.229])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600], max_size=1333),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales),
                ])
            ),
            normalize,
        ])
    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_sonar_transforms1(image_set):

    normalize = T.Compose([
        T.Normalize([0.485], [0.229])
    ])
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            normalize
        ])
    if image_set == 'val':
        return T.Compose([
            normalize,
        ])


def build_sonar(image_set, args):
    root = Path(args.sonar_path)
    assert root.exists(), f'provided COCO path {root} does not exist'

    annotation_filename = image_set + '.txt'
    annotation_path = root / Path(annotation_filename)
    with open(annotation_path, 'r') as f:
        train_lines = f.readlines()

    dataset = SonarDetection(train_lines,
                             transforms=make_sonar_transforms1(image_set),
                             )
    return dataset

