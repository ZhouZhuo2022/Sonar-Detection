from .coco import build_coco
from .sonar import build_sonar


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    elif args.dataset_file == 'sonar':
        return build_sonar(image_set, args)
    else:
        raise ValueError(f'dataset {args.dataset_file} not supported')