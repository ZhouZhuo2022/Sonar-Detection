import torch
from matplotlib import pyplot as plt
from .get_wh import wh
from .box_ops import box_cxcywh_to_xyxy


def bbox_to_rect(bbox, color):
    # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
    # ((左上x,左上y),宽,高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2)


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # @save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.show()
    # return axes


def show_image(img, bboxes, bbox_scaled):
    w, h = wh(img)

    if torch.is_tensor(img):
        fig = plt.imshow(img.squeeze(0).numpy())
    else:
        plt.imshow(img)

    if bbox_scaled:
        bboxes = box_cxcywh_to_xyxy(bboxes)
        bboxes *= torch.tensor([w, h, w, h])

    if torch.is_tensor(bboxes):
        bboxes = bboxes.numpy()

    for bbox in bboxes:
        rect = plt.Rectangle(
            xy=(bbox[0], bbox[1]),
            width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
            fill=False, edgecolor='red', linewidth=2)
        fig.axes.add_patch(rect)

    plt.show()


def show(dataset, index, bbox_scaled=False, m=False):
    if not m:
        data = dataset[index]
        img = data[0]
        bboxes = data[1]['boxes']
        show_image(img, bboxes, bbox_scaled)
    else:
        pass
