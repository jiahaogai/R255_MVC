import os
import sys
from pathlib import Path

import config
import cv2
import numpy as np
import pandas as pd
import torch as th
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import make_blobs

# from mmhb.utils import Config
# from mmhb.loader import *
DATA_DIR = Path(os.path.abspath(__file__)).parents[2] / "data"


def export_dataset(name, views, labels):
    processed_dir = DATA_DIR / "processed"

    os.makedirs(processed_dir, exist_ok=True)
    file_path = processed_dir / f"{name}.npz"
    npz_dict = {"labels": labels, "n_views": len(views)}
    for i, v in enumerate(views):
        npz_dict[f"view_{i}"] = v
    np.savez(file_path, **npz_dict)


def _concat_edge_image(img):
    img = np.array(img)
    dilation = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
    edge = dilation - img
    return np.stack((img, edge), axis=-1)


def _mnist(add_edge_img, dataset_class=torchvision.datasets.MNIST):
    img_transforms = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    if add_edge_img:
        img_transforms.insert(0, _concat_edge_image)
    transform = transforms.Compose(img_transforms)
    dataset = dataset_class(root=config.DATA_DIR / "raw", train=True, download=True, transform=transform)

    loader = th.utils.data.DataLoader(dataset, batch_size=len(dataset))
    data, labels = list(loader)[0]
    return data, labels


def mnist_mv():
    data, labels = _mnist(add_edge_img=True)
    views = np.split(data, data.shape[1], axis=1)
    export_dataset("mnist_mv", views=views, labels=labels)


def fmnist():
    data, labels = _mnist(add_edge_img=True, dataset_class=torchvision.datasets.FashionMNIST)
    views = np.split(data, data.shape[1], axis=1)
    export_dataset("fmnist", views=views, labels=labels)


def ccv():
    ccv_dir = config.DATA_DIR / "raw" / "CCV"

    def _load_train_test(typ, suffix="Feature"):
        if typ:
            typ += "-"
        train = np.loadtxt(ccv_dir / f"{typ}train{suffix}.txt")
        test = np.loadtxt(ccv_dir / f"{typ}test{suffix}.txt")
        return np.concatenate((train, test), axis=0)

    views = [_load_train_test(typ) for typ in ["STIP", "SIFT", "MFCC"]]
    labels = _load_train_test("", suffix="Label")

    # Only include videos with exactly one label
    row_mask = (labels.sum(axis=1) == 1)
    labels = labels[row_mask].argmax(axis=1)
    views = [v[row_mask] for v in views]
    export_dataset("ccv", views=views, labels=labels)


def coil():
    from skimage.io import imread

    data_dir = DATA_DIR / "raw" / "COIL"
    img_size = (1, 128, 128)
    n_objs = 20
    n_imgs = 72
    n_views = 3
    assert n_imgs % n_views == 0

    n = (n_objs * n_imgs) // n_views

    imgs = np.empty((n_views, n, *img_size))
    labels = []

    img_idx = np.arange(n_imgs)

    for obj in range(n_objs):
        obj_img_idx = np.random.permutation(img_idx).reshape(n_views, n_imgs // n_views)
        labels += (n_imgs // n_views) * [obj]

        for view, indices in enumerate(obj_img_idx):
            for i, idx in enumerate(indices):
                fname = data_dir / f"obj{obj + 1}__{idx}.png"
                img = imread(fname)[None, ...]
                imgs[view, ((obj * (n_imgs // n_views)) + i)] = img

    assert not np.isnan(imgs).any()
    views = [imgs[v] for v in range(n_views)]
    labels = np.array(labels)
    print(views[0].shape)
    # print(len(views))
    print(labels.shape)
    export_dataset("coil", views=views, labels=labels)


def blobs_overlap():
    nc = 1000
    ndim = 2
    view_1, l1 = make_blobs(n_samples=[nc, 2 * nc], n_features=ndim, cluster_std=1.0, shuffle=False)
    view_2, l2 = make_blobs(n_samples=[2 * nc, nc], n_features=ndim, cluster_std=1.0, shuffle=False)
    labels = l1 + l2
    export_dataset("blobs_overlap", views=[view_1, view_2], labels=labels)


def blobs_overlap_5():
    nc = 500
    ndim = 2
    view_1, _ = make_blobs(n_samples=[3 * nc, 2 * nc], n_features=ndim, cluster_std=1.0, shuffle=False)
    view_2, _ = make_blobs(n_samples=[1 * nc, 2 * nc, 2 * nc], n_features=ndim, cluster_std=1.0, shuffle=False)
    view_2[(2 * nc): (4 * nc)] = view_2[(2 * nc): (4 * nc)][::-1]
    labels = np.concatenate(([nc * [i] for i in range(5)]))
    export_dataset("blobs_overlap_5", views=[view_1, view_2], labels=labels)


def chest_xray():
    from skimage.io import imread
    data_dir = DATA_DIR / "raw" / "chestx"
    image_dir = data_dir / "images" / "images_normalized"
    projections_df = pd.read_csv(data_dir / "indiana_projections.csv")
    # Start by finding the rows with duplicate values in the uid column
    projections_df = projections_df[projections_df['uid'].duplicated(keep=False)]
    # print(projections_df.head())
    # projections_df = duplicated_rows
    n_views = 2
    views_1, views_2 = [], []
    labels = []
    image_size = (1, 2048, 2048)

    # 根据uid分组
    grouped = projections_df.groupby("uid")
    used_num = int(len(grouped) * 0.2)

    for uid, group in grouped:
        if uid >= used_num:
            continue
        if len(group) == n_views:
            labels.append(uid)
            for i, row in group.iterrows():
                # print(i)
                # path = image_dir / row['filename']
                # print((image_dir / row['filename']))
                img = cv2.imread(str(image_dir / row['filename']), cv2.IMREAD_GRAYSCALE)
                # img = img[None, ...]
                # 将图片的大小调整为(1, 2048, 2048)
                img = cv2.resize(img, (128, 128))
                # 转为(1, 2048, 2048)
                img = img[None, ...]
                # print(img.shape)
                if (i + 1) % 2 == 1:
                    views_1.append(img)
                else:
                    views_2.append(img)
    assert len(labels) == len(views_1) == len(views_2), f"{len(labels)}, {len(views_1)}, {len(views_2)}"
    views_1 = np.concatenate(views_1, axis=0)
    # 调整形状，增加一个维度在第二个位置
    views_1 = np.expand_dims(views_1, axis=1)
    views_2 = np.concatenate(views_2, axis=0)
    views_2 = np.expand_dims(views_2, axis=1)
    views = [views_1, views_2]
    labels = np.array(labels)
    print(views_1.shape)
    print(views_2.shape)
    print(labels.shape)
    export_dataset("chestx", views=views, labels=labels)


LOADERS = {
    "mnist_mv": mnist_mv,
    "ccv": ccv,
    "blobs_overlap": blobs_overlap,
    "blobs_overlap_5": blobs_overlap_5,
    "fmnist": fmnist,
    "coil": coil,
    "chestx": chest_xray
}


def main():
    export_sets = sys.argv[1:] if len(sys.argv) > 1 else LOADERS.keys()
    for name in export_sets:
        print(f"Exporting dataset '{name}'")
        LOADERS[name]()


if __name__ == '__main__':
    main()
