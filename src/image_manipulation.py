"""
Copyright 2021 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np


from skimage import data, io
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb, rgb2gray

from tempfile import NamedTemporaryFile

from PIL import Image


def join_images(
    images, col_wrap=0, bgcolor=(255, 255, 255, 0), img_size=None, padding=0
):
    if img_size:
        for im in images:
            im.thumbnail(img_size, Image.ANTIALIAS)
    n_images = len(images)
    widths, heights = zip(*(i.size for i in images))
    row_widths = []
    row_heights = []
    if col_wrap > 0:
        total_width = 0
        for i in range(0, n_images, col_wrap):
            row_widths.append(sum(widths[i : i + col_wrap]))
            row_heights.append(max(heights[i : i + col_wrap]))
    else:
        col_wrap = n_images
        row_widths.append(sum(widths))
        row_heights.append(max(heights))
    total_width = max(row_widths) + (padding * (col_wrap - 1))
    max_height = sum(row_heights) + (padding * (len(row_heights) - 1))
    row_y_offsets = [
        sum(row_heights[:i]) + i * padding for i, h in enumerate(row_heights)
    ]
    new_im = Image.new("RGB", (total_width, max_height), color=bgcolor)
    x_offset, y_offset = 0, 0
    for i, im in enumerate(images):
        row = i // col_wrap
        col = i % col_wrap
        y_offset = row_y_offsets[row]
        x_offset = 0 if col == 0 else x_offset
        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0] + padding
    return new_im


def saliency_points_to_image(
    all_salient_points, use_softmax=False, q=90, temperature=1
):
    x, y, z = zip(*sorted(all_salient_points, key=lambda x: (x[1], x[0])))
    saliency_img = np.array(z).reshape(len(set(y)), len(set(x)))
    if use_softmax:
        new_img = np.exp((saliency_img - saliency_img.max()) / temperature)
        new_img = new_img / new_img.sum()
    else:
        new_img = 1 / (1 + np.exp(-saliency_img / temperature))
    threshold = np.percentile(new_img, q)
    new_img = np.where(new_img > threshold, new_img, 0)
    new_img = np.ceil(new_img * 255).astype("uint8")
    return new_img, threshold


def segment_saliency_map(saliency_map_resized):
    with NamedTemporaryFile(suffix=".jpg") as fp:
        saliency_map_resized.save(fp, "JPEG")
        image = io.imread(fp.name)
    # apply threshold
    thresh = threshold_otsu(image)
    bw = closing(image >= thresh, square(3))
    # remove artifacts connected to image border
    cleared = clear_border(bw)
    cleared = bw
    # label image regions
    label_image = label(cleared)
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = label2rgb(
        label_image, image=image, bg_label=0, bg_color=None, kind="overlay"
    )
    regions = regionprops(label_image)
    return image_label_overlay, regions


def get_image_saliency_map(img_path, model, use_softmax=False, q=90, temperature=1):
    output = model.get_output(img_path)
    img = mpimg.imread(img_path)
    new_img, threshold = saliency_points_to_image(
        output["all_salient_points"],
        use_softmax=use_softmax,
        q=q,
        temperature=temperature,
    )
    n_img = Image.fromarray(new_img, "L")
    saliency_map_resized = n_img.resize(img.shape[::-1][1:], Image.NEAREST)
    image_label_overlay, regions = segment_saliency_map(saliency_map_resized)
    return img, image_label_overlay, regions, threshold


def plot_saliency_map(
    img, image_label_overlay, regions, ax=None, use_softmax=False, q=90, temperature=1
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, alpha=0.4, aspect="auto")
    ax.imshow(image_label_overlay, alpha=0.5, aspect="auto")
    for region in regions:
        # take regions with large enough areas
        if region.area >= 100:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = Rectangle(
                (minc, minr),
                maxc - minc,
                maxr - minr,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
            ax.add_patch(rect)
    ax.set_axis_off()


def plot_region_segments(img, regions, ax=None):
    n_regions = len(regions)
    if ax is None:
        fig, ax = plt.subplots(1, n_regions, figsize=(4 * n_regions, 4))
    for i, region in enumerate(regions):
        # take regions with large enough areas
        if region.area >= 100:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            segment_img = img[minr:maxr, minc:maxc]
            ax[i].imshow(segment_img, aspect="auto")
        ax[i].set_axis_off()


def process_image(img_path, model, w=2, h=2):
    img, image_label_overlay, regions, threshold = get_image_saliency_map(
        img_path, model
    )
    n_regions = len(regions)
    fig = plt.figure(constrained_layout=True, figsize=(w * n_regions, h * 3))
    gs = fig.add_gridspec(3, n_regions)
    ax = fig.add_subplot(gs[:2, :])
    ax.set_title(f"Threshold: {threshold:.3f}")
    plot_saliency_map(img, image_label_overlay, regions, ax=ax)
    ax = [fig.add_subplot(gs[2, i]) for i in range(n_regions)]
    plot_region_segments(img, regions, ax=ax)
    return n_regions
