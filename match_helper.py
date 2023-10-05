""""Helper functions for image matching."""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def imresize(img, scale):
    im = Image.fromarray(img)
    new_size = (np.array(im.size) * scale).astype(np.int)
    im = im.resize((*new_size,))
    return np.array(im)


def extract_corrs(img1, img2, bi_check=True, ratio_th=None, numkp=2000):
    # Create correspondences with SIFT
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=numkp, contrastThreshold=1e-5)

    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)
    xy1 = np.array([_kp.pt for _kp in kp1])
    xy2 = np.array([_kp.pt for _kp in kp2])

    # Memory efficient way to calculate l2 distance.
    distmat = np.sqrt(
        np.sum(desc1 ** 2, axis=1, keepdims=True)
        + np.sum(desc2 ** 2, axis=1)
        - 2 * np.dot(desc1, desc2.T)
    )
    idx_sort = np.argsort(distmat, axis=1)
    idx_top1 = idx_sort[:, 0]
    corrs = np.concatenate([xy1, xy2.take(idx_top1, axis=0)], axis=1)

    filter = np.ones(len(corrs)) > 0

    # Lowe's ratio test.
    if ratio_th is not None:
        idx_top2 = idx_sort[:, :2]
        distmat_top2 = np.take_along_axis(distmat, idx_top2, axis=1)
        ratio = distmat_top2[:, 0] / distmat_top2[:, 1]
        filter_ratio = ratio < ratio_th
        filter = np.all(np.stack([filter, filter_ratio], axis=0), axis=0)

    # Bidirectional check.
    if bi_check:
        idx_sort1 = np.argsort(distmat, axis=0)[
            0, :
        ]  # choose the best from 1 to 0
        filter_bi_check = idx_sort1[idx_top1] == np.arange(len(idx_top1))
        filter = np.all(np.stack([filter, filter_bi_check], axis=0), axis=0)
    return corrs[filter]


def visualize_corrs(img1, img2, corrs, mask=None):
    if mask is None:
        mask = np.ones(len(corrs)).astype(bool)

    scale1 = 1.0
    scale2 = 1.0
    if img1.shape[1] > img2.shape[1]:
        scale2 = img1.shape[1] / img2.shape[1]
        w = img1.shape[1]
    else:
        scale1 = img2.shape[1] / img1.shape[1]
        w = img2.shape[1]
    # Resize if too big
    max_w = 400
    if w > max_w:
        scale1 *= max_w / w
        scale2 *= max_w / w
    img1 = imresize(img1, scale1)
    img2 = imresize(img2, scale2)

    x1, x2 = corrs[:, :2], corrs[:, 2:]
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img = np.zeros((h1 + h2, max(w1, w2), 3), dtype=img1.dtype)
    img[:h1, :w1] = img1
    img[h1:, :w2] = img2
    # Move keypoints to coordinates to image coordinates
    x1 = x1 * scale1
    x2 = x2 * scale2
    # recompute the coordinates for the second image
    x2p = x2 + np.array([[0, h1]])
    fig = plt.figure(frameon=False)
    fig = plt.imshow(img)

    cols = [
        [0.0, 0.67, 0.0],
        [0.9, 0.1, 0.1],
    ]
    lw = 0.5
    alpha = 1

    # Draw outliers
    _x1 = x1[~mask]
    _x2p = x2p[~mask]
    xs = np.stack([_x1[:, 0], _x2p[:, 0]], axis=1).T
    ys = np.stack([_x1[:, 1], _x2p[:, 1]], axis=1).T
    plt.plot(
        xs,
        ys,
        alpha=alpha,
        linestyle="-",
        linewidth=lw,
        aa=False,
        color=cols[1],
    )

    # Draw Inliers
    _x1 = x1[mask]
    _x2p = x2p[mask]
    xs = np.stack([_x1[:, 0], _x2p[:, 0]], axis=1).T
    ys = np.stack([_x1[:, 1], _x2p[:, 1]], axis=1).T
    plt.plot(
        xs,
        ys,
        alpha=alpha,
        linestyle="-",
        linewidth=lw,
        aa=False,
        color=cols[0],
    )

    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()


def visualize_epipolar_lines(img1, img2, pts1, pts2, F, colors):
    fig = plt.imshow(img1)

    # For each point
    for (x1, y1), (x2, y2), c in zip(pts1, pts2, colors):
        # Different color for each point

        # Draw point
        plt.plot(
            np.array([x1]),
            np.array([y1]),
            "o",
            color=c,
        )

        # (5 points) TODO: Draw the corresponding epipolar line for this
        # point, using the coordinates x2, y2, and the fundamental matrix F. An
        # easy way to draw lines is to find the line equation, and find points
        # that the line should go through. In my implementation I simply find
        # the points at the borders of the image, and draw a line through them.
        # Note that I do manually enforce that the drawing area is restricted
        # to the image, meaning that you can also draw outside of the image and
        # get away with it.
        homogeneous_x2 = np.array([x2, y2, 1])
        epipolar_line_1 = F.T @ homogeneous_x2
        start_y = (-epipolar_line_1[0] * 0 - epipolar_line_1[2]) / epipolar_line_1[1]
        end_y = (-epipolar_line_1[0] * img1.shape[1] - epipolar_line_1[2]) / epipolar_line_1[1]
        plt.plot(np.array([0, img1.shape[1]]), np.array([start_y, end_y]), color=c)

    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    ax = plt.gca()
    ax.set_axis_off()
    plt.xlim([0, img1.shape[1]])
    plt.ylim([img1.shape[0], 0])
    plt.show()
