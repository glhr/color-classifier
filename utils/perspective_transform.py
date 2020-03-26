import math
import numpy as np
import matplotlib.pyplot as plt

from skimage import transform as tf
from skimage import io


DEFAULT_TRANSFORM = tf.ProjectiveTransform(matrix=np.array(
    [[ 7.10911421e-01, -4.30153672e-01,  3.50491973e+02],
     [-2.90549210e-16,  5.11613046e-01,  1.74003107e+02],
     [-1.86212869e-19, -6.76544115e-04,  1.24287934e+00]]))


def estimate_transform():
    # src= np.array([[0, 0], [0, 718], [1278, 718], [1278, 0]])
    # dst = np.array([[272, 122], [40, 718], [1265, 718], [1023, 122]])
    src = np.array([[0, 0], [0, 718], [1278, 718], [1278, 0]])
    dst = np.array([[282, 140], [55, 715], [1255, 715], [1013, 140]])

    tform3 = tf.ProjectiveTransform()
    tform3.estimate(src, dst)
    print(tform3)
    return tform3, dst


def apply_transform(image, tform=DEFAULT_TRANSFORM):
    warped = tf.warp(image, tform, output_shape=image.shape)
    return warped


if __name__ == '__main__':
    image = io.imread("test/green.png")
    tform, dst = estimate_transform()
    warped = apply_transform(image, tform)
    fig, ax = plt.subplots(nrows=2, figsize=(3, 4))

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].plot(dst[:, 0], dst[:, 1], '.r')
    ax[1].imshow(warped, cmap=plt.cm.gray)

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.savefig('test/transform.png', dpi=500)
