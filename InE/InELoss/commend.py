import numpy as np

def EN(img):  # entropy
    batch_size = img.shape[0]
    entropies = 0
    for i in range(batch_size):
        a = np.uint8(np.round(img[i, 0])).flatten()  # Flatten the single channel image
        h = np.bincount(a, minlength=256) / a.size  # Ensure all 256 bins are counted
        entropy = -sum(h * np.log2(h + (h == 0)))  # Calculate entropy
        entropies += entropy
    return entropies

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式

    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """

    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    return Y, Cb, Cr
