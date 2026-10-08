import numpy as np


def conv2d(image, kernel, padding, stride):
    c_in, h, w = image.shape
    c_out, _, k_h, k_w = kernel.shape

    padded_image = np.pad(image, ((0, 0), (padding, padding), (padding, padding)))

    out_h, out_w = (h + 2 * padding - k_h) // stride + 1, (w + 2 * padding - k_w) // stride + 1
    output = np.zeros((c_out, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            for filter in range(c_out):
                i_input, j_input = i * stride, j * stride
                patch = padded_image[:, i_input:i_input + k_h, j_input:j_input + k_w]

                output[filter, i, j] = np.sum(patch * kernel[filter])

    return output


def pooling(feature_map, pool_size, stride, strategy='max'):
    c, h, w = feature_map.shape

    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    output = np.zeros((c, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            i_input, j_input = i * stride, j * stride
            patch = feature_map[:, i_input:i_input + pool_size, j_input:j_input + pool_size]

            if strategy == 'max':
                output[:, i, j] = np.max(patch, axis=(1, 2))
            elif strategy == 'average':
                output[:, i, j] = np.mean(patch, axis=(1, 2))
            else:
                raise ValueError(f"Unknown pooling strategy: '{strategy}'. "
                                 f"Expected 'max' or 'average'.")

    return output


def relu(x):
    return np.maximum(0, x)


image = np.arange(25).reshape(5, 5)
rgb = np.stack([image] * 3, axis=-1)  # shape: (5, 5, 3)
rgb = np.moveaxis(rgb, -1, 0)  # shape: (3, 5, 5) -> channel-first
c_out, c_in, k_h, k_w = 5, 3, 3, 3
kernel = np.ones((c_out, c_in, k_h, k_w))
padding = 1
stride = 2

conv1 = conv2d(rgb, kernel, padding, stride)
relu1 = relu(conv1)
pool1 = pooling(relu1, 2, 1)
