import numpy as np


def bytes_to_array(image_data):
    width = image_data.width
    height = image_data.height
    pixel_length = image_data.pixel_length
    data = image_data.data
    assert len(data) == width * height * pixel_length
    a = np.zeros((height, width))
    r = np.zeros((height, width))
    g = np.zeros((height, width))
    b = np.zeros((height, width))
    for i in range(len(data)):
        row = int(i / (width * pixel_length))
        column = int(i % (width * pixel_length) / pixel_length)
        attribute = i % pixel_length
        if attribute == 0:
            a[row, column] = data[i]
        elif attribute == 1:
            r[row, column] = data[i]
        elif attribute == 2:
            g[row, column] = data[i]
        else:
            b[row, column] = data[i]
    return np.array((a, r, g, b))

