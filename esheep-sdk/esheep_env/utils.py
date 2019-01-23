import numpy as np


def to_np_array(image_data, dtype=np.int8):
    width = image_data.width
    height = image_data.height
    pixel_length = image_data.pixel_length
    data = image_data.data
    assert len(data) == width * height * pixel_length
    return np.frombuffer(data, dtype=dtype).reshape((height, width, pixel_length))

