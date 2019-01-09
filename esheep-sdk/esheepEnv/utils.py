import numpy as np
import ctypes

def to_np_array(mp_arr, shape: tuple, dtype=np.float32):
    return np.frombuffer(mp_arr.get_obj(), dtype=dtype).reshape(shape)


def create_shared_data(mp_ctx, shape: tuple, dtype: str):
    size = _cacl_size(shape, dtype)
    shared_arr = mp_ctx.Array(ctypes.c_byte, size)
    return shared_arr


def _cacl_size(shape: tuple, dtype: str):
    bytes_len = 0
    if dtype == 'int32' or \
            dtype == 'int' or \
            dtype == 'float' or \
            dtype == 'float32' or \
            dtype == 'float':
        bytes_len = 4
    elif dtype == 'int8' or dtype == 'uint8':
        bytes_len = 1
    elif dtype == 'float64' or dtype == 'int64':
        bytes_len = 8
    else:
        raise Exception("error dtype:", dtype)

    length = 1
    for x in shape:
        length *= x

    size = bytes_len * length
    return size

