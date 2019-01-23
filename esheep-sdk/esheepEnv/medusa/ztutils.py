# Author: Taoz
# Date  : 7/1/2018
# Time  : 11:26 PM
# FileName: ztutils.py
import os


def mkdir_if_not_exist(path):
    print('mkdir_if_not_exist:[%s]' % path)
    if not os.path.exists(path):
        os.makedirs(path)
