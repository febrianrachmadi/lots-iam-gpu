#!/usr/bin/python

from IAM_GPU_lib import *
from iam_params import *

import sys

def main():
    print('Check OpenCV version: ' + cv2.__version__ + '\n')
    print(cuda.current_context().get_memory_info())
    print('Initialisation is done..\n')
        
    ## NOTE: Put parameters in iam_params.py
    ## Parameters are loaded by "from iam_params import *" line
    iam_lots_gpu_compute(
                     output_filedir   = output_filedir,
                     csv_filename     = csv_filename,
                     patch_size       = patch_size,
                     blending_weights = blending_weights,
                     num_sample       = num_samples_all,
                     alpha            = alpha,
                     bin_tresh        = bin_tresh,
                     save_jpeg        = save_jpeg,
                     delete_intermediary = delete_intermediary)

if __name__ == "__main__":
    main()