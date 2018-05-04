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
    iam_lots_gpu_compute(output_filedir,csv_filename,patch_size,num_samples_all,save_jpeg)

if __name__ == "__main__":
    main()