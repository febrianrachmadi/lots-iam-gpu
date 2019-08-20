#!/usr/bin/python

from IAM_GPU_NIFTI_2d_lib import *
from IAM_GPU_MAT_2d_lib import *
from iam_params import *

import sys

def main():
    ## NOTE: Put parameters in iam_params.py
    ## Parameters are loaded by "from iam_params import *" line

    # Check if using .mat or .nii.gz files
    mat_files, nifti_files = False, False
    with open(csv_filename, mode='r') as csv_file:
        reader = csv.reader(csv_file)

        for row in reader:
            if row[0][-4:] == ".mat":
                mat_files = True
            elif row[2][-7:] == ".nii.gz":
                nifti_files = True
    if mat_files and nifti_files:
        raise ValueError("Inconsistent input file formats.")

    elif mat_files:
        print('Check OpenCV version: ' + cv2.__version__ + '\n')
        print(cuda.current_context().get_memory_info())
        print('Initialisation is done..\n')
        iam_lots_gpu_mat_compute(csv_filename = csv_filename,
                         patch_size       = patch_size,
                         blending_weights = blending_weights,
                         num_sample       = num_samples_all,
                         alpha            = alpha,
                         bin_tresh        = bin_tresh,
                         save_jpeg        = save_jpeg,
                         delete_intermediary = delete_intermediary)

    elif nifti_files:
        print('Check OpenCV version: ' + cv2.__version__ + '\n')
        print(cuda.current_context().get_memory_info())
        print('Initialisation is done..\n')
        iam_lots_gpu_nifti_compute(output_filedir   = output_filedir,
                         csv_filename     = csv_filename,
                         patch_size       = patch_size,
                         blending_weights = blending_weights,
                         num_sample       = num_samples_all,
                         alpha            = alpha,
                         bin_tresh        = bin_tresh,
                         save_jpeg        = save_jpeg,
                         delete_intermediary = delete_intermediary,
                         nawm_preprocessing  = nawm_preprocessing)


if __name__ == "__main__":
    main()
