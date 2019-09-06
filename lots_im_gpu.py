#!/usr/bin/python

from LOTS_IM_GPU_FUNction import *
from lots_im_parameters import *

import sys

def main():
    print('Check OpenCV version: ' + cv2.__version__ + '\n')
    print(cuda.current_context().get_memory_info())
    print('Initialisation is done..\n')

    ## NOTE: Put parameters in lots_im_parameters.py
    ## Parameters are loaded by "from lots_im_parameters import *" line
    print("--- PARAMETERS - CHECKED ---")
    print('Patch size(s): ' + str(patch_size))
    print('Number of samples (all): ' + str(num_samples_all))
    print('Save JPEGs? ' + str(save_jpeg))
    print("--- PARAMETERS - CHECKED ---")

    print('--\nReady..')
    
    '''
    Name of csv file (note to user: you can change this variable)

    Default format of the CSV input file (NOTE: spaces are used to make the format clearer):

        input_file_path, output_folder_path, patient_code

    Example (NOTE: spaces are used to make the format clearer):

        W:/DMP01/V1/flair.mat,W:/results_lots_iam_3d,DMP01_V1
        W:/DMP01/V2/flair.mat,W:/results_lots_iam_3d,DMP01_V2
        W:/DMP01/V3/flair.mat,W:/results_lots_iam_3d,DMP01_V3

    '''
    csv_filename = "ADNI_small_test_pipeline.csv"

    with open(csv_filename, newline='') as csv_file:
        num_subjects = len(csv_file.readlines())
        print('Number of subject(s): ' + str(num_subjects))

    ''' For each experiments using different number of target patches.. '''
    for ii_s in range(0, len(num_samples_all)):
        with open(csv_filename, newline='') as csv_file:
            reader = csv.reader(csv_file)        

            ''' Set output directory based on the number of target patches '''
            num_samples = num_samples_all[ii_s]
            set_output_dir = output_filedir + '_' + str(num_samples) + 's'
        
            timer_idx = 0
            elapsed_times_all = np.zeros((num_subjects))
            elapsed_times_patch_all = []
            for row in reader:
                data = row[1]
                print('--\nNow processing data: ' + data)
                
                '''
                Read MRI files and do pre-processing.
                
                NOTE: The current version of demo reads the required MRI files and
                performs pre-processing outside the main LOTS-IM function.
                '''

                inputSubjectDir = row[0] + '/' + row[1]
                print('Input filename (full path): ' + inputSubjectDir)

                mri_nii = nib.load(row[2])
                icv_nii = nib.load(row[3])
                csf_nii = nib.load(row[4])

                mri_data = np.squeeze(mri_nii.get_data())
                icv_data = np.squeeze(icv_nii.get_data())
                csf_data = np.squeeze(csf_nii.get_data())
                print(' -> Loading FLAIR + ICV + CSF: OK!')

                ''' Make sure that all brain masks are binary masks, not probability '''
                bin_tresh = 0.5
                icv_data[icv_data > bin_tresh] = 1
                icv_data[icv_data <= bin_tresh] = 0
                csf_data[csf_data > bin_tresh] = 1
                csf_data[csf_data <= bin_tresh] = 0

                ''' Read and open NAWM data if available '''
                nawm_available = len(row) > 5 and row[5]
                if nawm_available:
                    print(" -> NAWM mask is avalilable!")
                    print(" --> " + row[5])
                    nawm_nii = nib.load(row[5])
                    nawm_data = np.squeeze(nawm_nii.get_data())
                    nawm_data[nawm_data > bin_tresh] = 1
                    nawm_data[nawm_data <= bin_tresh] = 0
                else:
                    print(" -> Cortex mask is NOT avalilable!")

                ''' Read and open Cortex data if available '''
                cortex_available = len(row) > 6 and row[6]
                if cortex_available:
                    print(" -> Cortex mask is avalilable!")
                    print(" --> " + row[6])
                    cortex_nii = nib.load(row[6])
                    cortex_data = np.squeeze(cortex_nii.get_data())
                    cortex_data[cortex_data > bin_tresh] = 1
                    cortex_data[cortex_data <= bin_tresh] = 0
                else:
                    print(" -> Cortex mask is NOT avalilable!")

                del csf_nii   # Free memory

                ''' ICV Erosion '''
                print(' -> ICV Erosion -- note: skimage library')
                print(' -> ICV shape:' + str(icv_data.shape) + '\n')
                for ii in range(0, icv_data.shape[2]):
                    kernel = kernel_sphere(7)
                    icv_data[:,:,ii] = skimorph.erosion(icv_data[:,:,ii],kernel)
                    kernel = kernel_sphere(11)
                    icv_data[:,:,ii] = skimorph.erosion(icv_data[:,:,ii],kernel)
                    if nawm_available:
                        kernel = kernel_sphere(3)
                        nawm_data[:,:,ii] = scimorph.binary_fill_holes(nawm_data[:,:,ii])
                        nawm_data[:,:,ii] = skimorph.erosion(nawm_data[:,:,ii],kernel)

                csf_data = csf_data.astype(bool)
                csf_data = ~csf_data

                mask_data = np.multiply(csf_data, icv_data)
                brain_data = np.multiply(mask_data, mri_data)

                nawm_preprocessing = False
                if nawm_available and nawm_preprocessing:
                    print("NAWM pre-processing..")
                    nawm_data = nawm_data.astype(bool)
                    mask_data = np.multiply(mask_data, nawm_data)
                    nawm_data = np.int16(nawm_data)
                    brain_data  = np.multiply(brain_data, nawm_data)

                if cortex_available:
                    cortex_data = cortex_data.astype(bool)
                    cortex_data = ~cortex_data
                    mask_data = np.multiply(mask_data, cortex_data)
                    cortex_data = np.int16(cortex_data)
                    brain_data  = np.multiply(brain_data, cortex_data)
                ''' Pre-processing is done '''
                
                ''' Call LOTS-IM GPU for each MRI data '''
                timer_data, timer_patches = lots_im_function_compute(
                                        mri_code         = data,
                                        input_mri_data   = brain_data,
                                        output_filedir   = set_output_dir,
                                        nifti_header     = mri_nii.affine,
                                        set_3d           = set_3d,
                                        patch_size       = patch_size,
                                        blending_weights = blending_weights,
                                        num_sample       = num_samples,
                                        alpha            = alpha,
                                        thrsh_patches    = thrsh_patches,
                                        save_jpeg        = save_jpeg,
                                        delete_intermediary = delete_intermediary)

                ''' Save the elapsed time for each MRI data '''
                elapsed_times_all[timer_idx] = timer_data
                elapsed_times_patch_all.append(timer_patches)
                timer_idx += 1

            ''' Save the elapsed time for all MRI data '''
            ''' Save all elapsed times '''
            sio.savemat(set_output_dir + '/elapsed_times_all_test.mat',
                        {'elapsed_times_all':elapsed_times_all})
            sio.savemat(set_output_dir + '/elapsed_times_patch_all_test.mat',
                        {'elapsed_times_patch_all':elapsed_times_patch_all})

if __name__ == "__main__":
    main()