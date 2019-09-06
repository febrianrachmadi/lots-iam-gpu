import matplotlib
matplotlib.use('Agg')

from numba import cuda
from timeit import default_timer as timer
from matplotlib import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from LOTS_IM_GPU_lib import *

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.ndimage.morphology as scimorph

import math, numba, cv2, csv, gc
import os, errno, sys, shutil

import code, time

# Turn interactive plotting off
plt.ioff()

def lots_im_function_compute(mri_code="", input_mri_data="", output_filedir="", nifti_header="", set_3d=True,
                        patch_size=[1,2,4,8], blending_weights=[0.65,0.2,0.1,0.05], num_sample=512,
                        alpha=0.5, thrsh_patches = None, save_jpeg=True, delete_intermediary=True):
    '''
    FUNCTION'S SUMMARY:
    Main function of the LOTS-IM algorithm running on GPU. This function produces (i.e. saving)
    irregularity maps that indicate level of irregularity of voxels in brain FLAIR MRI, either
    in slice (2D) or volume (3D). This function read MRI data from a single subject (i.e., only
    1 MRI data for each run). Please note that the data is retrieved as a 3D numpy array (i.e.,
    whole MRI volume).

    By default, the irregularity maps are calculated by using four different sizes of source/target
    patches (i.e. 1x1, 2x2, 4x4, and 8x8) and 64 target samples. Furthermore, all intermediary
    files are saved in .mat (Matlab) while the irregularity map can be saved as either JPEG files
    (optional) or Nifti file. For now, only Nifti is supported for output file.

    INPUT PARAMETERS:
    This function's behavior can be set by using input parameters below.

        1. mri_code         : Name code of the subject.

        2. input_mri_data   : Numpy 3D array which consists MRI data of the subject. Note that the
                              data needs to be pre-processed before entering this function (i.e.,
                              only brain tissues needed for LOTS-IM's calculation).

        3. output_filedir   : Path of directory to save the results.

        4. nifti_header     : Needed to save the results as Nifti file.

        5. set_3d           : Set `True` if 3D LOTS-IM is used for calculation (default).
                              Set `False` if 2D LOTS-IM is used for calculation.

        6. patch_size       : Size of source/target patches for IM's computation. Default:
                              [1,2,4,8] to calculate irregularity maps from four different 
                              sizes of source/target patches i.e. 1x1, 2x2, 4x4, and 8x8.
                              The sizes of source/target patches must be in the form of 
                              python'slist.

        7. blending_weights : Weights used for blending irregularity maps produced by different 
                              size of source/target patches. The weights must be in the form of
                              python's list, summed to 1, and its length must be the same as 
                              `patch_size`. Default: `blending_weights=[0.65,0.2,0.1,0.05]`

        8. num_sample       : A number used for randomly sampling target patches to be
                              used in the LOTS-IM calculation. Default: 512. Available
                              values: 64, 128, 256, 512, 1024 and 2048]. Some important notes:

                                a. Smaller number will make computation faster.
                                b. Input the numbers as a list to automatically produce
                                   irregularity maps by using all different numbers of target patches.
                                   The software will automatically create different output
                                   folders for different number of target samples.
                                c. For this version, only 64, 128, 256, 512, 1024, and 2048
                                   can be used as input numbers (error will be raised if other
                                   numbers are used).

        9. alpha            : Weight of distance function to blend maximum difference and
                              average difference between source and target patches. Default:
                              0.5. Input value should be between 0 and 1 (i.e. floating points).
                              The current distance function being used is:

                                  d = (alpha . |max(s - t)|) + ((1 - alpha) . |mean(s - t)|)

                              where d is distance value, s is source patch, and t is target patch.

        10. thrsh_patches   : Thresholds the target patches during extraction phase to prevent
                              the inclusions of patches containing prior knowledge of WMH (early 
                              automatic detection/estimation using confidence interval (CI) of
                              95%). Note, this only works for 3D LOTS-IM.

                              Available parameter values:
                              a. False --> (default)
                              b. Between 0 to 1 --> Recommendation: 0.05 (i.e., reject target
                                 patches which contain 5% of early detected WMH).

        11. save_jpeg       : True  --> Save all JPEG files for visualisation.
                              False --> Do not save the JPEG files.

        12. delete_intermediary : False --> Save all intermediary .mat files (default).
                                  True  --> Delete all intermediary files, saving some spaces in
                                            the hard disk drive.

    OUTPUT:
    This function will automatically create a new folder for the current subject.
    Please make sure that the directory is accessible and writable.

    Inside the experiment’s folder, each subject/MRI data will have its own folder. In default,
    there are 5 sub-folders which are:
    1. 1: Contains irregularity maps of each slice generated by using 1x1 patch (in .mat files).
    2. 2: Contains irregularity maps of each slice generated by using 2x2 patch (in .mat files).
    3. 4: Contains irregularity maps of each slice generated by using 4x4 patch (in .mat files).
    4. 8: Contains irregularity maps of each slice generated by using 8x8 patch (in .mat files).
    5. JPEGs: Contains two sub-folders file:
        a. Patch: contains visualisation of irregularity maps of each slices in JPEG files, and
        b. Combined: contains visualisation of the final output of LOTS-IM’s computation.

    Furthermore, two Nifti files will also be produced inside each subject's folder.

    Note: If parameter value of `delete_intermediary` is `True`, then folders number 1 to 4 
    listed above will be deleted, except for folder `JPEGs` and its contents.

    MORE HELP:
    Please read README.md file provided in:
        https://github.com/febrianrachmadi/lots-iam-gpu 

    VERSION (dd/mm/yyyy):
    LOTS-IM running on GPU
    - 05/09/2019 : Moved out the pre-processing from the function. Now, the function can do either
                   2D or 3D computation, set by an input parameter (set_3d).
    - 05/07/2019 : Modification of the original work to extend it to 3D processing with different input.
    - 31/05/2018b: NAWM and Cortical brain masks are now optional input (will be used if available).
    - 31/05/2018a: Fix header information of the LOTS-IAM-GPU's result.
    - 08/05/2018 : Add lines to cutting off probability mask and deleting intermediary folders.
    - 07/05/2018 : Initial release mri_code.
    '''

    ## Check the name of input data (MRI code)
    if mri_code == "":
        raise ValueError("Please set the name of input data/subject. See: help(lots_im_function_compute)")
        return 0
        
    ## Check the name of input data (MRI code)
    if input_mri_data == "":
        raise ValueError("Please set the MRI data (3D Numpy array). See: help(lots_im_function_compute)")
        return 0
        
    ## Check the name of input data (MRI code)
    if nifti_header == "":
        raise ValueError("Please set Nifti's header file so that the result can be saved as Nifti.")
        return 0

    ## Check availability of output files. If it is empty, set output files in the working directory.
    if output_filedir == "":
        output_filedir = os.path.dirname(os.path.realpath(__file__)) + "/output"
        print("!!! WARNING: OUTPUT FILES HAS BEEN SET TO THE WORKING DIRECTORY !!! --> ")
        print("Output directory: " + output_filedir)

    ## Check compatibility between 'patch_size' and 'blending_weights'
    if len(patch_size) != len(blending_weights):
        raise ValueError("Lengths of 'patch_size' and 'blending_weights' variables are not the same. Length of 'patch_size' is " + 
            str(len(patch_size)) + ", while 'blending_weights' is " + str(len(blending_weights)) + ".")
        return 0

    print("--- PARAMETERS - CHECKED ---")
    print('Output file dir: ' + output_filedir)
    print('Patch size(s): ' + str(patch_size))
    print('Use patch selection: ' + str(thrsh_patches))
    print('Save JPEGs? ' + str(save_jpeg))
    print("--- PARAMETERS - CHECKED ---\n")

    num_samples = num_sample
    num_mean_samples = set_mean_sample_number(num_samples)
    print('Number of samples for IAM: ' + str(num_samples))
    print('Number of mean samples for IAM: ' + str(num_mean_samples))

    elapsed_times_all = np.zeros((1))
    elapsed_times_patch_all = np.zeros((1, len(patch_size)))

    print('--\nNow processing data: ' + mri_code)

    dirOutput = output_filedir
    dirOutData = dirOutput + '/' + mri_code
    print('Output path: ' + dirOutData)

    ''' Create output folder(s) '''
    try:
        create_output_folders(dirOutput, mri_code)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    '''
    -----------------------------------
    LOADING THE DATA
    '''

    ## Remove empty slices at the top and bottom of mri volume (z-axis)
    input_mri_data, index_start, original_index_end = keep_relevant_slices(input_mri_data)

    ## Get MRI measurements
    [x_len, y_len, z_len] = input_mri_data.shape

    one_data = timer()
    for xyz in range(0, len(patch_size)):
        print('>>> Processing patch-size: ' + str(patch_size[xyz]) + ' <<<\n')

        try:
            os.makedirs(dirOutData + '/' + str(patch_size[xyz]))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        one_patch = timer()
        if set_3d:
            ''''''''''''''''''''''''''''''''''''
            ''''' IF 3D processing IS USED '''''
            ''''''''''''''''''''''''''''''''''''
            
            ## Set the counter for each axis
            counter_y = int(y_len / patch_size[xyz])
            counter_x = int(x_len / patch_size[xyz])
            counter_z = int(z_len / patch_size[xyz])
            source_patch_len = counter_x * counter_y * counter_z
            irregularity_values_all = np.zeros(source_patch_len)

            ''' Source & target patches generation (3D) '''
            source_patches_all, target_patches_all, idx_source, idx_mapping = \
                gen_3d_source_target_patches(input_mri_data, patch_size[xyz], num_samples, thrsh_patches)

            ''' GPU (CUDA) calculation of irregularity values '''
            irregularity_values_valid = calculate_irregularity_values(source_patches_all, target_patches_all,
                                    num_mean_samples, idx_source, alpha)

            ''' Mapping from irregularity_value_valid to irregularity value_all '''
            index = 0
            for idx_val in idx_mapping:
                if idx_val != -1:
                    irregularity_values_all[index] = irregularity_values_valid[int(idx_val)]
                index += 1

            ''' Normalisation to irregularity map (0...1) '''
            if (np.max(irregularity_values_all) - np.min(irregularity_values_all)) == 0:
                all_mean_distance_normed = irregularity_values_all
            else:
                all_mean_distance_normed = np.divide((irregularity_values_all - np.min(irregularity_values_all)),
                    (np.max(irregularity_values_all) - np.min(irregularity_values_all)))

            ''' SAVE Result '''
            whole_irregularity_map = np.zeros([counter_x,counter_y,counter_z])
            index = 0
            for ix in range(0, counter_x):
                for iy in range(0, counter_y):
                    for iz in range(0, counter_z):
                        whole_irregularity_map[ix,iy, iz] = all_mean_distance_normed[index]
                        index += 1
            
            # Save data
            for zz in range(input_mri_data.shape[2]):
                try:
                    sio.savemat(dirOutData + '/' + str(patch_size[xyz]) + '/' + str(zz) + '_dat.mat',
                        {'slice_irregularity_map':whole_irregularity_map[:, :, zz//patch_size[xyz]]})
                except:
                    pass
            
            ## Flushing GPU
            print('Check GPU memory: ' + str(cuda.current_context().get_memory_info()))
            print('GPU flushing..\n--\n')
            numba.cuda.profile_stop()
            elapsed_times_patch_all[0,xyz] = timer() - one_patch
            print('IAM for MRI data ID: ' + mri_code + ' with patch size: ' + str(patch_size[xyz])
                    + ' elapsed for: ' + str(elapsed_times_patch_all[0,xyz]))
        else:
            ''''''''''''''''''''''''''''''''''''
            ''''' IF 2D processing IS USED '''''
            ''''''''''''''''''''''''''''''''''''
            # output_mri_data = np.zeros(input_mri_data.shape)
            for zz in range(0, input_mri_data.shape[2]):
                print('---> Slice number: ' + str(zz) + ' <---')

                '''
                -------------------------------------
                This version still does per slice operation for extracting brain tissues.
                Two important variables used in the next part of the code are:
                1. mask_slice --->  Combination of ICV & CSF masks. It is used to find valid source patches
                                    for LOTS-IAM-GPU computation (i.e. brain tissues' source patches).
                2. brain_slice -->  Brain tissues' information from FLAIR slice.
                '''

                mask_slice = np.nan_to_num(input_mri_data[:, :, zz])
                mask_slice[mask_slice > 0] = 1

                brain_slice = np.nan_to_num(input_mri_data[:, :, zz])

                # Vol distance threshold
                ## Proportion of brain slice compared to full image
                vol_slice = np.count_nonzero(brain_slice) / (x_len * y_len)                     
                print('DEBUG-Patch: brain_slice - ' + str(np.count_nonzero(brain_slice)) +
                        ', x_len * y_len - ' + str(x_len * y_len) + ', vol: ' + str(vol_slice)) ## x_len/y_len = 512 here

                # Patch's sampling number treshold
                TRSH = 0.50
                if patch_size[xyz] == 1:
                    if vol_slice < 0.010: TRSH = 0
                    elif vol_slice < 0.035: TRSH = 0.15
                    elif vol_slice < 0.070 and vol_slice >= 0.035: TRSH = 0.60
                    elif vol_slice >= 0.070: TRSH = 0.80
                elif patch_size[xyz] == 2:
                    if vol_slice < 0.010: TRSH = 0
                    elif vol_slice < 0.035: TRSH = 0.15
                    elif vol_slice < 0.070 and vol_slice >= 0.035: TRSH = 0.60
                    elif vol_slice >= 0.070: TRSH = 0.80
                elif patch_size[xyz] == 4 or patch_size[xyz] == 8:
                    if vol_slice < 0.035: TRSH = 0

                print('DEBUG-Patch: Size - ' + str(patch_size[xyz]) + ', slice - ' + str(zz) +
                        ', vol: ' + str(vol_slice) + ', TRSH: ' + str(TRSH))

                counter_y = int(y_len / patch_size[xyz])        ## counter_y = 512 if patch of size 1 and image of size 512x512
                counter_x = int(x_len / patch_size[xyz])
                source_patch_len = counter_x * counter_y        ## How many source patches are neede (e.g. for 1, we need one for each pixel)
                irregularity_values_all = np.zeros(source_patch_len)     ## Irregularity Map that will be filled with the actual values

                ''' Process valid MRI slice that has 'enough' brain tissues available '''
                if ((vol_slice >= 0.008 and vol_slice < 0.035) and (patch_size[xyz] == 1 or patch_size[xyz] == 2)) or \
                    ((vol_slice >= 0.035 and vol_slice < 0.065) and (patch_size[xyz] == 1 or patch_size[xyz] == 2 or \
                        patch_size[xyz] == 4)) or (vol_slice > 0.065):

                    ''' Source & target patches generation (2D) '''
                    source_patches_all, target_patches_all, idx_source, idx_mapping = \
                        gen_2d_source_target_patches(brain_slice, patch_size[xyz], num_samples, TRSH)
                    irregularity_values_valid = np.zeros(idx_source)

                    ''' GPU (CUDA) calculation of irregularity values '''
                    irregularity_values_valid = calculate_irregularity_values(source_patches_all, target_patches_all,
                                            num_mean_samples, idx_source, alpha)

                    ''' Mapping from irregularity_value_valid to irregularity value_all '''
                    index = 0
                    for idx_val in idx_mapping:
                        if idx_val != -1:
                            irregularity_values_all[index] = irregularity_values_valid[int(idx_val)]
                        index += 1

                ''' Normalisation to probabilistic map (0...1) '''
                if (np.max(irregularity_values_all) - np.min(irregularity_values_all)) == 0:
                    all_mean_distance_normed = irregularity_values_all
                else:
                    all_mean_distance_normed = np.divide((irregularity_values_all - np.min(irregularity_values_all)),
                        (np.max(irregularity_values_all) - np.min(irregularity_values_all)))

                ''' SAVE Result '''
                slice_irregularity_map = np.zeros([counter_x,counter_y])
                index = 0
                for ix in range(0, counter_x):
                    for iy in range(0, counter_y):
                        slice_irregularity_map[ix,iy] = all_mean_distance_normed[index]
                        index += 1

                ## Save mri_data
                sio.savemat(dirOutData + '/' + str(patch_size[xyz]) + '/' + str(zz) + '_dat.mat',
                            {'slice_irregularity_map':slice_irregularity_map})

                print('Check GPU memory: ' + str(cuda.current_context().get_memory_info()))
                print('GPU flushing..\n--\n')
                numba.cuda.profile_stop()
            elapsed_times_patch_all[0,xyz] = timer() - one_patch
            print('IAM for MRI ID: ' + mri_code + ' with patch size: ' + str(patch_size[xyz])
                    + ' elapsed for: ' + str(elapsed_times_patch_all[0,xyz]))

    elapsed_times_all = timer() - one_data
    print('IAM for MRI data ID: ' + mri_code + ' elapsed for: ' + str(elapsed_times_all))

    ''' IAM's (GPU Part) Computation ENDS here '''

    '''
    KEY POINT: IAM's Combination, Penalisation, and Post-processing - START
    -----------------------------------------------------------------------
    Part 0 - Saving output results in .mat and JPEG files.
    Part 1 - Combination of multiple irregularity maps.
    Part 2 - Global normalisation and penalisation of irregularity maps based on brain tissues.

    Hint: You can search the keys of Part 0/1/2.
    '''

    combined_irregularity_map_mri = np.zeros((x_len, y_len, z_len))
    combined_irregularity_map_mri_penalised = np.zeros((x_len, y_len, z_len))
    combined_irregularity_map_mri_penalised_normed = np.zeros((x_len, y_len, z_len))
    for zz in range(0, input_mri_data.shape[2]):
        mri_slice = input_mri_data[:,:,zz]
        mask_slice = np.nan_to_num(mri_slice)
        mask_slice[mask_slice > 0] = 1
        penalty_slice = np.nan_to_num(mri_slice)   # Penalty for FLAIR

        # Uncomment next line if using t1-weighted scan
        #penalty_slice = 1 - np.nan_to_num(mri_slice)   # Penalty for t1-weighted

        slice_irregularity_map_all = np.zeros((len(patch_size), x_len, y_len))
        dirOutData = dirOutput + '/' + mri_code
        for xyz in range(0, len(patch_size)):
            try:
                mat_contents = sio.loadmat(dirOutData + '/' + str(patch_size[xyz]) + '/' + str(zz) + '_dat.mat')
            except:
                break
            slice_irregularity_map_all[xyz,:,:] = get_slice_irregularity_map(patch_size[xyz], mat_contents, mask_slice)
        slice_irregularity_map_all = np.nan_to_num(slice_irregularity_map_all)

        if save_jpeg:
            ''' >>> Part 0 <<<'''
            ''' Show all irregularity maps based on patch's size and saving the data '''
            fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
            fig.set_size_inches(10, 10)
            fig.suptitle('All Patches have been Gaussian Filtered', fontsize=16)

            axes[0,0].set_title('Patch 1 x 1')
            im1 = axes[0,0].imshow(np.rot90(slice_irregularity_map_all[0,:,:]), cmap="jet", vmin=0, vmax=1)
            divider1 = make_axes_locatable(axes[0,0])
            cax1 = divider1.append_axes("right", size="7%", pad=0.05)
            cbar1 = plt.colorbar(im1, ticks=[0, 0.5, 1], cax=cax1)

            if len(patch_size) > 1:
                axes[0,1].set_title('Patch 2 x 2')
                im2 = axes[0,1].imshow(np.rot90(slice_irregularity_map_all[1,:,:]), cmap="jet", vmin=0, vmax=1)
                divider2 = make_axes_locatable(axes[0,1])
                cax2 = divider2.append_axes("right", size="7%", pad=0.05)
                cbar2 = plt.colorbar(im2, ticks=[0, 0.5, 1], cax=cax2)

                if len(patch_size) > 2:
                    axes[1,0].set_title('Patch 4 x 4')
                    im3 = axes[1,0].imshow(np.rot90(slice_irregularity_map_all[2,:,:]), cmap="jet", vmin=0, vmax=1)
                    divider3 = make_axes_locatable(axes[1,0])
                    cax3 = divider3.append_axes("right", size="7%", pad=0.05)
                    cbar3 = plt.colorbar(im3, ticks=[0, 0.5, 1], cax=cax3)

                    if len(patch_size) > 3:
                        axes[1,1].set_title('Patch 8 x 8')
                        im4 = axes[1,1].imshow(np.rot90(slice_irregularity_map_all[3,:,:]), cmap="jet", vmin=0, vmax=1)
                        divider4 = make_axes_locatable(axes[1,1])
                        cax4 = divider4.append_axes("right", size="7%", pad=0.05)
                        cbar4 = plt.colorbar(im4, ticks=[0, 0.5, 1], cax=cax4)

            plt.tight_layout()
            plt.subplots_adjust(top=0.95)

            ''' Save data in *_all.jpg '''
            dirOutData = dirOutput + '/' + mri_code + '/JPEGs/Patch/'
            fig.savefig(dirOutData + str(zz) + '_all.jpg', dpi=100)
            print('Saving files: ' + dirOutData + str(zz) + '_all.jpg')
            plt.close()

        ''' >>> Part 1 <<< '''
        ''' Combined all patches irregularity map information '''

        combined_irregularity_map = 0
        for bi in range(len(patch_size)):
            combined_irregularity_map += np.multiply(blending_weights[bi],slice_irregularity_map_all[bi,:,:])
        combined_irregularity_map_mri[:,:,zz] = combined_irregularity_map

        ''' PENALTY - saving needed data '''
        combined_irregularity_map_mri_penalised[:,:,zz] = np.multiply(
            np.multiply(combined_irregularity_map, penalty_slice), mask_slice)

    ''' >>> Part 2 <<< '''
    ''' Penalty + Global Normalisation (GN) '''
    combined_irregularity_map_mri_normed = np.divide((combined_irregularity_map_mri - np.min(combined_irregularity_map_mri)),\
                                (np.max(combined_irregularity_map_mri) - np.min(combined_irregularity_map_mri)))
    combined_irregularity_map_mri_penalised_normed = np.divide((combined_irregularity_map_mri_penalised - np.min(combined_irregularity_map_mri_penalised)),\
                                (np.max(combined_irregularity_map_mri_penalised) - np.min(combined_irregularity_map_mri_penalised)))

    if save_jpeg:
        for zz in range(0, input_mri_data.shape[2]):
            fig2, axes2 = plt.subplots(1, 3)
            fig2.set_size_inches(16,5)

            axes2[0].set_title('Combined and normalised')
            im1 = axes2[0].imshow(np.rot90(np.nan_to_num(combined_irregularity_map_mri_normed[:,:,zz])), cmap="jet", vmin=0, vmax=1)
            divider1 = make_axes_locatable(axes2[0])
            cax1 = divider1.append_axes("right", size="7%", pad=0.05)
            cbar1 = plt.colorbar(im1, ticks=[0, 0.5, 1], cax=cax1)

            axes2[1].set_title('Combined, penalised and normalised')
            im2 = axes2[1].imshow(np.rot90(np.nan_to_num(combined_irregularity_map_mri_penalised_normed[:,:,zz])), cmap="jet", vmin=0, vmax=1)
            divider2 = make_axes_locatable(axes2[1])
            cax2 = divider2.append_axes("right", size="7%", pad=0.05)
            cbar2 = plt.colorbar(im2, ticks=[0, 0.5, 1], cax=cax2)

            axes2[2].set_title('Original MRI slice')
            im3 = axes2[2].imshow(np.rot90(input_mri_data[:,:,zz]), cmap="gray")
            divider3 = make_axes_locatable(axes2[2])
            cax3 = divider3.append_axes("right", size="7%", pad=0.05)
            cbar3 = plt.colorbar(im3, cax=cax3)

            plt.tight_layout()
            # Make space for title
            plt.subplots_adjust(top=0.95)

            ''' Save data in *_combined.jpg '''
            dirOutData = dirOutput + '/' + mri_code + '/JPEGs/Combined/'
            fig2.savefig(dirOutData + str(zz) + '_combined.jpg', dpi=100)
            print('Saving files: ' + dirOutData + str(zz) + '_combined.jpg')
            plt.close()

    ## Reshaping into original dimensions
    combined_irregularity_map_mri_normed = reshape_original_dimensions(combined_irregularity_map_mri_normed, index_start, original_index_end)
    combined_irregularity_map_mri_penalised_normed = reshape_original_dimensions(combined_irregularity_map_mri_penalised_normed, index_start, original_index_end)

    dirOutData = dirOutput + '/' + mri_code
    combined_irregularity_map_mri_img = nib.Nifti1Image(combined_irregularity_map_mri_normed, nifti_header)
    nib.save(combined_irregularity_map_mri_img, str(dirOutData + '/IAM_GPU.nii.gz'))

    combined_irregularity_map_mri_GN_img = nib.Nifti1Image(combined_irregularity_map_mri_penalised_normed, nifti_header)
    nib.save(combined_irregularity_map_mri_GN_img, str(dirOutData + '/IAM_GPU_GloballyNormalised.nii.gz'))

    if delete_intermediary:
        if not save_jpeg:
            shutil.rmtree(dirOutput + '/' + mri_code + '/JPEGs', ignore_errors=True)
        for xyz in range(0, len(patch_size)):
            shutil.rmtree(dirOutData + '/' + str(patch_size[xyz]), ignore_errors=True)

    '''
    ---------------------------------------------------------------------
    KEY POINT: IAM's Combination, Penalisation, and Post-processing - END
    '''

    del idx_mapping, source_patches_all, target_patches_all   # Free memory
    gc.collect()

    return elapsed_times_all, elapsed_times_patch_all

        
