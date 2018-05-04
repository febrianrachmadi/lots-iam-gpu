import matplotlib
matplotlib.use('Agg')

from numba import cuda
from timeit import default_timer as timer
from matplotlib import pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable

from IAM_lib import *

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.io as sio
import skimage.morphology as skimorph
import skimage.filters as skifilters
import scipy.ndimage.morphology as scimorph

import math, numba, cv2, csv, gc
import os, errno, sys

# Turn interactive plotting off
plt.ioff()

def iam_lots_gpu_compute(output_filedir="",csv_filename="",patch_size=[1,2,4,8],num_sample=[64],save_jpeg=True):
    '''
    Test: test1
    '''
    
    if output_filedir == "" or csv_filename == "":
        raise ValueError("Please set output folder's name and CSV data filename. See: help(iam_lots_gpu)")
        return 0
    
    #### Set number of mean samples automatically
    ## num_samples_all = [64, 128, 256, 512, 1024, 2048]
    ## num_mean_samples_all = [16, 32, 32, 64, 128, 128]
    num_samples_all = num_sample
    num_mean_samples_all = []
    for sample in num_samples_all:
        if sample == 64:
            num_mean_samples_all.append(16)
        elif sample == 128:
            num_mean_samples_all.append(32)
        elif sample == 256:
            num_mean_samples_all.append(32)
        elif sample == 512:
            num_mean_samples_all.append(64)
        elif sample == 1024:
            num_mean_samples_all.append(128)
        elif sample == 2048:
            num_mean_samples_all.append(128)
        else:
            raise ValueError("Number of samples must be either 64, 128, 256, 512, 1024 or 2048!")
            return 0
    
    print("--- PARAMETERS - CHECKED ---")
    print('Output file dir: ' + output_filedir)
    print('CSV data filename: ' + csv_filename)
    print('Patch size(s): ' + str(patch_size))
    print('Number of samples (all): ' + str(num_samples_all))
    print('Number of mean samples (all): ' + str(num_mean_samples_all))
    print('Save JPEGs? ' + str(save_jpeg))
    print("--- PARAMETERS - CHECKED ---\n")
    
    for ii_s in range(0, len(num_samples_all)):
        num_samples = num_samples_all[ii_s]
        num_mean_samples = num_mean_samples_all[ii_s]
        print('Number of samples for IAM: ' + str(num_samples))
        print('Number of mean samples for IAM: ' + str(num_mean_samples))

        dirOutput = output_filedir + '_' + str(num_samples) + 's' + str(num_mean_samples) + 'm'    
        print('Output dir: ' + dirOutput + '\n--')

        try:
            os.makedirs(dirOutput)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        with open(csv_filename, newline='') as csv_file:        
            num_subjects = len(csv_file.readlines())
            print('Number of subject(s): ' + str(num_subjects))

        with open(csv_filename, newline='') as csv_file:
            reader = csv.reader(csv_file)

            timer_idx = 0
            elapsed_times_all = np.zeros((num_subjects))
            elapsed_times_patch_all = np.zeros((num_subjects, len(patch_size)))
            for row in reader:            
                data = row[1]
                print('--\nNow processing data: ' + data)

                inputSubjectDir = row[0] + '/' + row[1]
                print('Input filename (full path): ' + inputSubjectDir)

                ''' Create output folder(s) '''
                dirOutData = dirOutput + '/' + data
                dirOutDataCom = dirOutput + '/' + data + '/IAM_combined_python/'
                dirOutDataFin = dirOutput + '/' + data + '/IAM_GPU_nifti_python/'
                dirOutDataPatch = dirOutput + '/' + data + '/IAM_combined_python/Patch/'
                dirOutDataCombined = dirOutput + '/' + data + '/IAM_combined_python/Combined/'
                try:
                    os.makedirs(dirOutData)
                    os.makedirs(dirOutDataCom)
                    os.makedirs(dirOutDataFin)
                    os.makedirs(dirOutDataPatch)
                    os.makedirs(dirOutDataCombined)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

                mri_nii = nib.load(row[2])
                icv_nii = nib.load(row[3])
                csf_nii = nib.load(row[4])
                nawm_nii = nib.load(row[5])

                mri_data = np.squeeze(mri_nii.get_data())
                icv_data = np.squeeze(icv_nii.get_data())
                csf_data = np.squeeze(csf_nii.get_data())   
                nawm_data = np.squeeze(nawm_nii.get_data()) 
                print(' -> Loading FLAIR + ICV + CSF: OK!')

                del icv_nii, csf_nii   # Free memory

                ''' ICV Erosion '''
                print(' -> ICV Erosion -- note: skimage library')
                print(' -> ICV shape:' + str(icv_data.shape) + '\n')
                for ii in range(0, icv_data.shape[2]):
                    kernel = kernel_sphere(7)
                    icv_data[:,:,ii] = skimorph.erosion(icv_data[:,:,ii],kernel)
                    kernel = kernel_sphere(11)
                    icv_data[:,:,ii] = skimorph.erosion(icv_data[:,:,ii],kernel)   
                    nawm_data[:,:,ii] = scimorph.binary_fill_holes(nawm_data[:,:,ii])
                    kernel = kernel_sphere(3)
                    nawm_data[:,:,ii] = skimorph.erosion(nawm_data[:,:,ii],kernel)
                nawm_data = nawm_data.astype(int)     

                [x_len, y_len, z_len] = mri_data.shape

                one_data = timer()
                for xy in range(0, len(patch_size)):
                    print('>>> Processing patch-size: ' + str(patch_size[xy]) + ' <<<\n')

                    try:
                        os.makedirs(dirOutData + '/' + str(patch_size[xy]))
                    except OSError as e:
                        if e.errno != errno.EEXIST:
                            raise        

                    one_patch = timer()
                    for zz in range(0, mri_data.shape[2]):
                        print('---> Slice number: ' + str(zz) + ' <---')
                        mri_slice = mri_data[:,:,zz]
                        icv_slice = icv_data[:,:,zz]
                        csf_slice = csf_data[:,:,zz]
                        csf_slice = csf_slice.astype(bool)
                        csf_slice = ~csf_slice

                        mask_slice = np.multiply(csf_slice, icv_slice)
                        mri_slice = np.int16(mri_slice)
                        brain_slice = np.multiply(mask_slice, mri_slice)

                        # Vol distance threshold
                        vol_slice = np.count_nonzero(brain_slice) / (x_len * y_len)

                        # Patch's sampling number treshold
                        TRSH = 0.50
                        if patch_size[xy] == 1:
                            if vol_slice < 0.010: TRSH = 0
                            elif vol_slice < 0.035: TRSH = 0.15
                            elif vol_slice < 0.070 and vol_slice >= 0.035: TRSH = 0.60
                            elif vol_slice >= 0.070: TRSH = 0.80
                        elif patch_size[xy] == 2:
                            if vol_slice < 0.010: TRSH = 0
                            elif vol_slice < 0.035: TRSH = 0.15
                            elif vol_slice < 0.070 and vol_slice >= 0.035: TRSH = 0.60
                            elif vol_slice >= 0.070: TRSH = 0.80
                        elif patch_size[xy] == 4 or patch_size[xy] == 8:
                            if vol_slice < 0.035: TRSH = 0

                        print('DEBUG-Patch: Size - ' + str(patch_size[xy]) + ', slice - ' + str(zz) +
                              ', vol: ' + str(vol_slice) + ', TRSH: ' + str(TRSH))

                        counter_y = int(y_len / patch_size[xy])
                        counter_x = int(x_len / patch_size[xy])
                        source_patch_len = counter_x * counter_y
                        age_values_all = np.zeros(source_patch_len)

                        valid = 0
                        if ((vol_slice >= 0.008 and vol_slice < 0.035) and (patch_size[xy] == 1 or patch_size[xy] == 2)) or \
                            ((vol_slice >= 0.035 and vol_slice < 0.065) and (patch_size[xy] == 1 or patch_size[xy] == 2 or \
                             patch_size[xy] == 4)) or (vol_slice > 0.065):
                            valid = 1

                            ## Creating grid-patch 'xy-by-xy'
                            #  -- Column
                            y_c = np.ceil(patch_size[xy] / 2)
                            y_c_sources = np.zeros(int(y_len / patch_size[xy]))
                            for iy in range(0, int(y_len / patch_size[xy])):
                                y_c_sources[iy] = (iy * patch_size[xy]) + y_c - 1

                            #  -- Row
                            x_c = np.ceil(patch_size[xy] / 2)
                            x_c_sources = np.zeros(int(x_len / patch_size[xy]))
                            for ix in range(0, int(x_len / patch_size[xy])):
                                x_c_sources[ix] = (ix * patch_size[xy]) + x_c - 1

                            ''' Extracting Source Patches '''
                            area_source_patch = np.zeros([1,patch_size[xy],patch_size[xy]])
                            center_source_patch = np.zeros([1,2])
                            icv_source_flag = np.zeros([source_patch_len])
                            icv_source_flag_valid = np.ones([source_patch_len])
                            index_mapping = np.ones([source_patch_len]) * -1

                            flag = 1
                            index = 0
                            index_source= 0
                            for isc in range(0, counter_x):
                                for jsc in range(0, counter_y):
                                    icv_source_flag[index] = mask_slice[int(x_c_sources[isc]), int(y_c_sources[jsc])]

                                    if icv_source_flag[index] == 1:

                                        if flag:
                                            flag = 0                              
                                            temp = np.matrix([x_c_sources[isc], y_c_sources[jsc]])
                                            center_source_patch[0,:] = temp
                                            temp = get_area(x_c_sources[isc], y_c_sources[jsc],
                                                            patch_size[xy], patch_size[xy], brain_slice)
                                            area_source_patch[0,:,:] = temp
                                        else:
                                            temp = np.matrix([x_c_sources[isc], y_c_sources[jsc]])
                                            center_source_patch = np.concatenate((center_source_patch, temp))                                
                                            temp = get_area(x_c_sources[isc], y_c_sources[jsc],
                                                            patch_size[xy], patch_size[xy], brain_slice)
                                            temp = np.reshape(temp, (1, patch_size[xy], patch_size[xy]))
                                            area_source_patch = np.concatenate((area_source_patch, temp))

                                        index_mapping[index] = index_source
                                        index_source += 1                            
                                    index += 1

                            icv_source_flag_valid = icv_source_flag_valid[0:index_source]
                            age_values_valid = np.zeros(index_source)

                            ''' Extracting Target Patches '''
                            target_patches = []
                            index_debug = 0
                            index_population = 0
                            for iii in range(0, x_len):
                                for jjj in range(0, y_len):
                                    index_population += 1
                                    if mask_slice[iii,jjj] != 0 and np.random.rand(1) > TRSH:
                                        area = get_area(iii, jjj, patch_size[xy], patch_size[xy], brain_slice)
                                        if area.size == patch_size[xy] * patch_size[xy]:
                                            target_patches.append(area)
                                            index_debug += 1

                            target_patches_np = np.array(target_patches)
                            np.random.shuffle(target_patches_np)
                            target_patches_np = target_patches_np[0:num_samples,:,:]
                            print('Sampling finished: ' + ' with: ' + str(index_debug) + ' samples from: ' 
                                  + str(index_population))
                            area = []

                            ''''''
                            ''' Reshaping array data '''
                            area_source_patch_cuda_all = np.reshape(area_source_patch,(area_source_patch.shape[0],
                                                            area_source_patch.shape[1] * area_source_patch.shape[2]))
                            target_patches_np_cuda_all = np.reshape(target_patches_np, (target_patches_np.shape[0],
                                                            target_patches_np.shape[1] * target_patches_np.shape[2]))

                            source_len = icv_source_flag_valid.shape[0]
                            loop_len = 512 # def: 512
                            loop_num = int(np.ceil(source_len / loop_len))
                            print('\nLoop Information:')
                            print('Total number of source patches: ' + str(source_len))
                            print('Number of voxels processed in one loop: ' + str(loop_len))
                            print('Number of loop needed: ' + str(loop_num))
                            print('Check GPU memory: ' + str(cuda.current_context().get_memory_info()))

                            for il in range(0, loop_num):
                                ''' Debug purposed printing '''
                                print('.', end='')
                                if np.remainder(il+1, 32) == 0:
                                    print(' ' + str(il+1) + '/' + str(loop_num)) # Print newline

                                ''' Only process sub-array '''
                                source_patches_loop = area_source_patch_cuda_all[il*loop_len:(il*loop_len)+loop_len,:]

                                '''  SUBTRACTION '''
                                sub_result_gm = cuda.device_array((source_patches_loop.shape[0],
                                                                   target_patches_np_cuda_all.shape[0],
                                                                   target_patches_np_cuda_all.shape[1]))                    
                                TPB = (4,256)
                                BPGx = int(math.ceil(source_patches_loop.shape[0] / TPB[0]))
                                BPGy = int(math.ceil(target_patches_np_cuda_all.shape[0] / TPB[1]))
                                BPGxy = (BPGx,BPGy)
                                cu_sub_st[BPGxy,TPB](source_patches_loop, target_patches_np_cuda_all, sub_result_gm)

                                '''  MAX-MEAN-ABS '''
                                sub_max_mean_result = cuda.device_array((source_patches_loop.shape[0],
                                                                         target_patches_np_cuda_all.shape[0],2))
                                cu_max_mean_abs[BPGxy,TPB](sub_result_gm, sub_max_mean_result)
                                sub_result_gm = 0  # Free memory

                                '''  DISTANCE '''    
                                distances_result = cuda.device_array((source_patches_loop.shape[0],
                                                                      target_patches_np_cuda_all.shape[0]))
                                cu_distances[BPGxy,TPB](sub_max_mean_result,
                                                        icv_source_flag_valid[il*loop_len:(il*loop_len)+loop_len],
                                                        distances_result, 0.5)
                                sub_max_mean_result = 0  # Free memory

                                ''' SORT '''
                                TPB = 256
                                BPG = int(math.ceil(distances_result.shape[0] / TPB))
                                cu_sort_distance[BPG,TPB](distances_result)

                                ''' MEAN (AGE-VALUE) '''
                                idx_start = 8 # Starting index of mean calculation (to avoid bad example)
                                distances_result_for_age = distances_result[:,idx_start:idx_start+num_mean_samples] 
                                distances_result = 0  # Free memory
                                cu_age_value[BPG,TPB](distances_result_for_age,
                                                      age_values_valid[il*loop_len:(il*loop_len)+loop_len])
                                distances_result_for_age = 0  # Free memory
                                del source_patches_loop  # Free memory
                            print(' - Finished!\n')

                        ''' Mapping from age_value_valid to age value_all '''
                        if valid == 1:
                            index = 0
                            for idx_val in index_mapping:
                                if idx_val != -1:
                                    age_values_all[index] = age_values_valid[int(idx_val)]
                                index += 1

                        ''' Normalisation to probabilistic map (0...1) '''
                        if (np.max(age_values_all) - np.min(age_values_all)) == 0:
                            all_mean_distance_normed = age_values_all
                        else:
                            all_mean_distance_normed = np.divide((age_values_all - np.min(age_values_all)),
                                (np.max(age_values_all) - np.min(age_values_all)))

                        ''' SAVE Result (JPG) '''
                        slice_age_map = np.zeros([counter_x,counter_y])
                        index = 0
                        for ix in range(0, counter_x):
                            for iy in range(0, counter_y):
                                slice_age_map[ix,iy] = all_mean_distance_normed[index]
                                index += 1

                        ## Save data
                        sio.savemat(dirOutData + '/' + str(patch_size[xy]) + '/' + str(zz) + '_dat.mat',
                                    {'slice_age_map':slice_age_map})

                        print('Check GPU memory: ' + str(cuda.current_context().get_memory_info()))
                        print('GPU flushing..\n--\n')
                        numba.cuda.profile_stop()
                    elapsed_times_patch_all[timer_idx,xy] = timer() - one_patch
                    print('IAM for MRI data ID: ' + data + ' with patch size: ' + str(patch_size[xy]) 
                          + ' elapsed for: ' + str(elapsed_times_patch_all[timer_idx,xy]))           

                elapsed_times_all[timer_idx] = timer() - one_data
                print('IAM for MRI data ID: ' + data + ' elapsed for: ' + str(elapsed_times_all[timer_idx]))
                timer_idx += 1

                ## Save all elapsed times
                sio.savemat(dirOutput + '/elapsed_times_all_' + str(num_samples) + 's' + str(num_mean_samples) + 'm.mat',
                            {'elapsed_times_all':elapsed_times_all})
                sio.savemat(dirOutput + '/elapsed_times_patch_all_' + str(num_samples) + 's' + str(num_mean_samples) + 'm.mat',
                            {'elapsed_times_patch_all':elapsed_times_patch_all})
                ''' IAM's (GPU Part) Computation ENDS here '''

                ''' IAM's Combination, Penalisation, and Post-processing START here '''
                combined_age_map_mri = np.zeros((x_len, y_len, z_len))
                combined_age_map_mri_mult = np.zeros((x_len, y_len, z_len))
                combined_age_map_mri_mult_normed = np.zeros((x_len, y_len, z_len))  
                for zz in range(0, mri_data.shape[2]): 
                    mri_slice = mri_data[:,:,zz]
                    icv_slice = icv_data[:,:,zz]
                    icv_slice = icv_slice.astype(int)
                    brain_slice = np.multiply(icv_slice, mri_slice)

                    slice_age_map_all = np.zeros((len(patch_size), x_len, y_len))        

                    dirOutData = dirOutput + '/' + data
                    for xy in range(0, len(patch_size)):
                        mat_contents = sio.loadmat(dirOutData + '/' + str(patch_size[xy]) + '/' + str(zz) + '_dat.mat')
                        slice_age_map = mat_contents['slice_age_map']
                        slice_age_map_res = cv2.resize(slice_age_map, None, fx=patch_size[xy],
                                                       fy=patch_size[xy], interpolation=cv2.INTER_CUBIC)
                        slice_age_map_res = skifilters.gaussian(slice_age_map_res,sigma=0.5,truncate=2.0)
                        slice_age_map_res = np.multiply(icv_slice, slice_age_map_res)
                        slice_age_map_all[xy,:,:] = slice_age_map_res
                    slice_age_map_all = np.nan_to_num(slice_age_map_all)

                    if save_jpeg:
                        ''' >>> <<<'''
                        ''' Show all age maps based on patch's size and saving the data '''
                        ''' >>> <<< '''                
                        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
                        fig.set_size_inches(10, 10)
                        fig.suptitle('All Patches Gaussian Filtered', fontsize=16)

                        axes[0,0].set_title('Patch 1 x 1')
                        im1 = axes[0,0].imshow(np.rot90(slice_age_map_all[0,:,:]), cmap="jet", vmin=0, vmax=1)
                        divider1 = make_axes_locatable(axes[0,0])
                        cax1 = divider1.append_axes("right", size="7%", pad=0.05)
                        cbar1 = plt.colorbar(im1, ticks=[0, 0.5, 1], cax=cax1)

                        axes[0,1].set_title('Patch 2 x 2')
                        im2 = axes[0,1].imshow(np.rot90(slice_age_map_all[1,:,:]), cmap="jet", vmin=0, vmax=1)
                        divider2 = make_axes_locatable(axes[0,1])
                        cax2 = divider2.append_axes("right", size="7%", pad=0.05)
                        cbar2 = plt.colorbar(im2, ticks=[0, 0.5, 1], cax=cax2)

                        axes[1,0].set_title('Patch 4 x 4')
                        im3 = axes[1,0].imshow(np.rot90(slice_age_map_all[2,:,:]), cmap="jet", vmin=0, vmax=1)
                        divider3 = make_axes_locatable(axes[1,0])
                        cax3 = divider3.append_axes("right", size="7%", pad=0.05)
                        cbar3 = plt.colorbar(im3, ticks=[0, 0.5, 1], cax=cax3)

                        axes[1,1].set_title('Patch 8 x 8')
                        im4 = axes[1,1].imshow(np.rot90(slice_age_map_all[3,:,:]), cmap="jet", vmin=0, vmax=1)
                        divider4 = make_axes_locatable(axes[1,1])
                        cax4 = divider4.append_axes("right", size="7%", pad=0.05)
                        cbar4 = plt.colorbar(im4, ticks=[0, 0.5, 1], cax=cax4)

                        plt.tight_layout()
                        plt.subplots_adjust(top=0.95)

                        ''' >>> <<<'''
                        ''' Save data in *_all.jpg '''
                        dirOutData = dirOutput + '/' + data + '/IAM_combined_python/Patch/'
                        fig.savefig(dirOutData + str(zz) + '_all.jpg', dpi=100)
                        print('Saving files: ' + dirOutData + str(zz) + '_all.jpg')
                        plt.close()

                    ''' >>> <<< '''
                    ''' Combined all patches age map information '''
                    BLEND_1 = 0.65
                    BLEND_2 = 0.2
                    BLEND_3 = 0.1
                    BLEND_4 = 0.05
                    combined_age_map = np.multiply(BLEND_1,slice_age_map_all[0,:,:]) + \
                                np.multiply(BLEND_2,slice_age_map_all[1,:,:]) + \
                                np.multiply(BLEND_3,slice_age_map_all[2,:,:]) + \
                                np.multiply(BLEND_4,slice_age_map_all[3,:,:])
                    combined_age_map_mri[:,:,zz] = combined_age_map        

                    ''' Global Normalisation - saving needed data '''
                    combined_age_map_mri_mult[:,:,zz] = np.multiply(np.multiply(combined_age_map, mri_slice), icv_slice)      
                    normed_only = np.divide((combined_age_map_mri[:,:,zz] - np.min(combined_age_map_mri[:,:,zz])),\
                                            (np.max(combined_age_map_mri[:,:,zz]) - np.min(combined_age_map_mri[:,:,zz])))
                    normed_mult = np.multiply(np.multiply(normed_only, mri_slice), icv_slice)
                    normed_mult_normed = np.divide((normed_mult - np.min(normed_mult)), \
                                            (np.max(normed_mult) - np.min(normed_mult)))
                    combined_age_map_mri_mult_normed[:,:,zz] = normed_mult_normed

                    ## Save data in *.mat
                    print('Saving files: ' + dirOutData + 'c' + str(zz) + '_combined.mat\n')
                    sio.savemat(dirOutData + 'c' + str(zz) + '_combined.mat', {'slice_age_map_all':slice_age_map_all,
                                                                'combined_age_map':normed_only,
                                                                'mri_slice_mul_normed':normed_mult_normed,
                                                                'combined_mult':combined_age_map_mri_mult[:,:,zz]})

                ''' Penalty + Global Normalisation (GN) '''
                combined_age_map_mri_normed = np.divide((combined_age_map_mri - np.min(combined_age_map_mri)),\
                                            (np.max(combined_age_map_mri) - np.min(combined_age_map_mri)))
                combined_age_map_mri_mult_normed = np.divide((combined_age_map_mri_mult - np.min(combined_age_map_mri_mult)),\
                                            (np.max(combined_age_map_mri_mult) - np.min(combined_age_map_mri_mult)))

                if save_jpeg:
                    for zz in range(0, mri_data.shape[2]):
                        fig2, axes2 = plt.subplots(1, 3)
                        fig2.set_size_inches(16,5)
                        fig2.suptitle('Combined results', fontsize=16)

                        axes2[0].set_title('Combined and normalised')
                        im1 = axes2[0].imshow(np.rot90(combined_age_map_mri_normed[:,:,zz]), cmap="jet", vmin=0, vmax=1)
                        divider1 = make_axes_locatable(axes2[0])
                        cax1 = divider1.append_axes("right", size="7%", pad=0.05)
                        cbar1 = plt.colorbar(im1, ticks=[0, 0.5, 1], cax=cax1)

                        axes2[1].set_title('Combined, penalised and normalised')
                        im2 = axes2[1].imshow(np.rot90(combined_age_map_mri_mult_normed[:,:,zz]), cmap="jet", vmin=0, vmax=1)
                        divider2 = make_axes_locatable(axes2[1])
                        cax2 = divider2.append_axes("right", size="7%", pad=0.05)
                        cbar2 = plt.colorbar(im2, ticks=[0, 0.5, 1], cax=cax2)

                        axes2[2].set_title('Original MRI slice')
                        im3 = axes2[2].imshow(np.rot90(mri_data[:,:,zz]), cmap="gray")
                        divider3 = make_axes_locatable(axes2[2])
                        cax3 = divider3.append_axes("right", size="7%", pad=0.05)
                        cbar3 = plt.colorbar(im3, cax=cax3)

                        plt.tight_layout()
                        # Make space for title
                        plt.subplots_adjust(top=0.95)

                        ## Save data in *_combined.jpg
                        dirOutData = dirOutput + '/' + data + '/IAM_combined_python/Combined/'
                        fig2.savefig(dirOutData + str(zz) + '_combined.jpg', dpi=100)
                        print('Saving files: ' + dirOutData + str(zz) + '_combined.jpg')
                        plt.close()

                ## Save data in *.mat
                sio.savemat(dirOutDataCombined + '/all_slice_dat.mat', {'combined_age_map_all_slice':combined_age_map_mri,
                                                   'mri_slice_mul_all_slice':combined_age_map_mri_mult,
                                                   'combined_age_map_mri_normed':combined_age_map_mri_normed,
                                                   'combined_age_map_mri_mult_normed':combined_age_map_mri_mult_normed})

                combined_age_map_mri_img = nib.Nifti1Image(combined_age_map_mri_normed, mri_nii.affine, mri_nii.header)
                nib.save(combined_age_map_mri_img, str(dirOutDataFin + '/IAM_GPU_COMBINED.nii.gz'))

                combined_age_map_mri_GN_img = nib.Nifti1Image(combined_age_map_mri_mult_normed, mri_nii.affine, mri_nii.header)
                nib.save(combined_age_map_mri_GN_img, str(dirOutDataFin + '/IAM_GPU_GN.nii.gz'))

                combined_age_map_mri_mult_normed = np.multiply(combined_age_map_mri_mult_normed,nawm_data)
                combined_age_map_mri_GN_img = nib.Nifti1Image(combined_age_map_mri_mult_normed, mri_nii.affine, mri_nii.header)
                nib.save(combined_age_map_mri_GN_img, str(dirOutDataFin + '/IAM_GPU_GN_postprocessed.nii.gz'))

                del temp
                del mri_nii, nawm_nii
                del mri_slice, icv_slice, csf_slice
                del mri_data, icv_data, csf_data, nawm_data
                del center_source_patch, icv_source_flag
                del icv_source_flag_valid, index_mapping
                del area_source_patch, target_patches_np   # Free memory
                del area_source_patch_cuda_all, target_patches_np_cuda_all   # Free memory
                gc.collect()

        ## Print the elapsed time information
        print('\n--\nSpeed statistics of this run..')
        print('mean elapsed time  : ' + str(np.mean(elapsed_times_all)) + ' seconds')
        print('std elapsed time   : ' + str(np.std(elapsed_times_all)) + ' seconds')
        print('median elapsed time : ' + str(np.median(elapsed_times_all)) + ' seconds')
        print('min elapsed time   : ' + str(np.min(elapsed_times_all)) + ' seconds')
        print('max elapsed time   : ' + str(np.max(elapsed_times_all)) + ' seconds')