from numba import cuda
import numpy as np
import math, numba, cv2
import os, random
import skimage.morphology as skimorph
import skimage.filters as skifilters
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal
import code
from timeit import default_timer as timer

## CUDA FUNCTIONS
@cuda.jit
def cu_sub_st(source, target, result):
    si, ti = cuda.grid(2)

    if si < source.shape[0] and ti < target.shape[0]:
        for ii in range(0, result.shape[2]):
            result[si,ti,ii] = source[si,ii] - target[ti,ii]
        cuda.syncthreads()

@cuda.jit
def cu_sub_sqr_st(source, target, result):
    si, ti = cuda.grid(2)

    if si < source.shape[0] and ti < target.shape[0]:
        for ii in range(0, result.shape[2]):
            result[si,ti,ii] = (source[si,ii] - target[ti,ii]) * (source[si,ii] - target[ti,ii])
        cuda.syncthreads()

@cuda.jit(device=True)
def cu_max_abs_1d(array):
    temp = -9999
    for i in range(0, array.shape[0]):
        if array[i] > temp:
            temp = array[i]
    if temp < 0: temp *= -1
    return temp

@cuda.jit(device=True)
def cu_mean_abs_1d(array):
    temp = 0
    for i in range(array.shape[0]):
        temp += array[i]
    if temp < 0: temp *= -1
    return temp / array.size

@cuda.jit
def cu_max_mean_abs(inputs, results):
    si, ti = cuda.grid(2)

    if si < results.shape[0] and ti < results.shape[1]:
        results[si,ti,0] = cu_max_abs_1d(inputs[si,ti,:])
        results[si,ti,1] = cu_mean_abs_1d(inputs[si,ti,:])
        cuda.syncthreads()
    cuda.syncthreads()

@cuda.jit
def cu_distances(inputs, flag, outputs, alpha):
    si, ti = cuda.grid(2)

    if si < outputs.shape[0] and ti < outputs.shape[1]:
        outputs[si,ti] = flag[si] * (alpha * inputs[si,ti,0] + (1 - alpha) * inputs[si,ti,1])
        cuda.syncthreads()
    cuda.syncthreads()

@cuda.jit
def cu_sort_distance(array):
    i = cuda.grid(1)

    if i < array.shape[0]:
        for passnum in range(len(array[i,:]) - 1, 0, -1):
            for j in range(passnum):
                if array[i,j] > array[i,j + 1]:
                    temp = array[i,j]
                    array[i,j] = array[i,j + 1]
                    array[i,j + 1] = temp
    cuda.syncthreads()

@cuda.jit
def cu_age_value(arrays, results):
    i = cuda.grid(1)

    if i < results.shape[0]:
        results[i] = cu_mean_abs_1d(arrays[i,:])
        cuda.syncthreads()
    cuda.syncthreads()


## NON-CUDA FUNCTIONS
def set_mean_sample_number(num_samples_all):
    '''
    Set number of target patches used to calculate irregularity map.
    '''
    if num_samples_all == 64:
        return 16
    elif num_samples_all == 128:
        return 32
    elif num_samples_all == 256:
        return 32
    elif num_samples_all == 512:
        return 64
    elif num_samples_all == 1024:
        return 128
    elif num_samples_all == 2048:
        return 256
    else:
        raise ValueError("Number of samples must be either 64, 128, 256, 512, 1024 or 2048!")
        return 0

def gen_2d_source_target_patches(brain_slice, patch_size, num_samples, TRSH):
    ''' 
    Generate 2D source and target patches for LOTS-IM calculation 
    '''
    [x_len, y_len] = brain_slice.shape
    counter_y = int(y_len / patch_size)        ## counter_y = 512 if patch of size 1 and image of size 512x512
    counter_x = int(x_len / patch_size)
    source_patch_len = counter_x * counter_y        ## How many source patches are neede (e.g. for 1, we need one for each pixel)

    mask_slice = np.nan_to_num(brain_slice)
    mask_slice[mask_slice > 0] = 1

    ## Creating grid-patch 'xy-by-xy'
    #  -- Column
    y_c = np.ceil(patch_size / 2)
    y_c_sources = np.zeros(int(y_len / patch_size))
    for iy in range(0, int(y_len / patch_size)):
        y_c_sources[iy] = (iy * patch_size) + y_c - 1

    #  -- Row
    x_c = np.ceil(patch_size / 2)
    x_c_sources = np.zeros(int(x_len / patch_size))
    for ix in range(0, int(x_len / patch_size)):
        x_c_sources[ix] = (ix * patch_size) + x_c - 1

    ''' Extracting Source Patches '''
    area_source_patch = np.zeros([1,patch_size,patch_size])
    icv_source_flag = np.zeros([source_patch_len])
    idx_mapping = np.ones([source_patch_len]) * -1

    index = 0
    idx_source= 0

    if patch_size == 1:
        area_source_patch = brain_slice[mask_slice == 1]
        area_source_patch = area_source_patch.reshape([area_source_patch.shape[0], 1, 1])
        index = source_patch_len
        idx_source = area_source_patch.shape[0]
        icv_source_flag = mask_slice.flatten()
        positive_indices = (np.where(brain_slice.flatten() > 0))[0]
        index = 0
        for i in positive_indices:
            idx_mapping[i] = index
            index += 1
    else:
        area_source_patch = []
        for isc in range(0, counter_x):
            for jsc in range(0, counter_y):
                    icv_source_flag[index] = mask_slice[int(x_c_sources[isc]), int(y_c_sources[jsc])]
                    if icv_source_flag[index] == 1:
                        temp = get_area(x_c_sources[isc], y_c_sources[jsc],
                                        patch_size, patch_size, brain_slice)
                        area_source_patch.append(temp.tolist())
                        idx_mapping[index] = idx_source
                        idx_source += 1

                    index += 1
        area_source_patch = np.asarray(area_source_patch)

    ''' Extracting Target Patches '''
    target_patches = []
    index_debug = 0
    random_array = np.random.randint(10, size=(x_len, y_len))
    index_possible = np.zeros(brain_slice.shape)
    index_possible[(mask_slice != 0) & (random_array > TRSH*10)] = 1
    index_possible = np.argwhere(index_possible)

    for index_chosen in index_possible:
        x, y = index_chosen
        area = get_area(x, y, patch_size, patch_size, brain_slice)
        if area.size == patch_size * patch_size:
            if np.random.randint(low=1, high=10)/10 < (100/(x*y)) * num_samples:
                pass
            target_patches.append(area)
            index_debug += 1

    target_patches_np = get_shuffled_patches(target_patches, num_samples)
    target_patches_np = target_patches_np[0:num_samples,:,:]
    print('Sampling finished: ' + ' with: ' + str(index_debug) + ' samples from: ' + str(x_len * y_len))
    area = []

    ''''''
    ''' Reshaping array data for GPU (CUDA) calculation '''
    source_patches_all = np.reshape(area_source_patch,(area_source_patch.shape[0],
                                    area_source_patch.shape[1] * area_source_patch.shape[2]))
    target_patches_all = np.reshape(target_patches_np, (target_patches_np.shape[0],
                                    target_patches_np.shape[1] * target_patches_np.shape[2]))
    
    return source_patches_all, target_patches_all, idx_source, idx_mapping

def gen_3d_source_target_patches(input_mri_data, patch_size, num_samples, thrsh_patches=None):
    ''' 
    Generate 3D source and target patches for LOTS-IM calculation 
    '''
    ## Get MRI measurements
    [x_len, y_len, z_len] = input_mri_data.shape
    whole_volume = x_len * y_len * z_len

    ## Create mask for whole brain
    mri_mask = np.nan_to_num(input_mri_data)
    mri_mask[mri_mask > 0] = 1

    vol_slice = np.count_nonzero(input_mri_data) / whole_volume
    print('DEBUG-Patch: brain - ' + str(np.count_nonzero(input_mri_data)) +
        ', x_len * y_len * z_len - ' + str(whole_volume) + ', vol: ' + str(round(vol_slice, 5)))

    ## Set the counter for each axis
    counter_y = int(y_len / patch_size)
    counter_x = int(x_len / patch_size)
    counter_z = int(z_len / patch_size)
    source_patch_len = counter_x * counter_y * counter_z

    ## Creating grid-patch 'x-by-y-by-z'
    #  -- Column
    y_c = np.ceil(patch_size / 2)
    y_c_sources = np.zeros(int(y_len / patch_size))
    for iy in range(0, int(y_len / patch_size)):
        y_c_sources[iy] = (iy * patch_size) + y_c - 1

    #  -- Row
    x_c = np.ceil(patch_size / 2)
    x_c_sources = np.zeros(int(x_len / patch_size))
    for ix in range(0, int(x_len / patch_size)):
        x_c_sources[ix] = (ix * patch_size) + x_c - 1

    #  -- Depth
    z_c = np.ceil(patch_size / 2)
    z_c_sources = np.zeros(int(z_len / patch_size))
    for iz in range(0, int(z_len / patch_size)):
        z_c_sources[iz] = (iz * patch_size) + z_c - 1

    # Patch's sampling number treshold
    TRSH = 0.50
    if patch_size == 1 or patch_size == 2:
        if vol_slice < 0.010: TRSH = 0
        elif vol_slice < 0.035: TRSH = 0.15
        elif vol_slice < 0.070 and vol_slice >= 0.035: TRSH = 0.60
        elif vol_slice >= 0.070: TRSH = 0.80
    elif patch_size == 4 or patch_size == 8:
        if vol_slice < 0.035: TRSH = 0

    ''' Extracting Source Patches '''
    print("Extracting source patches.")
    print(str(source_patch_len) + " source patches to extract...")
    icv_source_flag = np.zeros([source_patch_len])
    index_mapping = np.ones([source_patch_len]) * -1

    index = 0
    index_source= 0

    ## If patch_size == 1, avoid heavy computation
    if patch_size == 1:
        area_source_patch = input_mri_data[mri_mask == 1]
        area_source_patch = area_source_patch.reshape([area_source_patch.shape[0], 1, 1, 1])
        index = source_patch_len
        index_source = area_source_patch.shape[0]
        icv_source_flag = mri_mask.flatten()
        positive_indices = (np.where(input_mri_data.flatten() > 0))[0]
        index = 0
        for i in positive_indices:
            index_mapping[i] = index
            index += 1
    else:
        area_source_patch = []
        for isc in range(0, counter_x):
            for jsc in range(0, counter_y):
                for ksc in range(0, counter_z):
                    icv_source_flag[index] = mri_mask[int(x_c_sources[isc]), int(y_c_sources[jsc]), int(z_c_sources[ksc])]
                    if icv_source_flag[index] == 1:
                        temp = get_volume(x_c_sources[isc], y_c_sources[jsc], z_c_sources[ksc],
                                        patch_size, patch_size, patch_size, input_mri_data)
                        area_source_patch.append(temp.tolist())
                        index_mapping[index] = index_source
                        index_source += 1

                    index += 1
        area_source_patch = np.asarray(area_source_patch)
    print("Source patch extraction completed.")

    ''' Extracting Target Patches '''
    print("Extracting target patches.")
    ## Note: target patches are chosen according to mri_mask and threshold
    ## if thresholding is enabled, get a thresholded volume of the brain (WMH)
    if thrsh_patches != None:
        thresholded_brain = get_thresholded_brain(input_mri_data)
        patches_rejected = 0

    target_patches = []
    index_debug = 0
    random_array = np.random.randint(10, size=(x_len, y_len, z_len))
    index_possible = np.zeros(input_mri_data.shape)
    index_possible[(mri_mask != 0) & (random_array > TRSH*10)] = 1
    index_possible = np.argwhere(index_possible)

    for index_chosen in index_possible:
        x, y, z = index_chosen
        volume = get_volume(x, y, z, patch_size, patch_size, patch_size, input_mri_data)
        if volume.size == patch_size * patch_size * patch_size:
            if np.random.randint(low=1, high=10)/10 < (100/(x*y*z)) * num_samples:
                pass
            if thrsh_patches != None:
                thrsh_filter = threshold_filter(thresholded_brain, patch_size, index_chosen, thrsh_patches)
                if thrsh_filter == True:
                    target_patches.append(volume)
                    index_debug += 1
                else:
                    patches_rejected += 1
            else:
                target_patches.append(volume)
                index_debug += 1

    if thrsh_patches != None:
        percentage_rejected = round((patches_rejected/index_debug)*100, 1)
        print("Number of patches rejected: " + str(patches_rejected) + " (" + str(percentage_rejected) + "%).")

    target_patches_np = get_shuffled_patches(target_patches, num_samples)
    print('Sampling finished with: ' + str(target_patches_np.shape[0]) + ' samples from: '
            + str(len(target_patches)))
    volume = []
    ''' 3D processing until here'''

    ''' Reshaping array data for GPU (CUDA) calculation '''
    source_patches_all = np.reshape(area_source_patch,(area_source_patch.shape[0],
                                    area_source_patch.shape[1] * area_source_patch.shape[2] * target_patches_np.shape[3]))
    target_patches_all = np.reshape(target_patches_np, (target_patches_np.shape[0],
                                    target_patches_np.shape[1] * target_patches_np.shape[2] * target_patches_np.shape[3]))

    return source_patches_all, target_patches_all, index_source, index_mapping

def calculate_irregularity_values(source_patches, target_patches, num_mean_samples, 
                                    index_source, alpha=0.5):
    '''
    Calculate irregularity values on GPU (CUDA)
    '''
    age_values_valid = np.zeros(index_source)
    brain_mask = np.ones(index_source)
    source_len = index_source
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
        source_patches_loop = source_patches[il*loop_len:(il*loop_len)+loop_len,:]

        '''  SUBTRACTION '''
        sub_result_gm = cuda.device_array((source_patches_loop.shape[0],
                                            target_patches.shape[0],
                                            target_patches.shape[1]))
        TPB = (4,256)
        BPGx = int(math.ceil(source_patches_loop.shape[0] / TPB[0]))
        BPGy = int(math.ceil(target_patches.shape[0] / TPB[1]))
        BPGxy = (BPGx,BPGy)
        cu_sub_st[BPGxy,TPB](source_patches_loop, target_patches, sub_result_gm)


        '''  MAX-MEAN-ABS '''
        sub_max_mean_result = cuda.device_array((source_patches_loop.shape[0],
                                                    target_patches.shape[0],2))
        cu_max_mean_abs[BPGxy,TPB](sub_result_gm, sub_max_mean_result)
        sub_result_gm = 0  # Free memory

        '''  DISTANCE '''
        distances_result = cuda.device_array((source_patches_loop.shape[0],
                                                target_patches.shape[0]))
        cu_distances[BPGxy,TPB](sub_max_mean_result,
                                brain_mask[il*loop_len:(il*loop_len)+loop_len],
                                distances_result, alpha)
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
    return age_values_valid

def create_output_folders(dirOutput, mri_code):
    ''' 
    Create output folders (directories) 
    '''
    dirOutData = dirOutput + '/' + mri_code
    dirOutDataCom = dirOutput + '/' + mri_code + '/JPEGs/'
    dirOutDataPatch = dirOutput + '/' + mri_code + '/JPEGs/Patch/'
    dirOutDataCombined = dirOutput + '/' + mri_code + '/JPEGs/Combined/'

    os.makedirs(dirOutData)
    os.makedirs(dirOutDataCom)
    os.makedirs(dirOutDataPatch)
    os.makedirs(dirOutDataCombined)

def keep_relevant_slices(mri_data):
    '''
    Exclude empty slices 
    '''
    original_index_end = mri_data.shape[2]
    index_start = 0
    index_end = original_index_end-1

    for index in range(0, original_index_end):
        if np.count_nonzero(~np.isnan(mri_data[:, :, index])) == 0:
            index_start = index
        else:
            break

    for index in range(original_index_end - 1, -1, -1):
        if np.count_nonzero(~np.isnan(mri_data[:, :, index])) == 0:
            index_end = index
        else:
            break
    print("Only considering relevant slices between indices: [" + str(index_start) + "-" + str(index_end) + "]")
    mri_data = mri_data[:, :, index_start:index_end+1]
    mri_data = np.nan_to_num(mri_data)
    return mri_data, index_start, original_index_end

def reshape_original_dimensions(modified_array, index_start, original_index_end):
    '''
    Restore the empty slices back.
    '''
    [x_len, y_len, z_len] = modified_array.shape
    index_end = original_index_end - z_len - index_start
    top_empty_slices = np.zeros([x_len, y_len, index_start])
    bottom_empty_slices = np.zeros([x_len, y_len, index_end])
    reshaped_array = np.concatenate((top_empty_slices,modified_array), axis=2)
    reshaped_array = np.concatenate((reshaped_array, bottom_empty_slices), axis=2)
    return reshaped_array


def kernel_sphere(vol):
    '''
    Kernel sphere for Gaussian noise (OpenCV library).
    '''
    if vol == 1 or vol == 2:
        return np.array([[1]])
    elif vol == 3 or vol == 4:
        return np.array([[0,1,0],[1,1,1],[0,1,0]])
    elif vol == 5 or vol == 6:
        return np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]])
    elif vol == 7 or vol == 8:
        return np.array([[0,0,0,1,0,0,0],[0,1,1,1,1,1,0],[0,1,1,1,1,1,0],[1,1,1,1,1,1,1],
                         [0,1,1,1,1,1,0],[0,1,1,1,1,1,0],[0,0,0,1,0,0,0]])
    elif vol == 9 or vol == 10:
        return np.array([[0,0,0,0,1,0,0,0,0],[0,0,1,1,1,1,1,0,0],[0,1,1,1,1,1,1,1,0],[0,1,1,1,1,1,1,1,0],
                         [1,1,1,1,1,1,1,1,1],[0,1,1,1,1,1,1,1,0],[0,1,1,1,1,1,1,1,0],[0,0,1,1,1,1,1,0,0],
                         [0,0,0,0,1,0,0,0,0]])
    elif vol == 11  or vol > 11:
        return np.array([[0,0,0,0,0,1,0,0,0,0,0],[0,0,1,1,1,1,1,1,1,0,0],[0,1,1,1,1,1,1,1,1,1,0],
                         [0,1,1,1,1,1,1,1,1,1,0],[0,1,1,1,1,1,1,1,1,1,0],[1,1,1,1,1,1,1,1,1,1,1],
                         [0,1,1,1,1,1,1,1,1,1,0],[0,1,1,1,1,1,1,1,1,1,0],[0,1,1,1,1,1,1,1,1,1,0],
                         [0,0,1,1,1,1,1,1,1,0,0],[0,0,0,0,0,1,0,0,0,0,0]])


def get_area(x_c, y_c, x_dist, y_dist, img):
    '''
    Get MRI's intensities (2D).
    '''
    [x_len, y_len] = img.shape
    even_x = np.mod(x_dist, 2) - 2
    even_y = np.mod(y_dist, 2) - 2

    x_top = x_c - np.floor(x_dist / 2) - (even_x + 1)
    x_low = x_c + np.floor(x_dist / 2)
    y_left = y_c - np.floor(y_dist / 2) - (even_y + 1)
    y_rght = y_c + np.floor(y_dist / 2)

    if x_top < 0: x_top = 0
    if x_low >= x_len: x_low = x_len
    if y_left < 0: y_left = 0
    if y_rght >= y_len: y_rght = y_len

    area = img[int(x_top):int(x_low+1),int(y_left):int(y_rght+1)]

    return area

def get_volume(x_c, y_c, z_c, x_dist, y_dist, z_dist, brain):
    '''
    Get MRI's intensities (3D).
    '''
    [x_len, y_len, z_len] = brain.shape
    even_x = np.mod(x_dist, 2) - 2
    even_y = np.mod(y_dist, 2) - 2
    even_z = np.mod(z_dist, 2) - 2

    x_top = x_c - np.floor(x_dist / 2) - (even_x + 1)
    x_low = x_c + np.floor(x_dist / 2)
    y_left = y_c - np.floor(y_dist / 2) - (even_y + 1)
    y_rght = y_c + np.floor(y_dist / 2)
    z_front = z_c - np.floor(z_dist / 2) - (even_z + 1)
    z_back = z_c + np.floor(z_dist / 2)

    if x_top < 0: x_top = 0
    if x_low >= x_len: x_low = x_len
    if y_left < 0: y_left = 0
    if y_rght >= y_len: y_rght = y_len
    if z_front < 0: z_front = 0
    if z_back >= z_len: z_back = z_len

    volume = brain[int(x_top):int(x_low+1),int(y_left):int(y_rght+1),int(z_front):int(z_back+1)]
    return volume


def get_thresholded_brain(mri_data):
    '''
    Early estimate the WMH using confidence interval (CI).
    '''
    mri_data = mri_data/np.nanmax(np.nanmax(np.nanmax(mri_data)))    
    scan_mean = np.sum(mri_data[mri_data > 0]) / np.sum(mri_data > 0)
    scan_std = np.std(mri_data[mri_data > 0])

    mri_std = np.true_divide((mri_data-scan_mean), scan_std)
    WMH = np.zeros(mri_data.shape)
    iWMH = np.zeros(mri_data.shape)

    WMH[np.nan_to_num(mri_data) >= (scan_mean + (1.282 * scan_std))] = 1      # Less intense regions
    iWMH[np.nan_to_num(mri_data) >= (scan_mean + (1.69 * scan_std))] = 1     # Very intense regions

    for zz in range(WMH.shape[2]):
        layer_iWMH = iWMH[:, :, zz]
        kernel = np.ones((2,2),np.uint8)
        layer_iWMH = cv2.erode(layer_iWMH, kernel, iterations = 1)
        iWMH[:, :, zz] = layer_iWMH
    iWMH[iWMH > 0] = 1

    return iWMH

def gaussian_3d(thresholded_brain):
    '''
    Calculate 3D Gaussian blur.
    '''
    ## Based on https://stackoverflow.com/questions/45723088/
    ## how-to-blur-3d-array-of-points-while-maintaining-their-original-values-python

    sigma = 1.0
    x = np.arange(-1,2,1)
    y = np.arange(-1,2,1)
    z = np.arange(-1,2,1)
    xx, yy, zz = np.meshgrid(x,y,z)
    kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    return signal.convolve(thresholded_brain, kernel, mode="same")

def threshold_filter(thresholded_brain, patch_size, index_chosen, threshold):
    '''
    Reject/accept target patch which has a certain number of early estimated WMH.
    '''
    threshold = (patch_size*patch_size*patch_size) * threshold
    x, y, z = index_chosen
    WMH_volume = get_volume(x, y, z, patch_size, patch_size, patch_size, thresholded_brain)

    if np.count_nonzero(WMH_volume) > threshold:
        return False
    return True

def get_shuffled_patches(target_patches_list, num_samples):
    '''
    Shuffle the target patches.
    '''
    shuffled_list = [target_patches_list[index] for index in random.sample(range(len(target_patches_list)), num_samples)]
    shuffled_array = np.asarray(shuffled_list)
    return shuffled_array

def get_slice_irregularity_map(patch_size, mat_contents, mask_slice):
    '''
    Read irregularity map of each file from intermediary .mat file.
    '''
    slice_age_map = mat_contents['slice_irregularity_map']
    slice_age_map_res = cv2.resize(slice_age_map, None, fx=patch_size,
                                    fy=patch_size, interpolation=cv2.INTER_CUBIC)
    slice_age_map_res = skifilters.gaussian(slice_age_map_res,sigma=0.5,truncate=2.0)
    slice_age_map_res = np.multiply(mask_slice, slice_age_map_res)
    return slice_age_map_res