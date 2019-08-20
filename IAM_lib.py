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
    return num_mean_samples_all

def create_output_folders(dirOutput, mri_code):
    dirOutData = dirOutput + '/' + mri_code
    dirOutDataCom = dirOutput + '/' + mri_code + '/IAM_combined_python/'
    dirOutDataPatch = dirOutput + '/' + mri_code + '/IAM_combined_python/Patch/'
    dirOutDataCombined = dirOutput + '/' + mri_code + '/IAM_combined_python/Combined/'

    os.makedirs(dirOutData)
    os.makedirs(dirOutDataCom)
    os.makedirs(dirOutDataPatch)
    os.makedirs(dirOutDataCombined)

def keep_relevant_slices(mri_data):
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
    [x_len, y_len, z_len] = modified_array.shape
    index_end = original_index_end - z_len - index_start
    top_empty_slices = np.zeros([x_len, y_len, index_start])
    bottom_empty_slices = np.zeros([x_len, y_len, index_end])
    reshaped_array = np.concatenate((top_empty_slices,modified_array), axis=2)
    reshaped_array = np.concatenate((reshaped_array, bottom_empty_slices), axis=2)
    return reshaped_array



def kernel_sphere(vol):
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
    [x_len, y_len] = img.shape
    even_x = np.mod(x_dist, 2) - 2;
    even_y = np.mod(y_dist, 2) - 2;

    x_top = x_c - np.floor(x_dist / 2) - (even_x + 1);
    x_low = x_c + np.floor(x_dist / 2);
    y_left = y_c - np.floor(y_dist / 2) - (even_y + 1);
    y_rght = y_c + np.floor(y_dist / 2);

    if x_top < 0: x_top = 0
    if x_low >= x_len: x_low = x_len
    if y_left < 0: y_left = 0
    if y_rght >= y_len: y_rght = y_len

    area = img[int(x_top):int(x_low+1),int(y_left):int(y_rght+1)]

    return area


def get_volume(x_c, y_c, z_c, x_dist, y_dist, z_dist, brain):
    [x_len, y_len, z_len] = brain.shape
    even_x = np.mod(x_dist, 2) - 2;
    even_y = np.mod(y_dist, 2) - 2;
    even_z = np.mod(z_dist, 2) - 2;

    x_top = x_c - np.floor(x_dist / 2) - (even_x + 1);
    x_low = x_c + np.floor(x_dist / 2);
    y_left = y_c - np.floor(y_dist / 2) - (even_y + 1);
    y_rght = y_c + np.floor(y_dist / 2);
    z_front = z_c - np.floor(z_dist / 2) - (even_z + 1);
    z_back = z_c + np.floor(z_dist / 2);

    if x_top < 0: x_top = 0
    if x_low >= x_len: x_low = x_len
    if y_left < 0: y_left = 0
    if y_rght >= y_len: y_rght = y_len
    if z_front < 0: z_front = 0
    if z_back >= z_len: z_back = z_len

    #print('IDX: ' + str(int(x_top)) + ', ' + str(int(x_low)) + ', ' + str(int(y_left)) + ', ' + str(int(y_rght)) + ', ' + str(int(z_front)) + ', ' + str(int(z_back)))
    #print("From x:", x_c, "and y", y_c, "and z", z_c)

    volume = brain[int(x_top):int(x_low+1),int(y_left):int(y_rght+1),int(z_front):int(z_back+1)]
    return volume


def get_thresholded_brain(mri_data):
    mri_data = mri_data/np.nanmax(np.nanmax(np.nanmax(mri_data)))
    scan_mean = np.sum(mri_data[mri_data > 0]) / np.sum(mri_data > 0)
    scan_std = np.std(mri_data[mri_data > 0])

    mri_std = np.true_divide((mri_data-scan_mean), scan_std)
    WMH = np.zeros(mri_data.shape)
    iWMH = np.zeros(mri_data.shape)

    WMH[np.nan_to_num(mri_data) >= (scan_mean + (1.282 * scan_std))] = 1       # Less intense regions
    iWMH[np.nan_to_num(mri_data) >= (scan_mean + (1.69 * scan_std))] = 1     # Very intense regions

    for zz in range(WMH.shape[2]):
        layer_iWMH = iWMH[:, :, zz]
        kernel = np.ones((2,2),np.uint8)
        layer_iWMH = cv2.erode(layer_iWMH, kernel, iterations = 1)
        #layer_iWMH = cv2.blur(layer_iWMH,(2, 2))
        #layer_iWMH[layer_iWMH > 0] = 1
        iWMH[:, :, zz] = layer_iWMH
    iWMH = gaussian_3d(iWMH)
    iWMH[iWMH > 0] = 1

    return iWMH


def gaussian_3d(thresholded_brain):
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
    threshold = (patch_size*patch_size*patch_size) * threshold
    x, y, z = index_chosen
    WMH_volume = get_volume(x, y, z, patch_size, patch_size, patch_size, thresholded_brain)

    if np.count_nonzero(WMH_volume) > threshold:
        return False
    return True


def get_shuffled_patches(target_patches_list, num_samples):
    shuffled_list = [target_patches_list[index] for index in random.sample(range(len(target_patches_list)), num_samples)]
    shuffled_array = np.asarray(shuffled_list)
    return shuffled_array

def get_slice_age_map(patch_size, mat_contents, mask_slice):
    slice_age_map = mat_contents['slice_age_map']
    slice_age_map_res = cv2.resize(slice_age_map, None, fx=patch_size,
                                    fy=patch_size, interpolation=cv2.INTER_CUBIC)
    slice_age_map_res = skifilters.gaussian(slice_age_map_res,sigma=0.5,truncate=2.0)
    slice_age_map_res = np.multiply(mask_slice, slice_age_map_res)
    return slice_age_map_res
