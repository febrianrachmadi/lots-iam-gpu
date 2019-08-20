## Name of csv file (note to user: you can change this variable)
csv_filename = "input.csv"

## Size of source and target patches.
## Must be in the form of python's list data structure.
## Default: patch_size = [1,2,4,8]
patch_size = [1,2,4,8]

## Weights for age map blending produced by different size of source/target patches
## Must be in the form of python's list data structure.
## Its length must be the same as 'patch_size' variable.
## Default: blending_weights = [0.65,0.2,0.1,0.05]
blending_weights = [0.65,0.2,0.1,0.05]

## Used only for automatic calculation for all number of samples
## NOTE: Smaller number of samples makes computation faster (please refer to the manuscript).
## Samples used for IAM calculation
## Default: num_samples_all = [512]
num_samples_all = [64]
## Uncomment line below and comment line above if you want to run all different number of samples
# num_samples_all = [64, 128, 256, 512, 1024, 2048]

## Weight of distance function to blend maximum difference and average difference between source
## and target patches. Default: alpha=0.5. Input value should be between 0 and 1 (i.e. floating).
alpha = 0.5

## Thresholds the target patches to prevent including patches containing hyper-intensities.
## Default : threshold_patches = None.
thrsh_patches = 0.05

## Threshold value for cutting of probability values of brain masks, if probability masks
## are given instead of binary masks.
bin_tresh = 0.5

## Save JPEG outputs
save_jpeg = True

## Delete all intermediary files/folders, saving some spaces in the hard disk drive.
delete_intermediary = False
