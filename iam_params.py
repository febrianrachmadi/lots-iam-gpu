## General output full path (note to user: you can change this variable)
output_filedir = "results/IAM_GPU_pipeline_test"

## Name of csv file (note to user: you can change this variable)
csv_filename = "IAM_GPU_pipeline_test_v2.csv"

# Save JPEG outputs
save_jpeg = True

## Size of source and target patches
## Default: patch_size = [1,2,4,8]
patch_size = [1,2,4,8]

## Used only for automatic calculation for all number of samples
## NOTE: Smaller number of samples makes computation faster (please refer to the manuscript).
## Samples used for IAM calculation 
## Default: num_samples_all = [512]
num_samples_all = [64]
## Uncomment line below and comment line above if you want to run all different number of samples 
# num_samples_all = [64, 128, 256, 512, 1024, 2048]