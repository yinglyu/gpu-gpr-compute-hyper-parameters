module load intel/2017A CUDA
nvcc -arch=compute_35 -code=sm_35 -o gpr gpr.cu

