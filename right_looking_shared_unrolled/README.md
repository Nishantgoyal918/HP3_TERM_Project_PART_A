Before Compilation, please edit the values of N and TILE_SIZE in the "pyexpander_functions.txt" file

Then generate the unrolled "aux_1.cu" using the following command:
python3 -m expander.py pyexpander_functions.txt > aux_1.cu

After that, compile the "driver_code.cu" file using the following command:
nvcc driver_code.cu -o right_looking_unrolled

#Please note that, the max value of TILE_SIZE should not exceed 16 in completely Unrolled version, and also the max value of N should not exceed 16 times the value of TILE_SIZE

To use TILE_SIZE 32 in the unrolled version, you should have to define a loop which should run twice, i.e, 16 unrolled statement running twice
