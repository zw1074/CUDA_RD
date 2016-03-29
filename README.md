# CUDA RD solution

Data: A large tridiagonal matrix

Platform: CUDA C++

Requirenment: [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit)

## Abstract

[RD(Recursive Doubling)](http://www.cs.ucsb.edu/~omer/DOWNLOADABLE/3diagonal89.pdf) is a very efficient paralle algorithm for solving tridiagonal matrix. If you have enough core, you can decrease the running time to O(log n), which is extremely fast compared to LU decomposition O(n). 

## How to run

Besure that you have `CUDA toolkit` and add `nvcc` in your library path. Then you can just simply type
```bash
$ nvcc RD_solution.cu -o solution.out; ./solution.out
```
