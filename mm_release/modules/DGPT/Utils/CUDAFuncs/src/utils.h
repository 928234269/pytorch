#ifndef _UTIL_H
#define _UTIL_H

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


#ifdef __cplusplus
	extern "C" {
#endif

long4 get_stride(THCState * state, THCudaTensor * tensor);
long4 get_size(THCState * state, THCudaTensor * tensor);

#ifdef __cplusplus
	}
#endif



#define TENSOR_INFO(in) (in).data<float>(),make_long4((in).size(0),(in).size(1),(in).size(2),(in).size(3)), make_long4((in).stride(0),(in).stride(1),(in).stride(2),(in).stride(3))


#endif