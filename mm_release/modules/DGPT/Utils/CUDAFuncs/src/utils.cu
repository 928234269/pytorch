#include <THC.h>
#include <THCGeneral.h>

#ifdef __cplusplus
	extern "C" {
#endif

long4 get_stride(THCState * state, THCudaTensor * tensor)
{
    int stride[4];
    for (int i=0; i< 4; i++)
    {
        stride[i]= THCudaTensor_stride(state,tensor,i);

    }
    return make_long4(stride[0],stride[1],stride[2],stride[3]);
}

long4 get_size(THCState * state, THCudaTensor * tensor)
{
    int size[4];
    for (int i=0; i< 4; i++)
    {
        size[i]= THCudaTensor_size(state,tensor,i);

    }
    return make_long4(size[0],size[1],size[2],size[3]);
}

#ifdef __cplusplus
	}
#endif