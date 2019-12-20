#include <THC.h>
#include <THCGeneral.h>

#include "utils.h"

#ifdef __cplusplus
	extern "C" {
#endif



const int BLOCK_DIM =16;
#define CLAMP_TO_EDGE 1

__global__ void d_transpose(float *odata, float *idata, int width, int height)
{
/*
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
*/
    __shared__ float block[BLOCK_DIM][BLOCK_DIM+1];

    // read the matrix tile into shared memory
    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;

    if ((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

//   cg::sync(cta);
	__syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;

    if ((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

void transpose(float *d_src, float *d_dest, unsigned int width, int height)
{
    dim3 grid((width+BLOCK_DIM-1)/BLOCK_DIM, (height+BLOCK_DIM-1)/BLOCK_DIM, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
    d_transpose<<< grid, threads >>>(d_dest, d_src, width, height);

}


__global__ void
d_recursiveGaussian(float *id, float *od, int w, int h, float a0, float a1, float a2, float a3, float b1, float b2, float coefp, float coefn)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

    if (x >= w) return;

    id += x;    // advance pointers to correct column
    od += x;

    // forward pass
    float xp = (0.0f);  // previous input
    float yp = (0.0f);  // previous output
    float yb = (0.0f);  // previous output by 2
#if CLAMP_TO_EDGE
    xp = (*id);
    yb = coefp*xp;
    yp = yb;
#endif

    for (int y = 0; y < h; y++)
    {
        float xc = (*id);
        float yc = a0*xc + a1*xp - b1*yp - b2*yb;
        *od = (yc);
        id += w;
        od += w;    // move to next row
        xp = xc;
        yb = yp;
        yp = yc;
    }

    // reset pointers to point to last element in column
    id -= w;
    od -= w;

    // reverse pass
    // ensures response is symmetrical
    float xn = 0.0f;
    float xa = 0.0f;
    float yn = 0.0f;
    float ya = 0.0f;
#if CLAMP_TO_EDGE
    xn = xa = (*id);
    yn = coefn*xn;
    ya = yn;
#endif

    for (int y = h-1; y >= 0; y--)
    {
        float xc = (*id);
        float yc = a2*xn + a3*xa - b1*yn - b2*ya;
        xa = xn;
        xn = xc;
        ya = yn;
        yn = yc;
        *od += yc;
        id -= w;
        od -= w;  // move to previous row
    }
}



void GaussianBlur_kernel_forward(
	at::Tensor input,
	at::Tensor temp,
	at::Tensor output,
	float  sigma
) {

    // compute filter coefficients
    const float
	    nsigma = sigma < 0.1f ? 0.1f : sigma,
	    alpha = 1.695f / nsigma,
	    ema = (float)std::exp(-alpha),
	    ema2 = (float)std::exp(-2*alpha),
	    b1 = -2*ema,
	    b2 = ema2;

    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, coefp = 0, coefn = 0;

	int order = 0;//test this first 
    switch (order)
    {
        case 0:
            {
                const float k = (1-ema)*(1-ema)/(1+2*alpha*ema-ema2);
                a0 = k;
                a1 = k*(alpha-1)*ema;
                a2 = k*(alpha+1)*ema;
                a3 = -k*ema2;
            }
            break;

        case 1:
            {
                const float k = (1-ema)*(1-ema)/ema;
                a0 = k*ema;
                a1 = a3 = 0;
                a2 = -a0;
            }
            break;

        case 2:
            {
                const float
                ea = (float)std::exp(-alpha),
                k = -(ema2-1)/(2*alpha*ema),
                kn = (-2*(-1+3*ea-3*ea*ea+ea*ea*ea)/(3*ea+1+3*ea*ea+ea*ea*ea));
                a0 = kn;
                a1 = -kn*(1+k*alpha)*ema;
                a2 = kn*(1-k*alpha)*ema;
                a3 = -kn*ema2;
            }
            break;

        default:
            fprintf(stderr, "gaussianFilter: invalid order parameter!\n");
            return;
    }

    coefp = (a0+a1)/(1+b1+b2);
    coefn = (a2+a3)/(1+b1+b2);





	float * in = input.data<float>();
	float * tmp = temp.data<float>();
	float * out = output.data<float>();
	int width = input.size(3);
	int height = input.size(2);

/*
	float * in = THCudaTensor_data(state, input);
	float * tmp = THCudaTensor_data(state, temp);
	float * out = THCudaTensor_data(state,output);

	int width = THCudaTensor_size(state, input,3);
	int height = THCudaTensor_size(state, input,2);
	int inputsize0 = THCudaTensor_size(state, input,0);
	int inputstride0 = THCudaTensor_stride(state, input,0);
	int tempstride0 = THCudaTensor_stride(state, temp,0);
	int outputstride0 = THCudaTensor_stride(state, output,0);

	int inputsize1 = THCudaTensor_size(state, input,1);
	int inputstride1 = THCudaTensor_stride(state, input,1);
	int tempstride1 = THCudaTensor_stride(state, temp,1);
	int outputstride1 = THCudaTensor_stride(state, output,1);
*/


	for (int b = 0; b < input.size(0); b++, tmp+=temp.stride(0),in+= input.stride(0), out+= output.stride(0))
	{
	    float * t = tmp;
	    float * i = in;
	    float * o = out;
        for (int ch = 0; ch< input.size(1); ch++, t+=temp.stride(1),i+= input.stride(1),o+=output.stride(1))
        {

            // process columns
            d_recursiveGaussian<<< (width+512-1)/512, 512,0 /*, THCState_getCurrentStream(state)*/ >>>(i, t, width, height, a0, a1, a2, a3, b1, b2, coefp, coefn);

            transpose(t, o, width, height);

            // process rows
            d_recursiveGaussian<<<  (height+512-1)/512, 512,0/*, THCState_getCurrentStream(state)*/  >>>(o, t, height, width, a0, a1, a2, a3, b1, b2, coefp, coefn);
            transpose(t, o, height, width);
        }
    }

	THCudaCheck(cudaGetLastError());
}








#ifdef __cplusplus
	}
#endif
