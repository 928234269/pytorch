#include <THC.h>
#include <THCGeneral.h>

#include "utils.h"

#ifdef __cplusplus
	extern "C" {
#endif



const int BLOCK_DIM =16;
#define CLAMP_TO_EDGE 1


//transpose float 

__global__ void d_transpose(float *odata, float *idata, int width, int height);
void transpose(float *d_src, float *d_dest, unsigned int width, int height);


//tranpose float2

__global__ void d_transpose2(float2 *odata, float2 *idata, int width, int height)
{
/*
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
*/
    __shared__ float2 block[BLOCK_DIM][BLOCK_DIM+1];

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

void transpose2(float2 *d_src, float2 *d_dest, unsigned int width, int height)
{
    dim3 grid((width+BLOCK_DIM-1)/BLOCK_DIM, (height+BLOCK_DIM-1)/BLOCK_DIM, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
    d_transpose2<<< grid, threads >>>(d_dest, d_src, width, height);

}



//linear intropolation for gap 

__global__ void
d_recursiveLinear(float2 * flow, float * mask /* 0 for gap, >0.0 for valid*/, float2 * value, float * pos,  int w, int h)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

    if (x >= w) return;

	mask += x; 
	flow += x; 
	pos += x; 
	value +=x; 
	
	float last_pos = -1000000.0f; // pos value < 0.0f means it's invalid
	float2 last_value ;
	
	
//forward: from top to bottom 
	for (int y=0; y< h;y++,mask+=w,flow+=w,pos+=w,value+=w)
	{
		if (*mask > 0.0f ) 
		{
			last_pos = 0.0f;  //0.0f for pos means "spot-on" 
			last_value = *flow;
		}
		else 
		{
			last_pos += 1.0f;
		}
		
		*pos = last_pos;
		*value = last_value; 
		
	}

//backward: from bottom to top 
	flow -=w; 
	mask -=w;
	pos -=w;
	value -=w;
	
	last_pos = -1000000.0f;
	
	for (int y=h ; y> 0 ; y--, mask-=w, flow-=w, pos-=w, value-=w)
	{

		if (*mask > 0.0f )
		{
			last_pos = 0.0f;
			last_value = *flow;
			//spot on , no need to change values in value and pos
		}
		else 
		{
			last_pos += 1.0f;
			if (last_pos > 0.0f && *pos > 0.0f ) 
			{
			//has both side 
				 
				(*value).x = ((*value).x * last_pos +  last_value.x * *pos )/(last_pos + *pos) ;
				(*value).y = ((*value).y * last_pos + last_value.y * *pos) / (last_pos + *pos) ;
				*pos = last_pos + *pos; 
			}
			else 
			{
				*pos = 10000000.0f;
			}
		}
	}
	
}


__global__ void
d_merge(float2 * value0, float * pos0 , float2 * value1,float * pos1,  int w, int h)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

    if (x >= w) return;

	pos0 += x; 
	value0 += x; 
	pos1 += x; 
	value1 +=x; 
	
	
	
	for (int y=0; y< h ; y++)
	{
		if (* pos0 < * pos1 )
		{ 
			*value1 = *value0;
			*pos1 = *pos0;
		}

		if (*pos1 > 100000.0f)
			*pos1 = 0.0f;
		else 
			*pos1 = 1.0f; 

		pos0 += w; 
		value0 += w; 
		pos1 += w; 
		value1 +=w; 		
	}
	
}

void FlowBlur_kernel_forward(
	at::Tensor in_flow,
	at::Tensor in_mask,
	at::Tensor out_flow,
	at::Tensor out_mask, 
	at::Tensor tmp1_flow,
	at::Tensor tmp1_mask, 
	at::Tensor tmp2_flow,
	at::Tensor tmp2_mask
	
) 
//flow shape HxWx2
//mask shape HxW

{

	float2 * v_flow  = (float2*) in_flow.data<float>();
	float2 * v_value  = (float2*) out_flow.data<float>();
	float2 * h_flow  = (float2*) tmp1_flow.data<float>();
	float2 * h_value  = (float2*) tmp2_flow.data<float>();
	
	float * v_mask = in_mask.data<float>();
	float * v_pos = out_mask.data<float>();
	float * h_mask = tmp1_mask.data<float>();
	float * h_pos = tmp2_mask.data<float>();
	
	
	int width = in_mask.size(1);
	int height = in_mask.size(0);
	
    // process columns
    d_recursiveLinear<<< (width+512-1)/512, 512>>>
    	(v_flow, v_mask, h_value, h_pos, width, height);
	
	//transpose input to h_flow and h_mask
    transpose2(v_flow, h_flow, width, height);
    transpose(v_mask , h_mask, width, height);
    
    //transpose first output to v_value, v_pos
    transpose2(h_value, v_value, width, height);
    transpose(h_pos , v_pos, width, height);




    // process rows
            
    d_recursiveLinear<<< (width+512-1)/512, 512>>>
    	(h_flow, h_mask, h_value, h_pos, height, width);

	//second output in h_value,h_pos, first output in v_value, v_pos.  Merge them to h_value, h_pos
    d_merge<<< (width+512-1)/512, 512 >>>
    	(v_value, v_pos, h_value, h_pos, height, width);

	//transpose final result from h_value, h_pos to v_value, v_pos
    transpose2(h_value, v_value, height, width);
    transpose(h_pos , v_pos, height, width);
    

	//THCudaCheck(cudaGetLastError());
}








#ifdef __cplusplus
	}
#endif
