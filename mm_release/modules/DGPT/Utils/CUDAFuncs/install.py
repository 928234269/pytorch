import os
import torch
# import torch.utils.ffi
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools import setup

strBasepath = os.path.split(os.path.abspath(__file__))[0] + '/'
strHeaders = []
strSources = []
strDefines = []
strObjects = []

if torch.cuda.is_available() == True:
	# strHeaders += ['src/SeparableConvolution_cuda.h']
	strSources += ['src/SeparableConvolution_cuda.cpp']
	strDefines += [('WITH_CUDA', None)]
	strObjects += ['src/utils.o',
					'src/SeparableConvolution_kernel.o',
				   'src/FlowMover_kernel.o',
				   'src/GaussianBlur_kernel.o',
                   'src/EigenAnalysis_kernel.o',
                   'src/FlowChecker_kernel.o',
				   'src/FlowBlur_kernel.o',

                   ]

# end
if __name__ == '__main__':
	# print(os.path.expandvars('$CUDA_HOME') + '/include')
	setup(
		name='DGPTCUDA',
		ext_modules=[
			CUDAExtension('DGPTCUDA',
						  [
							'src/SeparableConvolution_cuda.cpp',
							'src/utils.cu',
							'src/EigenAnalysis_kernel.cu',
							'src/FlowBlur_kernel.cu',
							'src/FlowChecker_kernel.cu',
							'src/FlowMover_kernel.cu',
							'src/GaussianBlur_kernel.cu',
							'src/SeparableConvolution_kernel.cu',
							# 'EigenAnalysis_kernel.cu',
							# 'EigenAnalysis_kernel.cu',
							# 'src/utils.o',
							# 'src/SeparableConvolution_kernel.o',
							# 'src/FlowMover_kernel.o',
							# 'src/GaussianBlur_kernel.o',
							# 'src/EigenAnalysis_kernel.o',
							# 'src/FlowChecker_kernel.o',
							# 'src/FlowBlur_kernel.o',
							# 'lltm_cuda_kernel.cu',
						],
						  # include_dirs=[os.path.expandvars('$CUDA_HOME') + '/include'],
						  # library_dirs=[os.path.expandvars('$CUDA_HOME') + '/lib/x64', strBasepath],

			),


			# CUDAExtension(
			# 	name='DGPTCUDA',
			# 	# headers=strHeaders,
			# 	sources=strSources,
			# 	# verbose=False,
			# 	# with_cuda=any(strDefine[0] == 'WITH_CUDA' for strDefine in strDefines),
			# 	# package=False,
			# 	# relative_to=strBasepath,
			# 	include_dirs=[os.path.expandvars('$CUDA_HOME') + '/include'],
			# 	# define_macros=strDefines,
			# 	library_dirs=[os.path.expandvars('$CUDA_HOME') + '/lib/x64', strBasepath],
			# 	# libraries=['cudart', "caffe2", "caffe2_gpu"],
			# 	extra_objects=[os.path.join(strBasepath, strObject) for strObject in strObjects]
			# )
		],
		cmdclass={
			'build_ext': BuildExtension
		})

# objectExtension = torch.utils.cpp_extension.(
# 	name='_ext.cunnex',
# 	headers=strHeaders,
# 	sources=strSources,
# 	verbose=False,
# 	with_cuda=any(strDefine[0] == 'WITH_CUDA' for strDefine in strDefines),
# 	package=False,
# 	relative_to=strBasepath,
# 	include_dirs=[os.path.expandvars('$CUDA_HOME') + '/include'],
# 	define_macros=strDefines,
# 	library_dirs=[os.path.expandvars('$CUDA_HOME') + '/lib/x64', strBasepath],
#     # libraries=['cudart', "caffe2", "caffe2_gpu"],
# 	extra_objects=[os.path.join(strBasepath, strObject) for strObject in strObjects]
#
# )

# if __name__ == '__main__':
# 	objectExtension.build()
# end
