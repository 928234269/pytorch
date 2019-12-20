REM run "c:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.11

set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0

set TORCHCMD=python -c "import os; import torch; print(os.path.dirname(torch.__file__))"
set TORCH
for /f "usebackq delims=" %%a in (`%TORCHCMD%`) do set TORCH=%%a

set GENDEF="d:\Program Files\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin\gendef.exe"

REM %GENDEF% %TORCH%\lib\TH.dll
REM %GENDEF% %TORCH%\lib\THC.dll
REM %GENDEF% %TORCH%\lib\ATen.dll
REM lib /machine:x64 /def:TH.def 
REM lib /machine:x64 /def:THC.def 
REM lib /machine:x64 /def:ATen.def

SET CC_OPTS=--gpu-architecture=compute_52 --gpu-code=compute_52 -I %TORCH%/lib/include/TH -I %TORCH%/lib/include/THC -I %TORCH%/lib/include -Xcompiler "/MD /wd 4819"


nvcc  -c -o src/utils.o src/utils.cu %CC_OPTS%
nvcc -c -o src/SeparableConvolution_kernel.o src/SeparableConvolution_kernel.cu %CC_OPTS%
nvcc -c -o src/FlowMover_kernel.o src/FlowMover_kernel.cu %CC_OPTS%
nvcc -c -o src/FlowChecker_kernel.o src/FlowChecker_kernel.cu %CC_OPTS%
nvcc -c -o src/GaussianBlur_kernel.o src/GaussianBlur_kernel.cu %CC_OPTS%
nvcc -c -o src/EigenAnalysis_kernel.o src/EigenAnalysis_kernel.cu %CC_OPTS%
nvcc -c -o src/FlowBlur_kernel.o src/FlowBlur_kernel.cu %CC_OPTS%

python install_win_cuda90.py
