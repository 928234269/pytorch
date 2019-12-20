FROM nvcr.io/nvidia/pytorch:19.04-py3
RUN pip install pyyaml  opencv-python -i https://mirrors.aliyun.com/pypi/simple/
RUN pip uninstall -y torch && pip list
RUN git clone --recursive https://github.com/pytorch/pytorch && \
    cd pytorch  && \
    git submodule sync && \
    git submodule update --init --recursive
COPY mm_release /workspace/mm_release
RUN easy_install /workspace/mm_release/DGPTCUDA-0.0.0-py3.6-linux-x86_64.egg
RUN easy_install /workspace/mm_release/mmcudnnsvc-0.0.1b0-py3.6-linux-x86_64.egg