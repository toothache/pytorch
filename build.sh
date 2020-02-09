export TORCH_CUDA_ARCH_LIST="Kepler;Pascal;7.0(6.0)"
export USE_OPENCV=1
export BUILD_TORCH=ON
export CMAKE_PREFIX_PATH="/usr/bin/"
export USE_CUDA=1
export USE_NNPACK=1
export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

python3 setup.py bdist_wheel
