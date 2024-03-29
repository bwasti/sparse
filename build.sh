PYTORCH_DIR=$(python -c 'import os, torch; print(os.path.dirname(os.path.realpath(torch.__file__)))')
mkdir -p build && cd build
cmake .. -DPYTORCH_DIR=${PYTORCH_DIR} -DCMAKE_BUILD_TYPE=Release ${@}
make -j 24
