#!/bin/sh

if [ ! -d "build" ]; then

	mkdir build
fi

export EIGEN3_PATH=/Users/lix/codes/OpenMS/openms/lib/eigen-3.3.9

cmake -B build \
	-DENABLE_MPI=ON \
	-DENABLE_MEEP=ON \
	-DENABLE_TA=ON \
	-DENABLE_TASCALAPACK=OFF \
	-DENABLE_TACUDA=OFF \
	-DENABLE_TAPYTHON=ON \
	-DENABLE_TBB=ON \
	-Dtoolchainpath=cmake/vg/toolchains/

cd build
make 
cd ..

