install:
	pip3 install -r requirements.txt

	mkdir build
	
	cd build && git clone https://github.com/gilbertmike/combinatorics.git && pip3 install ./combinatorics
	cd build && git clone --recurse-submodules https://github.com/Accelergy-Project/accelergy-timeloop-infrastructure.git
	cd build/accelergy-timeloop-infrastructure && make pull && make install_accelergy && make install_timeloop

	cd build && git clone https://github.com/Accelergy-Project/timeloop-python.git
	cd timeloop-python && git checkout bugfix
	cd build/timeloop-python && TIMELOOP_INCLUDE_PATH=/home/fastfusion/build/accelergy-timeloop-infrastructure/src/timeloop/include \
	   TIMELOOP_LIB_PATH=/home/fastfusion/build/accelergy-timeloop-infrastructure/src/timeloop/lib \
	   pip3 install -e .
