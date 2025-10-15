NTL_VER := 11.5.1
BARVINOK_VER := 0.41.8
ISLPY_VER := 2024.2

# Function to check and get platform
check_platform:
	@if [ -z "$$DOCKER_ARCH" ]; then \
		echo "Please set DOCKER_ARCH environment variable to amd64 or arm64"; \
		echo "Option A: export DOCKER_ARCH=amd64 ; make build_docker"; \
		echo "Option B: make build_docker DOCKER_ARCH=amd64"; \
		exit 1; \
	fi
	@if [ "$$DOCKER_ARCH" != "amd64" ] && [ "$$DOCKER_ARCH" != "arm64" ]; then \
		echo "Invalid platform '$$DOCKER_ARCH'. Must be amd64 or arm64."; \
		exit 1; \
	fi
	@echo "Using platform: $$DOCKER_ARCH"


build_docker: check_platform
	sudo chmod -R 777 .dependencies
	docker build -t fastfusion/fastfusion-infrastructure:latest-$$DOCKER_ARCH .

run_docker: check_platform
	sudo chmod -R 777 .dependencies
	DOCKER_ARCH=$$DOCKER_ARCH docker-compose up
	
install-dependencies:
	git clone --recurse-submodules https://github.com/Accelergy-Project/hwcomponents.git
	cd hwcomponents && make install-submodules
	cd hwcomponents && pip3 install .

install-ntl:
	wget -O libraries/ntl-$(NTL_VER).tar.gz https://libntl.org/ntl-$(NTL_VER).tar.gz
	cd libraries/ && tar -xvzf ntl-$(NTL_VER).tar.gz
	cd libraries/ntl-$(NTL_VER)/src && ./configure NTL_GMP_LIP=on SHARED=on NATIVE=off && make
	cd libraries/ntl-$(NTL_VER)/src && make install

install-barvinok: install-ntl
	wget -O libraries/barvinok-$(BARVINOK_VER).tar.gz https://barvinok.sourceforge.io/barvinok-$(BARVINOK_VER).tar.gz
	cd libraries && tar -xvzf barvinok-$(BARVINOK_VER).tar.gz
	cd libraries/barvinok-$(BARVINOK_VER) && ./configure --enable-shared-barvinok
	cd libraries/barvinok-$(BARVINOK_VER) && make
	cd libraries/barvinok-$(BARVINOK_VER) && make install

install-islpy:
	wget -O libraries/islpy-$(ISLPY_VER).tar.gz https://github.com/inducer/islpy/archive/refs/tags/v$(ISLPY_VER).tar.gz
	cd libraries/ && tar -xvzf islpy-$(ISLPY_VER).tar.gz
	cd libraries/islpy-$(ISLPY_VER) && rm -f siteconf.py
	cd libraries/islpy-$(ISLPY_VER) && ./configure.py --use-barvinok --isl-inc-dir=/usr/local/include --isl-lib-dir=/usr/local/lib --no-use-shipped-isl --no-use-shipped-imath
	cd libraries/islpy-$(ISLPY_VER) && pip3 install .

docs:
    # pip install sphinx-autobuild sphinx_autodoc_typehints sphinx-rtd-theme
    LC_ALL=C.UTF-8 LANG=C.UTF-8 sphinx-apidoc -o docs fastfusion
    LC_ALL=C.UTF-8 LANG=C.UTF-8 sphinx-autobuild docs docs/_build/html
