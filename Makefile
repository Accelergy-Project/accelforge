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

clone_dependencies:
	mkdir -p .dependencies
	cd .dependencies && git clone git@github.com:Accelergy-Project/fastfusion.git
	cd .dependencies && git clone --recurse-submodules git@github.com:Accelergy-Project/hwcomponents.git
	cd .dependencies && git clone git@github.com:gilbertmike/combinatorics.git
	chmod -R 777 .dependencies

install_dependencies:
	if [ ! -f /.dockerenv ]; then \
		echo "Not in a container. Please run 'make run_docker' first."; \
		exit 1; \
	fi
	pip3 install -e ./.dependencies/fastfusion --break-system-packages
	pip3 install -e ./.dependencies/hwcomponents/models/* --break-system-packages
	pip3 install -e ./.dependencies/combinatorics --break-system-packages
	pip3 install -e ./.dependencies/hwcomponents --break-system-packages

pull_dependencies:
	cd .dependencies/fastfusion && git pull
	cd .dependencies/hwcomponents && git pull
	cd .dependencies/hwcomponents && git submodule update --init --recursive
	cd .dependencies/combinatorics && git pull

build_docker: check_platform
	sudo chmod -R 777 .dependencies
	docker build -t fastfusion/fastfusion-infrastructure:latest-$$DOCKER_ARCH .

run_docker: check_platform
	sudo chmod -R 777 .dependencies
	DOCKER_ARCH=$$DOCKER_ARCH docker-compose up