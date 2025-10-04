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
