DOCKER_EXE ?= docker
DOCKER_NAME ?= accelforge
DOCKER_BUILD ?= ${DOCKER_EXE} buildx build --load

VERSION := 0.1

USER    := timeloopaccelergy
REPO    := accelforge

NAME    := ${USER}/${REPO}
TAG     := $$(git log -1 --pretty=%h)
IMG     := ${NAME}:${TAG}

ALTTAG  := latest
ALTIMG  := ${NAME}:${ALTTAG}

# Install hwcomponents packages from PyPI for Docker builds.
.PHONY: install-hwcomponents
install-hwcomponents:
	python3 -m pip install --no-cache-dir hwcomponents hwcomponents-adc hwcomponents-cacti hwcomponents-library hwcomponents-neurosim

# Login to docker hub
login:
	"${DOCKER_EXE}" login --username ${DOCKER_NAME} --password ${DOCKER_PASS}

# Build and tag docker image
build-amd64:
	${DOCKER_BUILD} ${BUILD_FLAGS} --platform linux/amd64 \
          --build-arg BUILD_DATE=`date -u +"%Y-%m-%dT%H:%M:%SZ"` \
          --build-arg VCS_REF=${TAG} \
          --build-arg BUILD_VERSION=${VERSION} \
          -t ${IMG}-amd64 \
          -t ${ALTIMG}-amd64 .

build-arm64:
	${DOCKER_BUILD} ${BUILD_FLAGS} --platform linux/arm64 \
          --build-arg BUILD_DATE=`date -u +"%Y-%m-%dT%H:%M:%SZ"` \
          --build-arg VCS_REF=${TAG} \
          --build-arg BUILD_VERSION=${VERSION} \
          -t ${IMG}-arm64 \
          -t ${ALTIMG}-arm64 .

# Push docker image
push-amd64:
	@echo "Pushing ${NAME}:${ALTTAG}-amd64"
	#Push Amd64 version
	"${DOCKER_EXE}" push ${NAME}:${ALTTAG}-amd64
	#Combine Amd64 version into multi-architecture docker image.
	"${DOCKER_EXE}" manifest create \
		${NAME}:${ALTTAG} \
		--amend ${NAME}:${ALTTAG}-amd64 \
	  --amend ${NAME}:${ALTTAG}-arm64
	"${DOCKER_EXE}" manifest push ${NAME}:${ALTTAG}

push-arm64:
	@echo "Pushing ${NAME}:${ALTTAG}-arm64"
	#Push Arm64 version
	"${DOCKER_EXE}" push ${NAME}:${ALTTAG}-arm64
	#Combine Arm64 version into multi-architecture docker image.
	"${DOCKER_EXE}" manifest create \
		${NAME}:${ALTTAG} \
		--amend ${NAME}:${ALTTAG}-amd64 \
	  --amend ${NAME}:${ALTTAG}-arm64
	"${DOCKER_EXE}" manifest push ${NAME}:${ALTTAG}

run-docker:
	docker-compose up

.PHONY: generate-docs
generate-docs:
    # pip install sphinx-autobuild sphinx_autodoc_typehints sphinx-copybutton pydata-sphinx-theme
	rm -r docs/_build/html
	rm docs/source/accelforge.*.rst
	LC_ALL=C.UTF-8 LANG=C.UTF-8 sphinx-apidoc -f -o docs/source/ --tocfile accelforge accelforge
	LC_ALL=C.UTF-8 LANG=C.UTF-8 sphinx-autobuild -a docs/source docs/_build/html
