DOCKER_EXE ?= docker
DOCKER_NAME ?= accelforge

VERSION := 0.1

USER    := accelforge
REPO    := accelforge

NAME    := ${USER}/${REPO}
TAG     := $$(git log -1 --pretty=%h)
IMG     := ${NAME}:${TAG}

ALTTAG  := latest
ALTIMG  := ${NAME}:${ALTTAG}

# Build and tag docker image
build-amd64:
	"${DOCKER_EXE}" build ${BUILD_FLAGS} --platform linux/amd64 \
          --build-arg BUILD_DATE=`date -u +"%Y-%m-%dT%H:%M:%SZ"` \
          --build-arg VCS_REF=${TAG} \
          --build-arg BUILD_VERSION=${VERSION} \
          -t ${IMG}-amd64 .
	"${DOCKER_EXE}" tag ${IMG}-amd64 ${ALTIMG}-amd64

build-arm64:
	"${DOCKER_EXE}" build ${BUILD_FLAGS} --platform linux/arm64 \
          --build-arg BUILD_DATE=`date -u +"%Y-%m-%dT%H:%M:%SZ"` \
          --build-arg VCS_REF=${TAG} \
          --build-arg BUILD_VERSION=${VERSION} \
          -t ${IMG}-arm64 .
	"${DOCKER_EXE}" tag ${IMG}-arm64 ${ALTIMG}-arm64

run-docker:
	docker-compose up

.PHONY: generate-docs
generate-docs:
    # pip install sphinx-autobuild sphinx_autodoc_typehints sphinx-copybutton pydata-sphinx-theme
	rm -r docs/_build/html
	rm docs/source/accelforge.*.rst
	LC_ALL=C.UTF-8 LANG=C.UTF-8 sphinx-apidoc -f -o docs/source/ --tocfile accelforge accelforge
	LC_ALL=C.UTF-8 LANG=C.UTF-8 sphinx-autobuild -a docs/source docs/_build/html
