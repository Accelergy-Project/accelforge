NTL_VER := 11.5.1
BARVINOK_VER := 0.41.8
ISLPY_VER := 2024.2

LIB_DIR := libraries


.PHONY: install-hwcomponents install-ntl install-barvinok install-islpy

install-hwcomponents:
	mkdir -p $(LIB_DIR)
	# install hwcomponents from GitHub (pinned commit recommended)
	git clone --depth 1 https://github.com/Accelergy-Project/hwcomponents.git $(LIB_DIR)/hwcomponents || true
	cd $(LIB_DIR)/hwcomponents && make install-submodules
	cd $(LIB_DIR)/hwcomponents && pip3 install .

install-ntl:
	mkdir -p $(LIB_DIR)
	# If the file does not exist, download it
	if [ ! -f $(LIB_DIR)/ntl-$(NTL_VER).tar.gz ]; then \
		wget -O $(LIB_DIR)/ntl-$(NTL_VER).tar.gz https://libntl.org/ntl-$(NTL_VER).tar.gz; \
	fi
	tar -xvzf $(LIB_DIR)/ntl-$(NTL_VER).tar.gz -C $(LIB_DIR)
	cd $(LIB_DIR)/ntl-$(NTL_VER)/src && ./configure NTL_GMP_LIP=on SHARED=on NATIVE=off
	cd $(LIB_DIR)/ntl-$(NTL_VER)/src && make -j$$(nproc) && make install

install-barvinok:
	mkdir -p $(LIB_DIR)
	if [ ! -f $(LIB_DIR)/barvinok-$(BARVINOK_VER).tar.gz ]; then \
		wget -O $(LIB_DIR)/barvinok-$(BARVINOK_VER).tar.gz https://barvinok.sourceforge.io/barvinok-$(BARVINOK_VER).tar.gz; \
	fi
	tar -xvzf $(LIB_DIR)/barvinok-$(BARVINOK_VER).tar.gz -C $(LIB_DIR)
	cd $(LIB_DIR)/barvinok-$(BARVINOK_VER) && ./configure --enable-shared-barvinok
	cd $(LIB_DIR)/barvinok-$(BARVINOK_VER) && make -j$$(nproc) && make install

install-islpy:
	mkdir -p $(LIB_DIR)
	if [ ! -f $(LIB_DIR)/islpy-$(ISLPY_VER).tar.gz ]; then \
		wget -O $(LIB_DIR)/islpy-$(ISLPY_VER).tar.gz https://github.com/inducer/islpy/archive/refs/tags/v$(ISLPY_VER).tar.gz; \
	fi
	tar -xvzf $(LIB_DIR)/islpy-$(ISLPY_VER).tar.gz -C $(LIB_DIR)
	cd $(LIB_DIR)/islpy-$(ISLPY_VER) && rm -f siteconf.py
	cd $(LIB_DIR)/islpy-$(ISLPY_VER) && ./configure.py --use-barvinok --isl-inc-dir=/usr/local/include --isl-lib-dir=/usr/local/lib --no-use-shipped-isl --no-use-shipped-imath
	cd $(LIB_DIR)/islpy-$(ISLPY_VER) && pip3 install .

build-docker:
	docker build -t fastfusion/fastfusion-infrastructure .

run-docker:
	docker-compose up

.PHONY: generate-docs
generate-docs:
    # pip install sphinx-autobuild sphinx_autodoc_typehints sphinx-rtd-theme
	# rm -r docs/_build
	# sphinx-build -nW docs/source docs/_build/html
	LC_ALL=C.UTF-8 LANG=C.UTF-8 sphinx-apidoc -f -o docs/source/ fastfusion
	LC_ALL=C.UTF-8 LANG=C.UTF-8 sphinx-autobuild -a docs/source docs/_build/html

	# LC_ALL=C.UTF-8 LANG=C.UTF-8 sphinx-apidoc -f -o docs/source/ fastfusion ; rm -r docs/_build/html ; LC_ALL=C.UTF-8 LANG=C.UTF-8 sphinx-autobuild -a docs/source docs/_build/html