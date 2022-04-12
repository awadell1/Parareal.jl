.ONESHELL:
ROOT_DIR := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))

# Target to just install
install: Manifest.toml test/Manifest benchmark/Manifest

# Recipe for installing manifests
Manifest%toml: Project%toml
	julia --project=$(@D) -e "\
		using Pkg;\
		Pkg.Registry.update();\
		Pkg.instantiate();\
		Pkg.build();\
		Pkg.precompile();\
		"

.PHONY: test
test: Manifest.toml test/Manifest.toml
	julia --project --eval 'using Pkg; Pkg.test()'

benchmark: THREADS ?= 4
benchmark: Manifest.toml benchmark/Manifest
	julia --startup-file=no -O3 --threads ${THREADS} \
    	-e 'using Pkg; Pkg.add("PkgJogger"); using PkgJogger; PkgJogger.ci()'
