DOTNET ?= /usr/lib64/dotnet/dotnet
PROJECT := src/ModelRouter/ModelRouter.csproj
TESTS := src/ModelRouter.Tests/ModelRouter.Tests.csproj

export DOTNET

.PHONY: all build publish test ab setup run e2e clean

all: build

build:
	$(DOTNET) build $(PROJECT) -c Release

publish:
	$(DOTNET) publish $(PROJECT) -c Release -r linux-x64

test:
	$(DOTNET) test $(TESTS) -c Release

ab:
	python3 tests/ab/ab.py

setup:
	bash scripts/setup-service.sh

run:
	bash scripts/run-router.sh

e2e:
	$(DOTNET) run probes/e2e.cs

clean:
	rm -rf src/ModelRouter/bin src/ModelRouter/obj \
	       src/ModelRouter.Tests/bin src/ModelRouter.Tests/obj \
	       tests/ab/AbRunner/bin tests/ab/AbRunner/obj
