DOTNET ?= /usr/lib64/dotnet/dotnet
# Tools that host MSBuild (roslynator) enumerate SDKs under DOTNET_ROOT, not
# under DOTNET: with DOTNET_ROOT on an older SDK they load net11 assets with
# SDK 10 and fail every reference. Keep it aligned with the SDK in use.
DOTNET_ROOT := $(patsubst %/,%,$(dir $(DOTNET)))
PROJECT := src/ModelRouter/ModelRouter.csproj
TESTS := src/ModelRouter.Tests/ModelRouter.Tests.csproj
SOLUTION := ModelRouter.slnx

export DOTNET
export DOTNET_ROOT

.PHONY: all build publish test analyze setup run e2e clean

all: build

build:
	$(DOTNET) build $(PROJECT) -c Release

publish:
	$(DOTNET) publish $(PROJECT) -c Release -r linux-x64

test:
	$(DOTNET) test $(TESTS) -c Release

analyze:
	$(DOTNET) roslynator analyze $(SOLUTION)

setup:
	bash scripts/setup-service.sh

run:
	bash scripts/run-router.sh

e2e:
	$(DOTNET) run probes/e2e.cs

clean:
	rm -rf src/ModelRouter/bin src/ModelRouter/obj \
	       src/ModelRouter.Tests/bin src/ModelRouter.Tests/obj
