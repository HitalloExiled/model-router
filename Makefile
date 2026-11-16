# Let the environment resolve the SDK: `dotnet` comes from PATH, and
# DOTNET/DOTNET_ROOT can be supplied by the environment.
DOTNET ?= dotnet
PROJECT := src/ModelRouter/ModelRouter.csproj
TESTS := src/ModelRouter.Tests/ModelRouter.Tests.csproj
SOLUTION := ModelRouter.slnx

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
