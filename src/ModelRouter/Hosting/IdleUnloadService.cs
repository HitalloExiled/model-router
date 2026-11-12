using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;

namespace ModelRouter.Hosting;

public sealed class IdleUnloadService : BackgroundService
{
    private readonly BackendManager _backend;
    public IdleUnloadService(BackendManager backend) => _backend = backend;

    protected override Task ExecuteAsync(CancellationToken stoppingToken)
        => _backend.IdleWatchAsync(stoppingToken);
}
