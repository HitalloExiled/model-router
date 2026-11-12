using ModelRouter;
using ModelRouter.Chat;
using ModelRouter.Configuration;
using ModelRouter.Endpoints;
using ModelRouter.Hosting;
using ModelRouter.Relay;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddSingleton<BackendManager>();
builder.Services.AddSingleton(_ => new HttpClient
{
    Timeout = RouterDefaults.ForwardTimeout,
    BaseAddress = new Uri($"http://127.0.0.1:{BackendManager.BackendPort}"),
});
builder.Services.AddSingleton<UpstreamForwarder>();
builder.Services.AddSingleton<RelayService>();
builder.Services.AddSingleton<ChatRequestProcessor>();
builder.Services.AddHostedService<IdleUnloadService>();

var app = builder.Build();

// Mirror python __main__: bind 127.0.0.1:ROUTER_PORT (not ASP.NET's default).
app.Urls.Add($"http://127.0.0.1:{BackendManager.RouterPort}");

// Adopt an already-running backend instead of rebooting (mirror __main__).
await app.Services.GetRequiredService<BackendManager>().AdoptAsync();

app.MapRouterEndpoints();
app.Run();
