using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using ModelRouter.Chat;
using ModelRouter.Relay;

namespace ModelRouter.Endpoints;

public static class RouterEndpoints
{
    private static readonly string[] AllMethods =
        { "GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS" };

    private static readonly string[] ReadMethods = { "GET", "POST" };

    private static readonly string[] WriteMethods = { "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS" };

    public static IEndpointRouteBuilder MapRouterEndpoints(this IEndpointRouteBuilder app)
    {
        // The slot API can save/restore files on the backend and is never part
        // of the public surface: the router talks to it directly instead.
        app.MapMethods("/slots/{**rest}", AllMethods, ProxyEndpoint.ForbiddenAsync);

        app.MapGet("/health", (HttpContext ctx, BackendManager backend) =>
            HealthEndpoint.HandleAsync(ctx, backend));

        app.MapGet("/v1/models", (HttpContext ctx, BackendManager backend, RelayService relay) =>
            ModelsEndpoint.HandleAsync(ctx, ctx.Request.Path.Value ?? "/v1/models", backend, relay));

        app.MapPost("/v1/chat/completions", (HttpContext ctx, ChatRequestProcessor processor, RelayService relay) =>
            ChatEndpoint.HandleAsync(ctx, processor, relay));

        app.MapPost("/v1/messages", (HttpContext ctx, ChatRequestProcessor processor, RelayService relay) =>
            ChatEndpoint.HandleAsync(ctx, processor, relay));

        app.MapMethods("/{**path}", ReadMethods, (HttpContext ctx, RelayService relay) =>
            ProxyEndpoint.ProxyAsync(ctx, relay));

        app.MapMethods("/{**path}", WriteMethods, (HttpContext ctx) =>
            ProxyEndpoint.NotImplementedAsync(ctx));

        return app;
    }
}
