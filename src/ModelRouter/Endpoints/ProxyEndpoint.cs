using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using ModelRouter.Http;
using ModelRouter.Relay;

namespace ModelRouter.Endpoints;

public static class ProxyEndpoint
{
    public static async Task ForbiddenAsync(HttpContext ctx)
    {
        ctx.Response.StatusCode = 403;
        ctx.Response.ContentType = "application/json";
        await ctx.Response.WriteAsync("{\"error\": \"slot API is not proxied\"}");
    }

    public static async Task ProxyAsync(HttpContext ctx, RelayService relay)
    {
        string path = ctx.Request.Path.Value ?? "/";
        string method = ctx.Request.Method;
        byte[]? body = method == "POST" ? await RequestBodyReader.ReadAsync(ctx.Request) : null;
        await relay.RelayAsync(ctx, method, path, null, false, body, false);
    }

    public static Task NotImplementedAsync(HttpContext ctx)
    {
        ctx.Response.StatusCode = 501;
        return Task.CompletedTask;
    }
}
