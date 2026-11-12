
namespace ModelRouter.Endpoints;

public static class HealthEndpoint
{
    public static async Task HandleAsync(HttpContext ctx, BackendManager backend)
    {
        string p = backend.Profile is null ? "null" : $"\"{backend.Profile}\"";
        var json = $"{{\"status\": \"router-ok\", \"backend\": {p}, \"busy\": {backend.Busy.ToString().ToLowerInvariant()}}}";
        ctx.Response.ContentType = "application/json";
        await ctx.Response.WriteAsync(json);
    }
}
