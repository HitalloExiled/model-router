using ModelRouter.Configuration;
using ModelRouter.Relay;

namespace ModelRouter.Endpoints;

public static class ModelsEndpoint
{
    public static async Task HandleAsync(HttpContext ctx, string path, BackendManager backend, RelayService relay)
    {
        bool up = backend.Profile is not null
            && ((await backend.GetBackendContainerAsync()) is not null || await backend.IsBackendHealthyAsync());
        if (up)
        {
            await relay.RelayAsync(ctx, "GET", path, null, false, null, false);
            return;
        }
        ctx.Response.ContentType = "application/json";
        await ctx.Response.WriteAsync(StaticModels.Json);
    }
}
