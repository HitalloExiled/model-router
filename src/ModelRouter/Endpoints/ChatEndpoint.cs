using ModelRouter.Chat;
using ModelRouter.Relay;

namespace ModelRouter.Endpoints;

public static class ChatEndpoint
{
    public static async Task HandleAsync(HttpContext ctx, ChatRequestProcessor processor, RelayService relay)
    {
        string path = ctx.Request.Path.Value ?? "/v1/chat/completions";
        var plan = await processor.ProcessAsync(path, ctx.Request);
        await relay.RelayAsync(ctx, "POST", path, plan.Profile, plan.WantMmproj, plan.Body, true);
    }
}
