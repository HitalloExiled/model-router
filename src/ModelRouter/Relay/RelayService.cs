using System.Text;
using System.Text.Json.Nodes;

namespace ModelRouter.Relay;

public sealed class RelayService
{
    private readonly BackendManager _backend;
    private readonly UpstreamForwarder _forwarder;

    public RelayService(BackendManager backend, UpstreamForwarder forwarder)
    {
        _backend = backend;
        _forwarder = forwarder;
    }

    public async Task RelayAsync(HttpContext ctx, string method, string path,
        string? profile, bool wantMmproj, byte[]? body, bool retryable)
    {
        if (!await _backend.BeginAsync(profile, wantMmproj))
        {
            string pj = profile is null ? "null" : $"\"{profile}\"";
            ctx.Response.StatusCode = 502;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync($"{{\"error\": \"backend profile {pj} failed to start\"}}");
            return;
        }
        try
        {
            // Gate only chat/messages POST: an arrival may be held while another
            // slot prefills. Parsing is best effort; a bad body must not fail it.
            if (retryable && body is { Length: > 0 })
            {
                try
                {
                    if (JsonNode.Parse(Encoding.UTF8.GetString(body)) is JsonObject decoded)
                        await _backend.AdmitPrefillAsync(decoded, body, profile);
                }
                catch { }
            }
            await _forwarder.ForwardAsync(ctx, method, path, body, retryable);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[relay] {ex}");
            try
            {
                ctx.Response.StatusCode = 502;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync("{\"error\": \"backend error\"}");
            }
            catch { }
        }
        finally
        {
            await _backend.EndAsync();
        }
    }
}
