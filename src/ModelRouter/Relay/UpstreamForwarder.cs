using ModelRouter.Configuration;
using ModelRouter.Http;

namespace ModelRouter.Relay;

public sealed class UpstreamForwarder
{
    private readonly HttpClient _http;
    public UpstreamForwarder(HttpClient http) => _http = http;

    public async Task ForwardAsync(HttpContext ctx, string method, string path, byte[]? body, bool retryable)
    {
        for (int attempt = 0; attempt <= RouterDefaults.MaxRetries; attempt++)
        {
            var req = new HttpRequestMessage(new HttpMethod(method), path);
            foreach (var header in ctx.Request.Headers)
            {
                if (HeaderPolicy.IsSkipped(header.Key)) continue;
                try
                {
                    foreach (var v in header.Value) req.Headers.Add(header.Key, v);
                }
                catch (Exception ex)
                {
                    // A single rejected header must not fail the whole turn.
                    Console.WriteLine($"[relay] dropped header '{header.Key}': {ex.Message}");
                }
            }
            if (body is { Length: > 0 })
            {
                var content = new ByteArrayContent(body);
                content.Headers.Add("Content-Type", ctx.Request.ContentType ?? "application/json");
                req.Content = content;
            }

            HttpResponseMessage? resp;
            try
            {
                resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[relay] forward failed: {ex}");
                ctx.Response.StatusCode = 502;
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync("{\"error\": \"cannot connect to backend\"}");
                return;
            }

            if ((int)resp.StatusCode == 200)
            {
                await ResponseStreamer.CopyAsync(ctx, resp);
                return;
            }

            byte[] errBody = await ResponseStreamer.ReadUpToAsync(resp.Content, 65536);
            if (RetryPolicy.IsRetryable((int)resp.StatusCode, errBody, retryable, attempt))
            {
                await Task.Delay(RetryPolicy.Wait(attempt));
                continue;
            }

            string ctype = resp.Headers.TryGetValues("Content-Type", out var ct) ? ct.First() : "application/octet-stream";
            ctx.Response.StatusCode = (int)resp.StatusCode;
            ctx.Response.ContentType = ctype;
            ctx.Response.Headers["Content-Length"] = errBody.Length.ToString();
            await ctx.Response.Body.WriteAsync(errBody, 0, errBody.Length);
            return;
        }

        ctx.Response.StatusCode = 502;
        ctx.Response.ContentType = "application/json";
        await ctx.Response.WriteAsync("{\"error\": \"backend error\"}");
    }
}
