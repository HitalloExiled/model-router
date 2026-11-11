using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using ModelRouter;

// Idle: advertise both variants so pickers work cold (python STATIC_MODELS).
const string StaticModels = "{\"object\": \"list\", \"data\": [{\"id\": \"qwen3.8-gguf-duo128\", \"object\": \"model\", \"owned_by\": \"llama.cpp\", \"name\": \"qwen3.8-gguf-duo128\", \"context_window\": 131072, \"max_model_len\": 131072, \"modalities\": {\"vision\": false}}, {\"id\": \"qwen3.8-gguf-quad\", \"object\": \"model\", \"owned_by\": \"llama.cpp\", \"name\": \"qwen3.8-gguf-quad\", \"context_window\": 65536, \"max_model_len\": 65536, \"modalities\": {\"vision\": false}}, {\"id\": \"qwen3.8-gguf-single200\", \"object\": \"model\", \"owned_by\": \"llama.cpp\", \"name\": \"qwen3.8-gguf-single200\", \"context_window\": 204800, \"max_model_len\": 204800, \"modalities\": {\"vision\": false}}, {\"id\": \"qwen3.8-gguf-max\", \"object\": \"model\", \"owned_by\": \"llama.cpp\", \"name\": \"qwen3.8-gguf-max\", \"context_window\": 262144, \"max_model_len\": 262144, \"modalities\": {\"vision\": false}}, {\"id\": \"bonsai-2-27b-max\", \"object\": \"model\", \"owned_by\": \"prism\", \"name\": \"bonsai-2-27b-max\", \"context_window\": 262144, \"max_model_len\": 262144, \"modalities\": {\"vision\": false}}, {\"id\": \"bonsai-2-27b-duo256\", \"object\": \"model\", \"owned_by\": \"prism\", \"name\": \"bonsai-2-27b-duo256\", \"context_window\": 262144, \"max_model_len\": 262144, \"modalities\": {\"vision\": false}}, {\"id\": \"bonsai-2-27b-trio256\", \"object\": \"model\", \"owned_by\": \"prism\", \"name\": \"bonsai-2-27b-trio256\", \"context_window\": 262144, \"max_model_len\": 262144, \"modalities\": {\"vision\": false}}, {\"id\": \"bonsai-2-27b-trio200\", \"object\": \"model\", \"owned_by\": \"prism\", \"name\": \"bonsai-2-27b-trio200\", \"context_window\": 204800, \"max_model_len\": 204800, \"modalities\": {\"vision\": false}}, {\"id\": \"bonsai-2-27b-duo128\", \"object\": \"model\", \"owned_by\": \"prism\", \"name\": \"bonsai-2-27b-duo128\", \"context_window\": 131072, \"max_model_len\": 131072, \"modalities\": {\"vision\": false}}, {\"id\": \"bonsai-2-27b-quad128\", \"object\": \"model\", \"owned_by\": \"prism\", \"name\": \"bonsai-2-27b-quad128\", \"context_window\": 131072, \"max_model_len\": 131072, \"modalities\": {\"vision\": false}}, {\"id\": \"bonsai-2-27b-penta128\", \"object\": \"model\", \"owned_by\": \"prism\", \"name\": \"bonsai-2-27b-penta128\", \"context_window\": 131072, \"max_model_len\": 131072, \"modalities\": {\"vision\": false}}, {\"id\": \"bonsai-2-27b-hexa128\", \"object\": \"model\", \"owned_by\": \"prism\", \"name\": \"bonsai-2-27b-hexa128\", \"context_window\": 131072, \"max_model_len\": 131072, \"modalities\": {\"vision\": false}}, {\"id\": \"bonsai-2-27b-octo64\", \"object\": \"model\", \"owned_by\": \"prism\", \"name\": \"bonsai-2-27b-octo64\", \"context_window\": 65536, \"max_model_len\": 65536, \"modalities\": {\"vision\": false}}, {\"id\": \"bonsai-2-27b-vision\", \"object\": \"model\", \"owned_by\": \"prism\", \"name\": \"bonsai-2-27b-vision\", \"context_window\": 65536, \"max_model_len\": 65536, \"modalities\": {\"vision\": true}}]}";

var builder = WebApplication.CreateBuilder();
var backend = new BackendManager();
builder.Services.AddSingleton(backend);
var app = builder.Build();

// Long timeout for the forward leg (python http.client timeout=3600).
var fwd = new HttpClient { Timeout = TimeSpan.FromHours(1) };
fwd.BaseAddress = new Uri($"http://127.0.0.1:{BackendManager.BackendPort}");

// Adopt an already-running backend instead of rebooting (mirror __main__).
await backend.AdoptAsync();

// Idle watch (daemon thread).
var idleCts = new CancellationTokenSource();
_ = backend.IdleWatchAsync(idleCts.Token);

// One pipeline, mirrors do_GET / do_POST / _relay.
app.Use(async (HttpContext ctx, RequestDelegate next) =>
{
    var path = ctx.Request.Path.Value ?? "/";
    var method = ctx.Request.Method;

    // The slot API can save/restore files on the backend and is never part of
    // the public surface: the router talks to it directly instead.
    if (path.StartsWith("/slots"))
    {
        ctx.Response.StatusCode = 403;
        ctx.Response.ContentType = "application/json";
        await ctx.Response.WriteAsync("{\"error\": \"slot API is not proxied\"}");
        return;
    }

    if (method == "GET" && path == "/health")
    {
        string p = backend.Profile is null ? "null" : $"\"{backend.Profile}\"";
        var json = $"{{\"status\": \"router-ok\", \"backend\": {p}, \"busy\": {backend.Busy.ToString().ToLowerInvariant()}}}";
        ctx.Response.ContentType = "application/json";
        await ctx.Response.WriteAsync(json);
        return;
    }

    if (method == "GET" && path == "/v1/models")
    {
        bool up = backend.Profile is not null
            && ((await backend.GetBackendContainerAsync()) is not null || await backend.IsBackendHealthyAsync());
        if (up)
        {
            await RelayAsync(ctx, "GET", path, null, false, null, false, backend, fwd);
            return;
        }
        ctx.Response.ContentType = "application/json";
        await ctx.Response.WriteAsync(StaticModels);
        return;
    }

    if (method == "POST" && (path == "/v1/chat/completions" || path == "/v1/messages"))
    {
        await HandleChatAsync(ctx, path, backend, fwd);
        return;
    }

    if (method == "GET" || method == "POST")
    {
        var body = method == "POST" ? await ReadBodyAsync(ctx.Request) : null;
        await RelayAsync(ctx, method, path, null, false, body, false, backend, fwd);
        return;
    }

    ctx.Response.StatusCode = 501; // unimplemented method
});

app.Run();

static async Task<byte[]> ReadBodyAsync(HttpRequest request)
{
    if (request.ContentLength is not > 0) return Array.Empty<byte>();
    var ms = new MemoryStream();
    var buf = new byte[8192];
    int n;
    while ((n = await request.Body.ReadAsync(buf, 0, buf.Length)) > 0)
        ms.Write(buf, 0, n);
    return ms.ToArray();
}

static async Task HandleChatAsync(HttpContext ctx, string path, BackendManager bm, HttpClient fwd)
{
    byte[] body;
    try { body = await ReadBodyAsync(ctx.Request); }
    catch { body = Array.Empty<byte>(); }

    string? profile = null;
    bool wantMmproj = false;
    byte[] forwardBody = body;

    try
    {
        JsonNode? decoded = JsonNode.Parse(body.Length > 0 ? Encoding.UTF8.GetString(body) : "{}");
        var route = Routing.For(decoded);
        bool needVision = Media.Has(decoded);
        double est = body.Length > 0 ? (double)body.Length / 3.5 : 0.0;
        string requested = route.Profile;
        string? current = bm.Profile;
        string reused = Routing.EffectiveProfile(requested, current, est, needVision);
        profile = reused;
        var norm = Effort.Normalize(path, body, route, reused);
        forwardBody = norm.NewBody;
        wantMmproj = route.WantMmproj || norm.WantMmproj;
        if (needVision && Profiles.LlamaAll.Contains(profile))
            wantMmproj = true;
    }
    catch
    {
        profile = null;
        wantMmproj = false;
        forwardBody = body;
    }

    await RelayAsync(ctx, "POST", path, profile, wantMmproj, forwardBody, true, bm, fwd);
}

static async Task RelayAsync(HttpContext ctx, string method, string path,
    string? profile, bool wantMmproj, byte[]? body, bool retryable,
    BackendManager bm, HttpClient fwd)
{
    if (!await bm.BeginAsync(profile, wantMmproj))
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
                    await bm.AdmitPrefillAsync(decoded, body, profile);
            }
            catch { }
        }
        await ForwardAsync(ctx, method, path, body, retryable, fwd);
    }
    catch (Exception)
    {
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
        await bm.EndAsync();
    }
}

static async Task ForwardAsync(HttpContext ctx, string method, string path,
    byte[]? body, bool retryable, HttpClient fwd)
{
    const int maxRetries = 3;
    var hop = new HashSet<string> { "connection", "keep-alive", "transfer-encoding", "upgrade" };

    for (int attempt = 0; attempt <= maxRetries; attempt++)
    {
        var req = new HttpRequestMessage(new HttpMethod(method), path);
        foreach (var header in ctx.Request.Headers)
        {
            if (hop.Contains(header.Key.ToLowerInvariant())) continue;
            foreach (var v in header.Value) req.Headers.Add(header.Key, v);
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
            resp = await fwd.SendAsync(req, HttpCompletionOption.ResponseHeadersRead);
        }
        catch
        {
            ctx.Response.StatusCode = 502;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync("{\"error\": \"cannot connect to backend\"}");
            return;
        }

        if ((int)resp.StatusCode == 200)
        {
            await StreamToClientAsync(ctx, resp);
            return;
        }

        byte[] errBody = await ReadUpToAsync(resp.Content, 65536);
        bool statusRetry = (int)resp.StatusCode == 429 || (int)resp.StatusCode == 503;
        bool markerHit = HasRetryableMarker(errBody);
        if (retryable && attempt < maxRetries && (statusRetry || markerHit))
        {
            int waitMs = 5000 * (attempt + 1);
            await Task.Delay(waitMs);
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

static async Task StreamToClientAsync(HttpContext ctx, HttpResponseMessage resp)
{
    var contentType = resp.Content?.Headers.ContentType?.MediaType ?? "application/octet-stream";
    ctx.Response.ContentType = contentType;
    bool isSse = contentType == "text/event-stream";
    if (!isSse && resp.Content is not null
        && resp.Content.Headers.TryGetValues("Content-Length", out var len))
        ctx.Response.Headers["Content-Length"] = len.First();

    var stream = await resp.Content!.ReadAsStreamAsync();
    if (isSse)
    {
        var reader = new StreamReader(stream);
        string? line;
        while ((line = await reader.ReadLineAsync()) is not null)
        {
            try { await ctx.Response.WriteAsync(line + "\n"); await ctx.Response.Body.FlushAsync(); }
            catch { return; }
        }
    }
    else
    {
        var buf = new byte[65536];
        int n;
        while ((n = await stream.ReadAsync(buf, 0, buf.Length)) > 0)
        {
            try { await ctx.Response.Body.WriteAsync(buf, 0, n); await ctx.Response.Body.FlushAsync(); }
            catch { return; }
        }
    }
}

static async Task<byte[]> ReadUpToAsync(HttpContent? content, int max)
{
    if (content is null) return Array.Empty<byte>();
    using var ms = new MemoryStream();
    var buf = new byte[Math.Min(8192, max)];
    int total = 0;
    int n;
    using var stream = await content.ReadAsStreamAsync();
    while (total < max && (n = await stream.ReadAsync(buf, 0, buf.Length)) > 0)
    {
        int toCopy = Math.Min(n, max - total);
        ms.Write(buf, 0, toCopy);
        total += toCopy;
    }
    return ms.ToArray();
}

static bool HasRetryableMarker(byte[] errBody)
{
    if (errBody.Length == 0) return false;
    string[] markers = { "expired", "admission", "overloaded", "busy", "try again", "queue", "unavailable" };
    string s = Encoding.UTF8.GetString(errBody).ToLowerInvariant();
    foreach (var m in markers)
        if (s.Contains(m))
            return true;
    return false;
}
