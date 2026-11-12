
namespace ModelRouter.Http;

public static class ResponseStreamer
{
    public static async Task CopyAsync(HttpContext ctx, HttpResponseMessage resp)
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

    public static async Task<byte[]> ReadUpToAsync(HttpContent? content, int max)
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
}
