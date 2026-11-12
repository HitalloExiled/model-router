using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;

namespace ModelRouter.Http;

public static class RequestBodyReader
{
    public static async Task<byte[]> ReadAsync(HttpRequest request)
    {
        if (request.ContentLength is not > 0) return Array.Empty<byte>();
        var ms = new MemoryStream();
        var buf = new byte[8192];
        int n;
        while ((n = await request.Body.ReadAsync(buf, 0, buf.Length)) > 0)
            ms.Write(buf, 0, n);
        return ms.ToArray();
    }
}
