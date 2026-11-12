using System;
using ModelRouter.Configuration;

namespace ModelRouter.Http;

public static class HeaderPolicy
{
    // Hop-by-hop and Content-* headers cannot be copied onto
    // HttpRequestMessage.Headers (Content-* are HttpContent headers).
    public static bool IsSkipped(string name)
        => RouterDefaults.HopHeaders.Contains(name)
           || name.StartsWith("Content-", StringComparison.OrdinalIgnoreCase);
}
