using System;
using System.Collections.Generic;

namespace ModelRouter.Configuration;

public static class RouterDefaults
{
    public static readonly TimeSpan ForwardTimeout = TimeSpan.FromHours(1);
    public const int MaxRetries = 3;

    public static readonly HashSet<string> HopHeaders =
        new(StringComparer.OrdinalIgnoreCase)
        {
            "connection", "keep-alive", "transfer-encoding", "upgrade",
        };

    public static readonly string[] RetryableMarkers =
        { "expired", "admission", "overloaded", "busy", "try again", "queue", "unavailable" };
}
