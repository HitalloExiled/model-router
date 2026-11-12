using System.Text;
using ModelRouter.Configuration;

namespace ModelRouter.Relay;

public static class RetryPolicy
{
    public static bool IsRetryable(int status, byte[] errBody, bool requestRetryable, int attempt)
    {
        if (!requestRetryable || attempt >= RouterDefaults.MaxRetries) return false;
        if (status is 429 or 503) return true;
        return HasMarker(errBody);
    }

    public static TimeSpan Wait(int attempt) => TimeSpan.FromSeconds(5 * (attempt + 1));

    public static bool HasMarker(byte[] errBody)
    {
        if (errBody.Length == 0) return false;
        string s = Encoding.UTF8.GetString(errBody).ToLowerInvariant();
        foreach (var m in RouterDefaults.RetryableMarkers)
            if (s.Contains(m)) return true;
        return false;
    }
}
