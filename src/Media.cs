using System.Text.Json.Nodes;

namespace ModelRouter;

public static class Media
{
    /// <summary>True when any message carries image/video content parts.
    /// Mirrors Python has_media.</summary>
    public static bool Has(JsonNode? root)
    {
        if (root is not JsonObject rootObj)
            return false;
        if (rootObj["messages"] is not JsonArray msgArr)
            return false;
        foreach (var msg in msgArr)
        {
            if (msg is not JsonObject m)
                continue;
            if (m["content"] is not JsonArray arr)
                continue;
            foreach (var part in arr)
            {
                if (part is not JsonObject p)
                    continue;
                if (p["type"]?.GetValue<string>() is "image_url" or "input_image"
                    or "image" or "video_url" or "input_video")
                {
                    return true;
                }
                if (p["source"] is JsonObject src
                    && (src["media_type"]?.GetValue<string>() ?? "").StartsWith("image/"))
                {
                    return true;
                }
            }
        }
        return false;
    }
}
