using System;
using System.Collections.Generic;
using System.Text.Json.Nodes;

namespace ModelRouter;

public static class Routing
{
    public sealed record RouteRequest(string Profile, string? Effort, bool WantMmproj, List<string> Notes);

    /// <summary>Mirrors Python route_request(decoded).
    /// Media on a GGUF/Bonsai tier wants mmproj on that SAME tier (lazy upgrade);
    /// media on a ninfer text tier forces 35B vision.</summary>
    public static RouteRequest For(JsonNode? root)
    {
        string model = root?["model"]?.GetValue<string>() ?? "";
        var mp = ModelParser.Parse(model);
        var notes = new List<string>();
        bool wantMmproj = mp.Profile is "gguf-vision" or "bonsai-vision";
        if (Media.Has(root))
        {
            if (Profiles.LlamaAll.Contains(mp.Profile))
            {
                // Same tier, lazy encoder: reboot it with mmproj, tier stays.
                wantMmproj = true;
                notes.Add("media wants mmproj");
            }
            else if (!Profiles.VisionCapable.Contains(mp.Profile))
            {
                // Image/video needs a vision-capable backend; ninfer text -> 35B vision.
                mp = new ModelParser.Result("vision", mp.Effort);
                notes.Add("media forces vision");
            }
        }
        return new RouteRequest(mp.Profile, mp.Effort, wantMmproj, notes);
    }
}
