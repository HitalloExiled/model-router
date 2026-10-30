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
    // Per-request reuse decision: prefers the running backend (prefix cache)
    // unless family differs, media needs vision, window overflows, or slots
    // are insufficient. Mirrors python effective_profile.
    public static string EffectiveProfile(string wanted, string? current, double estTokens, bool needVision)
    {
        if (current is not null && Profiles.Windows.TryGetValue(current, out var window))
        {
            if (needVision && !Profiles.VisionCapable.Contains(current))
                return wanted;
            if (Profiles.Family(current) != Profiles.Family(wanted))
                return wanted;
            int curSlots = Profiles.Slots.TryGetValue(current, out var cs) ? cs : 1;
            int wantSlots = Profiles.Slots.TryGetValue(wanted, out var ws) ? ws : 1;
            bool enoughSlots = !Profiles.RespectSlots || curSlots >= wantSlots;
            if (enoughSlots && estTokens < 0.9 * window)
                return current;
        }
        return wanted;
    }

    // Encoder (mmproj) state is sticky: resident encoder also serves text.
    // Mirrors python same_backend.
    public static bool SameBackend(string? curName, bool curMmproj, string profile, bool wantMmproj)
        => curName == profile && (curMmproj || !wantMmproj);

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
