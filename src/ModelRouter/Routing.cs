using System.Text.Json.Nodes;

namespace ModelRouter;

public static class Routing
{
    public sealed record RouteRequest(string Profile, string? Effort, bool WantMmproj, List<string> Notes);

    // Prefer the resident backend unless the requested tier is much faster.
    // A switch costs a boot and, across weights groups, a full re-prefill;
    // inside a group the slot dump restores the KV. Mirrors python
    // effective_profile (speed-aware tier reuse).
    public static string EffectiveProfile(string wanted, string? current, double estTokens, bool needVision)
    {
        if (current is not null && Profiles.Windows.TryGetValue(current, out var window))
        {
            // Every GGUF/Bonsai backend carries mmproj: media never forces a
            // switch off them.
            if (needVision && !Profiles.VisionCapable.Contains(current))
                return wanted;
            if (Profiles.Family(current) != Profiles.Family(wanted))
                return wanted;
            int curSlots = Profiles.Slots.TryGetValue(current, out var cs) ? cs : 1;
            int wantSlots = Profiles.Slots.TryGetValue(wanted, out var ws) ? ws : 1;
            bool enoughSlots = !Profiles.RespectSlots || curSlots >= wantSlots;
            if (enoughSlots && estTokens < 0.9 * window)
            {
                // The requested tier can only justify a switch if it also fits.
                if (Profiles.Windows.TryGetValue(wanted, out var wantWindow)
                    && estTokens < 0.9 * wantWindow)
                {
                    int curSpeed = Profiles.Speed.TryGetValue(current, out var csp) ? csp : 0;
                    int wantSpeed = Profiles.Speed.TryGetValue(wanted, out var wsp) ? wsp : 0;
                    bool sameGroup = Profiles.Group.TryGetValue(current, out var cg)
                        && Profiles.Group.TryGetValue(wanted, out var wg) && cg == wg;
                    double ratio = sameGroup
                        ? Profiles.SPEED_SWITCH_RATIO
                        : Profiles.SPEED_SWITCH_CROSS;
                    if (wantSpeed > curSpeed * ratio)
                        return wanted;
                }
                return current;
            }
        }
        return wanted;
    }

    // Encoder (mmproj) state is sticky: resident encoder also serves text.
    // Mirrors python same_backend.
    public static bool SameBackend(string? curName, bool curMmproj, string profile, bool wantMmproj)
        => curName == profile && (curMmproj || !wantMmproj);

    // Returns (profile, suffix_effort, want_mmproj, notes). Media on any
    // GGUF/Bonsai tier wants mmproj on that SAME tier (lazy upgrade), never a
    // tier switch: every backend here can carry the encoder.
    public static RouteRequest For(JsonNode? root)
    {
        string model = root is JsonObject o && o["model"] is JsonValue v
            && v.TryGetValue<string>(out var s) ? s : "";
        var mp = ModelParser.Parse(model);
        var notes = new List<string>();
        bool wantMmproj = mp.Profile is "gguf-vision" or "bonsai-vision";
        if (Media.Has(root) && Profiles.LlamaAll.Contains(mp.Profile))
        {
            wantMmproj = true;
            notes.Add("media wants mmproj");
        }
        return new RouteRequest(mp.Profile, mp.Effort, wantMmproj, notes);
    }
}
