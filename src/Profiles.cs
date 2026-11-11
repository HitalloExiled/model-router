using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace ModelRouter;

public static class Profiles
{
    // Weights file per tier. GGUF and Bonsai are the only families: NInfer was
    // dropped from main, so an unknown model falls back to the general-purpose
    // Qwen tier (gguf-range).
    public static readonly IReadOnlyDictionary<string, string> GgufFiles =
        new Dictionary<string, string>
        {
            ["gguf-speed"] = "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
            ["gguf-range"] = "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",
            ["gguf-max"] = "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",
            ["gguf-duo"] = "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",
            ["gguf-duo128"] = "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",
            ["gguf-quad"] = "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",
            ["gguf-vision"] = "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf",
            ["gguf-single200"] = "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
        };

    public static readonly IReadOnlyDictionary<string, string> BonsaiFiles =
        new Dictionary<string, string>
        {
            ["bonsai-max"] = "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
            ["bonsai-duo256"] = "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
            ["bonsai-trio256"] = "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
            ["bonsai-trio200"] = "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
            ["bonsai-duo128"] = "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
            ["bonsai-quad128"] = "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
            ["bonsai-penta128"] = "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
            ["bonsai-hexa128"] = "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
            ["bonsai-octo64"] = "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
            ["bonsai-vision"] = "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        };

    public static readonly IReadOnlyCollection<string> GgufAll =
        ["gguf-speed", "gguf-range", "gguf-max", "gguf-duo", "gguf-duo128",
         "gguf-quad", "gguf-vision", "gguf-single200"];

    public static readonly IReadOnlyCollection<string> BonsaiAll =
        ["bonsai-max", "bonsai-duo256", "bonsai-trio256", "bonsai-trio200",
         "bonsai-duo128", "bonsai-quad128", "bonsai-penta128", "bonsai-hexa128",
         "bonsai-octo64", "bonsai-vision"];

    // Everything served by a llama.cpp-family image: budgets apply here.
    public static readonly IReadOnlyCollection<string> LlamaAll;

    // Every GGUF/Bonsai backend can carry mmproj (lazy load), so media never
    // forces a tier switch.
    public static readonly IReadOnlyCollection<string> VisionCapable;

    // Full set of known profiles (GGUF + Bonsai), mirrors python PROFILES.
    public static readonly IReadOnlyCollection<string> ProfilesSet;

    public static bool RespectSlots =>
        Environment.GetEnvironmentVariable("MODEL_ROUTER_RESPECT_SLOTS") != "0";

    // Per-slot context window (a client sees one slot, not the whole pool).
    public static readonly IReadOnlyDictionary<string, int> Windows = new Dictionary<string, int>
    {
        ["gguf-speed"] = 65536, ["gguf-range"] = 98304, ["gguf-max"] = 262144,
        ["gguf-duo"] = 98304, ["gguf-duo128"] = 131072, ["gguf-quad"] = 65536,
        ["gguf-vision"] = 65536, ["gguf-single200"] = 204800,
        ["bonsai-max"] = 262144, ["bonsai-duo256"] = 262144,
        ["bonsai-trio256"] = 262144, ["bonsai-trio200"] = 204800,
        ["bonsai-duo128"] = 131072, ["bonsai-quad128"] = 131072,
        ["bonsai-penta128"] = 131072, ["bonsai-hexa128"] = 131072,
        ["bonsai-octo64"] = 65536, ["bonsai-vision"] = 65536,
    };

    // Parallel slots each profile boots with.
    public static readonly IReadOnlyDictionary<string, int> Slots = new Dictionary<string, int>
    {
        ["gguf-speed"] = 1, ["gguf-range"] = 1, ["gguf-max"] = 1,
        ["gguf-duo"] = 2, ["gguf-duo128"] = 2, ["gguf-quad"] = 4,
        ["gguf-vision"] = 1, ["gguf-single200"] = 1,
        ["bonsai-max"] = 1, ["bonsai-duo256"] = 2, ["bonsai-trio256"] = 3,
        ["bonsai-trio200"] = 3, ["bonsai-duo128"] = 2, ["bonsai-quad128"] = 4,
        ["bonsai-penta128"] = 5, ["bonsai-hexa128"] = 6, ["bonsai-octo64"] = 8,
        ["bonsai-vision"] = 1,
    };

    // Measured decode throughput (tok/s). Speed dominates write/edit, so reuse
    // prefers a resident tier unless the requested one is materially faster.
    public static readonly IReadOnlyDictionary<string, int> Speed = new Dictionary<string, int>
    {
        ["gguf-speed"] = 65, ["gguf-single200"] = 65,
        ["gguf-range"] = 33, ["gguf-max"] = 33, ["gguf-duo"] = 33,
        ["gguf-duo128"] = 33, ["gguf-quad"] = 33, ["gguf-vision"] = 33,
        ["bonsai-max"] = 81, ["bonsai-duo256"] = 66, ["bonsai-trio256"] = 66,
        ["bonsai-trio200"] = 66, ["bonsai-duo128"] = 66, ["bonsai-quad128"] = 66,
        ["bonsai-penta128"] = 66, ["bonsai-hexa128"] = 66, ["bonsai-octo64"] = 66,
        ["bonsai-vision"] = 66,
    };

    // A switch costs a boot and, across weights groups, a full re-prefill;
    // inside a group the slot dump restores the KV.
    public const double SPEED_SWITCH_RATIO = 1.25;
    public const double SPEED_SWITCH_CROSS = 1.5;

    // family == python family(): returns "bonsai" | "gguf" | null.
    public static string? Family(string profile) =>
        BonsaiAll.Contains(profile) ? "bonsai" :
        GgufAll.Contains(profile) ? "gguf" : null;

    // Stable per-weights-file key: the unit a slot dump can be restored into.
    public static string WeightsKey(string profile)
    {
        string name = GgufFiles.TryGetValue(profile, out var g) ? g
            : BonsaiFiles.TryGetValue(profile, out var b) ? b
            : profile;
        byte[] hash = SHA1.HashData(Encoding.UTF8.GetBytes(name));
        return $"{Family(profile) ?? "x"}-{Convert.ToHexString(hash).ToLowerInvariant()[..10]}";
    }

    // Weights group per tier: tiers that share a file share slot dumps.
    public static readonly IReadOnlyDictionary<string, string> Group;

    static Profiles()
    {
        Group = ProfilesSetFor(ProfilesSetSource());
        LlamaAll = new HashSet<string>(GgufAll).Union(BonsaiAll).ToImmutableArray();
        VisionCapable = new HashSet<string>(GgufAll).Union(BonsaiAll).ToImmutableArray();
        ProfilesSet = new HashSet<string>(GgufAll).Union(BonsaiAll).ToImmutableArray();
    }

    private static IReadOnlyCollection<string> ProfilesSetSource()
        => new HashSet<string>(GgufAll).Union(BonsaiAll).ToImmutableArray();

    private static IReadOnlyDictionary<string, string> ProfilesSetFor(IReadOnlyCollection<string> profiles)
        => profiles.ToDictionary(p => p, WeightsKey);
}
