using System.Collections.Generic;
using System.Collections.Immutable;

namespace ModelRouter;

public static class Profiles
{
    // NInfer (não llama.cpp) tiers.
    // Reuse requires the resident backend to have >= requested slots.
    // MODEL_ROUTER_RESPECT_SLOTS=0 restores blind tier reuse.
    public static bool RespectSlots =>
        Environment.GetEnvironmentVariable("MODEL_ROUTER_RESPECT_SLOTS") != "0";

    public static readonly IReadOnlyCollection<string> NInferAll =
        ["balanced", "max", "c1", "ultra", "vision"];

    // GGUF (llama.cpp standard) tiers.
    public static readonly IReadOnlyCollection<string> GgufAll =
        ["gguf-speed", "gguf-range", "gguf-max", "gguf-duo", "gguf-duo128",
         "gguf-quad", "gguf-vision", "gguf-single192"];

    // Bonsai (ternary) tiers.
    public static readonly IReadOnlyCollection<string> BonsaiAll =
        ["bonsai-max", "bonsai-duo256", "bonsai-trio256", "bonsai-trio200",
         "bonsai-duo128", "bonsai-quad128", "bonsai-penta128", "bonsai-hexa128",
         "bonsai-octo64", "bonsai-vision"];

    // Todo backend llama.cpp-family (GGUF + Bonsai): budget se aplica.
    public static readonly IReadOnlyCollection<string> LlamaAll;

    // Todos os profiles que carregam mmproj (GGUF + Bonsai + vision).
    public static readonly IReadOnlyCollection<string> VisionCapable;

    // Per-slot context window (a client sees one slot, not the whole pool).
    public static readonly IReadOnlyDictionary<string, int> Windows = new Dictionary<string, int>
    {
        ["c1"] = 65536, ["balanced"] = 131072, ["max"] = 196608, ["ultra"] = 262144, ["vision"] = 16384,
        ["gguf-speed"] = 65536, ["gguf-range"] = 98304, ["gguf-max"] = 262144, ["gguf-duo"] = 98304,
        ["gguf-duo128"] = 131072, ["gguf-quad"] = 65536, ["gguf-vision"] = 65536, ["gguf-single192"] = 196608,
        ["bonsai-max"] = 262144, ["bonsai-duo256"] = 262144, ["bonsai-trio256"] = 262144,
        ["bonsai-trio200"] = 204800, ["bonsai-duo128"] = 131072, ["bonsai-quad128"] = 131072,
        ["bonsai-penta128"] = 131072, ["bonsai-hexa128"] = 131072, ["bonsai-octo64"] = 65536,
        ["bonsai-vision"] = 65536,
    };

    // Parallel slots each profile boots with.
    public static readonly IReadOnlyDictionary<string, int> Slots = new Dictionary<string, int>
    {
        ["c1"] = 1, ["balanced"] = 1, ["max"] = 1, ["ultra"] = 1, ["vision"] = 1,
        ["gguf-speed"] = 1, ["gguf-range"] = 1, ["gguf-max"] = 1, ["gguf-duo"] = 2,
        ["gguf-duo128"] = 2, ["gguf-quad"] = 4, ["gguf-vision"] = 1, ["gguf-single192"] = 1,
        ["bonsai-max"] = 1, ["bonsai-duo256"] = 2, ["bonsai-trio256"] = 3,
        ["bonsai-trio200"] = 3, ["bonsai-duo128"] = 2, ["bonsai-quad128"] = 4,
        ["bonsai-penta128"] = 5, ["bonsai-hexa128"] = 6, ["bonsai-octo64"] = 8, ["bonsai-vision"] = 1,
    };

    // family == python family(): returns "bonsai" | "gguf" | "ninfer" | null.
    public static string? Family(string profile) =>
        BonsaiAll.Contains(profile) ? "bonsai" :
        GgufAll.Contains(profile)   ? "gguf"  :
        NInferAll.Contains(profile) ? "ninfer"
        : null;

    // Full set of known profiles (ninfer + gguf + bonsai), mirrors python PROFILES.
    public static readonly IReadOnlyCollection<string> ProxiesSet =
        new HashSet<string>(NInferAll).Union(GgufAll).Union(BonsaiAll).ToImmutableArray();

    static Profiles()
    {
        var gguf = GgufAll;
        var bonsai = BonsaiAll;
        LlamaAll = new HashSet<string>(gguf).Union(bonsai).ToImmutableArray();
        VisionCapable = new HashSet<string>(gguf)
            .Union(bonsai)
            .Union(new HashSet<string> { "vision" })
            .ToImmutableArray();
    }
}
