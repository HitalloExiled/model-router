using System.Collections.Generic;
using System.Collections.Immutable;

namespace ModelRouter;

public static class Profiles
{
    // NInfer (não llama.cpp) tiers.
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
