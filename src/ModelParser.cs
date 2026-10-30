using System;

namespace ModelRouter;

public sealed class ModelParser
{
    private static readonly string[] EffortSuffixes =
    [
        "-xhigh", "-high", "-medium", "-low", "-none", "-minimal",
    ];

    private static readonly string[] BonsaiTiers =
    [
        "trio256", "trio200", "duo256", "hexa128", "penta128",
        "octo64", "quad128", "duo128", "vision", "max",
    ];

    public record Result(string Profile, string? Effort);

    public static Result Parse(string model)
    {
        string rest = model ?? "";
        string? effort = null;
        foreach (var suffix in EffortSuffixes)
        {
            if (rest.EndsWith(suffix))
            {
                effort = suffix[1..];
                rest = rest[..^suffix.Length];
                break;
            }
        }

        if (rest.Contains("bonsai"))
        {
            foreach (var tier in BonsaiTiers)
            {
                if (rest.Contains(tier))
                    return new Result($"bonsai-{tier}", effort);
            }
            return new Result("bonsai-max", effort);
        }
        if (rest.Contains("gguf-vision"))
            return new Result("gguf-vision", effort);
        if (rest.Contains("vision") || rest.Contains("35b"))
            return new Result("vision", effort);
        if (rest.Contains("gguf-speed"))
            return new Result("gguf-speed", effort);
        if (rest.Contains("gguf-duo128"))
            return new Result("gguf-duo128", effort);
        if (rest.Contains("gguf-duo"))
            return new Result("gguf-duo", effort);
        if (rest.Contains("gguf-quad"))
            return new Result("gguf-quad", effort);
        if (rest.Contains("gguf-single192"))
            return new Result("gguf-single192", effort);
        if (rest.Contains("gguf-max"))
            return new Result("gguf-max", effort);
        if (rest.Contains("gguf-range") || rest.Contains("gguf"))
            return new Result("gguf-range", effort);
        if (rest.StartsWith("qwen3.8-27b-max") || rest.Contains("192k"))
            return new Result("max", effort);
        if (rest.Contains("256k"))
            return new Result("ultra", effort);
        if (rest.Contains("64k"))
            return new Result("c1", effort);
        return new Result("balanced", effort);
    }
}
