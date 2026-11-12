using System;

namespace ModelRouter;

public sealed class ModelParser
{
    // Order matters: "-xhigh" must be tested before "-high".
    private static readonly string[] effortSuffixes =
    [
        "-xhigh", "-high", "-medium", "-low", "-none", "-minimal",
    ];

    private static readonly string[] bonsaiTiers =
    [
        "trio256", "trio200", "duo256", "hexa128", "penta128",
        "octo64", "quad128", "duo128", "vision", "max",
    ];

    public record Result(string Profile, string? Effort);

    public static Result Parse(string model)
    {
        string rest = model ?? "";
        string? effort = null;
        foreach (var suffix in effortSuffixes)
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
            // Match the tier word, not the full id: the catalog uses
            // bonsai-2-27b-<tier>.
            foreach (var tier in bonsaiTiers)
            {
                if (rest.Contains(tier))
                    return new Result($"bonsai-{tier}", effort);
            }
            return new Result("bonsai-max", effort);
        }
        if (rest.Contains("gguf-vision"))
            return new Result("gguf-vision", effort);
        if (rest.Contains("gguf-speed"))
            return new Result("gguf-speed", effort);
        if (rest.Contains("gguf-duo128"))
            return new Result("gguf-duo128", effort);
        if (rest.Contains("gguf-duo"))
            return new Result("gguf-duo", effort);
        if (rest.Contains("gguf-quad"))
            return new Result("gguf-quad", effort);
        if (rest.Contains("gguf-single200"))
            return new Result("gguf-single200", effort);
        if (rest.Contains("gguf-max"))
            return new Result("gguf-max", effort);
        if (rest.Contains("gguf-range") || rest.Contains("gguf"))
            return new Result("gguf-range", effort);
        // Unknown names land on the general-purpose Qwen tier: the server
        // accepts arbitrary model strings, so routing them anywhere else would
        // silently pick the wrong weights.
        return new Result("gguf-range", effort);
    }
}
