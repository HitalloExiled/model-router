using System;
using System.Collections.Generic;
using System.Linq;

namespace ModelRouter;

// Pure profile-inference from a running backend's argv (llama --alias /
// ninfer --max-context / shape heuristics). Mirrors python infer_profile.
public static class Infer
{
    // ninfer --max-context <n> -> tier.
    private static readonly Dictionary<string, string> ContextWindows = new()
    {
        ["65536"] = "c1",
        ["131072"] = "balanced",
        ["196608"] = "max",
        ["262144"] = "ultra",
        ["32768"] = "vision",
    };

    public static string? InferProfile(string? args)
    {
        if (string.IsNullOrWhiteSpace(args))
            return null;
        string[] toks = args.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        // ninfer-serve tier: --max-context <n> (space or '=' form).
        string? win = null;
        for (int i = 0; i < toks.Length; i++)
        {
            string t = toks[i];
            if (t.StartsWith("--max-context"))
            {
                if (t.Contains("="))
                {
                    int eq = t.IndexOf('=');
                    win = t[(eq + 1)..];
                }
                else if (i + 1 < toks.Length)
                {
                    win = toks[i + 1];
                }
                break;
            }
        }
        if (win is not null)
            return ContextWindows.TryGetValue(win, out string tier) ? tier : null;

        // llama.cpp: routing name rides on --alias.
        for (int i = 0; i < toks.Length; i++)
        {
            string t = toks[i];
            if (t == "--alias" && i + 1 < toks.Length)
                return ModelParser.Parse(toks[i + 1]).Profile;
            if (t.StartsWith("--alias="))
            {
                int eq = t.IndexOf('=');
                return ModelParser.Parse(t[(eq + 1)..]).Profile;
            }
        }

        // llama.cpp without an alias: fall back to shape heuristics.
        if (args.Contains("Ternary-Bonsai"))
        {
            if (args.Contains("--parallel 8"))
                return "bonsai-octo64";
            if (args.Contains("--parallel 4"))
                return "bonsai-quad128";
            if (args.Contains("--parallel 6"))
                return "bonsai-hexa128";
            if (args.Contains("--parallel 5"))
                return "bonsai-penta128";
            if (args.Contains("--parallel 3"))
                return (args.Contains("-c 614400") || args.Contains("--ctx-size 614400"))
                    ? "bonsai-trio200" : "bonsai-trio256";
            if (args.Contains("--parallel 2"))
                return (args.Contains("-c 524288") || args.Contains("--ctx-size 524288"))
                    ? "bonsai-duo256" : "bonsai-duo128";
            if (args.Contains("mmproj"))
                return "bonsai-vision";
            return "bonsai-max";
        }

        if (args.Contains("--parallel 8"))
            return args.Contains("Ternary-Bonsai") ? "bonsai-octo64" : "gguf-quad";
        if (args.Contains("--parallel 4"))
            return "gguf-quad";
        if (args.Contains("--parallel 2"))
            return args.Contains("262144") ? "gguf-duo128" : "gguf-duo";
        if (args.Contains("-c 262144") || args.Contains("--ctx-size 262144"))
            return "gguf-max";
        if (args.Contains("mtp.gguf"))
            return "gguf-speed";
        if (args.Contains("mmproj"))
            return "gguf-vision";
        if (args.Contains("-m "))
            return "gguf-range";
        return null;
    }
}