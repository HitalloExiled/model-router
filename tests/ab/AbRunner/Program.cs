using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using ModelRouter;

// Reads a cases.json corpus and runs the same routing + effort pipeline the
// relay uses, emitting one JSON line per case. The Python driver compares this
// output with the reference from model-router.py.

if (args.Length < 2)
{
    Console.Error.WriteLine("usage: AbRunner <cases.json> <out.jsonl>");
    Environment.ExitCode = 2;
    return;
}

var options = new JsonSerializerOptions { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
var cases = JsonNode.Parse(File.ReadAllText(args[0]));
var outs = new List<string>();

if (cases is JsonArray arr)
{
    foreach (var node in arr)
    {
        if (node is not JsonObject c) continue;
        string label = c["label"]?.GetValue<string>() ?? "";
        string path = c["path"]?.GetValue<string>() ?? "/v1/chat/completions";
        string body = c["body"]?.GetValue<string>() ?? "{}";
        string? current = c["currentProfile"] is JsonValue cv ? cv.GetValue<string>() : null;
        string? finalProfile = c["finalProfile"] is JsonValue fv ? fv.GetValue<string>() : null;

        JsonNode? decoded = JsonNode.Parse(body);
        var route = Routing.For(decoded);
        bool needVision = Media.Has(decoded);
        double est = Encoding.UTF8.GetByteCount(body) / 3.5;
        string reused = Routing.EffectiveProfile(route.Profile, current, est, needVision);
        string target = finalProfile ?? reused;
        var norm = Effort.Normalize(path, Encoding.UTF8.GetBytes(body), route, target);

        var result = new
        {
            label,
            profile = norm.Profile,
            note = norm.Note,
            wantMmproj = norm.WantMmproj,
            body = Canon(JsonNode.Parse(Encoding.UTF8.GetString(norm.NewBody)), options),
            meta = new
            {
                effort = norm.Meta.Effort,
                origin = norm.Meta.Origin,
                template = norm.Meta.Template,
                budget = norm.Meta.Budget,
                injected = norm.Meta.InjectedBudget,
                toolLoop = norm.Meta.ToolLoop,
                style = norm.Meta.Style,
            },
        };
        outs.Add(JsonSerializer.Serialize(result, options));
    }
}

File.WriteAllLines(args[1], outs);
Console.WriteLine($"wrote {outs.Count} results");

static string Canon(JsonNode? n, JsonSerializerOptions options)
{
    if (n is null) return "null";
    if (n is JsonObject obj)
    {
        var keys = new List<string>();
        foreach (var kv in obj) keys.Add(kv.Key);
        keys.Sort(StringComparer.Ordinal);
        var sb = new StringBuilder("{");
        bool first = true;
        foreach (var k in keys)
        {
            if (!first) sb.Append(',');
            first = false;
            sb.Append(JsonValue.Create(k)!.ToJsonString(options));
            sb.Append(':');
            sb.Append(Canon(obj[k], options));
        }
        sb.Append('}');
        return sb.ToString();
    }
    if (n is JsonArray arr)
    {
        var sb = new StringBuilder("[");
        for (int i = 0; i < arr.Count; i++)
        {
            if (i > 0) sb.Append(',');
            sb.Append(Canon(arr[i], options));
        }
        sb.Append(']');
        return sb.ToString();
    }
    return n.ToJsonString(options);
}
