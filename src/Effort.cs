using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace ModelRouter;

public static class Effort
{
    private const string ChatPath = "/v1/chat/completions";
    private const string MessagesPath = "/v1/messages";
    private static readonly string[] BudgetKeys = { "reasoning_budget_tokens", "thinking_budget_tokens" };

    private static readonly IReadOnlyDictionary<string, string> Templates = new Dictionary<string, string>
    {
        ["none"] = "none",
        ["low"] = "medium",
        ["medium"] = "medium",
        ["high"] = "xhigh",
        ["xhigh"] = "xhigh",
    };
    private static readonly IReadOnlyDictionary<string, string> Aliases = new Dictionary<string, string>
    {
        ["minimal"] = "low",
    };

    public sealed record EffortPolicy(string Template, int? Budget);
    public sealed record EffortMeta(string? Effort, string? Origin, string? Template, object? Budget, bool InjectedBudget);

    public static EffortPolicy Policy(string tier) => tier switch
    {
        "none"  => new EffortPolicy("none", null),
        "low"   => new EffortPolicy("medium", 512),
        "medium"=> new EffortPolicy("medium", 2048),
        "high"  => new EffortPolicy("xhigh", 8192),
        "xhigh" => new EffortPolicy("xhigh", null),
        _ => new EffortPolicy("medium", null),
    };

    // Safe string getter: returns "" for null / non-string values (mirrors Python's
    // isinstance(effort, str) gate — a number/bool counts as "not a string").
    private static string SafeString(JsonNode? node)
    {
        try
        {
            return node?.GetValue<string>() ?? "";
        }
        catch
        {
            return "";
        }
    }

    // ---- resolve_effort ----
    // (canonical tier, origin) where origin is "client" | "client-kwargs" | "suffix"
    public static (string? tier, string? origin) Resolve(JsonNode? root, string? suffixEffort)
    {
        string effort;
        string origin;
        effort = SafeString(root?["reasoning_effort"]);
        origin = "client";
        if (string.IsNullOrWhiteSpace(effort))
        {
            var ctk = root?["chat_template_kwargs"];
            effort = ctk is JsonObject ? SafeString(ctk["reasoning_effort"]) : "";
            origin = "client-kwargs";
        }
        if (string.IsNullOrWhiteSpace(effort))
        {
            effort = suffixEffort ?? "";
            origin = "suffix";
        }
        if (string.IsNullOrWhiteSpace(effort))
            return (null, null);
        string tier = effort.Trim().ToLowerInvariant();
        if (Aliases.TryGetValue(tier, out string aliased))
            tier = aliased;
        return (tier, origin);
    }

    // ---- client_thinking_mode ----
    public static string? ThinkingMode(JsonNode? root, string path)
    {
        if (root is not JsonObject r)
            return null;
        if (path.StartsWith(MessagesPath)
            && r["thinking"] is JsonObject th
            && th["type"]?.GetValue<string>() is "enabled" or "disabled")
        {
            return th["type"]!.GetValue<string>();
        }
        if (r["chat_template_kwargs"] is JsonObject k && k["enable_thinking"] is JsonNode en)
        {
            string s = en.ToString();
            if (s == "true")
                return "enabled";
            if (s == "false")
                return "disabled";
        }
        return null;
    }

    // ---- has_explicit_budget ----
    public static bool HasExplicitBudget(JsonNode? root, string path)
    {
        if (root is not JsonObject r)
            return false;
        foreach (var key in BudgetKeys)
            if (r.ContainsKey(key))
                return true;
        if (path.StartsWith(MessagesPath))
        {
            if (r["thinking"] is JsonObject th
                && th["type"]?.GetValue<string>() == "enabled"
                && th.ContainsKey("budget_tokens"))
            {
                return true;
            }
            if (r["extra_params"] is JsonObject ep)
            {
                foreach (var key in BudgetKeys)
                    if (ep.ContainsKey(key))
                        return true;
            }
        }
        return false;
    }
}
