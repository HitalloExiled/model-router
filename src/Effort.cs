using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace ModelRouter;

public static class Effort
{
    private const string ChatPath = "/v1/chat/completions";
    private const string MessagesPath = "/v1/messages";
    private static readonly string[] BudgetKeys = { "reasoning_budget_tokens", "thinking_budget_tokens" };

    // Injected right before the forced end-of-thinking tag when the budget runs
    // out, so the model doesn't emit a bare </think> and end the turn with EOS.
    private const string BudgetMessage =
        "\n\nOkay, I have thought enough. Time to write the final answer.";

    private static readonly Dictionary<string, string> Templates = new()
    {
        ["none"] = "none",
        ["low"] = "medium",
        ["medium"] = "medium",
        ["high"] = "xhigh",
        ["xhigh"] = "xhigh",
    };
    private static readonly Dictionary<string, string> Aliases = new()
    {
        ["minimal"] = "low",
    };

    public sealed record EffortPolicy(string Template, int? Budget);
    public sealed record EffortMeta(string? Effort, string? Origin, string? Template, object? Budget, bool InjectedBudget);

    // Backend family -> tier -> (template, budget). Mirrors python EFFORT_POLICY
    // + POLICY_UNCAPPED via effort_policy(family).
    private static EffortPolicy GetPolicy(string? family, string tier)
    {
        string template;
        int? budget;
        if (family == "gguf")
        {
            budget = tier switch
            {
                "low" => 2048,
                "medium" => 3072,
                "high" => 8192,
                "none" => null,
                _ => null,
            };
            template = tier switch { "none" => "none", "high" => "xhigh", "xhigh" => "xhigh", _ => "medium" };
        }
        else if (family == "bonsai")
        {
            budget = tier switch
            {
                "low" => 4096,
                "medium" => 6144,
                "high" => 8192,
                "none" => null,
                _ => null,
            };
            template = tier switch { "none" => "none", "high" => "xhigh", "xhigh" => "xhigh", _ => "medium" };
        }
        else
        {
            // POLICY_UNCAPPED (ninfer / unknown): template hint only, no budget.
            budget = null;
            template = tier switch { "none" => "none", "high" => "xhigh", "xhigh" => "xhigh", _ => "medium" };
        }
        return new EffortPolicy(template, budget);
    }

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

    // get-or-create a nested JsonObject; AOT-safe (indexer + ContainsKey only).
    private static JsonObject GetOrCreate(JsonObject obj, string key)
    {
        if (!obj.ContainsKey(key))
            obj[key] = new JsonObject();
        return (JsonObject)obj[key]!;
    }

    // ---- resolve_effort ----
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
            && SafeString(th["type"]) is "enabled" or "disabled")
        {
            return SafeString(th["type"]);
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
                && SafeString(th["type"]) == "enabled"
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

    // ---- apply_effort_policy (mutates) ----
    private static bool ApplyEffortPolicy(JsonObject decoded, string path, string effort,
        string? family, ref EffortMeta meta, List<string> notes)
    {
        var policy = GetPolicy(family, effort);
        string template = policy.Template;
        int? budget = policy.Budget;
        meta = meta with { Template = template };
        string? mode = ThinkingMode(decoded, path);
        if (mode == "disabled")
        {
            notes.Add("thinking disabled by client");
            return false;
        }
        if (effort == "none")
        {
            if (mode == "enabled")
            {
                notes.Add("thinking enabled by client");
                return false;
            }
            if (path.StartsWith(MessagesPath))
            {
                var ctk = GetOrCreate(decoded, "chat_template_kwargs");
                ctk["enable_thinking"] = false;
                ctk.Remove("reasoning_effort");
                decoded.Remove("reasoning_effort");
            }
            else
            {
                decoded["reasoning_effort"] = "none";
            }
            notes.Add("effort none");
            return true;
        }
        if (path.StartsWith(MessagesPath))
        {
            var ctk = GetOrCreate(decoded, "chat_template_kwargs");
            ctk["enable_thinking"] = true;
            ctk["reasoning_effort"] = template;
            decoded.Remove("reasoning_effort");
            notes.Add($"effort {effort}->{template}");
            if (budget is not null)
            {
                if (HasExplicitBudget(decoded, path))
                {
                    meta = meta with { Budget = "client" };
                    notes.Add("budget kept from client");
                }
                else
                {
                    var thinking = decoded["thinking"] as JsonObject;
                    if (thinking is not null && SafeString(thinking["type"]) == "enabled")
                    {
                        thinking["budget_tokens"] = budget;
                    }
                    else
                    {
                        decoded["thinking"] = new JsonObject
                        {
                            ["type"] = "enabled",
                            ["budget_tokens"] = budget,
                        };
                    }
                    meta = meta with { Budget = budget, InjectedBudget = true };
                    notes.Add($"budget {budget}");
                }
            }
            return true;
        }
        // chat path
        decoded["reasoning_effort"] = template;
        notes.Add($"effort {effort}->{template}");
        bool capped = HasExplicitBudget(decoded, path);
        if (capped)
        {
            meta = meta with { Budget = "client" };
            notes.Add("budget kept from client");
        }
        else if (budget is not null)
        {
            decoded["reasoning_budget_tokens"] = budget;
            meta = meta with { Budget = budget, InjectedBudget = true };
            capped = true;
            notes.Add($"budget {budget}");
        }
        if (capped)
            decoded["reasoning_budget_message"] = BudgetMessage;
        return true;
    }

    // ---- normalize_chat_body (mutates, returns) ----
    public sealed record NormalizeResult(string Profile, byte[] NewBody, string Note, bool WantMmproj, EffortMeta Meta);

    public static NormalizeResult Normalize(string path, byte[]? body, Routing.RouteRequest? route = null, string? finalProfile = null)
    {
        string json = body is { Length: > 0 } b ? Encoding.UTF8.GetString(b) : "{}";
        JsonObject decoded = JsonNode.Parse(json) is JsonObject o ? o : new JsonObject();

        Routing.RouteRequest r = route ?? Routing.For(decoded);
        string profile = finalProfile is { Length: > 0 } fp ? fp : r.Profile;
        bool modified = false;
        var meta = new EffortMeta(null, null, null, null, false);

        if (profile == "vision")
        {
            if (path.StartsWith(ChatPath))
            {
                if (r.Effort == "none" && !decoded.ContainsKey("reasoning_effort"))
                {
                    decoded["reasoning_effort"] = "none";
                    modified = true;
                    r.Notes.Add("injected none");
                }
                if (decoded.ContainsKey("reasoning_effort"))
                {
                    string re = SafeString(decoded["reasoning_effort"]);
                    if (re is not "none")
                    {
                        decoded.Remove("reasoning_effort");
                        r.Notes.Add($"vision dropped {re}");
                        modified = true;
                    }
                }
                foreach (var key in BudgetKeys)
                {
                    if (decoded.ContainsKey(key))
                    {
                        decoded.Remove(key);
                        modified = true;
                        r.Notes.Add($"vision dropped {key}");
                    }
                }
            }
        }
        else
        {
            var (effort, origin) = Resolve(decoded, r.Effort);
            meta = meta with { Effort = effort, Origin = origin };
            bool supported = effort is not null && Templates.ContainsKey(effort);
            if (effort is not null && !supported)
            {
                r.Notes.Add($"ignored unsupported effort {effort}");
                modified = true;
                if (path.StartsWith(ChatPath))
                    decoded.Remove("reasoning_effort");
                else
                    if (decoded["chat_template_kwargs"] is JsonObject ctk)
                        ctk.Remove("reasoning_effort");
                meta = meta with { Effort = null };
            }
            else if (effort is { } ef)
            {
                if (path.StartsWith(MessagesPath) && !Profiles.LlamaAll.Contains(profile))
                {
                    r.Notes.Add($"effort {effort} skipped for ninfer");
                }
                else
                {
                    bool changed = ApplyEffortPolicy(decoded, path, effort,
                        Profiles.Family(profile), ref meta, r.Notes);
                    modified = modified || changed;
                }
            }
        }

        byte[] newBody = modified ? Encoding.UTF8.GetBytes(decoded.ToJsonString()) : (body ?? Array.Empty<byte>());
        string note = r.Notes.Count > 0 ? " (" + string.Join(", ", r.Notes) + ")" : "";
        return new NormalizeResult(profile, newBody, note, r.WantMmproj, meta);
    }
}