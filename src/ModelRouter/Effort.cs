using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace ModelRouter;

public static class Effort
{
    private const string ChatPath = "/v1/chat/completions";
    private const string MessagesPath = "/v1/messages";
    private static readonly string[] budgetKeys = { "reasoning_budget_tokens", "thinking_budget_tokens" };

    // Injected right before the forced end-of-thinking tag when the budget runs
    // out. Without it llama.cpp forces a bare </think> mid-sentence and Bonsai
    // often emits EOS, ending the turn with no answer.
    private const string BUDGET_MESSAGE =
        "\n\nOkay, I have thought enough. Time to write the final answer.";

    // Bonsai in a tool loop needs room to finish the plan before it emits the
    // call; tested agent profile uses a 16k thinking budget.
    public const int TOOL_BUDGET_FLOOR = 16384;

    // Leave the model room to answer after the forced transition.
    public const int ANSWER_RESERVE = 1536;

    // One concise example beats an abstract instruction. Prepended to the
    // client system (gguf low/medium). Bonsai keeps no style.
    public const string FEWSHOT_STYLE =
        "Example of the desired style:\n" +
        "Question: What is 17*24?\n" +
        "17*24 = 17*20 + 17*4 = 408.\n" +
        "408\n\n" +
        "Solve the next task in the same concise style: at most one sentence of " +
        "reasoning, then the answer on its own line, without labels.";

    public sealed record EffortPolicy(string Template, int? Budget);

    public sealed record EffortMeta(
        string? Effort,
        string? Origin,
        string? Template,
        object? Budget,
        bool InjectedBudget,
        bool ToolLoop = false,
        bool Style = false);

    // Backend family -> {tier: (template reasoning_effort, configured budget)}.
    private static readonly Dictionary<string, Dictionary<string, EffortPolicy>> policy = new()
    {
        ["gguf"] = new Dictionary<string, EffortPolicy>
        {
            ["none"] = new("none", null),
            ["low"] = new("low", 1536),
            ["medium"] = new("medium", 3072),
            ["high"] = new("xhigh", 8192),
            ["xhigh"] = new("xhigh", null),
        },
        ["bonsai"] = new Dictionary<string, EffortPolicy>
        {
            ["none"] = new("none", null),
            ["low"] = new("medium", 1536),
            ["medium"] = new("medium", 6144),
            ["high"] = new("xhigh", 8192),
            ["xhigh"] = new("xhigh", null),
        },
    };

    private static readonly Dictionary<string, Dictionary<string, string>> styles = new()
    {
        ["gguf"] = new Dictionary<string, string>
        {
            ["low"] = FEWSHOT_STYLE,
            ["medium"] = FEWSHOT_STYLE,
        },
    };

    private static readonly Dictionary<string, string> aliases = new() { ["minimal"] = "low" };

    // Mirror python json.dumps(ensure_ascii=False): keep '+' and non-ASCII raw
    // instead of escaping them (\u002B, \u00e1) in the forwarded body.
    private static readonly JsonSerializerOptions jsonOptions = new()
    {
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    // Unknown family falls back to the gguf table: budgeting like Qwen is the
    // safe default.
    public static EffortPolicy GetPolicy(string? family, string tier)
    {
        var table = family is not null && policy.TryGetValue(family, out var t) ? t : policy["gguf"];
        return table.TryGetValue(tier, out var p) ? p : new EffortPolicy("medium", null);
    }

    private static bool Truthy(JsonNode? node)
    {
        if (node is null) return false;
        if (node is JsonArray a) return a.Count > 0;
        if (node is JsonObject o) return o.Count > 0;
        if (node is JsonValue v)
        {
            if (v.TryGetValue<bool>(out var b)) return b;
            if (v.TryGetValue<string>(out var s)) return s.Length > 0;
            if (v.TryGetValue<double>(out var d)) return d != 0;
            return true;
        }
        return false;
    }

    private static JsonNode? Clone(JsonNode? node)
        => node is null ? null : JsonNode.Parse(node.ToJsonString());

    private static string SafeString(JsonNode? node)
        => node is JsonValue v && v.TryGetValue<string>(out var s) ? s : "";

    private static JsonObject GetOrCreateObject(JsonObject obj, string key)
    {
        if (obj[key] is JsonObject existing) return existing;
        var created = new JsonObject();
        obj[key] = created;
        return created;
    }

    // ---- resolve_effort ----
    public static (string? effort, string? origin) Resolve(JsonNode? root, string? suffixEffort)
    {
        string effort = SafeString(root?["reasoning_effort"]);
        string origin = "client";
        if (string.IsNullOrWhiteSpace(effort))
        {
            var ctk = root?["chat_template_kwargs"];
            effort = ctk is JsonObject co ? SafeString(co["reasoning_effort"]) : "";
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
        if (aliases.TryGetValue(tier, out var a) && a is not null)
            tier = a;
        return (tier, origin);
    }

    // ---- client_thinking_mode ----
    public static string? ThinkingMode(JsonNode? root, string path)
    {
        if (root is not JsonObject r) return null;
        if (path.StartsWith(MessagesPath) && r["thinking"] is JsonObject th)
        {
            string t = SafeString(th["type"]);
            if (t is "enabled" or "disabled") return t;
        }
        if (r["chat_template_kwargs"] is JsonObject ctk && ctk["enable_thinking"] is JsonValue v)
        {
            if (v.TryGetValue<bool>(out var b)) return b ? "enabled" : "disabled";
            if (v.TryGetValue<string>(out var s))
            {
                if (s == "false") return "disabled";
                if (s == "true") return "enabled";
            }
        }
        return null;
    }

    // ---- has_explicit_budget ----
    public static bool HasExplicitBudget(JsonNode? root, string path)
    {
        if (root is not JsonObject r) return false;
        foreach (var key in budgetKeys)
            if (r.ContainsKey(key)) return true;
        if (path.StartsWith(MessagesPath))
        {
            if (r["thinking"] is JsonObject th
                && SafeString(th["type"]) == "enabled"
                && th.ContainsKey("budget_tokens"))
                return true;
            if (r["extra_params"] is JsonObject ep)
                foreach (var key in budgetKeys)
                    if (ep.ContainsKey(key)) return true;
        }
        return false;
    }

    // ---- effective_budget ----
    public static int? EffectiveBudget(int? configured, JsonObject decoded)
    {
        int? limit = null;
        if (decoded["max_tokens"] is JsonValue mv && mv.TryGetValue<int>(out var mt) && mt > 0)
            limit = mt;
        else if (decoded["max_completion_tokens"] is JsonValue cv && cv.TryGetValue<int>(out var mc) && mc > 0)
            limit = mc;
        if (limit is null) return configured;
        int room = limit.Value - ANSWER_RESERVE;
        if (room < 256) room = Math.Max(128, limit.Value / 2);
        return configured is null ? room : Math.Min(configured.Value, room);
    }

    // ---- inject_style ----
    public static void InjectStyle(JsonObject decoded, string path, string style)
    {
        if (path.StartsWith(MessagesPath))
        {
            var system = decoded["system"];
            if (system is JsonArray arr)
            {
                var first = new JsonObject { ["type"] = "text", ["text"] = style };
                var newArr = new JsonArray(first);
                foreach (var item in arr) newArr.Add(Clone(item));
                decoded["system"] = newArr;
            }
            else if (system is JsonValue sv && sv.TryGetValue<string>(out var s) && s.Length > 0)
            {
                decoded["system"] = style + "\n\n" + s;
            }
            else
            {
                decoded["system"] = style;
            }
            return;
        }
        if (decoded["messages"] is not JsonArray messages) return;
        foreach (var msgNode in messages)
        {
            if (msgNode is not JsonObject msg || SafeString(msg["role"]) != "system") continue;
            var content = msg["content"];
            if (content is JsonArray contentArr)
            {
                var first = new JsonObject { ["type"] = "text", ["text"] = style };
                var newArr = new JsonArray(first);
                foreach (var item in contentArr) newArr.Add(Clone(item));
                msg["content"] = newArr;
            }
            else if (content is JsonValue cv && cv.TryGetValue<string>(out var cs) && cs.Length > 0)
            {
                msg["content"] = style + "\n\n" + cs;
            }
            else
            {
                msg["content"] = style;
            }
            return;
        }
        messages.Insert(0, new JsonObject { ["role"] = "system", ["content"] = style });
    }

    // ---- has_tool_context ----
    public static bool HasToolContext(JsonNode? root)
    {
        if (root is not JsonObject r) return false;
        if (Truthy(r["tools"])) return true;
        if (r["messages"] is JsonArray messages)
        {
            foreach (var msgNode in messages)
            {
                if (msgNode is not JsonObject msg) continue;
                string role = SafeString(msg["role"]);
                if (role is "tool" or "function") return true;
                if (msg["content"] is JsonArray parts)
                {
                    foreach (var part in parts)
                        if (part is JsonObject p && SafeString(p["type"]) is "tool_result" or "tool_use")
                            return true;
                }
            }
        }
        return false;
    }

    // ---- apply_effort_policy (mutates) ----
    private static bool ApplyEffortPolicy(JsonObject decoded, string path, string effort,
        string? family, ref EffortMeta meta, List<string> notes)
    {
        var p = GetPolicy(family, effort);
        string template = p.Template;
        int? configured = p.Budget;
        meta = meta with { Template = template };
        bool toolLoop = family == "bonsai" && HasToolContext(decoded);
        if (toolLoop)
        {
            configured = TOOL_BUDGET_FLOOR;
            meta = meta with { ToolLoop = true };
            notes.Add($"tool loop budget {TOOL_BUDGET_FLOOR}");
        }
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
                var ctk = GetOrCreateObject(decoded, "chat_template_kwargs");
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
        if (styles.TryGetValue(family ?? "", out var famStyle) && famStyle.TryGetValue(effort, out var style))
        {
            InjectStyle(decoded, path, style);
            meta = meta with { Style = true };
            notes.Add("style injected");
        }
        int? budget = EffectiveBudget(configured, decoded);
        if (path.StartsWith(MessagesPath))
        {
            var ctk = GetOrCreateObject(decoded, "chat_template_kwargs");
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
                    if (decoded["thinking"] is JsonObject thinking
                        && SafeString(thinking["type"]) == "enabled")
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
        // A bare forced </think> lands mid-sentence; give the model a
        // transition first, whoever set the budget. An explicit client message
        // still wins. Tool loops skip it: the 16k floor means the natural
        // reasoning fits, and the message anchored malformed calls.
        if (capped && !toolLoop && !decoded.ContainsKey("reasoning_budget_message"))
            decoded["reasoning_budget_message"] = BUDGET_MESSAGE;
        return true;
    }

    // ---- normalize_chat_body (mutates, returns) ----
    public sealed record NormalizeResult(string Profile, byte[] NewBody, string Note, bool WantMmproj, EffortMeta Meta);

    public static NormalizeResult Normalize(string path, byte[]? body, Routing.RouteRequest? route = null, string? finalProfile = null)
    {
        string json = body is { Length: > 0 } b ? Encoding.UTF8.GetString(b) : "{}";
        JsonObject decoded = JsonNode.Parse(json) as JsonObject ?? new JsonObject();
        Routing.RouteRequest r = route ?? Routing.For(decoded);
        string profile = finalProfile ?? r.Profile;
        var notes = new List<string>(r.Notes);
        bool modified = false;
        var meta = new EffortMeta(null, null, null, null, false);

        var (effort, origin) = Resolve(decoded, r.Effort);
        meta = meta with { Effort = effort, Origin = origin };
        if (effort is not null && !policy["gguf"].ContainsKey(effort))
        {
            notes.Add($"ignored unsupported effort {effort}");
            if (path.StartsWith(ChatPath))
                decoded.Remove("reasoning_effort");
            else if (decoded["chat_template_kwargs"] is JsonObject ctk)
                ctk.Remove("reasoning_effort");
            modified = true;
            effort = null;
            meta = meta with { Effort = null };
        }
        if (effort is not null)
        {
            bool changed = ApplyEffortPolicy(decoded, path, effort, Profiles.Family(profile), ref meta, notes);
            modified = modified || changed;
        }

        byte[] newBody = modified ? Encoding.UTF8.GetBytes(decoded.ToJsonString(jsonOptions)) : (body ?? Array.Empty<byte>());
        string note = notes.Count > 0 ? " (" + string.Join(", ", notes) + ")" : "";
        return new NormalizeResult(profile, newBody, note, r.WantMmproj, meta);
    }
}
