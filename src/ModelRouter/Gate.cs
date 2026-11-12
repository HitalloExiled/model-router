using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;

namespace ModelRouter;

// Prefill admission gate: pure helpers. The HTTP side (backend_slot_state)
// and the waiting loop (admit_prefill) live in BackendManager; the relay calls
// them before forwarding. Mirrors python prompt_* / next_token_info /
// slot_phase / gate_decision.
public static class Gate
{
    public static int PrefillGateMin =>
        int.Parse(Environment.GetEnvironmentVariable("MODEL_ROUTER_PREFILL_GATE_MIN") ?? "1024");

    // ---- prompt prefix/request hashing ----
    public static string? PromptPrefixKey(JsonNode? root, string? profile = null)
    {
        if (root is not JsonObject obj || obj["messages"] is not JsonArray messages || messages.Count == 0)
            return null;
        var copy = new JsonObject();
        foreach (var kv in obj)
        {
            if (kv.Key == "messages") continue;
            copy[kv.Key] = Clone(kv.Value);
        }
        var prefixMessages = new JsonArray();
        for (int i = 0; i < messages.Count - 1; i++)
            prefixMessages.Add(Clone(messages[i]));
        copy["messages"] = prefixMessages;
        if (profile is not null) copy["_router_backend_profile"] = profile;
        return Sha1Hex(Canonical(copy));
    }

    public static string? PromptRequestKey(JsonNode? root, string? profile = null)
    {
        if (root is not JsonObject obj || obj["messages"] is not JsonArray messages || messages.Count == 0)
            return null;
        var copy = new JsonObject();
        foreach (var kv in obj)
            copy[kv.Key] = Clone(kv.Value);
        if (profile is not null) copy["_router_backend_profile"] = profile;
        return Sha1Hex(Canonical(copy));
    }

    // ---- next_token_info ----
    public static (int Decoded, bool HasNext) NextTokenInfo(JsonNode? slot)
    {
        if (slot is not JsonObject s || s["next_token"] is not JsonArray nt
            || nt.Count == 0 || nt[0] is not JsonObject first)
            return (0, false);
        int decoded = 0;
        if (first["n_decoded"] is JsonValue dv && dv.TryGetValue<int>(out var d))
            decoded = d;
        bool hasNext = first["has_next_token"] is JsonValue hv
            && hv.TryGetValue<bool>(out var h) && h;
        return (decoded, hasNext);
    }

    // ---- slot_phase ----
    public sealed record SlotState(
        int Id, bool Proc, int Decoded, int NPrompt, int PromptLen,
        int Cached, int Processed, int InFlight, int Queued,
        bool Prefilling, bool Generating);

    public static SlotState? SlotPhase(JsonNode? slot)
    {
        if (slot is not JsonObject s) return null;
        if (!TryIntField(s, "id", -1, out int id)) return null;
        bool proc = Truthy(s["is_processing"]);
        var (decoded, hasNext) = NextTokenInfo(s);
        if (!TryIntField(s, "n_prompt_tokens", 0, out int nPrompt)) return null;
        if (!TryIntField(s, "n_prompt_tokens_cache", 0, out int cached)) return null;
        if (!TryIntField(s, "n_prompt_tokens_processed", 0, out int processed)) return null;
        int promptLen = Math.Max(0, nPrompt - decoded);
        int inFlight = Math.Max(0, nPrompt - cached);
        int queued = Math.Max(0, promptLen - cached - processed);
        bool prefilling = proc && decoded == 0 && !hasNext && inFlight >= PrefillGateMin;
        bool generating = proc && (hasNext || decoded > 0);
        return new SlotState(id, proc, decoded, nPrompt, promptLen, cached, processed,
            inFlight, queued, prefilling, generating);
    }

    // ---- gate_decision ----
    public static string? GateDecision(bool prefilling, bool generating, bool coldLarge)
    {
        if (prefilling) return "prefill";
        if (coldLarge && generating) return "cold-vs-decode";
        return null;
    }

    // ---- canonical JSON (sorted keys, compact separators) ----
    private static string Canonical(JsonNode? node)
    {
        if (node is null) return "null";
        if (node is JsonObject obj)
        {
            var sb = new StringBuilder("{");
            bool first = true;
            foreach (var kv in obj.OrderBy(k => k.Key, StringComparer.Ordinal))
            {
                if (!first) sb.Append(',');
                first = false;
                sb.Append(JsonValue.Create(kv.Key)!.ToJsonString());
                sb.Append(':');
                sb.Append(Canonical(kv.Value));
            }
            sb.Append('}');
            return sb.ToString();
        }
        if (node is JsonArray arr)
        {
            var sb = new StringBuilder("[");
            for (int i = 0; i < arr.Count; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append(Canonical(arr[i]));
            }
            sb.Append(']');
            return sb.ToString();
        }
        return node.ToJsonString();
    }

    private static JsonNode? Clone(JsonNode? node)
        => node is null ? null : JsonNode.Parse(node.ToJsonString());

    private static string Sha1Hex(string raw)
        => Convert.ToHexString(SHA1.HashData(Encoding.UTF8.GetBytes(raw))).ToLowerInvariant();

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

    // Python int(x or 0): missing/null -> fallback; numeric -> value; a
    // non-numeric present value raises (returns false here).
    private static bool TryIntField(JsonObject s, string key, int fallback, out int value)
    {
        value = fallback;
        if (!s.TryGetPropertyValue(key, out var node) || node is null) return true;
        if (node is JsonValue v)
        {
            if (v.TryGetValue<int>(out value)) return true;
            if (v.TryGetValue<string>(out var str) && int.TryParse(str, out value)) return true;
        }
        return false;
    }
}

// LRU of recently admitted prompts. A prompt is "hot" when the same prefix was
// admitted within PREFILL_PREFIX_TTL; the entry stores the complete admitted
// request so the next turn (same conversation + one new message) matches.
public static class PrefixCache
{
    private static readonly object sync = new();
    private static readonly LinkedList<string> order = new();
    private static readonly Dictionary<string, (long Seen, LinkedListNode<string> Node)> map = new();

    private static long TtlSeconds =>
        (long)double.Parse(Environment.GetEnvironmentVariable("MODEL_ROUTER_PREFILL_PREFIX_TTL") ?? "3600");

    private static int Max =>
        int.Parse(Environment.GetEnvironmentVariable("MODEL_ROUTER_PREFILL_PREFIX_MAX") ?? "4096");

    private static long NowSeconds => Environment.TickCount64 / 1000;

    public static void Clear()
    {
        lock (sync)
        {
            order.Clear();
            map.Clear();
        }
    }

    public static bool IsHot(string? key)
    {
        if (key is null) return false;
        long now = NowSeconds;
        lock (sync)
        {
            while (order.First is { } first)
            {
                long seen = map[first.Value].Seen;
                if (now - seen <= TtlSeconds) break;
                order.RemoveFirst();
                map.Remove(first.Value);
            }
            if (!map.TryGetValue(key, out var entry)) return false;
            order.Remove(entry.Node);
            order.AddLast(entry.Node);
            map[key] = (now, entry.Node);
            return true;
        }
    }

    public static void Remember(string? key)
    {
        if (key is null) return;
        long now = NowSeconds;
        lock (sync)
        {
            if (map.TryGetValue(key, out var existing))
                order.Remove(existing.Node);
            var node = order.AddLast(key);
            map[key] = (now, node);
            while (map.Count > Max)
            {
                if (order.First is not { } first) break;
                order.RemoveFirst();
                map.Remove(first.Value);
            }
        }
    }
}
