using System.Text;
using System.Text.Json.Nodes;
using ModelRouter;
using Xunit;

public class EffortTests
{
    [Fact] public void ResolveClientWins()
    {
        var (tier, origin) = Effort.Resolve(JsonNode.Parse("{\"reasoning_effort\":\"HIGH\"}"), "low");
        Assert.Equal("high", tier);
        Assert.Equal("client", origin);
    }
    [Fact] public void ResolveSuffixAlias()
    {
        var (tier, origin) = Effort.Resolve(JsonNode.Parse("{}"), "minimal");
        Assert.Equal("low", tier);
        Assert.Equal("suffix", origin);
    }
    [Fact] public void BudgetConfigured() => Assert.Equal(1536, Effort.EffectiveBudget(1536, new JsonObject()));
    [Fact] public void BudgetDynamic() => Assert.Equal(2560, Effort.EffectiveBudget(3072, new JsonObject { ["max_tokens"] = 4096 }));
    [Fact] public void BudgetUnboundedDynamic() => Assert.Equal(2560, Effort.EffectiveBudget(null, new JsonObject { ["max_tokens"] = 4096 }));
    [Fact] public void HasToolContextTools() => Assert.True(Effort.HasToolContext(JsonNode.Parse("{\"tools\":[{\"type\":\"function\"}]}")));
    [Fact] public void HasToolContextToolRole() => Assert.True(Effort.HasToolContext(JsonNode.Parse("{\"messages\":[{\"role\":\"tool\",\"content\":\"ok\"}]}")));
    [Fact] public void NormalizeLowInjectsBudget()
    {
        var body = Encoding.UTF8.GetBytes("{\"model\":\"qwen3.8-gguf-max-low\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
        var r = Effort.Normalize("/v1/chat/completions", body);
        Assert.Equal("gguf-max", r.Profile);
        var decoded = JsonNode.Parse(Encoding.UTF8.GetString(r.NewBody))!;
        Assert.Equal(1536, (int)decoded["reasoning_budget_tokens"]!);
        Assert.Equal("low", (string)decoded["reasoning_effort"]!);
    }
    [Fact] public void NormalizeBonsaiToolFloor()
    {
        var body = Encoding.UTF8.GetBytes("{\"model\":\"bonsai-2-27b-quad128-low\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"tools\":[{\"type\":\"function\"}]}");
        var r = Effort.Normalize("/v1/chat/completions", body);
        var decoded = JsonNode.Parse(Encoding.UTF8.GetString(r.NewBody))!;
        Assert.Equal(16384, (int)decoded["reasoning_budget_tokens"]!);
        Assert.True(r.Meta.ToolLoop);
    }
}
