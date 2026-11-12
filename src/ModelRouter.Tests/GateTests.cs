using System.Text.Json.Nodes;
using ModelRouter;
using Xunit;

public class GateTests
{
    [Fact] public void GateDecision()
    {
        Assert.Null(Gate.GateDecision(false, false, true));
        Assert.Equal("prefill", Gate.GateDecision(true, false, false));
        Assert.Equal("cold-vs-decode", Gate.GateDecision(false, true, true));
    }
    [Fact] public void NextToken()
    {
        var s = JsonNode.Parse("{\"next_token\":[{\"has_next_token\":true,\"n_decoded\":154}]}");
        Assert.Equal((154, true), Gate.NextTokenInfo(s));
    }
    [Fact] public void SlotPhasePrefill()
    {
        var s = JsonNode.Parse("{\"id\":1,\"is_processing\":true,\"n_prompt_tokens\":4096,\"n_prompt_tokens_cache\":0,\"n_prompt_tokens_processed\":2048,\"next_token\":[{\"has_next_token\":false,\"n_decoded\":0}]}");
        var p = Gate.SlotPhase(s);
        Assert.NotNull(p);
        Assert.True(p!.Prefilling);
        Assert.False(p.Generating);
        Assert.Equal(4096, p.InFlight);
    }
    [Fact] public void PrefixKeySamePrefix()
    {
        var p2 = JsonNode.Parse("{\"model\":\"x\",\"stream\":true,\"messages\":[{\"role\":\"system\",\"content\":\"rules\"},{\"role\":\"user\",\"content\":\"first\"},{\"role\":\"assistant\",\"content\":\"answer\"},{\"role\":\"user\",\"content\":\"next\"}]}");
        var p3 = JsonNode.Parse("{\"model\":\"x\",\"stream\":true,\"messages\":[{\"role\":\"system\",\"content\":\"rules\"},{\"role\":\"user\",\"content\":\"first\"},{\"role\":\"assistant\",\"content\":\"answer\"},{\"role\":\"user\",\"content\":\"other\"}]}");
        Assert.Equal(Gate.PromptPrefixKey(p2), Gate.PromptPrefixKey(p3));
        Assert.Null(Gate.PromptPrefixKey(JsonNode.Parse("{\"messages\":[]}")));
    }
    [Fact] public void HotAfterRemember()
    {
        PrefixCache.Clear();
        var p1 = JsonNode.Parse("{\"model\":\"x\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"first\"}]}");
        var p2 = JsonNode.Parse("{\"model\":\"x\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"first\"},{\"role\":\"user\",\"content\":\"next\"}]}");
        Assert.False(PrefixCache.IsHot(Gate.PromptPrefixKey(p1)));
        PrefixCache.Remember(Gate.PromptRequestKey(p1));
        Assert.True(PrefixCache.IsHot(Gate.PromptPrefixKey(p2)));
    }
}
