using System.Text.Json.Nodes;
using ModelRouter;
using Xunit;

public class RoutingTests
{
    [Fact] public void TextPlain()
    {
        var r = Routing.For(JsonNode.Parse("{\"model\":\"qwen3.8-gguf-range\"}"));
        Assert.Equal("gguf-range", r.Profile);
        Assert.Null(r.Effort);
        Assert.False(r.WantMmproj);
        Assert.Empty(r.Notes);
    }
    [Fact] public void MediaKeepsTierWithMmproj()
    {
        var r = Routing.For(JsonNode.Parse("{\"model\":\"qwen3.8-gguf-max\",\"messages\":[{\"content\":[{\"type\":\"image_url\"}]}]}"));
        Assert.Equal("gguf-max", r.Profile);
        Assert.True(r.WantMmproj);
        Assert.Contains("media wants mmproj", r.Notes);
    }
    [Fact] public void UnknownModelFallsBackToRange()
    {
        var r = Routing.For(JsonNode.Parse("{\"model\":\"foo\"}"));
        Assert.Equal("gguf-range", r.Profile);
    }
    [Fact] public void SameBackendStickyEncoder() => Assert.True(Routing.SameBackend("bonsai-max", true, "bonsai-max", false));
    [Fact] public void SameBackendNeedsMmproj() => Assert.False(Routing.SameBackend("bonsai-max", false, "bonsai-max", true));
    [Fact] public void ReuseKeepsFasterResident() => Assert.Equal("gguf-single200", Routing.EffectiveProfile("gguf-range", "gguf-single200", 1000, false));
    [Fact] public void ReuseSwitchesToFasterTier() => Assert.Equal("gguf-single200", Routing.EffectiveProfile("gguf-single200", "gguf-range", 1000, false));
}
