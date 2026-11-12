using ModelRouter;
using Xunit;

public class ModelParserTests
{
    [Theory]
    [InlineData("qwen3.8-gguf", "gguf-range", null)]
    [InlineData("qwen3.8-gguf-speed", "gguf-speed", null)]
    [InlineData("qwen3.8-gguf-range-xhigh", "gguf-range", "xhigh")]
    [InlineData("qwen3.8-gguf-max", "gguf-max", null)]
    [InlineData("qwen3.8-gguf-duo", "gguf-duo", null)]
    [InlineData("qwen3.8-gguf-duo128-low", "gguf-duo128", "low")]
    [InlineData("qwen3.8-gguf-quad", "gguf-quad", null)]
    [InlineData("bonsai-2-27b-max", "bonsai-max", null)]
    [InlineData("bonsai-2-27b-duo256", "bonsai-duo256", null)]
    [InlineData("bonsai-2-27b-trio256-medium", "bonsai-trio256", "medium")]
    [InlineData("bonsai-2-27b-trio200-low", "bonsai-trio200", "low")]
    [InlineData("bonsai-2-27b-quad128-low", "bonsai-quad128", "low")]
    [InlineData("bonsai-2-27b-octo64-xhigh", "bonsai-octo64", "xhigh")]
    [InlineData("mystery-model", "gguf-range", null)]
    [InlineData("mystery-model-low", "gguf-range", "low")]
    public void Parse(string model, string profile, string? effort)
    {
        var r = ModelParser.Parse(model);
        Assert.Equal(profile, r.Profile);
        Assert.Equal(effort, r.Effort);
    }
}
