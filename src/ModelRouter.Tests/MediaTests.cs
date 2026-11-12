using System.Text.Json.Nodes;
using ModelRouter;
using Xunit;

public class MediaTests
{
    [Fact] public void NoMessages() => Assert.False(Media.Has(JsonNode.Parse("{\"model\":\"x\"}")));
    [Fact] public void StringContent() => Assert.False(Media.Has(JsonNode.Parse("{\"messages\":[{\"content\":\"hi\"}]}")));
    [Theory]
    [InlineData("image_url")] [InlineData("input_image")] [InlineData("image")]
    [InlineData("video_url")] [InlineData("input_video")]
    public void MediaPart(string type) =>
        Assert.True(Media.Has(JsonNode.Parse($"{{\"messages\":[{{\"content\":[{{\"type\":\"{type}\"}}]}}]}}")));
    [Fact] public void DocumentImageSource() =>
        Assert.True(Media.Has(JsonNode.Parse("{\"messages\":[{\"content\":[{\"type\":\"document\",\"source\":{\"media_type\":\"image/png\"}}]}]}")));
    [Fact] public void NonDictParts() => Assert.False(Media.Has(JsonNode.Parse("{\"messages\":[{\"content\":[\"x\",1]}]}")));
}
