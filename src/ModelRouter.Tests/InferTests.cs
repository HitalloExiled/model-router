using ModelRouter;
using Xunit;

public class InferTests
{
    [Theory]
    [InlineData("", null)]
    [InlineData("llama-server --alias qwen3.8-gguf-max-low", "gguf-max")]
    [InlineData("llama-server --alias=mystery-model", "gguf-range")]
    [InlineData("Ternary-Bonsai --parallel 8 --ctx-size 262144", "bonsai-octo64")]
    [InlineData("Ternary-Bonsai --parallel 3 --ctx-size 614400", "bonsai-trio200")]
    [InlineData("Ternary-Bonsai --parallel 3 --ctx-size 262144", "bonsai-trio256")]
    [InlineData("Ternary-Bonsai --ctx-size 262144 --mmproj", "bonsai-vision")]
    [InlineData("Ternary-Bonsai --ctx-size 262144", "bonsai-max")]
    [InlineData("llama-server --parallel 4 --ctx-size 131072", "gguf-quad")]
    [InlineData("llama-server --parallel 2 --ctx-size 262144", "gguf-duo128")]
    [InlineData("llama-server -c 262144 -m x", "gguf-max")]
    [InlineData("llama-server --ctx-size 131072 -m mtp.gguf", "gguf-speed")]
    [InlineData("llama-server --ctx-size 131072 -m something", "gguf-range")]
    [InlineData("no-recognizable-args", null)]
    public void InferProfileMatches(string args, string? profile) => Assert.Equal(profile, Infer.InferProfile(args));
}
