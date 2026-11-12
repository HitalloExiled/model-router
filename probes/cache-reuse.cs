using System.Diagnostics;
using System.Text;
using System.Text.Json.Nodes;

string baseUrl = Environment.GetEnvironmentVariable("E2E_BASE") ?? "http://127.0.0.1:18090";
string model = Environment.GetEnvironmentVariable("E2E_MODEL") ?? "qwen3.8-gguf-single200";
string prefix = string.Join("\n", Enumerable.Range(0, 400).Select(i => $"line {i}: the quick brown fox jumps over the lazy dog"));
using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
for (int i = 0; i < 2; i++)
{
    var body = new JsonObject
    {
        ["model"] = model,
        ["messages"] = new JsonArray(new JsonObject { ["role"] = "user", ["content"] = prefix + "\nReply OK." }),
        ["max_tokens"] = 8,
        ["stream"] = false,
    };
    var sw = Stopwatch.StartNew();
    var resp = await http.PostAsync($"{baseUrl}/v1/chat/completions",
        new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json"));
    string raw = await resp.Content.ReadAsStringAsync();
    sw.Stop();
    var n = JsonNode.Parse(raw);
    int prompt = n?["usage"]?["prompt_tokens"]?.GetValue<int>() ?? 0;
    int cached = n?["usage"]?["prompt_tokens_details"]?["cached_tokens"]?.GetValue<int>() ?? 0;
    double promptMs = n?["timings"]?["prompt_ms"]?.GetValue<double>() ?? 0;
    Console.WriteLine($"run {i}: status={(int)resp.StatusCode} {sw.ElapsedMilliseconds}ms prompt={prompt} cached={cached} prompt_ms={promptMs:F1}");
}
