using System.Diagnostics;
using System.Text;
using System.Text.Json.Nodes;

string baseUrl = Environment.GetEnvironmentVariable("E2E_BASE") ?? "http://127.0.0.1:18090";
string model = Environment.GetEnvironmentVariable("E2E_MODEL") ?? "qwen3.8-gguf-single200";
using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
Console.WriteLine($"effort probe: model={model} base={baseUrl}");
foreach (var tier in new[] { "none", "low", "medium", "high", "xhigh" })
{
    var body = new JsonObject
    {
        ["model"] = model,
        ["messages"] = new JsonArray(new JsonObject { ["role"] = "user", ["content"] = "What is 17*24? Answer briefly." }),
        ["reasoning_effort"] = tier,
        ["max_tokens"] = 256,
        ["stream"] = false,
    };
    var sw = Stopwatch.StartNew();
    var resp = await http.PostAsync($"{baseUrl}/v1/chat/completions",
        new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json"));
    string raw = await resp.Content.ReadAsStringAsync();
    sw.Stop();
    var n = JsonNode.Parse(raw);
    int completion = n?["usage"]?["completion_tokens"]?.GetValue<int>() ?? 0;
    int reasoning = n?["usage"]?["completion_tokens_details"]?["reasoning_tokens"]?.GetValue<int>() ?? 0;
    Console.WriteLine($"{tier,-6} status={(int)resp.StatusCode} {sw.ElapsedMilliseconds,6} ms completion={completion} reasoning={reasoning}");
}
