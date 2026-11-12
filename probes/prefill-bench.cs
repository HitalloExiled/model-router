using System.Diagnostics;
using System.Text;
using System.Text.Json.Nodes;

string baseUrl = Environment.GetEnvironmentVariable("E2E_BASE") ?? "http://127.0.0.1:18090";
string model = Environment.GetEnvironmentVariable("E2E_MODEL") ?? "qwen3.8-gguf-single200";
int lines = int.Parse(Environment.GetEnvironmentVariable("E2E_PREFILL_LINES") ?? "800");
string prompt = string.Join("\n", Enumerable.Range(0, lines).Select(i => $"line {i}: the quick brown fox jumps over the lazy dog"));
using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
var body = new JsonObject
{
    ["model"] = model,
    ["messages"] = new JsonArray(new JsonObject { ["role"] = "user", ["content"] = prompt + "\nReply OK." }),
    ["max_tokens"] = 8,
    ["stream"] = false,
};
var sw = Stopwatch.StartNew();
var resp = await http.PostAsync($"{baseUrl}/v1/chat/completions",
    new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json"));
string raw = await resp.Content.ReadAsStringAsync();
sw.Stop();
var n = JsonNode.Parse(raw);
int promptN = n?["usage"]?["prompt_tokens"]?.GetValue<int>() ?? 0;
double promptMs = n?["timings"]?["prompt_ms"]?.GetValue<double>() ?? 0;
double promptPs = n?["timings"]?["prompt_per_second"]?.GetValue<double>() ?? 0;
Console.WriteLine($"prefill: status={(int)resp.StatusCode} {sw.ElapsedMilliseconds}ms prompt_tokens={promptN} prompt_ms={promptMs:F1} tok/s={promptPs:F1}");
