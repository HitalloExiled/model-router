using System.Diagnostics;
using System.Text;
using System.Text.Json.Nodes;

string baseUrl = Environment.GetEnvironmentVariable("E2E_BASE") ?? "http://127.0.0.1:18090";
string model = Environment.GetEnvironmentVariable("E2E_MODEL") ?? "qwen3.8-gguf-single200";
int count = int.Parse(Environment.GetEnvironmentVariable("E2E_PARALLEL") ?? "2");
using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
var total = Stopwatch.StartNew();
var tasks = Enumerable.Range(0, count).Select(async i =>
{
    var sw = Stopwatch.StartNew();
    var body = new JsonObject
    {
        ["model"] = model,
        ["messages"] = new JsonArray(new JsonObject { ["role"] = "user", ["content"] = $"Request {i}: reply OK." }),
        ["max_tokens"] = 8,
        ["stream"] = false,
    };
    var resp = await http.PostAsync($"{baseUrl}/v1/chat/completions",
        new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json"));
    await resp.Content.ReadAsStringAsync();
    sw.Stop();
    return (i, status: (int)resp.StatusCode, ms: sw.ElapsedMilliseconds);
});
foreach (var r in await Task.WhenAll(tasks))
    Console.WriteLine($"req {r.i}: status={r.status} {r.ms}ms");
total.Stop();
Console.WriteLine($"parallel={count} wall={total.ElapsedMilliseconds}ms");
