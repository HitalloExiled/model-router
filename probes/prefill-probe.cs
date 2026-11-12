using System.Text;
using System.Text.Json.Nodes;

string baseUrl = Environment.GetEnvironmentVariable("E2E_BASE") ?? "http://127.0.0.1:18090";
string backend = Environment.GetEnvironmentVariable("E2E_BACKEND") ?? "http://127.0.0.1:8080";
string model = Environment.GetEnvironmentVariable("E2E_MODEL") ?? "qwen3.8-gguf-single200";
string prompt = string.Join("\n", Enumerable.Range(0, 800).Select(i => $"line {i}: the quick brown fox jumps over the lazy dog"));
using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
var body = new JsonObject
{
    ["model"] = model,
    ["messages"] = new JsonArray(new JsonObject { ["role"] = "user", ["content"] = prompt + "\nReply OK." }),
    ["max_tokens"] = 8,
    ["stream"] = false,
};
var request = http.PostAsync($"{baseUrl}/v1/chat/completions",
    new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json"));
while (!request.IsCompleted)
{
    try
    {
        var raw = await http.GetStringAsync($"{backend}/slots");
        var slots = JsonNode.Parse(raw);
        if (slots is JsonArray arr)
            foreach (var s in arr)
            {
                int id = s?["id"]?.GetValue<int>() ?? -1;
                bool proc = s?["is_processing"]?.GetValue<bool>() ?? false;
                int np = s?["n_prompt_tokens"]?.GetValue<int>() ?? 0;
                int cached = s?["n_prompt_tokens_cache"]?.GetValue<int>() ?? 0;
                Console.WriteLine($"slot {id} proc={proc} np={np} cached={cached} inflight={Math.Max(0, np - cached)}");
            }
    }
    catch { }
    await Task.Delay(250);
}
await request;
Console.WriteLine("request done");
