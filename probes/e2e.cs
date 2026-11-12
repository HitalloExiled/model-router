using System.Net.Http.Headers;
using System.Text;
using System.Text.Json.Nodes;

string baseUrl = Environment.GetEnvironmentVariable("E2E_BASE") ?? "http://127.0.0.1:18090";
string? model = Environment.GetEnvironmentVariable("E2E_MODEL");
using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };

try
{
    using var health = await http.GetAsync($"{baseUrl}/health");
    Console.WriteLine($"health: {(int)health.StatusCode}");
    if (!health.IsSuccessStatusCode) return 1;
}
catch (Exception ex)
{
    Console.WriteLine($"health failed: {ex.Message}");
    return 1;
}

model ??= await FirstModel(http, baseUrl);
Console.WriteLine($"model: {model}");

var body = new JsonObject
{
    ["model"] = model,
    ["messages"] = new JsonArray(new JsonObject { ["role"] = "user", ["content"] = "Reply with OK." }),
    ["max_tokens"] = 8,
    ["stream"] = true,
};
using var request = new HttpRequestMessage(HttpMethod.Post, $"{baseUrl}/v1/chat/completions")
{
    Content = new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json"),
};
request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));
using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
Console.WriteLine($"chat: {(int)response.StatusCode} {response.Content.Headers.ContentType}");
if (!response.IsSuccessStatusCode) return 1;
if (response.Content.Headers.ContentType?.MediaType != "text/event-stream")
{
    Console.WriteLine("expected text/event-stream");
    return 1;
}

using var stream = await response.Content.ReadAsStreamAsync();
using var reader = new StreamReader(stream);
int dataLines = 0;
bool done = false;
while (await reader.ReadLineAsync() is { } line)
{
    if (!line.StartsWith("data:")) continue;
    dataLines++;
    if (line.Contains("[DONE]"))
    {
        done = true;
        break;
    }
}
Console.WriteLine($"sse data={dataLines} done={done}");
return dataLines > 0 && done ? 0 : 1;

static async Task<string> FirstModel(HttpClient http, string baseUrl)
{
    try
    {
        string raw = await http.GetStringAsync($"{baseUrl}/v1/models");
        var node = JsonNode.Parse(raw);
        if (node?["data"] is JsonArray arr && arr.Count > 0)
        {
            string? id = arr[0]?["id"]?.GetValue<string>();
            if (!string.IsNullOrEmpty(id)) return id;
        }
    }
    catch { }
    return "qwen3.8-gguf-single200";
}
