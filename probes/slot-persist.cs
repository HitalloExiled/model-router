using System.Text;
using System.Text.Json.Nodes;

string backend = Environment.GetEnvironmentVariable("E2E_BACKEND") ?? "http://127.0.0.1:8080";
int slot = int.Parse(Environment.GetEnvironmentVariable("E2E_SLOT") ?? "0");
string file = Environment.GetEnvironmentVariable("E2E_SLOT_FILE") ?? "probe-slot.bin";
using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
foreach (var action in new[] { "save", "restore" })
{
    var body = new JsonObject { ["filename"] = file };
    var resp = await http.PostAsync($"{backend}/slots/{slot}?action={action}",
        new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json"));
    string raw = await resp.Content.ReadAsStringAsync();
    Console.WriteLine($"{action} slot={slot} file={file} status={(int)resp.StatusCode} {raw[..Math.Min(raw.Length, 200)]}");
}
