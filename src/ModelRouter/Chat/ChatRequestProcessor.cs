using System.Text;
using System.Text.Json.Nodes;
using ModelRouter.Http;

namespace ModelRouter.Chat;

public sealed record ChatPlan(string? Profile, bool WantMmproj, byte[] Body);

public sealed class ChatRequestProcessor
{
    private readonly BackendManager _backend;
    public ChatRequestProcessor(BackendManager backend) => _backend = backend;

    public async Task<ChatPlan> ProcessAsync(string path, HttpRequest request)
    {
        byte[] body;
        try { body = await RequestBodyReader.ReadAsync(request); }
        catch { body = Array.Empty<byte>(); }

        string? profile = null;
        bool wantMmproj = false;
        byte[] forwardBody = body;
        try
        {
            JsonNode? decoded = JsonNode.Parse(body.Length > 0 ? Encoding.UTF8.GetString(body) : "{}");
            var route = Routing.For(decoded);
            bool needVision = Media.Has(decoded);
            double est = body.Length > 0 ? (double)body.Length / 3.5 : 0.0;
            string requested = route.Profile;
            string? current = _backend.Profile;
            string reused = Routing.EffectiveProfile(requested, current, est, needVision);
            profile = reused;
            var norm = Effort.Normalize(path, body, route, reused);
            forwardBody = norm.NewBody;
            wantMmproj = route.WantMmproj || norm.WantMmproj;
            if (needVision && Profiles.LlamaAll.Contains(profile))
                wantMmproj = true;
        }
        catch
        {
            profile = null;
            wantMmproj = false;
            forwardBody = body;
        }
        return new ChatPlan(profile, wantMmproj, forwardBody);
    }
}
