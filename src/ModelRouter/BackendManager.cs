using System.Diagnostics;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json.Nodes;

namespace ModelRouter;

// Manages the single resident backend (llama.cpp GGUF or Bonsai). Mirrors
// python: state dict + lock, backend_containers / healthy / stop /
// launch_command / start_backend / ensure_backend / idle_watch / adopt, plus
// slot persistence and the prefill admission gate.
//
// Lock model: the relay (BeginAsync/EndAsync) holds `Lock` while it does
// ensure + active accounting. The idle watcher holds it while deciding to
// unload. The low-level backend ops (Ensure/Stop/Start) assume `Lock` is
// already held and only mutate state / fire docker+process.
public sealed class BackendManager : IAsyncDisposable
{
    // ---- configuration (env, same defaults as python) ----

    private static string ResolveRouterDir()
    {
        var fromEnv = Environment.GetEnvironmentVariable("MODEL_ROUTER_DIR");
        if (!string.IsNullOrEmpty(fromEnv)) return fromEnv;
        // Walk up from the assembly location to the repo root (stacks/ or Makefile).
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "stacks"))
                || File.Exists(Path.Combine(dir.FullName, "Makefile")))
                return dir.FullName;
            dir = dir.Parent;
        }
        return Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            "Projetos", "My", "model-router-dotnet-aot");
    }

    public static readonly string RouterDir = ResolveRouterDir();

    private static string ResolveStackDir(string name)
    {
        // New layout: <RouterDir>/stacks/<name>; fall back to the legacy
        // <RouterDir>/<name> (e.g. an uninitialized stacks/ in a worktree)
        // and then a sibling checkout (~/Projetos/My/<name>).
        var inside = Path.Combine(RouterDir, "stacks", name);
        if (Directory.Exists(inside) && Directory.EnumerateFileSystemEntries(inside).Any())
            return inside;
        var legacy = Path.Combine(RouterDir, name);
        if (Directory.Exists(legacy) && Directory.EnumerateFileSystemEntries(legacy).Any())
            return legacy;
        var parent = Path.GetDirectoryName(Path.GetDirectoryName(RouterDir)) ?? RouterDir;
        return Path.Combine(parent, name);
    }

    public static readonly string GgufServe = Environment.GetEnvironmentVariable(
        "MODEL_ROUTER_LLAMA_SERVE_SH")
        ?? Path.Combine(ResolveStackDir("gguf-stack"), "serve-llama.sh");
    public static readonly string BonsaiServe = Environment.GetEnvironmentVariable(
        "MODEL_ROUTER_BONSAI_SERVE_SH")
        ?? Path.Combine(ResolveStackDir("bonsai-stack"), "serve-bonsai.sh");
    public static readonly string LlamaImage = Environment.GetEnvironmentVariable(
        "MODEL_ROUTER_LLAMA_IMAGE") ?? "ghcr.io/ggml-org/llama.cpp:server-cuda-b12110";
    public static readonly string BonsaiImage = Environment.GetEnvironmentVariable(
        "MODEL_ROUTER_BONSAI_IMAGE") ?? "bonsai-llama-4090:sm89";

    public static readonly int RouterPort =
        int.Parse(Environment.GetEnvironmentVariable("MODEL_ROUTER_PORT") ?? "18089");
    public static readonly int BackendPort =
        int.Parse(Environment.GetEnvironmentVariable("MODEL_ROUTER_BACKEND_PORT") ?? "8080");
    public static readonly int IdleSec =
        int.Parse(Environment.GetEnvironmentVariable("MODEL_ROUTER_IDLE_SEC") ?? "21600");
    public static readonly int BootTimeout = 420;

    // Boot log lives under ~/.local/share (NOT /tmp): the router may run in a
    // mount namespace with a private /tmp, which would hide the log.
    private static readonly string LogDir =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "model-router");
    public static readonly string BootLog = Path.Combine(LogDir, "backend-boot.log");
    private static readonly string[] BackendImages = { LlamaImage, BonsaiImage };

    // ---- slot persistence ----
    public static readonly string SlotDir = Environment.GetEnvironmentVariable(
        "MODEL_ROUTER_SLOT_DIR") ?? Path.Combine(RouterDir, "slot-cache");
    private static readonly bool slotPersist = EnvironmentFlag("MODEL_ROUTER_SLOT_PERSIST", "1");
    private static readonly int slotDiskMib =
        int.Parse(Environment.GetEnvironmentVariable("MODEL_ROUTER_SLOT_DISK_MIB") ?? "32768");
    private static readonly int slotIoTimeout =
        int.Parse(Environment.GetEnvironmentVariable("MODEL_ROUTER_SLOT_IO_TIMEOUT") ?? "300");

    // ---- prefill admission gate ----
    private static readonly bool prefillGate = EnvironmentFlag("MODEL_ROUTER_PREFILL_GATE", "1");
    private static readonly bool prefillGateEnforce = EnvironmentFlag("MODEL_ROUTER_PREFILL_GATE_ENFORCE", "1");
    private static readonly double prefillGateTimeout =
        double.Parse(Environment.GetEnvironmentVariable("MODEL_ROUTER_PREFILL_GATE_TIMEOUT") ?? "10");
    private static readonly double prefillGatePoll =
        double.Parse(Environment.GetEnvironmentVariable("MODEL_ROUTER_PREFILL_GATE_POLL") ?? "0.5");
    private static readonly bool prefillGateCold = EnvironmentFlag("MODEL_ROUTER_PREFILL_GATE_COLD", "0");
    private static readonly int prefillGateColdMin =
        int.Parse(Environment.GetEnvironmentVariable("MODEL_ROUTER_PREFILL_GATE_COLD_MIN") ?? "8192");
    private static readonly bool prefillGateDiag = EnvironmentFlag("MODEL_ROUTER_PREFILL_GATE_DIAG", "1");
    private static readonly bool prefillGateDiagVerbose = EnvironmentFlag("MODEL_ROUTER_PREFILL_GATE_DIAG_VERBOSE", "0");

    private static bool EnvironmentFlag(string name, string def)
        => (Environment.GetEnvironmentVariable(name) ?? def) != "0";

    // ---- state (mirrors python `state` + `lock`) ----
    private readonly SemaphoreSlim _lock = new(1, 1);
    private string? _profile;
    private bool _profileMmproj;
    private bool _busy;
    private DateTime _lastUsed = new DateTime(1970, 1, 1);
    private int _active;
    private HttpClient? _http;
    private HttpClient? _slotHttp;

    public string? Profile => _profile;
    public bool ProfileMmproj => _profileMmproj;
    public bool Busy => _busy;

    public SemaphoreSlim Lock => _lock;

    private HttpClient HttpClient =>
        _http ??= new HttpClient { Timeout = TimeSpan.FromSeconds(3) };

    private HttpClient SlotHttp =>
        _slotHttp ??= new HttpClient { Timeout = TimeSpan.FromSeconds(slotIoTimeout) };

    // ---- container / health discovery (lock-free, external only) ----

    public async Task<List<string>> GetBackendContainersAsync()
    {
        var (rc, stdout) = await RunAsync(new[]
        { "docker", "ps", "--format", "{{.ID}} {{.Image}}" }, 60);
        if (rc != 0) return new();
        var found = new List<string>();
        foreach (var line in stdout.Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length >= 2 && BackendImages.Contains(parts[1]))
                found.Add(parts[0]);
        }
        return found;
    }

    public async Task<string?> GetBackendContainerAsync()
    {
        var ids = await GetBackendContainersAsync();
        return ids.Count > 0 ? ids[^1] : null;
    }

    public async Task<bool> IsBackendHealthyAsync()
    {
        try
        {
            using var resp = await HttpClient.GetAsync($"http://127.0.0.1:{BackendPort}/health", CancellationToken.None);
            return resp.StatusCode == System.Net.HttpStatusCode.OK;
        }
        catch
        {
            return false;
        }
    }

    // ---- low-level ops (assume Lock held) ----

    private async Task DoStopBackendAsync()
    {
        // Warm slots are saved before the stop: a restart otherwise costs a
        // full re-prefill. Best effort, skipped entirely when the backend is
        // already gone.
        if (await IsBackendHealthyAsync())
            await SaveSlotCacheAsync();
        var ids = await GetBackendContainersAsync();
        if (ids.Count > 0)
        {
            Log($"stopping backend {string.Join(" ", ids)}");
            var cmd = new[] { "docker", "stop" }.Concat(ids).ToArray();
            await RunAsync(cmd, 120);
        }
        _profile = null;
        _profileMmproj = false;
        PrefixCache.Clear();
    }

    public async Task StopBackendAsync()
    {
        await _lock.WaitAsync();
        try { await DoStopBackendAsync(); }
        finally { _lock.Release(); }
    }

    // Returns the argv for a profile, or null when the launcher is not wired.
    public List<string>? LaunchCommand(string profile, bool wantMmproj)
    {
        if (Profiles.BonsaiFiles.ContainsKey(profile))
        {
            if (!File.Exists(BonsaiServe)) return null;
            var args = new List<string>
            {
                BonsaiServe, "--profile", RemovePrefix(profile, "bonsai-"),
                "--port", BackendPort.ToString(),
            };
            if (wantMmproj) { args.Add("--mmproj"); args.Add("on"); }
            return args;
        }
        if (Profiles.GgufFiles.ContainsKey(profile))
        {
            if (!File.Exists(GgufServe)) return null;
            var args = new List<string>
            {
                GgufServe, "--profile", RemovePrefix(profile, "gguf-"),
                "--port", BackendPort.ToString(),
            };
            if (wantMmproj) { args.Add("--mmproj"); args.Add("on"); }
            return args;
        }
        return null;
    }

    // Mirror of python's str.removeprefix: only strips the prefix when it is at
    // the start. This build has no 3-arg Replace.
    private static string RemovePrefix(string s, string p) =>
        s.StartsWith(p, StringComparison.Ordinal) ? s.Substring(p.Length) : s;

    private async Task<bool> DoStartBackendAsync(string profile, bool wantMmproj)
    {
        Log($"starting profile {profile} want_mmproj={wantMmproj}");
        Directory.CreateDirectory(LogDir);
        var argv = LaunchCommand(profile, wantMmproj);
        if (argv is null || argv.Count == 0)
        {
            Log($"profile {profile} has no launcher wired yet");
            return false;
        }

        var psi = new ProcessStartInfo
        {
            FileName = argv[0],
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (var arg in argv.Skip(1)) psi.ArgumentList.Add(arg);
        // Both serve scripts mount this dir and pass --slot-save-path, so a
        // warm slot survives the restart.
        psi.Environment["MODEL_ROUTER_SLOT_CACHE_DIR"] = SlotDir;

        var proc = new Process { StartInfo = psi };

        // Launcher output goes to the boot log (evidence, never DEVNULL).
        // This build's FileStream.WriteAsync only accepts ReadOnlyMemory<byte>,
        // so we relay through a StreamWriter (string overload).
        using var logWriter = new StreamWriter(File.Create(BootLog));
        proc.Start();
        var outReader = proc.StandardOutput;
        var errReader = proc.StandardError;
        _ = Task.Run(async () =>
        {
            try
            {
                string? line;
                while ((line = await outReader.ReadLineAsync()) is not null)
                    await logWriter.WriteAsync(line + Environment.NewLine);
            }
            catch { }
            try
            {
                string? line;
                while ((line = await errReader.ReadLineAsync()) is not null)
                    await logWriter.WriteAsync(line + Environment.NewLine);
            }
            catch { }
        });

        var t0 = DateTime.UtcNow;
        while (DateTime.UtcNow - t0 < TimeSpan.FromSeconds(BootTimeout))
        {
            if (proc.HasExited)
            {
                Log($"launcher for {profile} exited rc={proc.ExitCode} (see {BootLog}); aborting boot");
                return false;
            }
            if (await IsBackendHealthyAsync())
            {
                _profile = profile;
                _profileMmproj = wantMmproj;
                Log($"profile {profile} listening");
                await RestoreSlotCacheAsync(profile);
                return true;
            }
            await Task.Delay(3000);
        }
        Log($"profile {profile} failed to listen in {BootTimeout}s");
        return false;
    }

    // Mirrors python ensure_backend. Assumes Lock held.
    public async Task<bool> EnsureBackendAsync(string profile, bool wantMmproj)
    {
        var name = _profile;
        var curMmproj = _profileMmproj;
        var container = await GetBackendContainerAsync();
        var healthy = await IsBackendHealthyAsync();
        var existing = container is not null || healthy;

        bool same = Routing.SameBackend(name, curMmproj, profile, wantMmproj);
        if (same && existing)
            return true;

        if (existing || _profile is not null)
            await DoStopBackendAsync();

        // Wait for the port to actually free (teardown races).
        var t0 = DateTime.UtcNow;
        while (DateTime.UtcNow - t0 < TimeSpan.FromSeconds(120))
        {
            if (await IsBackendHealthyAsync())
                await Task.Delay(2000);
            else
                break;
        }
        return await DoStartBackendAsync(profile, wantMmproj);
    }

    // ---- relay-facing API (takes/releases the lock itself) ----

    // Mirrors the relay's `with lock:` open block. Returns false (=> 502) if
    // the wanted profile cannot be made to serve.
    public async Task<bool> BeginAsync(string? profile, bool wantMmproj)
    {
        await _lock.WaitAsync();
        try
        {
            if (profile is not null && Profiles.ProfilesSet.Contains(profile))
            {
                _busy = true;
                try
                {
                    if (!await EnsureBackendAsync(profile, wantMmproj))
                        return false;
                }
                finally
                {
                    _busy = false;
                }
            }
            _active += 1;
            return true;
        }
        finally
        {
            _lock.Release();
        }
    }

    // Mirrors the relay's `finally` block.
    public async Task EndAsync()
    {
        await _lock.WaitAsync();
        try
        {
            _lastUsed = DateTime.UtcNow;
            _active = Math.Max(0, _active - 1);
        }
        finally
        {
            _lock.Release();
        }
    }

    // ---- idle watch (mirrors python idle_watch: one lock hold covers the
    // decision and the stop, so no request can sneak in between) ----
    public Task IdleWatchAsync(CancellationToken ct)
    {
        var timer = new PeriodicTimer(TimeSpan.FromSeconds(30));
        return Task.Run(async () =>
        {
            try
            {
                while (!ct.IsCancellationRequested)
                {
                    if (await timer.WaitForNextTickAsync(ct))
                    {
                        await _lock.WaitAsync(ct);
                        try
                        {
                            bool unload = _profile is not null && _active == 0
                                && DateTime.UtcNow - _lastUsed > TimeSpan.FromSeconds(IdleSec);
                            if (unload)
                            {
                                Log($"idle {IdleSec}s: unloading {_profile}");
                                await DoStopBackendAsync();
                            }
                        }
                        finally
                        {
                            _lock.Release();
                        }
                    }
                }
            }
            catch (OperationCanceledException) { }
        }, ct);
    }

    // ---- prefill gate ----

    public sealed record SlotStateSummary(bool Prefilling, bool Generating, List<Gate.SlotState> Slots);

    public async Task<SlotStateSummary?> BackendSlotStateAsync()
    {
        try
        {
            using var resp = await HttpClient.GetAsync(
                $"http://127.0.0.1:{BackendPort}/slots", CancellationToken.None);
            if (resp.StatusCode != System.Net.HttpStatusCode.OK) return null;
            using var stream = await resp.Content.ReadAsStreamAsync();
            var raw = await ReadUpToAsync(stream, 4 * 1024 * 1024);
            var node = JsonNode.Parse(Encoding.UTF8.GetString(raw));
            if (node is not JsonArray arr) return null;
            var slots = new List<Gate.SlotState>();
            foreach (var s in arr)
            {
                var phase = Gate.SlotPhase(s);
                if (phase is not null) slots.Add(phase);
            }
            return new SlotStateSummary(
                slots.Any(x => x.Prefilling),
                slots.Any(x => x.Generating),
                slots);
        }
        catch
        {
            return null;
        }
    }

    // Hold a request while another slot prefills (and optionally while a slot
    // decodes a cold large prompt). Best effort: never fails a request.
    public async Task AdmitPrefillAsync(JsonObject decoded, byte[]? body, string? profile)
    {
        if (!prefillGate) return;
        try
        {
            double est = (body?.Length ?? 0) / 3.5;
            bool hot = PrefixCache.IsHot(Gate.PromptPrefixKey(decoded, profile));
            bool coldLarge = prefillGateCold && !hot && est >= prefillGateColdMin;
            var phase = await BackendSlotStateAsync();
            if (phase is null)
            {
                if (prefillGateDiag && prefillGateDiagVerbose)
                    Log($"prefill gate diag slots=unavailable est={(int)est}");
                return;
            }
            string? reason = Gate.GateDecision(phase.Prefilling, phase.Generating, coldLarge);
            if (prefillGateDiag && (reason is not null || prefillGateDiagVerbose))
                Log($"prefill gate diag est={(int)est} hot={hot} cold={coldLarge} "
                    + $"prefilling={phase.Prefilling} generating={phase.Generating} "
                    + $"would_block={reason is not null} reason={reason ?? "pass"}");
            if (reason is not null && prefillGateEnforce)
            {
                var started = DateTime.UtcNow;
                bool waited = false;
                while (reason is not null)
                {
                    if ((DateTime.UtcNow - started).TotalSeconds >= prefillGateTimeout)
                    {
                        Log($"prefill gate timeout reason={reason} cold={coldLarge} est={(int)est}");
                        break;
                    }
                    if (!waited)
                    {
                        Log($"prefill gate wait reason={reason} est={(int)est}");
                        waited = true;
                    }
                    await Task.Delay(TimeSpan.FromSeconds(prefillGatePoll));
                    phase = await BackendSlotStateAsync();
                    if (phase is null) break;
                    reason = Gate.GateDecision(phase.Prefilling, phase.Generating, coldLarge);
                }
                if (waited)
                    Log($"prefill gate admit waited={(DateTime.UtcNow - started).TotalSeconds:F1}s");
            }
        }
        catch (Exception e)
        {
            Log($"prefill gate skipped: {e.Message}");
        }
        finally
        {
            try { PrefixCache.Remember(Gate.PromptRequestKey(decoded, profile)); } catch { }
        }
    }

    // ---- slot persistence ----

    private async Task SaveSlotCacheAsync()
    {
        if (!slotPersist) return;
        try
        {
            string? profile = _profile;
            var phases = await BackendSlotStateAsync();
            if (phases is null) return;
            var infos = new List<SlotSaveInfo>();
            foreach (var s in phases.Slots)
                infos.Add(new SlotSaveInfo(s.Id, s.Proc, s.PromptLen));
            var slots = Slots.ToSave(infos);
            if (slots.Count == 0 || profile is null || !Profiles.ProfilesSet.Contains(profile))
                return;
            Directory.CreateDirectory(SlotDir);
            long free = FreeBytes(SlotDir);
            if (free < 8L * 1024 * 1024 * 1024)
            {
                Log($"slot save skipped: only {free / (1024 * 1024 * 1024)} GiB free");
                return;
            }
            var kept = new HashSet<string>();
            foreach (var sid in slots)
            {
                string name = Slots.FileName(profile, sid);
                var (ok, raw) = await SlotActionAsync("save", sid, name);
                if (!ok)
                {
                    Log($"slot {sid} save failed: {Preview(raw)}");
                    continue;
                }
                kept.Add(name);
                long mib = 0;
                try
                {
                    var fi = new FileInfo(Path.Combine(SlotDir, name));
                    if (fi.Exists) mib = fi.Length / (1024 * 1024);
                }
                catch { }
                Log($"slot {sid} saved to {name} ({mib} MiB)");
            }
            Slots.Prune(SlotDir, slotDiskMib * 1024L * 1024L, kept, msg => Log(msg));
        }
        catch (Exception e)
        {
            Log($"slot save skipped: {e.Message}");
        }
    }

    private async Task RestoreSlotCacheAsync(string profile)
    {
        if (!slotPersist || !Profiles.ProfilesSet.Contains(profile)) return;
        string[] names;
        try
        {
            names = Directory.GetFiles(SlotDir)
                .Select(Path.GetFileName)
                .Where(n => n is not null)
                .Select(n => n!)
                .ToArray();
        }
        catch
        {
            return;
        }
        try
        {
            foreach (var (sid, name) in Slots.RestoreCandidates(profile, names))
            {
                var (ok, raw) = await SlotActionAsync("restore", sid, name);
                if (ok) Log($"slot {sid} restored from {name}");
                else Log($"slot {sid} restore failed: {Preview(raw)}");
            }
        }
        catch (Exception e)
        {
            Log($"slot restore skipped: {e.Message}");
        }
    }

    private async Task<(bool Ok, byte[] Raw)> SlotActionAsync(string action, int slotId, string filename)
    {
        try
        {
            var json = new JsonObject { ["filename"] = filename }.ToJsonString();
            var content = new ByteArrayContent(Encoding.UTF8.GetBytes(json));
            content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
            using var req = new HttpRequestMessage(
                HttpMethod.Post, $"http://127.0.0.1:{BackendPort}/slots/{slotId}?action={action}")
            { Content = content };
            using var resp = await SlotHttp.SendAsync(req, HttpCompletionOption.ResponseHeadersRead);
            using var stream = await resp.Content.ReadAsStreamAsync();
            var raw = await ReadUpToAsync(stream, 65536);
            return (resp.StatusCode == System.Net.HttpStatusCode.OK, raw);
        }
        catch
        {
            return (false, Array.Empty<byte>());
        }
    }

    // ---- adopt (infer profile from a running container on start) ----
    public async Task AdoptAsync()
    {
        var cid = await GetBackendContainerAsync();
        if (cid is null) return;
        var (rc, args) = await RunAsync(new[] { "docker", "inspect", "-f",
            "{{join .Args \" \"}}", cid }, 60);
        if (rc != 0) return;
        var prof = Infer.InferProfile(args);
        if (prof is null) return;
        await _lock.WaitAsync();
        try
        {
            _profile = prof;
            _profileMmproj = args.Contains("mmproj");
        }
        finally
        {
            _lock.Release();
        }
        Log($"adopted running backend {cid} as {_profile} mmproj={_profileMmproj}");
    }

    // ---- dispose ----
    public async ValueTask DisposeAsync()
    {
        await StopBackendAsync();
        _http?.Dispose();
        _slotHttp?.Dispose();
    }

    // ---- helpers ----
    private async Task<(int, string)> RunAsync(string[] cmd, int timeoutSeconds)
    {
        var psi = new ProcessStartInfo { FileName = cmd[0], UseShellExecute = false,
            RedirectStandardOutput = true, RedirectStandardError = true };
        foreach (var arg in cmd.Skip(1)) psi.ArgumentList.Add(arg);
        var proc = new Process { StartInfo = psi };
        try
        {
            proc.Start();
            var outputTask = proc.StandardOutput.ReadToEndAsync();
            var errTask = proc.StandardError.ReadToEndAsync();
            bool exited = await Task.Run(() => proc.WaitForExit(TimeSpan.FromSeconds(timeoutSeconds)));
            if (!exited)
            {
                try { proc.Kill(); } catch { }
                return (-1, string.Empty);
            }
            var stdout = await outputTask;
            var errout = await errTask;
            return (proc.ExitCode, stdout);
        }
        catch (Exception)
        {
            return (-1, string.Empty);
        }
        finally
        {
            try { proc.Kill(); } catch { }
            try { proc.Dispose(); } catch { }
        }
    }

    private static async Task<byte[]> ReadUpToAsync(Stream stream, int max)
    {
        var ms = new MemoryStream();
        var buf = new byte[Math.Min(8192, max)];
        int total = 0;
        int n;
        while (total < max && (n = await stream.ReadAsync(buf, 0, buf.Length)) > 0)
        {
            int toCopy = Math.Min(n, max - total);
            ms.Write(buf, 0, toCopy);
            total += toCopy;
        }
        return ms.ToArray();
    }

    private static long FreeBytes(string dir)
    {
        try
        {
            var root = Path.GetPathRoot(dir) ?? "/";
            return new DriveInfo(root).AvailableFreeSpace;
        }
        catch
        {
            return long.MaxValue;
        }
    }

    private static string Preview(byte[] raw)
    {
        string s = Encoding.UTF8.GetString(raw);
        return s.Length <= 160 ? s : s[..160];
    }

    private static void Log(string msg)
        => Console.WriteLine($"[router {DateTime.Now:HH:mm:ss}] {msg}");
}
