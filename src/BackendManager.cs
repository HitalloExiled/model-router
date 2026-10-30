using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace ModelRouter;

// Manages the single resident backend (ninfer / llama.cpp / Bonsai).
// Mirrors python: state dict + lock, backend_containers / healthy / stop /
// launch_command / start_backend / ensure_backend / idle_watch / adopt.
//
// Lock model: the relay (BeginAsync/EndAsync) holds `Lock` while it does
// ensure + active accounting. The idle watcher holds it while deciding to
// unload. The low-level backend ops (Ensure/Stop/Start) assume `Lock` is
// already held and only mutate state / fire docker+process — mirroring the
// python, where the lock is global and only the relay and idle_watch take it.
public sealed class BackendManager : IAsyncDisposable
{
    // ---- configuration (env, same defaults as python) ----

    private static string ResolveStackDir(string name)
    {
        // Inside the router repo first (submodule), then a sibling checkout
        // (~/Projetos/My/<name>), mirroring python stack_dir.
        var routerDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            "Projetos", "My", "model-router");
        var inside = Path.Combine(routerDir, name);
        if (Directory.Exists(inside)) return inside;
        var parent = Path.GetDirectoryName(Path.GetDirectoryName(routerDir)) ?? routerDir;
        return Path.Combine(parent, name);
    }

    public static readonly string Serve = Environment.GetEnvironmentVariable(
        "NINFER_SERVE_SH") ?? Path.Combine(ResolveStackDir("ninfer-stack"), "serve.sh");
    public static readonly string NInferModelsDir =
        Environment.GetEnvironmentVariable("NINFER_MODEL_DIR") ??
        Path.Combine(ResolveStackDir("ninfer-stack"), "models");
    public static readonly string ModelFile =
        Path.Combine(NInferModelsDir, "qwen3_8_27b.ninfer");
    public static readonly string VisionFile =
        Path.Combine(NInferModelsDir, "qwen3_6_35b_a3b.ninfer");
    public static readonly string Image =
        Environment.GetEnvironmentVariable("NINFER_DOCKER_IMAGE") ?? "ninfer-4090:sm89";
    public static readonly string LlamaImage = Environment.GetEnvironmentVariable(
        "NINFER_LLAMA_IMAGE") ?? "ghcr.io/ggml-org/llama.cpp:server-cuda-b11854";
    public static readonly string GgufDir = Environment.GetEnvironmentVariable(
        "NINFER_GGUF_DIR") ?? Path.Combine(ResolveStackDir("gguf-stack"), "models");
    public static readonly string LlamaServe = Environment.GetEnvironmentVariable(
        "LLAMA_SERVE_SH") ?? Path.Combine(ResolveStackDir("gguf-stack"), "serve-llama.sh");
    public static readonly string BonsaiImage = Environment.GetEnvironmentVariable(
        "NINFER_BONSAI_IMAGE") ?? "bonsai-llama-4090:sm89";
    public static readonly string BonsaiDir = Environment.GetEnvironmentVariable(
        "NINFER_BONSAI_DIR") ?? Path.Combine(ResolveStackDir("bonsai-stack"), "models");
    public static readonly string BonsaiServe = Environment.GetEnvironmentVariable(
        "BONSAI_SERVE_SH") ?? Path.Combine(ResolveStackDir("bonsai-stack"), "serve-bonsai.sh");

    public static readonly int RouterPort =
        int.Parse(Environment.GetEnvironmentVariable("MODEL_ROUTER_PORT") ?? "18089");
    public static readonly int BackendPort =
        int.Parse(Environment.GetEnvironmentVariable("NINFER_BACKEND_PORT") ?? "8080");
    public static readonly int IdleSec =
        int.Parse(Environment.GetEnvironmentVariable("MODEL_ROUTER_IDLE_SEC") ?? "900");
    public static readonly int BootTimeout = 420;

    private static readonly string LogDir =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ninfer");
    public static readonly string BootLog = Path.Combine(LogDir, "ninfer-backend-boot.log");
    private static readonly string[] BackendImages = { Image, LlamaImage, BonsaiImage };

    // ---- state (mirrors python `state` + `lock`) ----
    private readonly SemaphoreSlim _lock = new(1, 1);
    private string? _profile;
    private bool _profileMmproj;
    private bool _busy;
    private DateTime _lastUsed = new DateTime(1970, 1, 1);
    private int _active;
    private HttpClient? _http;

    public string? Profile => _profile;
    public bool ProfileMmproj => _profileMmproj;
    public bool Busy => _busy;

    public SemaphoreSlim Lock => _lock;

    private HttpClient HttpClient =>
        _http ??= new HttpClient { Timeout = TimeSpan.FromSeconds(3) };

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
        var ids = await GetBackendContainersAsync();
        if (ids.Count > 0)
        {
            Log($"stopping backend {string.Join(" ", ids)}");
            var cmd = new[] { "docker", "stop" }.Concat(ids).ToArray();
            await RunAsync(cmd, 120);
            _profile = null;
            _profileMmproj = false;
        }
    }

    public async Task StopBackendAsync()
    {
        await _lock.WaitAsync();
        try { await DoStopBackendAsync(); }
        finally { _lock.Release(); }
    }

    public string? LaunchCommand(string profile, bool wantMmproj)
    {
        if (Profiles.BonsaiAll.Contains(profile))
        {
            if (!File.Exists(BonsaiServe)) return null;
            var args = new[] { BonsaiServe, "--profile", RemovePrefix(profile, "bonsai-"),
                "--port", BackendPort.ToString() };
            if (wantMmproj) args = args.Concat(new[] { "--mmproj", "on" }).ToArray();
            return string.Join(" ", args);
        }
        if (Profiles.GgufAll.Contains(profile))
        {
            if (!File.Exists(LlamaServe)) return null;
            var args = new[] { LlamaServe, "--profile", RemovePrefix(profile, "gguf-"),
                "--port", BackendPort.ToString() };
            if (wantMmproj) args = args.Concat(new[] { "--mmproj", "on" }).ToArray();
            return string.Join(" ", args);
        }
        if (Profiles.ProxiesSet.Contains(profile))
        {
            var model = profile == "vision" ? VisionFile : ModelFile;
            return string.Join(" ", new[] { Serve, model, "--profile", profile,
                "--port", BackendPort.ToString() });
        }
        return null;
    }

    // Mirror of python's str.removeprefix: only strips the prefix when it is at
    // the start. This .NET build has no 3-arg Replace, so this is the faithful
    // replacement (exact same semantics, AOT-safe).
    private static string RemovePrefix(string s, string p) =>
        s.StartsWith(p, StringComparison.Ordinal) ? s.Substring(p.Length) : s;

    private async Task<bool> DoStartBackendAsync(string profile, bool wantMmproj)
    {
        Log($"starting profile {profile} want_mmproj={wantMmproj}");
        Directory.CreateDirectory(LogDir);
        var argv = LaunchCommand(profile, wantMmproj);
        if (argv is null)
        {
            Log($"profile {profile} has no launcher wired yet");
            return false;
        }

        var tokens = argv.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var psi = new ProcessStartInfo
        {
            FileName = tokens[0],
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (var arg in tokens.Skip(1)) psi.ArgumentList.Add(arg);

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
            if (profile is not null && Profiles.ProxiesSet.Contains(profile))
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

    private static void Log(string msg)
        => Console.WriteLine($"[router {DateTime.Now:HH:mm:ss}] {msg}");
}