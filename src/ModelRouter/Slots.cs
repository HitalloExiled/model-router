
namespace ModelRouter;

// Slot-cache helpers: a dump belongs to a weights file, not to a tier or a
// slot count. Mirrors python slot_weights_key / slot_file_name /
// slot_restore_candidates / slots_to_save / prune_slot_cache.
public static class Slots
{
    // Below this many tokens a re-prefill is cheaper than touching the disk.
    public static int SaveMin =>
        int.Parse(Environment.GetEnvironmentVariable("MODEL_ROUTER_SLOT_SAVE_MIN") ?? "2048");

    public static string WeightsKey(string profile) => Profiles.WeightsKey(profile);

    // Safe for the server's fs_validate_filename (no separators, no "..",
    // < 255 chars) and bound to the weights file.
    public static string FileName(string profile, int slotId)
        => $"{WeightsKey(profile)}-slot{slotId}.bin";

    // (slot id, filename) pairs to restore into a freshly booted profile.
    // Only slots the resident profile actually has are returned; duplicate
    // names collapse to their first occurrence (python dict.fromkeys).
    public static List<(int SlotId, string Name)> RestoreCandidates(string profile, IEnumerable<string> names)
    {
        if (!Profiles.Slots.TryGetValue(profile, out int nSlots))
            return new List<(int, string)>();
        var wanted = new Dictionary<string, int>();
        for (int i = 0; i < nSlots; i++)
            wanted[FileName(profile, i)] = i;
        var result = new List<(int SlotId, string Name)>();
        var seen = new HashSet<string>();
        foreach (var name in names)
        {
            if (!seen.Add(name)) continue;
            if (wanted.TryGetValue(name, out int id))
                result.Add((id, name));
        }
        return result;
    }

    // Slot ids worth persisting before a stop: idle and carrying a prompt.
    public static List<int> ToSave(IEnumerable<SlotSaveInfo>? slots)
    {
        var ids = new List<int>();
        if (slots is null) return ids;
        foreach (var s in slots)
        {
            if (s.Proc) continue;
            if ((s.PromptLen ?? 0) < SaveMin) continue;
            ids.Add(s.Id);
        }
        ids.Sort();
        return ids;
    }

    // Decide which files to unlink to get under the cap (oldest first). Pure:
    // the caller does the unlink.
    public static List<string> PrunePlan(
        IEnumerable<(string Path, long Size, DateTime Mtime)> files,
        long capBytes,
        ISet<string> keep)
    {
        var list = files.ToList();
        long total = list.Sum(f => f.Size);
        var plan = new List<string>();
        if (total <= capBytes) return plan;
        foreach (var f in list.OrderBy(f => f.Mtime))
        {
            if (total <= capBytes) break;
            if (keep.Contains(Path.GetFileName(f.Path))) continue;
            plan.Add(f.Path);
            total -= f.Size;
        }
        return plan;
    }

    // Cap the cache dir, oldest first. Unlinking needs write on the directory,
    // not on the file, so root-owned dumps written by the container are fine.
    public static void Prune(string dir, long capBytes, ISet<string> keep, Action<string>? log = null)
    {
        try
        {
            if (!Directory.Exists(dir)) return;
            var files = new List<(string, long, DateTime)>();
            foreach (var path in Directory.GetFiles(dir, "*.bin"))
            {
                var fi = new FileInfo(path);
                files.Add((path, fi.Length, fi.LastWriteTimeUtc));
            }
            foreach (var path in PrunePlan(files, capBytes, keep))
            {
                try
                {
                    File.Delete(path);
                    log?.Invoke($"slot cache pruned {Path.GetFileName(path)}");
                }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
            }
        }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }
}

public sealed record SlotSaveInfo(int Id, bool Proc, int? PromptLen);
