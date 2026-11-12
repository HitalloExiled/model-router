using ModelRouter;
using Xunit;

public class SlotsTests
{
    [Fact] public void SameWeightsSameKey() => Assert.Equal(Slots.WeightsKey("gguf-range"), Slots.WeightsKey("gguf-max"));
    [Fact] public void DifferentFileDifferentKey() => Assert.NotEqual(Slots.WeightsKey("gguf-speed"), Slots.WeightsKey("gguf-range"));
    [Fact] public void FamiliesDiffer() => Assert.NotEqual(Slots.WeightsKey("gguf-range"), Slots.WeightsKey("bonsai-trio200"));
    [Fact] public void FileNameShape()
    {
        var n = Slots.FileName("gguf-duo", 0);
        Assert.EndsWith("-slot0.bin", n);
        Assert.DoesNotContain("/", n);
    }
    [Fact] public void RestoreCandidates()
    {
        var names = new[] { Slots.FileName("gguf-duo", 0), Slots.FileName("gguf-duo", 1), Slots.FileName("gguf-duo", 0), Slots.FileName("gguf-quad", 3), "unrelated.bin" };
        var got = Slots.RestoreCandidates("gguf-duo", names);
        Assert.Equal(new[] { (0, Slots.FileName("gguf-duo", 0)), (1, Slots.FileName("gguf-duo", 1)) }, got.ToArray());
    }
    [Fact] public void ToSaveFilters()
    {
        var got = Slots.ToSave(new[] { new SlotSaveInfo(2, false, 5000), new SlotSaveInfo(0, true, 9000), new SlotSaveInfo(1, false, 100) });
        Assert.Equal(new[] { 2 }, got);
    }
}
