using ModelRouter;

var app = WebApplication.CreateBuilder().Build();

app.MapGet("/health", () => "ok");

app.Run();
