const std = @import("std");

// zig build                         build both programs into zig-out/bin
// zig build run-spam  -- data/sms.tsv "free prize, call now"
// zig build run-house -- data/housing.csv
// zig build -Doptimize=Debug        slow debug build with safety checks

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});

    // Fast by default.
    const optimize_mode = b.option(
        std.builtin.OptimizeMode,
        "optimize",
        "Debug, ReleaseSafe, ReleaseFast (default), ReleaseSmall",
    ) orelse .ReleaseFast;

    inline for (.{ "spam", "house" }) |program_name| {
        const executable = b.addExecutable(.{
            .name = program_name,
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/" ++ program_name ++ ".zig"),
                .target = target,
                .optimize = optimize_mode,
            }),
        });
        b.installArtifact(executable);

        const run_command = b.addRunArtifact(executable);
        if (b.args) |program_arguments| run_command.addArgs(program_arguments);
        b.step("run-" ++ program_name, "Run " ++ program_name).dependOn(&run_command.step);
    }
}
