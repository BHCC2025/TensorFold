//! The family sets in the full kernel contract: Nemotron's pocket, MiniMax-M3's, the dense EXL3 projections'.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const g = @import("core").contract.gates;

pub const Sets = struct {
    pocket: *@import("contract_pocket.zig").Set,
    minimax: *@import("contract_minimax.zig").Set,
    exl3: *@import("contract_exl3.zig").Set,
    a: std.mem.Allocator,

    /// Each set's kernels appended to `list`, in this order.
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !Sets {
        const pocket = try @import("contract_pocket.zig").Set.init(gpu, s, a, list);
        errdefer pocket.deinit();
        const minimax = try @import("contract_minimax.zig").Set.init(gpu, s, a, list);
        errdefer minimax.deinit();
        const exl3 = try a.create(@import("contract_exl3.zig").Set);
        exl3.* = .{};
        errdefer {
            exl3.deinit();
            a.destroy(exl3);
        }
        try exl3.add(gpu, s, list);
        return .{ .pocket = pocket, .minimax = minimax, .exl3 = exl3, .a = a };
    }

    pub fn deinit(sets: Sets) void {
        sets.exl3.deinit();
        sets.a.destroy(sets.exl3);
        sets.minimax.deinit();
        sets.pocket.deinit();
    }
};
