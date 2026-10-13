//! A TP rank's GLM-5.3-Flash weights from an EXL3 checkpoint: matrices cut at 128s, KDA and MoE as EXL3 has them.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const shard = @import("cuda_shard.zig");
const W = @import("cuda_weights.zig");

/// EXL3 matrix `name`'s share: inputs [klo, khi) (khi 0: all), `spans`' columns; tiles to `tiles`, suh+svh `scales`.
pub fn gather(ck: *core.Checkpoint, gpa: std.mem.Allocator, name: []const u8, klo: usize, khi: usize, spans: []const shard.Span, tiles: *std.ArrayList(u8), scales: *std.ArrayList(u8)) !cuda.exl3_linear.Matrix {
    var nb: [192]u8 = undefined;
    const t = try ck.get(try std.fmt.bufPrint(&nb, "{s}.trellis", .{name}));
    const suh = try ck.get(try std.fmt.bufPrint(&nb, "{s}.suh", .{name}));
    const svh = try ck.get(try std.fmt.bufPrint(&nb, "{s}.svh", .{name}));
    const book: cuda.exl3_linear.Codebook = if (ck.has(try std.fmt.bufPrint(&nb, "{s}.mul1", .{name}))) .mul1 else if (ck.has(try std.fmt.bufPrint(&nb, "{s}.mcg", .{name}))) .mcg else .inst3;
    if (t.dtype != .i16 or t.rank != 3 or t.dim(2) % 16 != 0) return error.UnexpectedTensor;
    const tn = t.dim(1);
    const words = t.dim(2);
    const k_all = t.dim(0) * 16;
    const k_hi = if (khi == 0) k_all else khi;
    if (klo % 128 != 0 or k_hi % 128 != 0 or k_hi > k_all) return error.Exl3RowCut;
    var n: usize = 0;
    for (spans) |c| {
        if (c.lo % 128 != 0 or c.hi % 128 != 0 or c.hi > tn * 16) return error.Exl3ColumnCut;
        n += c.len();
    }
    // trellis tiles: each 16-row group of the rank's inputs, its spans' 16-column tiles in order
    for (klo / 16..k_hi / 16) |r| for (spans) |c| {
        try tiles.appendSlice(gpa, t.bytes[(r * tn + c.lo / 16) * words * 2 ..][0 .. c.len() / 16 * words * 2]);
    };
    try scales.appendSlice(gpa, suh.bytes[klo * 2 .. k_hi * 2]);
    for (spans) |c| try scales.appendSlice(gpa, svh.bytes[c.lo * 2 .. c.hi * 2]);
    return .{ .trellis = 0, .suh = 0, .svh = 0, .k = k_hi - klo, .n = n, .bits = @intCast(words / 16), .codebook = book };
}

/// gather() of one matrix, uploaded as its trellis, suh and svh.
pub fn matrix(L: *W.Loader, ck: *core.Checkpoint, name: []const u8, klo: usize, khi: usize, spans: []const shard.Span) !cuda.qlinear.Weight {
    var scales: std.ArrayList(u8) = .empty;
    defer scales.deinit(L.gpa);
    var m = try gather(ck, L.gpa, name, klo, khi, spans, &L.host, &scales);
    m.trellis = try L.put("{s}.trellis", .{name});
    try L.host.appendSlice(L.gpa, scales.items[0 .. m.k * 2]);
    m.suh = try L.put("{s}.suh", .{name});
    try L.host.appendSlice(L.gpa, scales.items[m.k * 2 ..]);
    m.svh = try L.put("{s}.svh", .{name});
    return .{ .exl3 = m };
}

/// matrix() on the model's tensors, `fmt` naming the prefix of layer `i`.
pub fn layerMatrix(L: *W.Loader, comptime fmt: []const u8, i: u32, klo: usize, khi: usize, spans: []const shard.Span) !cuda.qlinear.Weight {
    var nb: [192]u8 = undefined;
    return matrix(L, L.ck, try std.fmt.bufPrint(&nb, W.prefix ++ fmt, .{i}), klo, khi, spans);
}

/// An EXL3 matrix's every column.
pub fn whole(L: *W.Loader, comptime fmt: []const u8, i: u32) !cuda.qlinear.Weight {
    var nb: [192]u8 = undefined;
    const name = try std.fmt.bufPrint(&nb, W.prefix ++ fmt, .{i});
    var nb2: [192]u8 = undefined;
    const t = try L.ck.get(try std.fmt.bufPrint(&nb2, "{s}.trellis", .{name}));
    return matrix(L, L.ck, name, 0, 0, &.{.{ .lo = 0, .hi = t.dim(1) * 16 }});
}

/// Loader.kda on an EXL3 checkpoint: qkv_proj by heads, [f_a | g_a | b] BF16, conv1d's q, k and v rows, o by inputs.
pub fn kda(L: *W.Loader, i: u32) !@import("cuda_kda_block.zig").Layer {
    const c = &L.w.config;
    const h = L.w.share.kda;
    const dk = c.kda_dim;
    const all = c.kda_heads * dk;
    const lo = h.lo * dk;
    const hi = h.hi * dk;
    const qkv = try layerMatrix(L, "layers.{d}.self_attn.qkv_proj", i, 0, 0, &.{ .{ .lo = lo, .hi = hi }, .{ .lo = all + lo, .hi = all + hi }, .{ .lo = 2 * all + lo, .hi = 2 * all + hi } });
    inline for (.{ "f_a", "g_a" }) |x| try L.whole(try L.get("layers.{d}.self_attn." ++ x ++ "_proj.weight", .{i}), .bf16);
    try L.rows(try L.get("layers.{d}.self_attn.b_proj.weight", .{i}), h.lo, h.hi, c.kda_heads, .bf16);
    const proj = try L.put("layers.{d}.kda.proj", .{i});
    try L.rows(try L.get("layers.{d}.self_attn.f_b_proj.weight", .{i}), lo, hi, all, .bf16);
    const fb = try L.put("layers.{d}.kda.fb", .{i});
    try L.rows(try L.get("layers.{d}.self_attn.g_b_proj.weight", .{i}), lo, hi, all, .bf16);
    const gb = try L.put("layers.{d}.kda.gb", .{i});
    const conv = try L.get("layers.{d}.self_attn.conv1d.weight", .{i});
    for (0..3) |j| try L.rows(conv, j * all + lo, j * all + hi, 3 * all, .f32);
    const conv_w = try L.put("layers.{d}.kda.conv", .{i});
    try L.rows(try L.get("layers.{d}.self_attn.A_log", .{i}), h.lo, h.hi, c.kda_heads, .f32);
    const a_log = try L.put("layers.{d}.kda.a_log", .{i});
    try L.rows(try L.get("layers.{d}.self_attn.dt_bias", .{i}), lo, hi, all, .f32);
    const dt = try L.put("layers.{d}.kda.dt_bias", .{i});
    try L.whole(try L.get("layers.{d}.self_attn.o_norm.weight", .{i}), .bf16);
    const norm = try L.put("layers.{d}.kda.norm", .{i});
    const o = try layerMatrix(L, "layers.{d}.self_attn.o_proj", i, lo, hi, &.{.{ .lo = 0, .hi = c.hidden }});
    return .{ .proj = proj, .qkv = qkv, .fb = fb, .gb = gb, .o = .{ .q = o }, .conv_w = conv_w, .a_log = a_log, .dt_bias = dt, .norm_w = norm, .eps = c.eps, .lower = c.lower_bound, .heads = h.len(), .dims = c.hidden };
}

/// Loader.moeLayer on EXL3: experts as exl3_experts stacks, gate and up by the rank's columns, down by its inputs.
pub fn moe(L: *W.Loader, i: u32) !@import("cuda_moe.zig").Layer {
    const c = &L.w.config;
    const d: usize = c.hidden;
    const cut = L.w.share.moe;
    try L.whole(try L.get("layers.{d}.mlp.gate.weight", .{i}), .bf16);
    const router = try L.put("layers.{d}.moe.router", .{i});
    try L.whole(try L.get("layers.{d}.mlp.gate.e_score_correction_bias", .{i}), .f32);
    const bias = try L.put("layers.{d}.moe.bias", .{i});
    var t_up: std.ArrayList(u8) = .empty;
    defer t_up.deinit(L.gpa);
    var h_up: std.ArrayList(u8) = .empty;
    defer h_up.deinit(L.gpa);
    var t_dn: std.ArrayList(u8) = .empty;
    defer t_dn.deinit(L.gpa);
    var h_dn: std.ArrayList(u8) = .empty;
    defer h_dn.deinit(L.gpa);
    var widths: cuda.exl3_experts.Widths = .{};
    var nb: [192]u8 = undefined;
    for (0..c.experts) |e| {
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |p, m| {
            const name = try std.fmt.bufPrint(&nb, W.prefix ++ "layers.{d}.mlp.experts.{d}.{s}", .{ i, e, p });
            const q = if (m < 2) try gather(L.ck, L.gpa, name, 0, 0, &.{cut}, &t_up, &h_up) else try gather(L.ck, L.gpa, name, cut.lo, cut.hi, &.{.{ .lo = 0, .hi = d }}, &t_dn, &h_dn);
            if (m < 2) widths.up = q.bits else widths.down = q.bits;
            widths.codebook = q.codebook;
        }
    }
    var ptrs: [4]u64 = undefined;
    for (&ptrs, [_][]const u8{ t_up.items, t_dn.items, h_up.items, h_dn.items }, [_][]const u8{ "up", "down", "up_scale", "down_scale" }) |*p, bytes, what| {
        try L.host.appendSlice(L.gpa, bytes);
        p.* = try L.put("layers.{d}.moe.experts.{s}", .{ i, what });
    }
    const sh = L.w.share.shared;
    return .{
        .router = router,
        .bias = bias,
        .routed = .{ .format = .exl3, .up = ptrs[0], .down = ptrs[1], .up_scale = ptrs[2], .down_scale = ptrs[3], .width = cut.len(), .dims = d, .experts = c.experts, .limit = c.swiglu_limit, .exl3 = widths },
        .count = c.experts,
        .shared_gu = 0,
        .shared_gate = try layerMatrix(L, "layers.{d}.mlp.shared_experts.gate_proj", i, 0, 0, &.{sh}),
        .shared_up = try layerMatrix(L, "layers.{d}.mlp.shared_experts.up_proj", i, 0, 0, &.{sh}),
        .shared_down = .{ .q = try layerMatrix(L, "layers.{d}.mlp.shared_experts.down_proj", i, sh.lo, sh.hi, &.{.{ .lo = 0, .hi = d }}) },
        .shared = sh.len(),
        .top_k = c.topk,
        .scale = c.routed_scale,
        .norm = c.norm_topk,
    };
}
