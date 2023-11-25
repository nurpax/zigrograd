const std = @import("std");

pub const Unop = enum {
    neg,
    exp,
    log,
};

pub const Binop = enum {
    add,
    sub,
    mul,
    div,
    relu_bw,
};

pub fn unopScalar(op: Unop, a: f32) f32 {
    return switch (op) {
        Unop.neg => -a,
        Unop.exp => @exp(a),
        Unop.log => @log(a),
    };
}

pub fn binopScalar(op: Binop, a: f32, b: f32) f32 {
    return switch (op) {
        Binop.add => a + b,
        Binop.sub => a - b,
        Binop.mul => a * b,
        Binop.div => a / b,
        Binop.relu_bw => if (a > 0) b else 0,
    };
}

pub fn unop(op: Unop, d: []f32, d_stride: usize, count: usize, a: []const f32, a_stride: usize) void {
    var d_offs: usize = 0;
    var a_offs: usize = 0;
    for (0..count) |_| {
        d[d_offs] = unopScalar(op, a[a_offs]);
        d_offs += d_stride;
        a_offs += a_stride;
    }
}

pub fn binop(op: Binop, d: []f32, d_stride: usize, count: usize, a: []const f32, a_stride: usize, b: []const f32, b_stride: usize) void {
    var d_offs: usize = 0;
    var a_offs: usize = 0;
    var b_offs: usize = 0;
    for (0..count) |_| {
        d[d_offs] = binopScalar(op, a[a_offs], b[b_offs]);
        d_offs += d_stride;
        a_offs += a_stride;
        b_offs += b_stride;
    }
}

pub fn binop_(op: Binop, d: []f32, d_stride: usize, count: usize, a: []const f32, a_stride: usize) void {
    var d_offs: usize = 0;
    var a_offs: usize = 0;
    for (0..count) |_| {
        d[d_offs] = binopScalar(op, d[d_offs], a[a_offs]);
        d_offs += d_stride;
        a_offs += a_stride;
    }
}

// dst[...] += src[...] * constant
pub fn addMulConstStride1_(d: []f32, count: usize, a: []const f32, c: f32) void {
    for (0..count) |i| {
        d[i] += a[i] * c;
    }
}

pub fn clipMin(d: []f32, d_stride: usize, count: usize, a: []const f32, a_stride: usize, clip_min: f32) void {
    var d_offs: usize = 0;
    var a_offs: usize = 0;
    for (0..count) |_| {
        d[d_offs] = @max(clip_min, a[a_offs]);
        d_offs += d_stride;
        a_offs += a_stride;
    }
}

pub fn sum(src: []f32, stride: usize, count: usize) f32 {
    var offs: usize = 0;
    var total = @as(f32, 0);
    for (0..count) |_| {
        total += src[offs];
        offs += stride;
    }
    return total;
}

pub fn innerProduct(a: []f32, a_stride: usize, count: usize, b: []f32, b_stride: usize) f32 {
    var a_offs: usize = 0;
    var b_offs: usize = 0;
    var total = @as(f32, 0);
    for (0..count) |_| {
        total += a[a_offs] * b[b_offs];
        a_offs += a_stride;
        b_offs += b_stride;
    }
    return total;
}

pub fn innerProductStride1(a: [*]f32, count: usize, b: [*]f32) f32 {
    const rem = count & 3;
    const count4 = count >> 2;
    var idx: usize = 0;
    var totalv4 = @Vector(4, f32){ 0, 0, 0, 0 };
    for (0..count4) |_| {
        const av = @Vector(4, f32){ a[idx + 0], a[idx + 1], a[idx + 2], a[idx + 3] };
        const bv = @Vector(4, f32){ b[idx + 0], b[idx + 1], b[idx + 2], b[idx + 3] };
        totalv4 += av * bv;
        idx += 4;
    }
    var totalf = @reduce(.Add, totalv4);
    for (0..rem) |_| {
        totalf += a[idx] * b[idx];
        idx += 1;
    }
    return totalf;
}
