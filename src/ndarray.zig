pub const std = @import("std");
pub const ndvec = @import("ndarray/vec_ops.zig");
pub const max_dims = 4;

const FixedBufSliceAlloc = struct {
    fba: std.heap.FixedBufferAllocator = undefined,

    pub fn init(buf: []u8) @This() {
        return @This(){
            .fba = std.heap.FixedBufferAllocator.init(buf),
        };
    }

    pub fn alloc(self: *@This(), comptime Type: type, n: usize) []usize {
        return (self.fba.allocator().alloc(Type, n) catch unreachable);
    }

    pub fn dupe(self: *@This(), comptime T: type, m: []const T) []T {
        return (self.fba.allocator().dupe(T, m) catch unreachable);
    }
};

pub fn compareShapes(a: []usize, b: []usize) bool {
    if (a.len != b.len) {
        return false;
    }
    for (a, b) |ad, bd| {
        if (ad != bd) {
            return false;
        }
    }
    return true;
}

pub const TransposeOpts = struct {
    axes: ?[]const usize = null,
};

pub const Conv2dOpts = struct {
    stride: usize = 1,
    pad: usize = 0,
};

pub const Conv2dGrads = struct {
    dw: Ndarray(f32),
    dx: Ndarray(f32),
};

pub const Maxpool2dRes = struct {
    out: Ndarray(f32),
    idx: []const u8,
};

pub fn Ndarray(comptime Dtype: type) type {
    return struct {
        buf: []Dtype,
        strides: []usize,
        shape: []usize,
        offs: usize,
        contiguous: bool,

        const Iter = struct {
            pub const Opts = struct {
                shape: ?[]usize = null, // broadcast to
            };

            pub fn next(self: *@This()) ?*Dtype {
                if (self.index >= self.size) {
                    return null;
                }
                self.index += 1;
                const ptr: *Dtype = @ptrFromInt(@intFromPtr(self.arr.buf.ptr) + self.data_ptr * @sizeOf(Dtype));

                var ii = self.nd_m1;
                while (ii >= 0) : (ii -= 1) {
                    const i: usize = @intCast(ii);
                    if (self.coords[i] < self.dims_m1[i]) {
                        self.coords[i] += 1;
                        self.data_ptr += self.strides[i];
                        break;
                    } else {
                        self.coords[i] = 0;
                        self.data_ptr -= self.backstrides[i];
                    }
                }
                return ptr;
            }

            // Private
            arr: *const Ndarray(Dtype),
            nd_m1: isize,
            index: usize, // iteration index, upto size
            size: usize, // broadcasted size
            data_ptr: usize,
            contiguous: bool,
            strides: [max_dims]usize = undefined,
            backstrides: [max_dims]usize = undefined,
            dims_m1: [max_dims]usize = undefined,
            coords: [max_dims]usize = .{0} ** max_dims,
        };

        pub fn equalShapes(a: @This(), b: @This()) bool {
            return compareShapes(a.shape, b.shape);
        }

        pub fn iterator(self: *const @This(), opts: Iter.Opts) Iter {
            var shape = self.shape;
            var diff = @as(usize, 0);
            if (opts.shape) |bcast| {
                std.debug.assert(self.shape.len <= bcast.len);
                diff = bcast.len - self.shape.len;
                var i: usize = 0;
                var j = diff;
                var compatible = true;
                while (i < self.shape.len) : ({
                    i += 1;
                    j += 1;
                }) {
                    if (self.shape[i] == 1) {
                        continue;
                    }
                    if (self.shape[i] != bcast[j]) {
                        compatible = false;
                        break;
                    }
                }
                std.debug.assert(compatible); // incompatible shapes for broadcast
                shape = bcast;
            }
            const nd: isize = @intCast(self.shape.len);
            var ret = Iter{
                .size = undefined,
                .contiguous = self.contiguous,
                .index = 0,
                .nd_m1 = nd - 1,
                .arr = self,
                .data_ptr = self.offset(.{}),
            };
            var size_prod: usize = 1;
            for (shape, 0..) |dim_i, i| {
                ret.dims_m1[i] = dim_i - 1;
                var k: isize = @intCast(i);
                k -= @intCast(diff);
                if (k < 0 or self.shape[@intCast(k)] != shape[i]) {
                    ret.contiguous = false;
                    ret.strides[i] = 0; // stretch the dim
                } else {
                    ret.strides[i] = self.strides[@intCast(k)];
                }
                ret.backstrides[i] = ret.strides[i] * ret.dims_m1[i];
                size_prod *= shape[i];
            }
            ret.size = size_prod;
            return ret;
        }

        fn initStrides(strides: []usize, shape: []const usize) usize {
            var stride_prod: usize = 1;
            var size_prod: usize = 1;
            for (0..shape.len) |i| {
                const irev = shape.len - 1 - i;
                strides[irev] = stride_prod;
                size_prod *= shape[irev];
                stride_prod *= shape[irev];
            }
            return stride_prod;
        }
        pub fn init(alloc: std.mem.Allocator, shape: []const usize) @This() {
            const strides = alloc.alloc(usize, shape.len) catch unreachable;
            const stride_prod = @This().initStrides(strides, shape);
            return @This(){
                .buf = alloc.alloc(Dtype, stride_prod) catch unreachable,
                .shape = alloc.dupe(usize, shape) catch unreachable,
                .strides = strides,
                .offs = 0,
                .contiguous = true,
            };
        }

        // TODO an allocator-less init would be cool too for read-only inputs
        // TODO shape should be inferable from 'init' value
        pub fn initFromSlice1d(alloc: std.mem.Allocator, initial: []const f32) @This() {
            var arr = @This().init(alloc, &[_]usize{initial.len});
            for (initial, 0..) |v, i| {
                arr.buf[i] = v;
            }
            return arr;
        }

        // TODO an allocator-less init would be cool too for read-only inputs
        // TODO shape should be inferable from 'init' value
        pub fn initFromSlice2d(alloc: std.mem.Allocator, initial: []const []const f32) @This() {
            var arr = @This().init(alloc, &[_]usize{ initial.len, initial[0].len });
            var c: usize = 0;
            for (0..arr.shape[0]) |i| {
                for (0..arr.shape[1]) |j| {
                    arr.buf[c] = initial[i][j];
                    c += 1;
                }
            }
            return arr;
        }

        pub fn shapeProd(shape: []usize) usize {
            var size_prod: usize = 1;
            for (shape) |s| {
                size_prod *= s;
            }
            return size_prod;
        }

        pub fn emptyLike(self: *const Ndarray(Dtype), alloc: std.mem.Allocator) @This() {
            return @This().init(alloc, self.shape);
        }

        pub fn zerosLike(self: *const Ndarray(Dtype), alloc: std.mem.Allocator) @This() {
            const arr = self.emptyLike(alloc);
            @memset(arr.buf, 0);
            return arr;
        }

        pub fn onesLike(self: *const Ndarray(Dtype), alloc: std.mem.Allocator) @This() {
            const arr = self.emptyLike(alloc);
            @memset(arr.buf, 1);
            return arr;
        }

        pub fn scalar(alloc: std.mem.Allocator, v: Dtype) @This() {
            var arr = @This().init(alloc, &[_]usize{});
            arr.buf[0] = v;
            return arr;
        }

        // idx can be either a tuple like .{ 0, 3} or a []usize slice.
        pub fn offset(self: *const @This(), idx: anytype) usize {
            const info = @typeInfo(@TypeOf(idx));
            var offs: usize = self.offs;
            if (info == .Struct) {
                if (!info.Struct.is_tuple) {
                    @compileError("Expected input idx to be a tuple-type");
                }
                inline for (idx, 0..) |idx_i, i| {
                    offs += idx_i * self.strides[i];
                }
                return offs;
            }
            // Not a tuple, assume it's a slice.
            for (idx, 0..) |idx_i, i| {
                offs += idx_i * self.strides[i];
            }
            return offs;
        }

        pub fn get(self: *const @This(), idx: anytype) @This() {
            const offs = self.offset(idx);
            return @This(){
                .strides = self.strides[idx.len..],
                .shape = self.shape[idx.len..],
                .offs = 0,
                .contiguous = self.contiguous,
                .buf = self.buf[offs..(offs + @This().shapeProd(self.shape[idx.len..]))],
            };
        }

        pub fn getItem(self: *const @This(), idx: anytype) Dtype {
            std.debug.assert(idx.len == self.shape.len);
            const offs = self.offset(idx);
            return self.buf[offs];
        }

        pub fn getContigousSlice(self: *const @This(), idx: anytype) []Dtype {
            const offs = self.offset(idx);
            std.debug.assert(self.contiguous);
            return self.buf[offs..(offs + @This().shapeProd(self.shape[idx.len..]))];
        }

        pub fn set(self: *const @This(), idx: anytype, v: Dtype) void {
            self.buf[self.offset(idx)] = v;
        }

        pub fn assign(self: *@This(), other: @This()) void {
            if (self.contiguous and other.contiguous and self.equalShapes(other)) {
                @memcpy(self.buf, other.buf);
                return;
            }

            var dst_it = self.iterator(.{});
            var src_it = other.iterator(.{ .shape = self.shape });
            while (dst_it.next()) |dst| {
                const src = src_it.next() orelse unreachable;
                dst.* = src.*;
            }
        }

        pub fn item(self: *const @This()) Dtype {
            std.debug.assert(self.shape.len == 0 or (self.shape.len == 1 and self.shape[0] == 1));
            return self.buf[self.offs];
        }

        pub fn fill(self: *const @This(), v: Dtype) void {
            if (self.contiguous) {
                @memset(self.buf[self.offs..], v);
                return;
            }
            var it = self.iterator(.{});
            while (it.next()) |d| {
                d.* = v;
            }
        }

        fn broadcastShape(alloc: std.mem.Allocator, shape_a: []const usize, shape_b: []const usize) []usize {
            var new_shape = alloc.alloc(usize, @max(shape_a.len, shape_b.len)) catch unreachable;

            var ai: isize = @intCast(shape_a.len);
            ai -= 1;
            var bi: isize = @intCast(shape_b.len);
            bi -= 1;

            for (0..new_shape.len) |ii| {
                const i = new_shape.len - 1 - ii;
                const s0 = if (ai >= 0) shape_a[@intCast(ai)] else 1;
                const s1 = if (bi >= 0) shape_b[@intCast(bi)] else 1;
                new_shape[i] = @max(s0, s1);
                ai -= 1;
                bi -= 1;
            }
            return new_shape;
        }

        fn binop(op: ndvec.Binop, alloc: std.mem.Allocator, a: @This(), b: @This()) @This() {
            if (a.contiguous and b.contiguous and a.equalShapes(b)) {
                const dst = a.emptyLike(alloc);
                ndvec.binop(op, dst.buf, 1, @This().shapeProd(dst.shape), a.buf[a.offs..], 1, b.buf[b.offs..], 1);
                return dst;
            }

            const shape = broadcastShape(alloc, a.shape, b.shape);
            var dst = @This().init(alloc, shape);
            var dst_it = dst.iterator(.{});
            var a_it = a.iterator(.{ .shape = shape });
            var b_it = b.iterator(.{ .shape = shape });
            while (dst_it.next()) |dp| {
                const ap = a_it.next() orelse unreachable;
                const bp = b_it.next() orelse unreachable;
                dp.* = ndvec.binopScalar(op, ap.*, bp.*);
            }
            return dst;
        }

        fn binop_(op: ndvec.Binop, dst: *@This(), src: @This()) void {
            const same_shape = dst.equalShapes(src);
            if (dst.contiguous and src.contiguous and same_shape) {
                ndvec.binop_(op, dst.buf[dst.offs..], 1, @This().shapeProd(dst.shape), src.buf[src.offs..], 1);
                return;
            }
            if (same_shape and dst.shape.len == 1) {
                ndvec.binop_(op, dst.buf[dst.offs..], dst.strides[0], @This().shapeProd(dst.shape), src.buf[src.offs..], src.strides[0]);
                return;
            }
            if (same_shape and dst.shape.len == 2) {
                var outer_d = dst.offs;
                var outer_s = src.offs;
                for (0..dst.shape[0]) |_| {
                    ndvec.binop_(op, dst.buf[outer_d..], dst.strides[1], dst.shape[1], src.buf[outer_s..], src.strides[1]);
                    outer_d += dst.strides[0];
                    outer_s += src.strides[0];
                }
                return;
            }

            var dst_it = dst.iterator(.{});
            var src_it = src.iterator(.{ .shape = dst.shape });
            while (dst_it.next()) |dp| {
                const ap = src_it.next() orelse unreachable;
                dp.* = ndvec.binopScalar(op, dp.*, ap.*);
            }
        }

        pub fn add(alloc: std.mem.Allocator, a: @This(), b: @This()) @This() {
            return binop(ndvec.Binop.add, alloc, a, b);
        }

        pub fn add_(self: *@This(), other: @This()) void {
            binop_(ndvec.Binop.add, self, other);
        }

        pub fn sub(alloc: std.mem.Allocator, a: @This(), b: @This()) @This() {
            return binop(ndvec.Binop.sub, alloc, a, b);
        }

        pub fn sub_(self: *@This(), other: @This()) void {
            binop_(ndvec.Binop.sub, self, other);
        }

        pub fn mul(alloc: std.mem.Allocator, a: @This(), b: @This()) @This() {
            return binop(ndvec.Binop.mul, alloc, a, b);
        }

        pub fn mul_(self: *@This(), other: @This()) void {
            binop_(ndvec.Binop.mul, self, other);
        }

        pub fn div(alloc: std.mem.Allocator, a: @This(), b: @This()) @This() {
            return binop(ndvec.Binop.div, alloc, a, b);
        }

        pub fn div_(self: *@This(), other: @This()) void {
            binop_(ndvec.Binop.div, self, other);
        }

        pub fn reluBackwards(self: *@This(), alloc: std.mem.Allocator, data: @This()) @This() {
            return binop(ndvec.Binop.relu_bw, alloc, data, self.*);
        }

        pub fn unop(op: ndvec.Unop, alloc: std.mem.Allocator, src: @This()) @This() {
            var dst = src.emptyLike(alloc);
            if (src.contiguous) {
                ndvec.unop(op, dst.buf, 1, @This().shapeProd(dst.shape), src.buf[src.offs..], 1);
                return dst;
            }

            var dst_it = dst.iterator(.{});
            var src_it = src.iterator(.{});
            while (dst_it.next()) |d| {
                const s = src_it.next() orelse unreachable;
                d.* = ndvec.unopScalar(op, s.*);
            }
            return dst;
        }

        pub fn neg(self: *const @This(), alloc: std.mem.Allocator) @This() {
            return unop(ndvec.Unop.neg, alloc, self.*);
        }

        pub fn clipMin(self: *const @This(), alloc: std.mem.Allocator, min: Dtype) @This() {
            var dst = self.emptyLike(alloc);
            if (self.contiguous) {
                ndvec.clipMin(dst.buf, 1, @This().shapeProd(dst.shape), self.buf[self.offs..], 1, min);
                return dst;
            }

            var dst_it = dst.iterator(.{});
            var src_it = self.iterator(.{});
            while (dst_it.next()) |d| {
                const s = src_it.next() orelse unreachable;
                d.* = @max(min, s.*);
            }
            return dst;
        }

        pub fn exp(self: *const @This(), alloc: std.mem.Allocator) @This() {
            return unop(ndvec.Unop.exp, alloc, self.*);
        }

        pub fn log(self: *const @This(), alloc: std.mem.Allocator) @This() {
            return unop(ndvec.Unop.log, alloc, self.*);
        }

        pub fn argmax(self: *const @This()) usize {
            std.debug.assert(self.shape.len == 1); // TODO this actually can't correctly return n-dim index
            var it = self.iterator(.{});
            var max_idx: usize = 0;
            var idx: usize = 0;
            var m = -std.math.inf(f32);
            while (it.next()) |src| : (idx += 1) {
                const val = src.*;
                if (val > m) {
                    m = val;
                    max_idx = idx;
                }
            }
            return max_idx;
        }

        pub fn sumOverAxis(self: *const @This(), alloc: std.mem.Allocator, axis: usize, keep_dims: bool) @This() {
            const axis_len = self.shape[axis];
            const axis_stride = self.strides[axis];

            var buf: [max_dims * 8 * @sizeOf(usize)]u8 = undefined;
            var fba = FixedBufSliceAlloc.init(&buf);

            std.debug.assert(self.shape.len != 0);
            const ndim = self.shape.len;
            const ndim_m1 = ndim -| 1;
            var backstrides = fba.alloc(usize, ndim_m1);
            var strides = fba.alloc(usize, ndim_m1);
            var shape = fba.alloc(usize, ndim_m1);
            var coords = fba.alloc(usize, ndim_m1);
            @memset(coords, 0);

            var count: usize = 0;
            for (0..ndim) |i| {
                if (i != axis) {
                    strides[count] = self.strides[i];
                    shape[count] = self.shape[i];
                    backstrides[count] = strides[count] * (shape[count] - 1);
                    count += 1;
                }
            }

            var dst_shape = shape;
            if (keep_dims) {
                dst_shape = fba.dupe(usize, self.shape);
                dst_shape[axis] = 1;
            }
            var dst = Ndarray(f32).init(alloc, dst_shape);

            var size: usize = Ndarray(f32).shapeProd(self.shape) / axis_len;
            var data_ptr: usize = self.offset(.{});
            var dst_it = dst.iterator(.{});
            while (size != 0) : (size -= 1) {
                // Perform a sum over 'axis'.
                const total = ndvec.sum(self.buf[data_ptr..], axis_stride, axis_len);
                (dst_it.next() orelse unreachable).* = total;

                // Iterate all the other dims that are not 'axis'
                for (0..coords.len) |ii| {
                    const i = coords.len - 1 - ii;
                    if (coords[i] < shape[i] - 1) {
                        coords[i] += 1;
                        data_ptr += strides[i];
                        break;
                    } else {
                        coords[i] = 0;
                        data_ptr -= backstrides[i];
                    }
                }
            }
            return dst;
        }

        pub const SumOpts = struct {
            axis: ?usize = null,
            keep_dims: bool = false,
        };

        pub fn sum(self: *const @This(), alloc: std.mem.Allocator, extra_args: SumOpts) @This() {
            if (extra_args.axis) |axis| {
                return self.sumOverAxis(alloc, axis, extra_args.keep_dims);
            } else {
                if (self.contiguous) {
                    const tot = ndvec.sum(self.buf[self.offs..], 1, @This().shapeProd(self.shape));
                    return Ndarray(f32).scalar(alloc, tot);
                }
                var src_it = self.iterator(.{});
                var s: Dtype = 0;
                while (src_it.next()) |src| {
                    s += src.*;
                }
                return Ndarray(f32).scalar(alloc, s);
            }
        }

        pub fn dot(alloc: std.mem.Allocator, a: @This(), b: @This()) @This() {
            if (a.shape.len == 1 and (b.shape.len == 0 or b.shape.len == 1)) {
                if (a.shape.len == b.shape.len) {
                    std.debug.assert(@This().equalShapes(a, b));
                }
                std.debug.assert(a.shape[0] == b.shape[0]);
                var a_it = a.iterator(.{});
                var b_it = b.iterator(.{ .shape = a.shape });
                var s: Dtype = 0;
                while (a_it.next()) |av| {
                    const bv = b_it.next() orelse unreachable;
                    s += av.* * bv.*;
                }
                return @This().scalar(alloc, s);
            } else if (a.shape.len == 2 and b.shape.len == 1) {
                var dst = @This().init(alloc, &[_]usize{a.shape[0]});
                std.debug.assert(compareShapes(a.shape[1..], b.shape));
                for (0..a.shape[0]) |i| {
                    const aa = a.get(.{i});
                    const s = ndvec.innerProduct(aa.buf, aa.strides[0], aa.shape[0], b.buf, b.strides[0]);
                    dst.set(.{i}, s);
                }
                return dst;
            } else if (a.shape.len == 2 and b.shape.len == 2) {
                var dst = @This().init(alloc, &[_]usize{ a.shape[0], b.shape[1] });
                var outer_a = a.offset(.{});
                var outer_d = dst.offset(.{});

                if (a.strides[1] == 1 and b.strides[0] == 1) {
                    for (0..a.shape[0]) |_| {
                        var outer_b = b.offset(.{});
                        var inner_d = outer_d;
                        const a_buf = a.buf[outer_a..];
                        for (0..b.shape[1]) |_| {
                            dst.buf[inner_d] = ndvec.innerProductStride1(a_buf.ptr, a.shape[1], b.buf[outer_b..].ptr);
                            inner_d += dst.strides[1];
                            outer_b += b.strides[1];
                        }
                        outer_a += a.strides[0];
                        outer_d += dst.strides[0];
                    }
                } else {
                    for (0..a.shape[0]) |_| {
                        var outer_b = b.offset(.{});
                        var inner_d = outer_d;
                        const a_buf = a.buf[outer_a..];
                        for (0..b.shape[1]) |_| {
                            dst.buf[inner_d] = ndvec.innerProduct(a_buf, a.strides[1], a.shape[1], b.buf[outer_b..], b.strides[0]);
                            inner_d += dst.strides[1];
                            outer_b += b.strides[1];
                        }
                        outer_a += a.strides[0];
                        outer_d += dst.strides[0];
                    }
                }

                return dst;
            }
            std.debug.print("a b {any} {any}\n", .{ a.shape, b.shape });
            unreachable; // TODO unimplemented
        }

        pub fn reshape(self: *const @This(), alloc: std.mem.Allocator, shape: []const usize) @This() {
            var new_arr: @This() = self.*;
            new_arr.shape = alloc.dupe(usize, shape) catch unreachable;
            new_arr.strides = alloc.alloc(usize, shape.len) catch unreachable;
            const stride_prod = @This().initStrides(new_arr.strides, shape);
            std.debug.assert(stride_prod == new_arr.buf.len);
            return new_arr;
        }

        pub fn transpose(self: *const @This(), alloc: std.mem.Allocator, opts: TransposeOpts) @This() {
            var new_arr: @This() = undefined;
            new_arr = self.*;
            new_arr.shape = alloc.alloc(usize, self.shape.len) catch unreachable;
            new_arr.strides = alloc.alloc(usize, self.shape.len) catch unreachable;
            new_arr.contiguous = false;
            if (opts.axes) |permute| {
                for (0..self.shape.len) |i| {
                    new_arr.strides[i] = self.strides[permute[i]];
                    new_arr.shape[i] = self.shape[permute[i]];
                }
            } else {
                for (0..self.shape.len) |i| {
                    const rev = self.shape.len - 1 - i;
                    new_arr.strides[i] = self.strides[rev];
                    new_arr.shape[i] = self.shape[rev];
                }
            }
            return new_arr;
        }

        pub fn padForConv2d(self: *const @This(), alloc: std.mem.Allocator, padding: usize) @This() {
            _ = alloc;
            // implement zero padding like: np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
            std.debug.assert(padding == 0); // TODO pad all sides
            return self.*;
        }

        // x: Input  (shape: N, C, H, W)
        // w: Weights (shape: F, C, HH, WW)
        pub fn conv2d(alloc: std.mem.Allocator, x: @This(), w: @This(), opts: Conv2dOpts) @This() {
            std.debug.assert(x.shape.len == 4);
            const N = x.shape[0];
            const C = x.shape[1];
            const H = x.shape[2];
            const W = x.shape[3];

            const C_out = w.shape[0];
            const C_in = w.shape[1];
            std.debug.assert(C_in == C);
            const kh = w.shape[2];
            const kw = w.shape[3];

            const stride = opts.stride;
            const h_out = 1 + (H + 2 * opts.pad - kh) / stride;
            const w_out = 1 + (W + 2 * opts.pad - kw) / stride;

            const x_padded = x.padForConv2d(alloc, opts.pad);
            const out = @This().init(alloc, &[_]usize{ N, C_out, h_out, w_out });

            const xtmp = alloc.alloc(Dtype, C * kh * kw) catch unreachable;
            std.debug.assert(x_padded.strides[3] == 1);

            // Try to tell the compiler that all the loops are non-zero length.
            if (N == 0 or C_out == 0 or C == 0 or h_out == 0 or w_out == 0 or kh == 0 or kw == 0) {
                unreachable;
            }

            std.debug.assert(kw == 3 and kh == 3); // TODO code below unrolled for 3x3

            const out_slice = out.getContigousSlice(.{});
            var out_idx: usize = 0;
            for (0..N) |n| {
                for (0..C_out) |f| {
                    const xp_n = x_padded.getContigousSlice(.{n});
                    const w_slice = w.getContigousSlice(.{f});
                    for (0..h_out) |i| {
                        for (0..w_out) |j| {
                            // Collect pixels for dot product between image patch and the filter kernel.
                            var xtmpd = xtmp.ptr;
                            var xpd = xp_n.ptr + (i * stride) * x_padded.strides[2] + (j * stride);
                            const xpd_stride = x_padded.strides[1] - x_padded.strides[2] * 2;
                            for (0..C) |_| {
                                xtmpd[0] = xpd[0];
                                xtmpd[1] = xpd[1];
                                xtmpd[2] = xpd[2];

                                xpd += x_padded.strides[2];
                                xtmpd[3] = xpd[0];
                                xtmpd[4] = xpd[1];
                                xtmpd[5] = xpd[2];

                                xpd += x_padded.strides[2];
                                xtmpd[6] = xpd[0];
                                xtmpd[7] = xpd[1];
                                xtmpd[8] = xpd[2];

                                xtmpd += 9;
                                xpd += xpd_stride;
                            }
                            out_slice[out_idx] = ndvec.innerProductStride1(xtmp.ptr, xtmp.len, w_slice.ptr);
                            out_idx += 1;
                        }
                    }
                }
            }
            return out;
        }

        // x: Input  (shape: N, C, H, W)
        // w: Weights (shape: F, C, HH, WW)
        pub fn conv2dBackwards(alloc: std.mem.Allocator, x: @This(), w: @This(), dout: @This(), opts: Conv2dOpts) Conv2dGrads {
            std.debug.assert(x.shape.len == 4);
            const N = x.shape[0];
            const C = x.shape[1];

            const C_out = w.shape[0];
            const C_in = w.shape[1];
            std.debug.assert(C_in == C);
            const kh = w.shape[2];
            const kw = w.shape[3];

            const stride = opts.stride;
            const h_out = dout.shape[2];
            const w_out = dout.shape[3];

            const x_padded = x.padForConv2d(alloc, opts.pad);
            const dx = x_padded.zerosLike(alloc);
            const dw = w.zerosLike(alloc);

            const xtmp = alloc.alloc(Dtype, C * kh * kw) catch unreachable;

            std.debug.assert(x_padded.strides[3] == 1);

            // Try to tell the compiler that all the loops are non-zero length.
            if (N == 0 or C_out == 0 or C == 0 or h_out == 0 or w_out == 0 or kh == 0 or kw == 0) {
                unreachable;
            }

            for (0..N) |n| {
                // dw
                for (0..h_out) |i| {
                    const is = i * stride;
                    for (0..w_out) |j| {
                        const js = j * stride;

                        var xtmp_idx: usize = 0;
                        var xpd = x_padded.getContigousSlice(.{ n, 0, is, js }).ptr;
                        for (0..x_padded.shape[1]) |_| {
                            xtmp[xtmp_idx + 0] = xpd[0];
                            xtmp[xtmp_idx + 1] = xpd[1];
                            xtmp[xtmp_idx + 2] = xpd[2];
                            xpd += x_padded.strides[2];

                            xtmp[xtmp_idx + 3] = xpd[0];
                            xtmp[xtmp_idx + 4] = xpd[1];
                            xtmp[xtmp_idx + 5] = xpd[2];
                            xpd += x_padded.strides[2];

                            xtmp[xtmp_idx + 6] = xpd[0];
                            xtmp[xtmp_idx + 7] = xpd[1];
                            xtmp[xtmp_idx + 8] = xpd[2];
                            xpd += x_padded.strides[1] - x_padded.strides[2] * 2;
                            xtmp_idx += 9;
                        }

                        for (0..w.shape[0]) |f| {
                            xtmp_idx = 0;
                            const dw_slice = dw.getContigousSlice(.{f});
                            const dout_nfij = dout.getItem(.{ n, f, i, j });
                            // TODO hardcoded for 3x3 kernel size
                            for (0..x_padded.shape[1]) |_| {
                                // x_nf_ij = x_nf[:, i*stride:i*stride+HH, j*stride:j*stride+WW]
                                // dw_nf += x_nf_ij * dout[n, f, i, j]
                                dw_slice[xtmp_idx + 0] += xtmp[xtmp_idx + 0] * dout_nfij;
                                dw_slice[xtmp_idx + 1] += xtmp[xtmp_idx + 1] * dout_nfij;
                                dw_slice[xtmp_idx + 2] += xtmp[xtmp_idx + 2] * dout_nfij;
                                dw_slice[xtmp_idx + 3] += xtmp[xtmp_idx + 3] * dout_nfij;
                                dw_slice[xtmp_idx + 4] += xtmp[xtmp_idx + 4] * dout_nfij;
                                dw_slice[xtmp_idx + 5] += xtmp[xtmp_idx + 5] * dout_nfij;
                                dw_slice[xtmp_idx + 6] += xtmp[xtmp_idx + 6] * dout_nfij;
                                dw_slice[xtmp_idx + 7] += xtmp[xtmp_idx + 7] * dout_nfij;
                                dw_slice[xtmp_idx + 8] += xtmp[xtmp_idx + 8] * dout_nfij;
                                xtmp_idx += 9;
                            }
                        }
                    }
                }

                for (0..w.shape[0]) |f| {
                    const wf_slice = w.getContigousSlice(.{f});
                    for (0..h_out) |i| {
                        const is = i * stride;
                        for (0..w_out) |j| {
                            const js = j * stride;
                            const dout_nfij = dout.getItem(.{ n, f, i, j });
                            var wf_idx: usize = 0;
                            var dxp = dx.getContigousSlice(.{ n, 0, is, js }).ptr;
                            // TODO hardcoded for 3x3 kernel size
                            for (0..x_padded.shape[1]) |_| {
                                dxp[0] += wf_slice[wf_idx + 0] * dout_nfij;
                                dxp[1] += wf_slice[wf_idx + 1] * dout_nfij;
                                dxp[2] += wf_slice[wf_idx + 2] * dout_nfij;
                                dxp += dx.strides[2];

                                dxp[0] += wf_slice[wf_idx + 3] * dout_nfij;
                                dxp[1] += wf_slice[wf_idx + 4] * dout_nfij;
                                dxp[2] += wf_slice[wf_idx + 5] * dout_nfij;
                                dxp += dx.strides[2];

                                dxp[0] += wf_slice[wf_idx + 6] * dout_nfij;
                                dxp[1] += wf_slice[wf_idx + 7] * dout_nfij;
                                dxp[2] += wf_slice[wf_idx + 8] * dout_nfij;

                                wf_idx += 9;
                                dxp += dx.strides[1] - dx.strides[2] * 2;
                            }
                        }
                    }
                }
            }
            return Conv2dGrads{
                .dw = dw,
                .dx = dx,
            };
        }

        // x: Input  (shape: N, C, H, W)
        // hardcoded to kernel size 2, stride 2
        pub fn avgpool2d(alloc: std.mem.Allocator, x: @This()) @This() {
            std.debug.assert(x.shape.len == 4);
            const N = x.shape[0];
            const C = x.shape[1];
            const H = x.shape[2];
            const W = x.shape[3];

            const h_out = H / 2;
            const w_out = W / 2;
            const out = @This().init(alloc, &[_]usize{ N, C, h_out, w_out });

            for (0..N) |n| {
                for (0..C) |c| {
                    for (0..h_out) |i| {
                        for (0..w_out) |j| {
                            var total: f32 = 0;
                            total += x.getItem(.{ n, c, i * 2, j * 2 });
                            total += x.getItem(.{ n, c, i * 2, j * 2 + 1 });
                            total += x.getItem(.{ n, c, i * 2 + 1, j * 2 });
                            total += x.getItem(.{ n, c, i * 2 + 1, j * 2 + 1 });
                            out.set(.{ n, c, i, j }, total * 0.25);
                        }
                    }
                }
            }
            return out;
        }

        // x: Input  (shape: N, C, H, W)
        pub fn avgpool2dBackwards(alloc: std.mem.Allocator, x: @This(), dout: @This()) @This() {
            std.debug.assert(x.shape.len == 4);
            const N = dout.shape[0];
            const C = dout.shape[1];
            const H = dout.shape[2];
            const W = dout.shape[3];

            const out = @This().init(alloc, &[_]usize{ N, C, x.shape[2], x.shape[3] });

            for (0..N) |n| {
                for (0..C) |c| {
                    for (0..H) |i| {
                        for (0..W) |j| {
                            const v = dout.getItem(.{ n, c, i, j }) * 0.25;
                            out.set(.{ n, c, i * 2, j * 2 }, v);
                            out.set(.{ n, c, i * 2 + 1, j * 2 }, v);
                            out.set(.{ n, c, i * 2, j * 2 + 1 }, v);
                            out.set(.{ n, c, i * 2 + 1, j * 2 + 1 }, v);
                        }
                    }
                }
            }
            return out;
        }

        // x: Input  (shape: N, C, H, W)
        // idx: Idx of the max element in 2x2 blocks of input
        // hardcoded to kernel size 2, stride 2
        pub fn maxpool2d(alloc: std.mem.Allocator, x: @This()) Maxpool2dRes {
            std.debug.assert(x.shape.len == 4);
            const N = x.shape[0];
            const C = x.shape[1];
            const H = x.shape[2];
            const W = x.shape[3];

            const h_out = H / 2;
            const w_out = W / 2;
            const out = @This().init(alloc, &[_]usize{ N, C, h_out, w_out });
            const idx = alloc.alloc(u8, N * C * h_out * w_out) catch unreachable;

            var out_slice = out.getContigousSlice(.{});
            var out_idx: usize = 0;

            for (0..N) |n| {
                for (0..C) |c| {
                    for (0..h_out) |ii| {
                        var src = x.getContigousSlice(.{ n, c, ii * 2 }).ptr;
                        for (0..w_out) |jj| {
                            var i: u8 = 0;
                            var a = src[0];
                            const s01 = if (jj * 2 + 1 < W) src[1] else -std.math.inf(f32);
                            const s10 = if (ii * 2 + 1 < H) src[x.strides[2]] else -std.math.inf(f32);
                            const s11 = if (jj * 2 + 1 < W and ii * 2 + 1 < H) src[1 + x.strides[2]] else -std.math.inf(f32);

                            if (s01 > a) {
                                a = s01;
                                i = 1;
                            }

                            if (s10 > a) {
                                a = s10;
                                i = 2;
                            }

                            if (s11 > a) {
                                a = s11;
                                i = 3;
                            }

                            out_slice[out_idx] = a;
                            idx[out_idx] = i;
                            out_idx += 1;
                            src += 2;
                        }
                    }
                }
            }
            return Maxpool2dRes{ .out = out, .idx = idx };
        }

        // x: Input  (shape: N, C, H, W)
        pub fn maxpool2dBackwards(alloc: std.mem.Allocator, x: @This(), max_idx: []const u8, dout: @This()) @This() {
            std.debug.assert(x.shape.len == 4);
            const N = dout.shape[0];
            const C = dout.shape[1];
            const H = dout.shape[2];
            const W = dout.shape[3];

            const xH = x.shape[2];
            const xW = x.shape[3];

            const out = @This().init(alloc, &[_]usize{ N, C, x.shape[2], x.shape[3] });
            var dout_idx: usize = 0;
            var dout_it = dout.iterator(.{});

            for (0..N) |n| {
                for (0..C) |c| {
                    const out_slice = out.getContigousSlice(.{ n, c });
                    for (0..H) |i| {
                        const out_idx = i * 2 * out.strides[2];
                        for (0..W) |j| {
                            const v = (dout_it.next() orelse unreachable).*;
                            const m = max_idx[dout_idx];
                            dout_idx += 1;

                            const v00 = if (m == 0) v else 0;
                            const v01 = if (m == 1) v else 0;
                            const v10 = if (m == 2) v else 0;
                            const v11 = if (m == 3) v else 0;

                            out_slice[out_idx + j * 2 + 0] = v00;
                            if (j * 1 + 1 < xW) {
                                out_slice[out_idx + j * 2 + 1] = v01;
                            }
                            if (i * 2 + 1 < xH) {
                                const s = out.strides[2];
                                out_slice[out_idx + j * 2 + s] = v10;
                                if (j * 2 + 1 < xW) {
                                    out_slice[out_idx + j * 2 + s + 1] = v11;
                                }
                            }
                        }
                    }
                }
            }
            return out;
        }
    };
}

test "tensor1" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var t2d = Ndarray(f32).init(arena.allocator(), &[_]usize{ 2, 4 });
    for (0..t2d.shape[0]) |yi| {
        for (0..t2d.shape[1]) |xi| {
            t2d.set(.{ yi, xi }, @floatFromInt(xi + yi * 10));
        }
    }

    for (t2d.buf) |v| {
        std.debug.print("{d} ", .{v});
    }
    std.debug.print("\n", .{});

    const ones = t2d.onesLike(arena.allocator());
    for (ones.buf) |v| {
        std.debug.print("{d} ", .{v});
    }
    std.debug.print("\n", .{});
}

test "2d" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var t2d = Ndarray(f32).init(arena.allocator(), &[_]usize{ 2, 4 });
    var cnt: usize = 0;
    for (0..t2d.shape[0]) |yi| {
        for (0..t2d.shape[1]) |xi| {
            t2d.set(.{ yi, xi }, @floatFromInt(cnt));
            cnt += 1;
        }
    }
    var tmp2 = t2d.emptyLike(arena.allocator());
    tmp2.fill(2);

    t2d.mul_(tmp2);

    cnt = 0;
    std.debug.print("t2d shape {any}\n", .{t2d.shape});
    for (0..t2d.shape[0]) |yi| {
        for (0..t2d.shape[1]) |xi| {
            std.debug.print("index {} {} {}\n", .{ yi, xi, t2d.get(.{ yi, xi }).item() });
            try std.testing.expectEqual(t2d.get(.{ yi, xi }).item(), @floatFromInt(cnt * 2));
            cnt += 1;
        }
    }

    var dst = t2d.get(.{0});
    const src = t2d.get(.{1});
    dst.assign(src);
}

test "zerod" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var t2d = Ndarray(f32).init(arena.allocator(), &[_]usize{ 2, 4 });
    var cnt: usize = 0;
    for (0..t2d.shape[0]) |yi| {
        for (0..t2d.shape[1]) |xi| {
            t2d.set(.{ yi, xi }, @floatFromInt(cnt));
            cnt += 1;
        }
    }

    var zerod = t2d.get(.{ 0, 0 });
    try std.testing.expect(zerod.shape.len == 0);
    try std.testing.expectEqual(@as(f32, 0), zerod.item());
    zerod = t2d.get(.{ 0, 1 });
    try std.testing.expectEqual(@as(f32, 1), zerod.item());
    zerod = t2d.get(.{ 1, 0 });
    try std.testing.expectEqual(@as(f32, 4), zerod.item());

    var sc = Ndarray(f32).scalar(arena.allocator(), 16.0);
    try std.testing.expect(sc.shape.len == 0);
    try std.testing.expectEqual(@as(f32, 16.0), sc.item());

    // Tests iterators on scalars
    var neg_sc = sc.neg(arena.allocator());
    try std.testing.expectEqual(@as(f32, -16), neg_sc.item());
}

test "inner_it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var t2d = Ndarray(f32).init(arena.allocator(), &[_]usize{ 2, 4 });
    for (0..t2d.shape[0]) |yi| {
        for (0..t2d.shape[1]) |xi| {
            t2d.set(.{ yi, xi }, @floatFromInt(xi + yi * 4));
        }
    }

    var it = t2d.iterator(.{});
    var count: usize = 0;
    while (it.next()) |elt| : (count += 1) {
        try std.testing.expectEqual(count, @intFromFloat(elt.*));
    }
    try std.testing.expectEqual(count, t2d.shape[0] * t2d.shape[1]);

    // broadcast scalar and access it with an iterator matching t2d shape
    var bc = Ndarray(f32).scalar(arena.allocator(), 12);
    count = 0;
    it = bc.iterator(.{ .shape = t2d.shape });
    while (it.next()) |elt| : (count += 1) {
        try std.testing.expectEqual(elt.*, 12);
    }
    try std.testing.expectEqual(count, 8);

    // broadcast scalar with add_
    var dst = Ndarray(f32).init(arena.allocator(), &[_]usize{ 3, 3 });
    dst.fill(10);
    const sc3 = Ndarray(f32).scalar(arena.allocator(), 3);
    dst.add_(sc3);
    count = 0;
    it = dst.iterator(.{});
    while (it.next()) |elt| : (count += 1) {
        try std.testing.expectEqual(elt.*, 13);
    }
    try std.testing.expect(count == 9);
}

test "slice init" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const arr = [_]f32{ 0, 1, 2, 3 };
    var tens1d = Ndarray(f32).initFromSlice1d(arena.allocator(), &arr);
    for (arr, 0..) |v, i| {
        try std.testing.expect(v == tens1d.get(&[_]usize{i}).item());
    }

    const mat2x4 = [_][]const f32{
        &[_]f32{ 1, 2, 3, 4 },
        &[_]f32{ 5, 5, 6, 7 },
    };
    var tens2d = Ndarray(f32).initFromSlice2d(arena.allocator(), &mat2x4);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 4 }, tens2d.shape);
    for (mat2x4, 0..) |r, i| {
        for (r, 0..) |v, j| {
            try std.testing.expect(v == tens2d.get(&[_]usize{ i, j }).item());
        }
    }
}

test "axis sum" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const arr = [_]f32{ 0, 1, 2, 3 };
    var tens1d = Ndarray(f32).initFromSlice1d(arena.allocator(), &arr);
    const sum0 = tens1d.sum(arena.allocator(), .{ .axis = 0, .keep_dims = false });
    try std.testing.expectEqual(@as(f32, 0 + 1 + 2 + 3), sum0.item());

    const mat2x4 = [_][]const f32{
        &[_]f32{ 1, 2, 3, 4 },
        &[_]f32{ 5, 5, 6, 7 },
    };
    var tens2d = Ndarray(f32).initFromSlice2d(arena.allocator(), &mat2x4);
    const sum1 = tens2d.sum(arena.allocator(), .{ .axis = 0, .keep_dims = false });
    try std.testing.expectEqualSlices(usize, &[_]usize{4}, sum1.shape);
    for (0..4) |i| {
        try std.testing.expectEqual(@as(f32, mat2x4[0][i] + mat2x4[1][i]), sum1.get(&[_]usize{i}).item());
    }

    const sum2 = tens2d.sum(arena.allocator(), .{ .axis = 1, .keep_dims = false });
    try std.testing.expectEqualSlices(usize, &[_]usize{2}, sum2.shape);
    for (0..2) |i| {
        const s = mat2x4[i][0] + mat2x4[i][1] + mat2x4[i][2] + mat2x4[i][3];
        try std.testing.expectEqual(s, sum2.get(&[_]usize{i}).item());
    }

    const sum3 = tens2d.sum(arena.allocator(), .{ .axis = 1, .keep_dims = true });
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 1 }, sum3.shape);
    for (0..2) |i| {
        const s = mat2x4[i][0] + mat2x4[i][1] + mat2x4[i][2] + mat2x4[i][3];
        try std.testing.expectEqual(s, sum2.get(&[_]usize{i}).item());
    }
}

test "reshape" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const mat2x4 = [_][]const f32{
        &[_]f32{ 1, 2, 3, 4 },
        &[_]f32{ 5, 6, 7, 8 },
    };
    var mat = Ndarray(f32).initFromSlice2d(arena.allocator(), &mat2x4);
    var res = mat.reshape(alloc, &[_]usize{8});
    try std.testing.expectEqualSlices(usize, &[_]usize{8}, res.shape);

    const ex = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    for (ex, 0..) |v, i| {
        try std.testing.expectEqual(v, res.get(&[_]usize{i}).item());
    }
}

test "dot prod" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const arr1 = [_]f32{ 0, 1, 2, 3 };
    const arr2 = [_]f32{ 1, 1, 2, 2 };
    var tens1 = Ndarray(f32).initFromSlice1d(arena.allocator(), &arr1);
    const tens2 = Ndarray(f32).initFromSlice1d(arena.allocator(), &arr2);
    const s0 = Ndarray(f32).dot(alloc, tens1, tens1.zerosLike(alloc));
    try std.testing.expectEqual(@as(f32, 0), s0.item());
    const s1 = Ndarray(f32).dot(alloc, tens1, tens2);
    try std.testing.expectEqual(@as(f32, 1 + 4 + 6), s1.item());

    const mat2x4 = [_][]const f32{
        &[_]f32{ 1, 2, 3, 4 },
        &[_]f32{ 5, 5, 6, 7 },
    };
    const mat = Ndarray(f32).initFromSlice2d(arena.allocator(), &mat2x4);
    const res = Ndarray(f32).dot(alloc, mat, tens1);
    try std.testing.expectEqualSlices(usize, &[_]usize{2}, res.shape);
    const ex = [_]f32{ 20, 38 };
    for (ex, 0..) |v, i| {
        try std.testing.expectEqual(v, res.get(&[_]usize{i}).item());
    }
}

test "dot prod 2" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const alloc = arena.allocator();
    defer arena.deinit();
    const arr1 = [_]f32{ 0, 1, 2, 3 };
    var tens1 = Ndarray(f32).initFromSlice1d(arena.allocator(), &arr1);
    tens1 = tens1.reshape(alloc, &[_]usize{ 1, 4 });

    const mat2x4 = [_][]const f32{
        &[_]f32{ 1, 2, 3, 4 },
        &[_]f32{ 5, 5, 6, 7 },
    };

    var mat = Ndarray(f32).initFromSlice2d(arena.allocator(), &mat2x4);
    const res = Ndarray(f32).dot(alloc, tens1, mat.transpose(alloc, .{ .axes = &[_]usize{ 1, 0 } }));
    const ex = [_]f32{ 20, 38 };
    for (ex, 0..) |v, i| {
        try std.testing.expectEqual(v, res.get(&[_]usize{ 0, i }).item());
    }
}

test "transpose" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const mat2x4 = [_][]const f32{
        &[_]f32{ 1, 2, 3, 4 },
        &[_]f32{ 5, 6, 7, 8 },
    };
    var tens2d = Ndarray(f32).initFromSlice2d(arena.allocator(), &mat2x4);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 4 }, tens2d.shape);
    const t = tens2d.transpose(arena.allocator(), TransposeOpts{ .axes = &[_]usize{ 1, 0 } });
    try std.testing.expectEqualSlices(usize, &[_]usize{ 4, 2 }, t.shape);
    for (0..2) |ri| {
        for (0..4) |ci| {
            try std.testing.expectEqual(mat2x4[ri][ci], t.get(&[_]usize{ ci, ri }).item());
        }
        std.debug.print("\n", .{});
    }

    const t2 = tens2d.transpose(arena.allocator(), .{});
    try std.testing.expectEqualSlices(usize, &[_]usize{ 4, 2 }, t2.shape);
}

test "tuple or slice access" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const mat2x4 = [_][]const f32{
        &[_]f32{ 1, 2, 3, 4 },
        &[_]f32{ 5, 6, 7, 8 },
    };
    var tens2d = Ndarray(f32).initFromSlice2d(arena.allocator(), &mat2x4);
    for (0..2) |ri| {
        for (0..4) |ci| {
            try std.testing.expectEqual(mat2x4[ri][ci], tens2d.get(&[_]usize{ ri, ci }).item());
            try std.testing.expectEqual(mat2x4[ri][ci], tens2d.get(.{ ri, ci }).item());
        }
        std.debug.print("\n", .{});
    }
}

test "conv2d" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const N = 1;
    const C = 1;
    const C_out = 1;
    var x = Ndarray(f32).init(arena.allocator(), &[_]usize{ N, C, 28, 28 });

    for (0..28) |i| {
        for (0..28) |j| {
            const f: f32 = @floatFromInt(j);
            const fi: f32 = @floatFromInt(i);
            x.set(.{ 0, 0, i, j }, (f + 0.5) / 27.5 + fi);
        }
    }

    const kernel = [_][]const f32{
        &[_]f32{ 1, 2, 3 },
        &[_]f32{ 4, 5, 6 },
        &[_]f32{ 7, 8, 9 },
    };
    const ww = Ndarray(f32).initFromSlice2d(arena.allocator(), &kernel);

    const w = Ndarray(f32).init(arena.allocator(), &[_]usize{ C_out, C, 3, 3 });
    var wv = w.get(.{ 0, 0 });
    wv.assign(ww);

    const out = Ndarray(f32).conv2d(arena.allocator(), x, w, .{});
    try std.testing.expectEqualSlices(usize, &[_]usize{ N, C_out, 28 - 2, 28 - 2 }, out.shape);

    const expected = [_][]const f32{
        &[_]f32{ 65.67273, 67.30909, 68.94545, 70.58182, 72.218185, 73.854546, 75.490906, 77.12727, 78.76363, 80.4, 82.03637, 83.67273, 85.30909, 86.94545, 88.581825, 90.21818, 91.854546, 93.49091, 95.12727, 96.76363, 98.399994, 100.03636, 101.67272, 103.3091, 104.94546, 106.58182 },
        &[_]f32{ 110.67273, 112.30909, 113.94545, 115.58182, 117.218185, 118.854546, 120.490906, 122.127266, 123.76363, 125.4, 127.03636, 128.67273, 130.3091, 131.94545, 133.5818, 135.21817, 136.85455, 138.4909, 140.12727, 141.76364, 143.40001, 145.03638, 146.67273, 148.3091, 149.94545, 151.58182 },
    };
    for (0..2) |ri| {
        var ov = out.get(.{ 0, 0, ri }).iterator(.{});
        var idx: usize = 0;
        while (ov.next()) |v| {
            try std.testing.expectApproxEqRel(expected[ri][idx], v.*, 0.00001);
            idx += 1;
        }
    }
}
