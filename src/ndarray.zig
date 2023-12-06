pub const std = @import("std");
pub const ndvec = @import("ndarray/vec_ops.zig");
pub const gemm = @import("ndarray/gemm.zig");
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

pub fn shapeProd(shape: []usize) usize {
    var size_prod: usize = 1;
    for (shape) |s| {
        size_prod *= s;
    }
    return size_prod;
}

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

pub fn inferShape(alloc: std.mem.Allocator, old_shape: []usize, shape: anytype) []usize {
    const info = @typeInfo(@TypeOf(shape));
    if (info == .Struct) {
        if (!info.Struct.is_tuple) {
            @compileError("Expected input idx to be a tuple-type");
        }
        const new_shape = alloc.alloc(usize, shape.len) catch unreachable;
        var unknown_idx: ?usize = null;
        var shape_prod: usize = 1;
        inline for (shape, 0..) |shape_i, i| {
            if (shape_i == -1) {
                std.debug.assert(unknown_idx == null); // only one unknown -1 size allowed
                unknown_idx = i;
            } else {
                new_shape[i] = shape_i;
                shape_prod *= shape_i;
            }
        }
        // Infer unknown size
        if (unknown_idx) |idx| {
            new_shape[idx] = shapeProd(old_shape) / shape_prod;
        }
        return new_shape;
    } else {
        return alloc.dupe(usize, shape) catch unreachable;
    }
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

pub const AddMmOpts = struct {
    transpose_a: bool = false,
    transpose_b: bool = false,
    alpha: f32 = 1, // GEMM alpha*(A @ B)
    beta: f32 = 1, // GEMM beta (C *= beta)
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

        pub fn clone(self: @This(), alloc: std.mem.Allocator) @This() {
            var dst = self.emptyLike(alloc);
            dst.assign(self);
            return dst;
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
                .buf = self.buf[offs..(offs + shapeProd(self.shape[idx.len..]))],
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
            return self.buf[offs..(offs + shapeProd(self.shape[idx.len..]))];
        }

        pub fn setItem(self: *const @This(), idx: anytype, v: Dtype) void {
            self.buf[self.offset(idx)] = v;
        }

        pub fn set(self: *const @This(), dst_idx: anytype, src: @This()) void {
            var view = self.get(dst_idx);
            std.debug.assert(view.equalShapes(src));
            view.assign(src);
        }

        pub fn assign(self: *@This(), other: @This()) void {
            if (self.contiguous and other.contiguous and self.equalShapes(other)) {
                @memcpy(self.buf, other.buf);
                return;
            }
            // Detect transpose fast path, it goes often together with matmuls so perf matters.
            // Transposes done through strides are typically made contiguous with the .clone()
            // method that calls here -- in this case self/dest is always contiguous.
            if (self.contiguous and self.equalShapes(other) and self.strides[1] == other.strides[0] and other.strides[1] == self.shape[0]) {
                std.debug.assert(self.strides[1] == 1);
                std.debug.assert(other.strides[0] == 1);
                ndvec.transpose2dStrided(self.getContigousSlice(.{}), self.shape[0], self.shape[1], other.buf[other.offset(.{})..], other.strides[0], other.strides[1]);
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

        fn binop(comptime op: ndvec.Binop, alloc: std.mem.Allocator, a: @This(), b: @This()) @This() {
            if (a.contiguous and b.contiguous and a.equalShapes(b)) {
                const dst = a.emptyLike(alloc);
                if (op == .add) {
                    ndvec.addContiguous(dst.buf, shapeProd(dst.shape), a.buf[a.offs..], b.buf[b.offs..]);
                } else {
                    ndvec.binop(op, dst.buf, 1, shapeProd(dst.shape), a.buf[a.offs..], 1, b.buf[b.offs..], 1);
                }
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

        fn binop_(comptime op: ndvec.Binop, dst: @This(), src: @This()) void {
            const same_shape = dst.equalShapes(src);
            if (dst.contiguous and src.contiguous and same_shape) {
                if (op == .add) {
                    ndvec.addContiguous_(dst.buf[dst.offs..], shapeProd(dst.shape), src.buf[src.offs..]);
                } else {
                    ndvec.binop_(op, dst.buf[dst.offs..], 1, shapeProd(dst.shape), src.buf[src.offs..], 1);
                }
                return;
            }
            if (same_shape and dst.shape.len == 1) {
                ndvec.binop_(op, dst.buf[dst.offs..], dst.strides[0], shapeProd(dst.shape), src.buf[src.offs..], src.strides[0]);
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

        pub fn add_(self: @This(), other: @This()) void {
            binop_(ndvec.Binop.add, self, other);
        }

        pub fn sub(alloc: std.mem.Allocator, a: @This(), b: @This()) @This() {
            return binop(ndvec.Binop.sub, alloc, a, b);
        }

        pub fn sub_(self: @This(), other: @This()) void {
            binop_(ndvec.Binop.sub, self, other);
        }

        pub fn mul(alloc: std.mem.Allocator, a: @This(), b: @This()) @This() {
            return binop(ndvec.Binop.mul, alloc, a, b);
        }

        pub fn mul_(self: @This(), other: @This()) void {
            binop_(ndvec.Binop.mul, self, other);
        }

        pub fn div(alloc: std.mem.Allocator, a: @This(), b: @This()) @This() {
            return binop(ndvec.Binop.div, alloc, a, b);
        }

        pub fn div_(self: @This(), other: @This()) void {
            binop_(ndvec.Binop.div, self, other);
        }

        pub fn reluBackwards(self: @This(), alloc: std.mem.Allocator, data: @This()) @This() {
            return binop(ndvec.Binop.relu_bw, alloc, data, self);
        }

        pub fn unop(comptime op: ndvec.Unop, alloc: std.mem.Allocator, src: @This()) @This() {
            var dst = src.emptyLike(alloc);
            if (src.contiguous) {
                ndvec.unop(op, dst.buf, 1, shapeProd(dst.shape), src.buf[src.offs..], 1);
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

        pub fn neg(self: @This(), alloc: std.mem.Allocator) @This() {
            return unop(ndvec.Unop.neg, alloc, self);
        }

        pub fn clipMin(self: @This(), alloc: std.mem.Allocator, min: Dtype) @This() {
            var dst = self.emptyLike(alloc);
            if (self.contiguous) {
                ndvec.clipMin(dst.buf, 1, shapeProd(dst.shape), self.buf[self.offs..], 1, min);
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

        pub fn exp(self: @This(), alloc: std.mem.Allocator) @This() {
            return unop(ndvec.Unop.exp, alloc, self);
        }

        pub fn log(self: @This(), alloc: std.mem.Allocator) @This() {
            return unop(ndvec.Unop.log, alloc, self);
        }

        pub fn argmax(self: @This()) usize {
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

        pub fn sumOverAxis(self: @This(), alloc: std.mem.Allocator, axis: usize, keep_dims: bool) @This() {
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
            var size: usize = shapeProd(self.shape) / axis_len;
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

        pub fn sum(self: @This(), alloc: std.mem.Allocator, extra_args: SumOpts) @This() {
            if (extra_args.axis) |axis| {
                return self.sumOverAxis(alloc, axis, extra_args.keep_dims);
            } else {
                if (self.contiguous) {
                    const tot = ndvec.sum(self.buf[self.offs..], 1, shapeProd(self.shape));
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

        // TODO this should be implemented in with gemm
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
                // TODO could do this with GEMM too
                std.debug.assert(compareShapes(a.shape[1..], b.shape));
                var dst = @This().init(alloc, &[_]usize{a.shape[0]});
                for (0..a.shape[0]) |i| {
                    const aa = a.get(.{i});
                    const s = ndvec.innerProduct(aa.buf, aa.strides[0], aa.shape[0], b.buf, b.strides[0]);
                    dst.setItem(.{i}, s);
                }
                return dst;
            } else if (a.shape.len == 2 and b.shape.len == 2) {
                // TODO b transposed is common
                var dst = @This().init(alloc, &[_]usize{ a.shape[0], b.shape[1] });
                const a_lin = if (a.contiguous) a else a.clone(alloc);
                const b_lin = if (b.contiguous) b else b.clone(alloc);
                dst.addmm_(a_lin, b_lin, .{ .beta = 0 });
                return dst;
            }
            std.debug.print("a b {any} {any}\n", .{ a.shape, b.shape });
            unreachable; // TODO unimplemented
        }

        pub fn addmm_(dest: @This(), a: @This(), b: @This(), opts: AddMmOpts) void {
            const M = if (opts.transpose_a) a.shape[1] else a.shape[0];
            const N = if (opts.transpose_b) b.shape[0] else b.shape[1];
            const K = if (opts.transpose_a) a.shape[0] else a.shape[1];

            const A = a.buf[a.offset(.{})..];
            const B = b.buf[b.offset(.{})..];
            const C = dest.buf[dest.offset(.{})..];

            std.debug.assert(dest.shape[0] == M and dest.shape[1] == N);
            std.debug.assert(a.strides[1] == 1 and b.strides[1] == 1);
            const lda = a.strides[0];
            const ldb = b.strides[0];
            const ldc = dest.strides[0];
            gemm.gemm(opts.transpose_a, opts.transpose_b, M, N, K, opts.alpha, A, lda, B, ldb, opts.beta, C, ldc);
        }

        // Shape can be either a []usize slice or a tuple.  Tuple allows specifying
        // one axis as "unknown" with -1 in which case its size will be inferred.
        //
        // Only the tuple path supports -1 since it's not convenient to make the shape
        // type signed integer.
        pub fn reshape(self: @This(), alloc: std.mem.Allocator, shape: anytype) @This() {
            var new_arr: @This() = self;
            new_arr.shape = inferShape(alloc, self.shape, shape);
            new_arr.strides = alloc.alloc(usize, shape.len) catch unreachable;
            const stride_prod = @This().initStrides(new_arr.strides, new_arr.shape);
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

        pub fn unfold(x: @This(), alloc: std.mem.Allocator, kw: usize, kh: usize, opts: Conv2dOpts) @This() {
            std.debug.assert(x.shape.len == 3); // only single sample support
            std.debug.assert(kw == 3 and kh == 3); // only 3x3 support for now

            const H = x.shape[1];
            const W = x.shape[2];
            const C = x.shape[0];

            const stride = opts.stride;
            const h_out = 1 + (H + 2 * opts.pad - kh) / stride;
            const w_out = 1 + (W + 2 * opts.pad - kw) / stride;
            const out = @This().init(alloc, &[_]usize{ kh * kw * x.shape[0], h_out * w_out });

            const xp_n = x.getContigousSlice(.{});
            for (0..h_out) |i| {
                for (0..w_out) |j| {
                    // Collect pixels for dot product between image patch and the filter kernel.
                    var xtmpd = out.getContigousSlice(.{ 0, j + w_out * i }).ptr;
                    const col_stride = out.strides[0]; // TODO minibatch dim xmat is sized for N matrices
                    var xpd = xp_n.ptr + (i * stride) * x.strides[1] + (j * stride);
                    const xpd_stride = x.strides[0] - x.strides[1] * 2;

                    // unfold 3x3 patch -> linear output
                    for (0..C) |_| {
                        xtmpd[0] = xpd[0];
                        xtmpd += col_stride;
                        xtmpd[0] = xpd[1];
                        xtmpd += col_stride;
                        xtmpd[0] = xpd[2];
                        xtmpd += col_stride;
                        xpd += x.strides[1];

                        xtmpd[0] = xpd[0];
                        xtmpd += col_stride;
                        xtmpd[0] = xpd[1];
                        xtmpd += col_stride;
                        xtmpd[0] = xpd[2];
                        xtmpd += col_stride;
                        xpd += x.strides[1];

                        xtmpd[0] = xpd[0];
                        xtmpd += col_stride;
                        xtmpd[0] = xpd[1];
                        xtmpd += col_stride;
                        xtmpd[0] = xpd[2];
                        xtmpd += col_stride;

                        xpd += xpd_stride;
                    }
                }
            }
            return out;
        }

        pub fn fold(x: @This(), alloc: std.mem.Allocator, output_size: []const usize, kw: usize, kh: usize, opts: Conv2dOpts) @This() {
            std.debug.assert(x.shape.len == 2); // only single sample support
            std.debug.assert(kw == 3 and kh == 3); // only 3x3 support for now

            const C = x.shape[0] / (kh * kw);
            const stride = opts.stride;
            const h_out = 1 + (output_size[0] + 2 * opts.pad - kh) / stride;
            const w_out = 1 + (output_size[1] + 2 * opts.pad - kw) / stride;

            const out = @This().init(alloc, &[_]usize{ C, output_size[0], output_size[1] });
            out.fill(0);
            const dst = out.getContigousSlice(.{});
            for (0..h_out) |i| {
                for (0..w_out) |j| {
                    // Collect pixels for dot product between image patch and the filter kernel.
                    var srcp = x.getContigousSlice(.{ 0, j + w_out * i }).ptr;
                    const src_col_stride = x.strides[0];
                    var dstp = dst.ptr + (i * stride) * out.strides[1] + (j * stride);
                    const out_stride = out.strides[0] - out.strides[1] * 2;

                    for (0..C) |_| {
                        dstp[0] += srcp[0];
                        srcp += src_col_stride;
                        dstp[1] += srcp[0];
                        srcp += src_col_stride;
                        dstp[2] += srcp[0];
                        srcp += src_col_stride;
                        dstp += out.strides[1];

                        dstp[0] += srcp[0];
                        srcp += src_col_stride;
                        dstp[1] += srcp[0];
                        srcp += src_col_stride;
                        dstp[2] += srcp[0];
                        srcp += src_col_stride;
                        dstp += out.strides[1];

                        dstp[0] += srcp[0];
                        srcp += src_col_stride;
                        dstp[1] += srcp[0];
                        srcp += src_col_stride;
                        dstp[2] += srcp[0];
                        srcp += src_col_stride;
                        dstp += out_stride;
                    }
                }
            }
            return out;
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

            // Try to tell the compiler that all the loops are non-zero length.
            if (N == 0 or C_out == 0 or C == 0 or h_out == 0 or w_out == 0 or kh == 0 or kw == 0) {
                unreachable;
            }

            const x_padded = x.padForConv2d(alloc, opts.pad);
            var out = @This().init(alloc, &[_]usize{ N, C_out, h_out, w_out });

            std.debug.assert(x_padded.strides[3] == 1);

            const w_col = w.reshape(alloc, &[_]usize{ C_out, w.shape[1] * w.shape[2] * w.shape[3] });
            for (0..N) |n| {
                const xmat = x_padded.get(.{n}).unfold(alloc, kw, kh, opts);
                const out_view = out.get(.{n}).reshape(alloc, &[_]usize{ C_out, h_out * w_out });
                out_view.addmm_(w_col, xmat, .{ .beta = 0 });
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

            const x_padded = x.padForConv2d(alloc, opts.pad);
            const dx = x_padded.zerosLike(alloc);
            var dw = w.zerosLike(alloc);

            std.debug.assert(x_padded.strides[3] == 1);

            // Try to tell the compiler that all the loops are non-zero length.
            if (N == 0 or C_out == 0 or C == 0 or kh == 0 or kw == 0) {
                unreachable;
            }

            const dw_col = @This().init(alloc, &[_]usize{ C_out, w.shape[1] * w.shape[2] * w.shape[3] });
            dw_col.fill(0);

            const w_col = w.reshape(alloc, &[_]usize{ C_out, w.shape[1] * w.shape[2] * w.shape[3] });
            const dx_col = @This().init(alloc, &[_]usize{ w.shape[1] * w.shape[2] * w.shape[3], dout.shape[2] * dout.shape[3] });

            for (0..N) |n| {
                const x_col = x_padded.get(.{n}).unfold(alloc, kw, kh, opts);
                const doutr = dout.get(.{n}).reshape(alloc, &[_]usize{ dout.shape[1], dout.shape[2] * dout.shape[3] });
                dw_col.addmm_(doutr, x_col, .{ .transpose_b = true });

                dx_col.addmm_(w_col, doutr, .{ .transpose_a = true, .beta = 0 });
                const dxtmp = dx_col.fold(alloc, x.shape[2..], kw, kh, opts);
                dx.get(.{n}).add_(dxtmp);
            }
            dw.add_(dw_col.reshape(alloc, &[_]usize{ dw_col.shape[0], C, kh, kw }));
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
                            out.setItem(.{ n, c, i, j }, total * 0.25);
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
                            out.setItem(.{ n, c, i * 2, j * 2 }, v);
                            out.setItem(.{ n, c, i * 2 + 1, j * 2 }, v);
                            out.setItem(.{ n, c, i * 2, j * 2 + 1 }, v);
                            out.setItem(.{ n, c, i * 2 + 1, j * 2 + 1 }, v);
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
            t2d.setItem(.{ yi, xi }, @floatFromInt(xi + yi * 10));
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
            t2d.setItem(.{ yi, xi }, @floatFromInt(cnt));
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
            t2d.setItem(.{ yi, xi }, @floatFromInt(cnt));
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
            t2d.setItem(.{ yi, xi }, @floatFromInt(xi + yi * 4));
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

test "reshape_tuple" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const mat2x4 = [_][]const f32{
        &[_]f32{ 1, 2, 3, 4 },
        &[_]f32{ 5, 6, 7, 8 },
    };
    var mat = Ndarray(f32).initFromSlice2d(arena.allocator(), &mat2x4);
    var res = mat.reshape(alloc, .{8});
    try std.testing.expectEqualSlices(usize, &[_]usize{8}, res.shape);

    const ex = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    for (ex, 0..) |v, i| {
        try std.testing.expectEqual(v, res.get(&[_]usize{i}).item());
    }

    const res2 = mat.reshape(alloc, .{ -1, 4 });
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 4 }, res2.shape);
    for (0..res2.shape[0]) |i| {
        for (0..res2.shape[1]) |j| {
            try std.testing.expectEqual(mat2x4[i][j], res2.getItem(.{ i, j }));
        }
    }

    const res3 = mat.reshape(alloc, .{ 8, -1 });
    try std.testing.expectEqualSlices(usize, &[_]usize{ 8, 1 }, res3.shape);
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
    const tens1 = Ndarray(f32).initFromSlice1d(arena.allocator(), &arr1).reshape(alloc, .{ 1, 4 });

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

test "matmul" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const alloc = arena.allocator();
    defer arena.deinit();

    var a_arr = Ndarray(f32).initFromSlice1d(alloc, &[_]f32{ 0.6307789174764159, 2.254766949581381, 1.5426314150437923, -1.525964788529883, -0.14363285701467568, -0.23310117089810367, 0.13662194783388937, 1.1187680761040366, -1.6470774640144064, -1.018455951688499, -0.2681332514219718, -0.09479973793631223, 0.4651122127730034, 0.2940099379962251, -0.17220421545117576, -0.15222648608130665, -1.2831481650597363, -1.001782199186841, -0.6221492713142152, -0.12495075169679382, 0.008655062849938796, 0.7656709593762457, -0.6872810468497651, 1.0193959366414957 });
    a_arr = a_arr.reshape(alloc, &[_]usize{ 3, 8 });
    var b_arr = Ndarray(f32).initFromSlice1d(alloc, &[_]f32{ -0.5306894614554938, 0.10236436814974047, -0.9462521123878044, 0.06348167961392685, -1.4383350528521912, 1.904673689705416, 1.9623861720338427, -2.30026911415194, -1.2734668168473275, -0.7667685148500176, 0.05429670457987651, 0.05408663915367491, -0.5919001070827915, -1.3085658827491222, 1.009861865981846, -0.22897211777699672, 2.5461802332944288, 3.0005600338918588, -1.8892754809772248, 0.14393464550669885, -2.2305866102702354, 1.5914295889568941, -0.7770524571014872, 1.5754215821419573 });
    b_arr = b_arr.reshape(alloc, &[_]usize{ 8, 3 });
    const out_expected_val = [_]f32{ 5.66640307661663, -8.065104700334489, 2.2639960603573406, 0.09640524844371084, 2.1413843061332014, 1.4512447555183916, 2.2325894204823524, 3.7810390112438204, 5.5368421076775505 };

    const out_flat = Ndarray(f32).dot(alloc, a_arr, b_arr).reshape(alloc, .{-1});
    for (out_expected_val, out_flat.getContigousSlice(.{})) |expected, actual| {
        try std.testing.expectApproxEqRel(expected, actual, 0.000001);
    }
}

test "fold_unfold" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    const alloc = arena.allocator();
    defer arena.deinit();

    const a_val = [_]f32{ 0.10430558456087256, 0.43858312641401176, 0.9517034413545684, -1.045581469990104, 0.7401740240417422, 0.70680632153984, 0.08035342731017923, 0.7370591228322054, 1.965933695912146, -0.626163761763856, -0.1867006577090551, 0.26160513349594894, 0.7733336542645307, -0.9740906358759013, -1.3666499415401236, 1.7335806798185045, -0.4732005127989997, 1.1557573937853085, -1.3824335399808696, 1.5137163413806853, 0.4285156823865879, 0.2674160373308898, 1.244436607523693, 0.09434513904203043, 0.5161933596949987, -0.3731630165976045, -0.4982230107026707, 0.7062638218935501, 0.5857309300288382, -1.380939005617481, 1.2156612427371825, 1.0486724269784742, 1.121348647148459, 1.039865713540383, -0.7699418327193321, -0.5801682650202291, 0.41100806977591203, 0.20979540271781613, 1.146365074948048, -1.5067822219487594, 0.7692493767605454, -0.7941960241110015, -1.035389985998136, 0.5195684357230007, -1.624105585690136, 0.2689617258622194, 0.19785331315940352, 1.715254721941367, 0.1466103429307763, -0.3855935307765039, 0.7022557818371753, -1.0004650491804234, 0.4396595665706977, 0.41848566176945134, 0.182242798875821, 0.7026610979789225, -0.4149968307564379, -0.19425092621905077, -0.7916284882230478, 0.5742561361933042, -0.9433121671434082, -0.43468338757801567, -0.27553852360584596, -1.5506806361835606, 0.09546461960963976, 0.10789659055020337, -0.9376164033152871, -0.6835947397076848, 0.52317219957338, -0.22203721713693025, 0.9861372375815816, 0.1089337191978776, 1.2074601561721672, 0.2646360316603576, 0.6497114126485019, 0.31495254054311606, -0.4748456366887811, -0.26323753553537405, -2.3641577275869645, -2.2972962510700046, -1.0551473594040908, -0.5901918016499518, 1.1518290970451521, -1.6029590626651902, 2.004218803829111, 0.6301546257539251, -0.48250684640650493, -1.888697319567144, 0.1600109115016712, -3.0102846561049525, 0.6907893850997603, -0.6371915508563151, 0.295076275794622, -0.461116771265289, 0.22345739523478866, -0.7395747136915637, -0.19152989062530745, 1.2534568809105129, -0.8543807710988324, -0.35966118169649103, 1.3953267753342649, -0.4172125717534239, 0.06223198052327424, -1.7234491712506523, -0.11940373418890424, -0.018639109518791296, 0.9853275839515692, -0.8552408845693293, 0.06640631394686243, 0.4653715207840639, 0.6126727177438529, -0.4395046883285174, -1.697582450021184, 0.5046594173523674, 1.194838727950116, 0.5003434358040134, 0.8375795650176676, -0.5439463081160844, -0.718314830924079, -0.4657463977367459, -1.5733684446225535, 0.24871537026778884, -0.6735391837177798, 0.9914773481922882, 0.5512706929631109, 0.2622969576363864, -1.4656361665674147, 0.8267578940117929, 0.7797659178417734, 0.062187959847728506, 0.20417645947704843, 2.681508405426129, -1.0506337737743108, -1.030067866866774, 0.6908233419998775, 1.49190557139001, 0.49305662430792546, -0.2024056725467825, -0.2560589846521373, -1.238242047512753, -0.10394213082848065, 0.19809224861611782, -0.9404348791966544, 0.9916941957913744, -1.048620097126247, 1.3147765007497652, 0.13730661199594202, 0.9198393768053104, 1.07479743569901, -0.17431664054536178, -1.5046290321068363, 0.8038400443222191, 1.3300822546648476, 0.2513507974970653, -0.5566002994013655, 1.8948613450943137, -0.7444561282448672, -0.9305600396299964, -0.9265993345286057, -0.43809095330465514 };
    var a_arr = Ndarray(f32).initFromSlice1d(alloc, &a_val);
    a_arr = a_arr.reshape(alloc, &[_]usize{ 2, 8, 10 });
    const out_expected_val = [_]f32{ 0.10430558456087256, 0.43858312641401176, 0.9517034413545684, -1.045581469990104, 0.7401740240417422, 0.70680632153984, 0.08035342731017923, 0.7370591228322054, -0.1867006577090551, 0.26160513349594894, 0.7733336542645307, -0.9740906358759013, -1.3666499415401236, 1.7335806798185045, -0.4732005127989997, 1.1557573937853085, 0.4285156823865879, 0.2674160373308898, 1.244436607523693, 0.09434513904203043, 0.5161933596949987, -0.3731630165976045, -0.4982230107026707, 0.7062638218935501, 1.2156612427371825, 1.0486724269784742, 1.121348647148459, 1.039865713540383, -0.7699418327193321, -0.5801682650202291, 0.41100806977591203, 0.20979540271781613, 0.7692493767605454, -0.7941960241110015, -1.035389985998136, 0.5195684357230007, -1.624105585690136, 0.2689617258622194, 0.19785331315940352, 1.715254721941367, 0.7022557818371753, -1.0004650491804234, 0.4396595665706977, 0.41848566176945134, 0.182242798875821, 0.7026610979789225, -0.4149968307564379, -0.19425092621905077, 0.43858312641401176, 0.9517034413545684, -1.045581469990104, 0.7401740240417422, 0.70680632153984, 0.08035342731017923, 0.7370591228322054, 1.965933695912146, 0.26160513349594894, 0.7733336542645307, -0.9740906358759013, -1.3666499415401236, 1.7335806798185045, -0.4732005127989997, 1.1557573937853085, -1.3824335399808696, 0.2674160373308898, 1.244436607523693, 0.09434513904203043, 0.5161933596949987, -0.3731630165976045, -0.4982230107026707, 0.7062638218935501, 0.5857309300288382, 1.0486724269784742, 1.121348647148459, 1.039865713540383, -0.7699418327193321, -0.5801682650202291, 0.41100806977591203, 0.20979540271781613, 1.146365074948048, -0.7941960241110015, -1.035389985998136, 0.5195684357230007, -1.624105585690136, 0.2689617258622194, 0.19785331315940352, 1.715254721941367, 0.1466103429307763, -1.0004650491804234, 0.4396595665706977, 0.41848566176945134, 0.182242798875821, 0.7026610979789225, -0.4149968307564379, -0.19425092621905077, -0.7916284882230478, 0.9517034413545684, -1.045581469990104, 0.7401740240417422, 0.70680632153984, 0.08035342731017923, 0.7370591228322054, 1.965933695912146, -0.626163761763856, 0.7733336542645307, -0.9740906358759013, -1.3666499415401236, 1.7335806798185045, -0.4732005127989997, 1.1557573937853085, -1.3824335399808696, 1.5137163413806853, 1.244436607523693, 0.09434513904203043, 0.5161933596949987, -0.3731630165976045, -0.4982230107026707, 0.7062638218935501, 0.5857309300288382, -1.380939005617481, 1.121348647148459, 1.039865713540383, -0.7699418327193321, -0.5801682650202291, 0.41100806977591203, 0.20979540271781613, 1.146365074948048, -1.5067822219487594, -1.035389985998136, 0.5195684357230007, -1.624105585690136, 0.2689617258622194, 0.19785331315940352, 1.715254721941367, 0.1466103429307763, -0.3855935307765039, 0.4396595665706977, 0.41848566176945134, 0.182242798875821, 0.7026610979789225, -0.4149968307564379, -0.19425092621905077, -0.7916284882230478, 0.5742561361933042, -0.1867006577090551, 0.26160513349594894, 0.7733336542645307, -0.9740906358759013, -1.3666499415401236, 1.7335806798185045, -0.4732005127989997, 1.1557573937853085, 0.4285156823865879, 0.2674160373308898, 1.244436607523693, 0.09434513904203043, 0.5161933596949987, -0.3731630165976045, -0.4982230107026707, 0.7062638218935501, 1.2156612427371825, 1.0486724269784742, 1.121348647148459, 1.039865713540383, -0.7699418327193321, -0.5801682650202291, 0.41100806977591203, 0.20979540271781613, 0.7692493767605454, -0.7941960241110015, -1.035389985998136, 0.5195684357230007, -1.624105585690136, 0.2689617258622194, 0.19785331315940352, 1.715254721941367, 0.7022557818371753, -1.0004650491804234, 0.4396595665706977, 0.41848566176945134, 0.182242798875821, 0.7026610979789225, -0.4149968307564379, -0.19425092621905077, -0.9433121671434082, -0.43468338757801567, -0.27553852360584596, -1.5506806361835606, 0.09546461960963976, 0.10789659055020337, -0.9376164033152871, -0.6835947397076848, 0.26160513349594894, 0.7733336542645307, -0.9740906358759013, -1.3666499415401236, 1.7335806798185045, -0.4732005127989997, 1.1557573937853085, -1.3824335399808696, 0.2674160373308898, 1.244436607523693, 0.09434513904203043, 0.5161933596949987, -0.3731630165976045, -0.4982230107026707, 0.7062638218935501, 0.5857309300288382, 1.0486724269784742, 1.121348647148459, 1.039865713540383, -0.7699418327193321, -0.5801682650202291, 0.41100806977591203, 0.20979540271781613, 1.146365074948048, -0.7941960241110015, -1.035389985998136, 0.5195684357230007, -1.624105585690136, 0.2689617258622194, 0.19785331315940352, 1.715254721941367, 0.1466103429307763, -1.0004650491804234, 0.4396595665706977, 0.41848566176945134, 0.182242798875821, 0.7026610979789225, -0.4149968307564379, -0.19425092621905077, -0.7916284882230478, -0.43468338757801567, -0.27553852360584596, -1.5506806361835606, 0.09546461960963976, 0.10789659055020337, -0.9376164033152871, -0.6835947397076848, 0.52317219957338, 0.7733336542645307, -0.9740906358759013, -1.3666499415401236, 1.7335806798185045, -0.4732005127989997, 1.1557573937853085, -1.3824335399808696, 1.5137163413806853, 1.244436607523693, 0.09434513904203043, 0.5161933596949987, -0.3731630165976045, -0.4982230107026707, 0.7062638218935501, 0.5857309300288382, -1.380939005617481, 1.121348647148459, 1.039865713540383, -0.7699418327193321, -0.5801682650202291, 0.41100806977591203, 0.20979540271781613, 1.146365074948048, -1.5067822219487594, -1.035389985998136, 0.5195684357230007, -1.624105585690136, 0.2689617258622194, 0.19785331315940352, 1.715254721941367, 0.1466103429307763, -0.3855935307765039, 0.4396595665706977, 0.41848566176945134, 0.182242798875821, 0.7026610979789225, -0.4149968307564379, -0.19425092621905077, -0.7916284882230478, 0.5742561361933042, -0.27553852360584596, -1.5506806361835606, 0.09546461960963976, 0.10789659055020337, -0.9376164033152871, -0.6835947397076848, 0.52317219957338, -0.22203721713693025, 0.4285156823865879, 0.2674160373308898, 1.244436607523693, 0.09434513904203043, 0.5161933596949987, -0.3731630165976045, -0.4982230107026707, 0.7062638218935501, 1.2156612427371825, 1.0486724269784742, 1.121348647148459, 1.039865713540383, -0.7699418327193321, -0.5801682650202291, 0.41100806977591203, 0.20979540271781613, 0.7692493767605454, -0.7941960241110015, -1.035389985998136, 0.5195684357230007, -1.624105585690136, 0.2689617258622194, 0.19785331315940352, 1.715254721941367, 0.7022557818371753, -1.0004650491804234, 0.4396595665706977, 0.41848566176945134, 0.182242798875821, 0.7026610979789225, -0.4149968307564379, -0.19425092621905077, -0.9433121671434082, -0.43468338757801567, -0.27553852360584596, -1.5506806361835606, 0.09546461960963976, 0.10789659055020337, -0.9376164033152871, -0.6835947397076848, 0.9861372375815816, 0.1089337191978776, 1.2074601561721672, 0.2646360316603576, 0.6497114126485019, 0.31495254054311606, -0.4748456366887811, -0.26323753553537405, 0.2674160373308898, 1.244436607523693, 0.09434513904203043, 0.5161933596949987, -0.3731630165976045, -0.4982230107026707, 0.7062638218935501, 0.5857309300288382, 1.0486724269784742, 1.121348647148459, 1.039865713540383, -0.7699418327193321, -0.5801682650202291, 0.41100806977591203, 0.20979540271781613, 1.146365074948048, -0.7941960241110015, -1.035389985998136, 0.5195684357230007, -1.624105585690136, 0.2689617258622194, 0.19785331315940352, 1.715254721941367, 0.1466103429307763, -1.0004650491804234, 0.4396595665706977, 0.41848566176945134, 0.182242798875821, 0.7026610979789225, -0.4149968307564379, -0.19425092621905077, -0.7916284882230478, -0.43468338757801567, -0.27553852360584596, -1.5506806361835606, 0.09546461960963976, 0.10789659055020337, -0.9376164033152871, -0.6835947397076848, 0.52317219957338, 0.1089337191978776, 1.2074601561721672, 0.2646360316603576, 0.6497114126485019, 0.31495254054311606, -0.4748456366887811, -0.26323753553537405, -2.3641577275869645, 1.244436607523693, 0.09434513904203043, 0.5161933596949987, -0.3731630165976045, -0.4982230107026707, 0.7062638218935501, 0.5857309300288382, -1.380939005617481, 1.121348647148459, 1.039865713540383, -0.7699418327193321, -0.5801682650202291, 0.41100806977591203, 0.20979540271781613, 1.146365074948048, -1.5067822219487594, -1.035389985998136, 0.5195684357230007, -1.624105585690136, 0.2689617258622194, 0.19785331315940352, 1.715254721941367, 0.1466103429307763, -0.3855935307765039, 0.4396595665706977, 0.41848566176945134, 0.182242798875821, 0.7026610979789225, -0.4149968307564379, -0.19425092621905077, -0.7916284882230478, 0.5742561361933042, -0.27553852360584596, -1.5506806361835606, 0.09546461960963976, 0.10789659055020337, -0.9376164033152871, -0.6835947397076848, 0.52317219957338, -0.22203721713693025, 1.2074601561721672, 0.2646360316603576, 0.6497114126485019, 0.31495254054311606, -0.4748456366887811, -0.26323753553537405, -2.3641577275869645, -2.2972962510700046, -1.0551473594040908, -0.5901918016499518, 1.1518290970451521, -1.6029590626651902, 2.004218803829111, 0.6301546257539251, -0.48250684640650493, -1.888697319567144, 0.6907893850997603, -0.6371915508563151, 0.295076275794622, -0.461116771265289, 0.22345739523478866, -0.7395747136915637, -0.19152989062530745, 1.2534568809105129, 1.3953267753342649, -0.4172125717534239, 0.06223198052327424, -1.7234491712506523, -0.11940373418890424, -0.018639109518791296, 0.9853275839515692, -0.8552408845693293, 0.6126727177438529, -0.4395046883285174, -1.697582450021184, 0.5046594173523674, 1.194838727950116, 0.5003434358040134, 0.8375795650176676, -0.5439463081160844, -1.5733684446225535, 0.24871537026778884, -0.6735391837177798, 0.9914773481922882, 0.5512706929631109, 0.2622969576363864, -1.4656361665674147, 0.8267578940117929, 0.20417645947704843, 2.681508405426129, -1.0506337737743108, -1.030067866866774, 0.6908233419998775, 1.49190557139001, 0.49305662430792546, -0.2024056725467825, -0.5901918016499518, 1.1518290970451521, -1.6029590626651902, 2.004218803829111, 0.6301546257539251, -0.48250684640650493, -1.888697319567144, 0.1600109115016712, -0.6371915508563151, 0.295076275794622, -0.461116771265289, 0.22345739523478866, -0.7395747136915637, -0.19152989062530745, 1.2534568809105129, -0.8543807710988324, -0.4172125717534239, 0.06223198052327424, -1.7234491712506523, -0.11940373418890424, -0.018639109518791296, 0.9853275839515692, -0.8552408845693293, 0.06640631394686243, -0.4395046883285174, -1.697582450021184, 0.5046594173523674, 1.194838727950116, 0.5003434358040134, 0.8375795650176676, -0.5439463081160844, -0.718314830924079, 0.24871537026778884, -0.6735391837177798, 0.9914773481922882, 0.5512706929631109, 0.2622969576363864, -1.4656361665674147, 0.8267578940117929, 0.7797659178417734, 2.681508405426129, -1.0506337737743108, -1.030067866866774, 0.6908233419998775, 1.49190557139001, 0.49305662430792546, -0.2024056725467825, -0.2560589846521373, 1.1518290970451521, -1.6029590626651902, 2.004218803829111, 0.6301546257539251, -0.48250684640650493, -1.888697319567144, 0.1600109115016712, -3.0102846561049525, 0.295076275794622, -0.461116771265289, 0.22345739523478866, -0.7395747136915637, -0.19152989062530745, 1.2534568809105129, -0.8543807710988324, -0.35966118169649103, 0.06223198052327424, -1.7234491712506523, -0.11940373418890424, -0.018639109518791296, 0.9853275839515692, -0.8552408845693293, 0.06640631394686243, 0.4653715207840639, -1.697582450021184, 0.5046594173523674, 1.194838727950116, 0.5003434358040134, 0.8375795650176676, -0.5439463081160844, -0.718314830924079, -0.4657463977367459, -0.6735391837177798, 0.9914773481922882, 0.5512706929631109, 0.2622969576363864, -1.4656361665674147, 0.8267578940117929, 0.7797659178417734, 0.062187959847728506, -1.0506337737743108, -1.030067866866774, 0.6908233419998775, 1.49190557139001, 0.49305662430792546, -0.2024056725467825, -0.2560589846521373, -1.238242047512753, 0.6907893850997603, -0.6371915508563151, 0.295076275794622, -0.461116771265289, 0.22345739523478866, -0.7395747136915637, -0.19152989062530745, 1.2534568809105129, 1.3953267753342649, -0.4172125717534239, 0.06223198052327424, -1.7234491712506523, -0.11940373418890424, -0.018639109518791296, 0.9853275839515692, -0.8552408845693293, 0.6126727177438529, -0.4395046883285174, -1.697582450021184, 0.5046594173523674, 1.194838727950116, 0.5003434358040134, 0.8375795650176676, -0.5439463081160844, -1.5733684446225535, 0.24871537026778884, -0.6735391837177798, 0.9914773481922882, 0.5512706929631109, 0.2622969576363864, -1.4656361665674147, 0.8267578940117929, 0.20417645947704843, 2.681508405426129, -1.0506337737743108, -1.030067866866774, 0.6908233419998775, 1.49190557139001, 0.49305662430792546, -0.2024056725467825, -0.10394213082848065, 0.19809224861611782, -0.9404348791966544, 0.9916941957913744, -1.048620097126247, 1.3147765007497652, 0.13730661199594202, 0.9198393768053104, -0.6371915508563151, 0.295076275794622, -0.461116771265289, 0.22345739523478866, -0.7395747136915637, -0.19152989062530745, 1.2534568809105129, -0.8543807710988324, -0.4172125717534239, 0.06223198052327424, -1.7234491712506523, -0.11940373418890424, -0.018639109518791296, 0.9853275839515692, -0.8552408845693293, 0.06640631394686243, -0.4395046883285174, -1.697582450021184, 0.5046594173523674, 1.194838727950116, 0.5003434358040134, 0.8375795650176676, -0.5439463081160844, -0.718314830924079, 0.24871537026778884, -0.6735391837177798, 0.9914773481922882, 0.5512706929631109, 0.2622969576363864, -1.4656361665674147, 0.8267578940117929, 0.7797659178417734, 2.681508405426129, -1.0506337737743108, -1.030067866866774, 0.6908233419998775, 1.49190557139001, 0.49305662430792546, -0.2024056725467825, -0.2560589846521373, 0.19809224861611782, -0.9404348791966544, 0.9916941957913744, -1.048620097126247, 1.3147765007497652, 0.13730661199594202, 0.9198393768053104, 1.07479743569901, 0.295076275794622, -0.461116771265289, 0.22345739523478866, -0.7395747136915637, -0.19152989062530745, 1.2534568809105129, -0.8543807710988324, -0.35966118169649103, 0.06223198052327424, -1.7234491712506523, -0.11940373418890424, -0.018639109518791296, 0.9853275839515692, -0.8552408845693293, 0.06640631394686243, 0.4653715207840639, -1.697582450021184, 0.5046594173523674, 1.194838727950116, 0.5003434358040134, 0.8375795650176676, -0.5439463081160844, -0.718314830924079, -0.4657463977367459, -0.6735391837177798, 0.9914773481922882, 0.5512706929631109, 0.2622969576363864, -1.4656361665674147, 0.8267578940117929, 0.7797659178417734, 0.062187959847728506, -1.0506337737743108, -1.030067866866774, 0.6908233419998775, 1.49190557139001, 0.49305662430792546, -0.2024056725467825, -0.2560589846521373, -1.238242047512753, -0.9404348791966544, 0.9916941957913744, -1.048620097126247, 1.3147765007497652, 0.13730661199594202, 0.9198393768053104, 1.07479743569901, -0.17431664054536178, 1.3953267753342649, -0.4172125717534239, 0.06223198052327424, -1.7234491712506523, -0.11940373418890424, -0.018639109518791296, 0.9853275839515692, -0.8552408845693293, 0.6126727177438529, -0.4395046883285174, -1.697582450021184, 0.5046594173523674, 1.194838727950116, 0.5003434358040134, 0.8375795650176676, -0.5439463081160844, -1.5733684446225535, 0.24871537026778884, -0.6735391837177798, 0.9914773481922882, 0.5512706929631109, 0.2622969576363864, -1.4656361665674147, 0.8267578940117929, 0.20417645947704843, 2.681508405426129, -1.0506337737743108, -1.030067866866774, 0.6908233419998775, 1.49190557139001, 0.49305662430792546, -0.2024056725467825, -0.10394213082848065, 0.19809224861611782, -0.9404348791966544, 0.9916941957913744, -1.048620097126247, 1.3147765007497652, 0.13730661199594202, 0.9198393768053104, -1.5046290321068363, 0.8038400443222191, 1.3300822546648476, 0.2513507974970653, -0.5566002994013655, 1.8948613450943137, -0.7444561282448672, -0.9305600396299964, -0.4172125717534239, 0.06223198052327424, -1.7234491712506523, -0.11940373418890424, -0.018639109518791296, 0.9853275839515692, -0.8552408845693293, 0.06640631394686243, -0.4395046883285174, -1.697582450021184, 0.5046594173523674, 1.194838727950116, 0.5003434358040134, 0.8375795650176676, -0.5439463081160844, -0.718314830924079, 0.24871537026778884, -0.6735391837177798, 0.9914773481922882, 0.5512706929631109, 0.2622969576363864, -1.4656361665674147, 0.8267578940117929, 0.7797659178417734, 2.681508405426129, -1.0506337737743108, -1.030067866866774, 0.6908233419998775, 1.49190557139001, 0.49305662430792546, -0.2024056725467825, -0.2560589846521373, 0.19809224861611782, -0.9404348791966544, 0.9916941957913744, -1.048620097126247, 1.3147765007497652, 0.13730661199594202, 0.9198393768053104, 1.07479743569901, 0.8038400443222191, 1.3300822546648476, 0.2513507974970653, -0.5566002994013655, 1.8948613450943137, -0.7444561282448672, -0.9305600396299964, -0.9265993345286057, 0.06223198052327424, -1.7234491712506523, -0.11940373418890424, -0.018639109518791296, 0.9853275839515692, -0.8552408845693293, 0.06640631394686243, 0.4653715207840639, -1.697582450021184, 0.5046594173523674, 1.194838727950116, 0.5003434358040134, 0.8375795650176676, -0.5439463081160844, -0.718314830924079, -0.4657463977367459, -0.6735391837177798, 0.9914773481922882, 0.5512706929631109, 0.2622969576363864, -1.4656361665674147, 0.8267578940117929, 0.7797659178417734, 0.062187959847728506, -1.0506337737743108, -1.030067866866774, 0.6908233419998775, 1.49190557139001, 0.49305662430792546, -0.2024056725467825, -0.2560589846521373, -1.238242047512753, -0.9404348791966544, 0.9916941957913744, -1.048620097126247, 1.3147765007497652, 0.13730661199594202, 0.9198393768053104, 1.07479743569901, -0.17431664054536178, 1.3300822546648476, 0.2513507974970653, -0.5566002994013655, 1.8948613450943137, -0.7444561282448672, -0.9305600396299964, -0.9265993345286057, -0.43809095330465514 };
    const out_shape_val = [_]usize{ 18, 48 };

    const unfolded = a_arr.unfold(alloc, 3, 3, .{});
    try std.testing.expectEqualSlices(usize, &out_shape_val, unfolded.shape);

    const flat_out = unfolded.reshape(alloc, .{-1});
    for (out_expected_val, flat_out.getContigousSlice(.{})) |expected, actual| {
        try std.testing.expectApproxEqRel(expected, actual, 0.000001);
    }

    // Fold it back
    const folded_expected_val = [_]f32{ 0.10430558456087256, 0.8771662528280235, 2.8551103240637055, -3.136744409970312, 2.2205220721252266, 2.12041896461952, 0.2410602819305377, 2.2111773684966165, 3.931867391824292, -0.626163761763856, -0.3734013154181102, 1.0464205339837958, 4.6400019255871845, -5.844543815255408, -8.199899649240741, 10.401484078911029, -2.839203076793998, 6.9345443627118515, -5.529734159923478, 3.0274326827613707, 1.2855470471597636, 1.6044962239853389, 11.199929467713236, 0.849106251378274, 4.645740237254988, -3.3584671493784404, -4.484007096324036, 6.356374397041952, 3.5143855801730295, -4.142817016852443, 3.6469837282115476, 6.292034561870844, 10.09213782433613, 9.358791421863447, -6.929476494473988, -5.221514385182063, 3.6990726279832082, 1.8881586244603452, 6.878190449688288, -4.5203466658462785, 2.3077481302816363, -4.765176144666009, -9.318509873983224, 4.676115921507006, -14.616950271211227, 2.4206555327599744, 1.780679818434632, 15.437292497472303, 0.8796620575846579, -1.1567805923295116, 2.1067673455115257, -6.002790295082541, 3.9569360991362794, 3.7663709559250615, 1.6401851898823894, 6.323949881810304, -3.7349714768079405, -1.7482583359714572, -4.749770929338287, 1.7227684085799124, -1.8866243342868163, -1.7387335503120627, -1.6532311416350756, -9.304083817101363, 0.5727877176578385, 0.6473795433012203, -5.625698419891722, -4.101568438246109, 2.09268879829352, -0.4440744342738605, 0.9861372375815816, 0.2178674383957552, 3.6223804685165018, 0.7939080949810728, 1.9491342379455054, 0.9448576216293482, -1.4245369100663434, -0.7897126066061222, -4.728315455173929, -2.2972962510700046, -1.0551473594040908, -1.1803836032999035, 3.4554872911354564, -4.80887718799557, 6.012656411487333, 1.8904638772617752, -1.4475205392195147, -5.666091958701432, 0.3200218230033424, -3.0102846561049525, 1.3815787701995206, -2.5487662034252603, 1.7704576547677318, -2.766700627591734, 1.340744371408732, -4.437448282149382, -1.1491793437518447, 7.520741285463076, -3.4175230843953295, -0.7193223633929821, 4.185980326002794, -2.5032754305205436, 0.5600878247094682, -15.511042541255875, -1.0746336077001382, -0.1677519856691217, 8.867948255564125, -7.697167961123963, 0.39843788368117455, 1.3961145623521918, 1.8380181532315587, -2.637028129971104, -15.278242050190654, 4.541934756171306, 10.753548551551045, 4.503090922236121, 7.538216085159011, -4.8955167730447595, -4.309888985544474, -1.3972391932102377, -4.720105333867661, 1.4922922216067331, -6.061852653460017, 8.923296133730593, 4.961436236667998, 2.360672618727477, -13.19072549910673, 7.440821046106137, 4.67859550705064, 0.1865638795431855, 0.6125293784311453, 16.089050432556775, -9.455703963968798, -9.270610801800967, 6.217410077998896, 13.42715014251009, 4.437509618771329, -1.8216510529210421, -1.5363539079128237, -3.7147261425382587, -0.2078842616569613, 0.7923689944644713, -5.642609275179927, 5.950165174748246, -6.291720582757482, 7.888659004498591, 0.8238396719756522, 5.519036260831862, 4.29918974279604, -0.34863328109072356, -1.5046290321068363, 1.6076800886444382, 3.990246763994543, 0.7540523924911959, -1.6698008982040964, 5.684584035282941, -2.2333683847346015, -2.7916801188899893, -1.8531986690572113, -0.43809095330465514 };
    const folded_shape_val = [_]usize{ 2, 8, 10 };
    const folded_out = unfolded.fold(alloc, a_arr.shape[1..], 3, 3, .{});
    try std.testing.expectEqualSlices(usize, &folded_shape_val, folded_out.shape);
    const flat_folded_out = folded_out.reshape(alloc, .{-1});
    for (folded_expected_val, flat_folded_out.getContigousSlice(.{})) |expected, actual| {
        try std.testing.expectApproxEqRel(expected, actual, 0.000001);
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
    const tc = t.clone(arena.allocator()); // does an internal transpose to linear memory
    try std.testing.expectEqualSlices(usize, &[_]usize{ 4, 2 }, t.shape);
    for (0..2) |ri| {
        for (0..4) |ci| {
            try std.testing.expectEqual(mat2x4[ri][ci], t.get(&[_]usize{ ci, ri }).item());
            try std.testing.expectEqual(mat2x4[ri][ci], tc.get(&[_]usize{ ci, ri }).item());
        }
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
            x.setItem(.{ 0, 0, i, j }, (f + 0.5) / 27.5 + fi);
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
            try std.testing.expectApproxEqRel(expected[ri][idx], v.*, 0.000001);
            idx += 1;
        }
    }
}
