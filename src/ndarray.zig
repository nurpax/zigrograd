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

pub const TransposeOpts = struct {
    axes: ?[]const usize = null,
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
                var ptr: *Dtype = @ptrFromInt(@intFromPtr(self.arr.buf.ptr) + self.data_ptr * @sizeOf(Dtype));

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
            if (a.shape.len != b.shape.len) {
                return false;
            }
            for (a.shape, b.shape) |ad, bd| {
                if (ad != bd) {
                    return false;
                }
            }
            return true;
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
            var strides = alloc.alloc(usize, shape.len) catch unreachable;
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
            var arr = self.emptyLike(alloc);
            @memset(arr.buf, 0);
            return arr;
        }

        pub fn onesLike(self: *const Ndarray(Dtype), alloc: std.mem.Allocator) @This() {
            var arr = self.emptyLike(alloc);
            @memset(arr.buf, 1);
            return arr;
        }

        pub fn scalar(alloc: std.mem.Allocator, v: Dtype) @This() {
            var arr = @This().init(alloc, &[_]usize{});
            arr.buf[0] = v;
            return arr;
        }

        pub fn offset(self: *const @This(), idx: anytype) usize {
            var offs: usize = self.offs;
            inline for (idx, 0..) |idx_i, i| {
                offs += idx_i * self.strides[i];
            }
            return offs;
        }

        // TODO combine tuple and slice
        pub fn offset2(self: *const @This(), idx: []const usize) usize {
            var offs: usize = self.offs;
            for (idx, 0..) |idx_i, i| {
                offs += idx_i * self.strides[i];
            }
            return offs;
        }

        pub fn get(self: *const @This(), idx: []const usize) @This() {
            const offs = self.offset2(idx);
            return @This(){
                .strides = self.strides[idx.len..],
                .shape = self.shape[idx.len..],
                .offs = 0,
                .contiguous = self.contiguous,
                .buf = self.buf[offs..(offs + @This().shapeProd(self.shape[idx.len..]))],
            };
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

        pub fn set(self: *const @This(), idx: anytype, v: Dtype) void {
            self.buf[self.offset(idx)] = v;
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
                var s0 = if (ai >= 0) shape_a[@intCast(ai)] else 1;
                var s1 = if (bi >= 0) shape_b[@intCast(bi)] else 1;
                new_shape[i] = @max(s0, s1);
                ai -= 1;
                bi -= 1;
            }
            return new_shape;
        }

        fn binop(op: ndvec.Binop, alloc: std.mem.Allocator, a: @This(), b: @This()) @This() {
            if (a.contiguous and b.contiguous and a.equalShapes(b)) {
                var dst = a.emptyLike(alloc);
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
                var val = src.*;
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
                std.debug.assert(@This().equalShapes(a.get(&[_]usize{0}), b));
                for (0..a.shape[0]) |i| {
                    const aa = a.get(&[_]usize{i});
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
                        var a_buf = a.buf[outer_a..];
                        for (0..b.shape[1]) |_| {
                            dst.buf[inner_d] = ndvec.innerProductStride1(a_buf, a.shape[1], b.buf[outer_b..]);
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
                        var a_buf = a.buf[outer_a..];
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

    var ones = t2d.onesLike(arena.allocator());
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
            std.debug.print("index {} {} {}\n", .{ yi, xi, t2d.get(&[_]usize{ yi, xi }).item() });
            try std.testing.expectEqual(t2d.get(&[_]usize{ yi, xi }).item(), @floatFromInt(cnt * 2));
            cnt += 1;
        }
    }

    var dst = t2d.get(&[_]usize{0});
    const src = t2d.get(&[_]usize{1});
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

    var zerod = t2d.get(&[_]usize{ 0, 0 });
    try std.testing.expect(zerod.shape.len == 0);
    try std.testing.expectEqual(@as(f32, 0), zerod.item());
    zerod = t2d.get(&[_]usize{ 0, 1 });
    try std.testing.expectEqual(@as(f32, 1), zerod.item());
    zerod = t2d.get(&[_]usize{ 1, 0 });
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
    var tens2 = Ndarray(f32).initFromSlice1d(arena.allocator(), &arr2);
    const s0 = Ndarray(f32).dot(alloc, tens1, tens1.zerosLike(alloc));
    try std.testing.expectEqual(@as(f32, 0), s0.item());
    const s1 = Ndarray(f32).dot(alloc, tens1, tens2);
    try std.testing.expectEqual(@as(f32, 1 + 4 + 6), s1.item());

    const mat2x4 = [_][]const f32{
        &[_]f32{ 1, 2, 3, 4 },
        &[_]f32{ 5, 5, 6, 7 },
    };
    var mat = Ndarray(f32).initFromSlice2d(arena.allocator(), &mat2x4);
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
