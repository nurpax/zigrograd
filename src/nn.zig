const std = @import("std");
const zg = @import("./zigrograd.zig");
const Ndarray = @import("./ndarray.zig").Ndarray;

const Value = zg.Value;

pub fn SliceIterator(comptime C: type) type {
    return struct {
        index: usize,
        arr: []const C,

        pub fn init(arr: []const C) @This() {
            return @This(){
                .index = 0,
                .arr = arr,
            };
        }

        pub fn len(self: @This()) usize {
            return self.arr.len;
        }

        pub fn next(self: *@This()) ?C {
            if (self.index < self.arr.len) {
                const idx = self.index;
                self.index += 1;
                return self.arr[idx];
            }
            return null;
        }
    };
}

pub fn NestedModuleIterator(comptime Module: type) type {
    return struct {
        inner_it: Module.ParamIterator,
        outer_it: SliceIterator(Module),

        pub fn init(outer: anytype) @This() {
            var outer_it = SliceIterator(Module).init(outer);
            if (outer_it.next()) |out| {
                return @This(){
                    .outer_it = outer_it,
                    .inner_it = out.parameters(),
                };
            }
            unreachable; // should not happen: empty iterator
        }

        pub fn next(self: *@This()) ?*zg.Tensor {
            if (self.inner_it.next()) |p| {
                return p;
            }
            if (self.outer_it.next()) |out| {
                self.inner_it = out.parameters();
                return self.inner_it.next();
            }
            return null;
        }

        pub fn len(self: @This()) usize {
            var copy = self;
            var count: usize = 0;
            while (copy.next()) |_| : (count += 1) {}
            return count;
        }
    };
}

pub const Layer = struct {
    pub const ParamIterator = SliceIterator(*zg.Tensor);

    w: *zg.Tensor,
    b: *zg.Tensor,

    relu: bool,
    params: []*zg.Tensor,

    pub fn init(pool: *zg.NodePool, num_in: usize, num_out: usize, relu: bool) Layer {
        const scl = 1.0 / @sqrt(@as(f32, @floatFromInt(num_in)));
        var w = Ndarray(f32).init(pool.arena.allocator(), &[_]usize{ num_out, num_in });
        var b = Ndarray(f32).init(pool.arena.allocator(), &[_]usize{num_out});
        for (0..num_out) |j| {
            b.set(.{j}, (zg.Random.uniform() * 2 - 1) * scl);
            for (0..num_in) |i| {
                w.set(.{ j, i }, (zg.Random.uniform() * 2 - 1) * scl);
            }
        }

        var ret = Layer{
            .w = pool.tensor(w),
            .b = pool.tensor(b),
            .params = pool.arena.allocator().alloc(*zg.Tensor, 2) catch unreachable,
            .relu = relu,
        };
        ret.params[0] = ret.b;
        ret.params[1] = ret.w;
        return ret;
    }

    pub fn forward(self: *@This(), p: *zg.NodePool, x: *const zg.Tensor) *zg.Tensor {
        var out = p.dot(x, p.transpose(self.w));
        out = p.add(out, self.b);
        return if (self.relu) p.relu(out) else out;
    }

    pub fn parameters(self: *const @This()) ParamIterator {
        return ParamIterator.init(self.params);
    }
};

pub const Conv2d = struct {
    pub const Options = struct {
        bias: bool = true,
    };
    pub const ParamIterator = SliceIterator(*zg.Tensor);

    w: *zg.Tensor,
    b: ?*zg.Tensor,

    params: []*zg.Tensor,

    pub fn init(pool: *zg.NodePool, num_in: usize, num_out: usize, kernel: usize, opts: Options) @This() {
        const fan_in: f32 = @floatFromInt(num_in * kernel * kernel);
        const scl = 1.0 / @sqrt(fan_in);
        var w = Ndarray(f32).init(pool.arena.allocator(), &[_]usize{ num_out, num_in, kernel, kernel });
        const b = if (opts.bias) Ndarray(f32).init(pool.arena.allocator(), &[_]usize{num_out}) else null;
        for (0..num_out) |i| {
            if (b) |bb| {
                bb.set(.{i}, (zg.Random.uniform() * 2 - 1) * scl);
            }
            for (0..num_in) |j| {
                for (0..kernel) |kj| {
                    for (0..kernel) |ki| {
                        w.set(.{ i, j, kj, ki }, (zg.Random.uniform() * 2 - 1) * scl);
                    }
                }
            }
        }

        var ret = @This(){
            .w = pool.tensor(w),
            .b = if (b) |bb| pool.tensor(bb) else null,
            .params = pool.arena.allocator().alloc(*zg.Tensor, if (opts.bias) 2 else 1) catch unreachable,
        };
        ret.params[0] = ret.w;
        if (ret.b) |bb| {
            ret.params[1] = bb;
        }
        return ret;
    }

    pub fn forward(self: *@This(), p: *zg.NodePool, x: *const zg.Tensor) *zg.Tensor {
        // TODO options
        var out = p.conv2d(x, self.w, .{});
        if (self.b) |bb| {
            out = p.add(out, p.reshape(bb, &[_]usize{ 1, self.w.shape()[0], 1, 1 }));
        }
        return out;
    }

    pub fn parameters(self: *const @This()) ParamIterator {
        return ParamIterator.init(self.params);
    }
};

test "iterator" {
    var arr = [_]i32{ 0, 1, 2, 3 };
    var it = SliceIterator(i32).init(&arr);
    var idx: usize = 0;
    while (it.next()) |v| {
        try std.testing.expect(v == arr[idx]);
        idx += 1;
    }
}
