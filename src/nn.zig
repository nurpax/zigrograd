const std = @import("std");
const zg = @import("./zigrograd.zig");

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
                    .inner_it = @constCast(&out).parameters(),
                };
            }
            unreachable; // should not happen: empty iterator
        }

        pub fn next(self: *@This()) ?*zg.Value {
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

pub const Neuron = struct {
    nonlin: bool,
    num_in: usize,
    params: []*zg.Value, // indexing: [0,num_in] = weights, [num_in]=bias

    pub const ParamIterator = SliceIterator(*zg.Value);

    pub fn init(pool: *zg.NodePool, num_in: usize, nonlin: bool) Neuron {
        const scl = 1.0 / @sqrt(@as(f32, @floatFromInt(num_in)));
        var params = pool.arena.allocator().alloc(*zg.Value, num_in + 1) catch unreachable;
        for (params) |*p| {
            const r = zg.Random.uniform() * 2 - 1; // -1,1 range
            p.* = pool.c(r * scl);
        }
        return Neuron{
            .nonlin = nonlin,
            .num_in = num_in,
            .params = params,
        };
    }

    pub fn forward(self: *@This(), p: *zg.NodePool, x: []*const zg.Value) *zg.Value {
        var sum = self.params[self.num_in]; // add bias
        for (0..self.num_in) |i| {
            sum = p.add(sum, p.mul(self.params[i], x[i]));
        }
        return if (self.nonlin) p.relu(sum) else sum;
    }

    pub fn parameters(self: *const @This()) ParamIterator {
        return ParamIterator.init(self.params);
    }
};

pub const Layer = struct {
    pub const ParamIterator = NestedModuleIterator(Neuron);

    neurons: []Neuron,
    out: []*zg.Value,

    pub fn init(pool: *zg.NodePool, num_in: usize, num_out: usize, nonlin: bool) Layer {
        var neurons = pool.arena.allocator().alloc(Neuron, num_out) catch unreachable;
        var out = pool.arena.allocator().alloc(*zg.Value, num_out) catch unreachable;
        for (0..num_out) |layer_idx| {
            neurons[layer_idx] = Neuron.init(pool, num_in, nonlin);
        }
        return Layer{
            .neurons = neurons,
            .out = out,
        };
    }

    pub fn forward(self: *@This(), fwd_pool: *zg.NodePool, x: []*const zg.Value) []*zg.Value {
        for (self.neurons, 0..) |*n, i| {
            std.debug.assert(n.num_in == x.len);
            self.out[i] = n.forward(fwd_pool, x);
        }
        return self.out;
    }

    pub fn parameters(self: *const @This()) ParamIterator {
        return ParamIterator.init(self.neurons);
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
