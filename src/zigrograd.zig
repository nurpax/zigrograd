const std = @import("std");
pub const PtrSet = @import("./ptrset.zig").PtrSet;
const time = @import("./time.zig");
const ndarray = @import("./ndarray.zig");
const Ndarray = ndarray.Ndarray;

const ExprKind = enum {
    imm,
    gather_slice,
    stack_slice,
    sum_axis,
    unop,
    binop,
};

const UnopKind = enum {
    neg,
    exp,
    log,
    relu,
    transpose2, // transpose last two dims
};

const BinopKind = enum {
    add,
    sub,
    mul,
    dot,
    // pow_imm,
};

const Expr = union(ExprKind) {
    imm: Ndarray(f32),
    gather_slice: struct {
        arg: *Tensor,
        idx: []usize,
    },
    stack_slice: struct {
        args: []const *Tensor,
    },
    sum_axis: struct {
        arg0: *Tensor,
        opts: Ndarray(f32).SumOpts,
    },
    unop: struct {
        op: UnopKind,
        arg0: *Tensor,
    },
    binop: struct {
        op: BinopKind,
        arg0: *Tensor,
        arg1: *Tensor,
    },
};

pub const Tensor = struct {
    data: Ndarray(f32),
    grad: Ndarray(f32),
    expr: Expr,

    pub fn shape(self: *const Tensor) []usize {
        return self.data.shape;
    }

    // See https://mostafa-samir.github.io/auto-diff-pt2/#unbroadcasting-adjoints
    fn unbroadcast(alloc: std.mem.Allocator, target: Ndarray(f32), grad: Ndarray(f32)) Ndarray(f32) {
        const broadcast_idx = 0;

        var g = grad;
        while (g.shape.len > target.shape.len) {
            g = g.sum(alloc, .{ .axis = broadcast_idx });
        }

        if (g.shape.len == target.shape.len) {
            for (target.shape, 0..) |si, i| {
                if (si == 1) {
                    g = g.sum(alloc, .{ .axis = i, .keep_dims = true });
                }
            }
        }
        return g;
    }

    pub fn backward_node(self: *Tensor, alloc: std.mem.Allocator) void {
        switch (self.expr) {
            Expr.imm => {},
            Expr.unop => |expr| {
                switch (expr.op) {
                    UnopKind.neg => {
                        expr.arg0.grad.sub_(self.grad);
                    },
                    UnopKind.exp => {
                        var e = expr.arg0.data.exp(alloc);
                        e.mul_(self.grad);
                        expr.arg0.grad.add_(e);
                    },
                    UnopKind.log => {
                        var g = expr.arg0.data.onesLike(alloc);
                        g.div_(expr.arg0.data);
                        g.mul_(self.grad);
                        expr.arg0.grad.add_(g);
                        //expr.arg0.grad += (1.0 / expr.arg0.data) * self.grad;
                    },
                    UnopKind.relu => {
                        const g = self.grad.reluBackwards(alloc, self.data);
                        expr.arg0.grad.add_(g);
                    },
                    UnopKind.transpose2 => {
                        var permute: []const usize = &[_]usize{0};
                        if (self.grad.shape.len == 2) {
                            permute = &[_]usize{ 1, 0 };
                        } else if (self.grad.shape.len == 3) {
                            permute = &[_]usize{ 0, 2, 1 };
                        } else {
                            std.debug.assert(false); // TODO
                        }
                        const g = self.grad.transpose(alloc, .{ .axes = permute });
                        expr.arg0.grad.add_(g);
                    },
                }
            },
            Expr.binop => |expr| {
                switch (expr.op) {
                    BinopKind.add => {
                        expr.arg0.grad.add_(unbroadcast(alloc, expr.arg0.grad, self.grad));
                        expr.arg1.grad.add_(unbroadcast(alloc, expr.arg1.grad, self.grad));
                    },
                    BinopKind.sub => {
                        expr.arg0.grad.add_(unbroadcast(alloc, expr.arg0.grad, self.grad));
                        expr.arg1.grad.sub_(unbroadcast(alloc, expr.arg1.grad, self.grad));
                    },
                    BinopKind.mul => {
                        const a = Ndarray(f32).mul(alloc, expr.arg1.data, self.grad);
                        expr.arg0.grad.add_(unbroadcast(alloc, expr.arg0.grad, a));
                        const b = Ndarray(f32).mul(alloc, expr.arg0.data, self.grad);
                        expr.arg1.grad.add_(unbroadcast(alloc, expr.arg1.grad, b));
                    },
                    BinopKind.dot => {
                        var op_a = expr.arg0.data;
                        var op_b = expr.arg1.data;
                        var prev_grad = self.grad;

                        const a = Ndarray(f32).dot(alloc, prev_grad, op_b.transpose(alloc, .{}));
                        std.debug.assert(Ndarray(f32).equalShapes(a, expr.arg0.data));
                        expr.arg0.grad.add_(a);

                        const b = Ndarray(f32).dot(alloc, op_a.transpose(alloc, .{}), prev_grad);
                        std.debug.assert(Ndarray(f32).equalShapes(b, expr.arg1.data));
                        expr.arg1.grad.add_(b);
                    },
                    // BinopKind.pow_imm => {
                    //     std.debug.assert(expr.arg1.expr == .imm); // only immediate field supported for now
                    //     const e = expr.arg1.expr.imm;
                    //     expr.arg0.grad += (e * std.math.pow(f32, expr.arg0.data, (e - 1))) * self.grad;
                    // },
                }
            },
            Expr.gather_slice => |expr| {
                for (expr.idx, 0..) |idx, ith| {
                    var arg_grad = expr.arg.grad.get(&[_]usize{ ith, idx });
                    arg_grad.add_(self.grad.get(&[_]usize{ith}));
                }
            },
            Expr.stack_slice => |expr| {
                for (0..expr.args.len) |i| {
                    var args_grad = expr.args[i].grad;
                    args_grad.add_(self.grad.get(&[_]usize{i}));
                }
            },
            Expr.sum_axis => |expr| {
                std.debug.assert(!expr.opts.keep_dims);
                if (expr.opts.axis == null) {
                    std.debug.assert(std.meta.eql(expr.opts, Ndarray(f32).SumOpts{}));
                    var ones = expr.arg0.data.onesLike(alloc);
                    ones.mul_(self.grad); // broadcasts self.grad
                    expr.arg0.grad.add_(ones);
                } else {
                    var sh = expr.arg0.data.shape;
                    var new_shape = alloc.dupe(usize, sh) catch unreachable;
                    const axis = expr.opts.axis orelse unreachable;
                    new_shape[axis] = 1;
                    var gg = expr.arg0.data.zerosLike(alloc);
                    gg.add_(self.grad.reshape(alloc, new_shape));
                    expr.arg0.grad.add_(gg);
                }
            },
        }
    }
    pub fn print(self: *const Tensor) void {
        // TODO works with only scalars
        std.debug.print("Value(data={d:.2}, grad={d:.2})\n", .{ self.data.get(&[_]usize{}).item(), self.grad.get(&[_]usize{}).item() });
    }
};

pub const Backward = struct {
    topo: std.ArrayList(*Tensor),
    visited: PtrSet(*const Tensor),

    pub fn init(alloc: std.mem.Allocator) Backward {
        return Backward{
            .topo = std.ArrayList(*Tensor).init(alloc),
            .visited = PtrSet(*const Tensor).init(alloc, 1024),
        };
    }

    pub fn deinit(self: *@This()) void {
        self.topo.deinit();
        self.visited.deinit();
    }

    fn backward_rec(self: *@This(), root: *Tensor) void {
        if (!self.visited.insert(root)) {
            switch (root.expr) {
                Expr.unop => |op| {
                    self.backward_rec(op.arg0);
                },
                Expr.binop => |op| {
                    self.backward_rec(op.arg0);
                    self.backward_rec(op.arg1);
                },
                Expr.gather_slice => |op| {
                    self.backward_rec(op.arg);
                },
                Expr.stack_slice => |op| {
                    for (op.args) |a| {
                        self.backward_rec(a);
                    }
                },
                Expr.sum_axis => |op| {
                    self.backward_rec(op.arg0);
                },
                Expr.imm => {},
            }
            self.topo.append(root) catch unreachable;
        }
    }

    pub fn backward(self: *@This(), alloc: std.mem.Allocator, root: *Tensor) void {
        // Compute topological sort.
        self.topo.clearRetainingCapacity();
        self.visited.clearRetainingCapacity();

        self.backward_rec(root);

        // Backprop.
        root.grad.fill(1);
        var topo = self.topo.items;
        for (self.topo.items, 0..) |_, idx| {
            topo[topo.len - idx - 1].backward_node(alloc);
        }
    }
};

pub const NodePool = struct {
    arena: std.heap.ArenaAllocator,

    pub fn init(alloc: std.mem.Allocator) NodePool {
        return NodePool{
            .arena = std.heap.ArenaAllocator.init(alloc),
        };
    }

    pub fn deinit(self: *@This()) void {
        self.arena.deinit();
    }

    pub fn reset(self: *@This()) void {
        _ = self.arena.reset(.retain_capacity);
    }

    pub fn new(self: *NodePool, expr: Expr, v: Ndarray(f32)) *Tensor {
        var n = self.arena.allocator().create(Tensor) catch unreachable;
        n.* = Tensor{
            .data = v,
            .grad = v.zerosLike(self.arena.allocator()),
            .expr = expr,
        };
        return n;
    }

    fn allocator(p: *NodePool) std.mem.Allocator {
        return p.arena.allocator();
    }

    pub fn nd_scalar(p: *NodePool, v: f32) Ndarray(f32) {
        return Ndarray(f32).scalar(p.allocator(), v);
    }

    pub fn scalar(p: *NodePool, v: f32) *Tensor {
        var arr = p.nd_scalar(v);
        return new(p, Expr{ .imm = arr }, arr);
    }

    pub fn tensor(p: *NodePool, v: Ndarray(f32)) *Tensor {
        return new(p, Expr{ .imm = v }, v);
    }

    fn unop(p: *NodePool, op: UnopKind, a: *const Tensor, data: Ndarray(f32)) *Tensor {
        return new(p, Expr{ .unop = .{ .op = op, .arg0 = @constCast(a) } }, data);
    }

    fn assertDimsEqual(a_shape: []usize, b_shape: []usize) void {
        std.debug.assert(a_shape.len == b_shape.len);
        for (a_shape, 0..) |adim_i, i| {
            std.debug.assert(adim_i == b_shape[i]);
        }
    }

    fn binop(p: *NodePool, op: BinopKind, a: *const Tensor, b: *const Tensor, data: Ndarray(f32)) *Tensor {
        return new(p, Expr{ .binop = .{ .op = op, .arg0 = @constCast(a), .arg1 = @constCast(b) } }, data);
    }

    pub fn add(p: *NodePool, a: *const Tensor, b: *const Tensor) *Tensor {
        return binop(p, .add, a, b, Ndarray(f32).add(p.allocator(), a.data, b.data));
    }

    pub fn mul(p: *NodePool, a: *const Tensor, b: *const Tensor) *Tensor {
        return binop(p, .mul, a, b, Ndarray(f32).mul(p.allocator(), a.data, b.data));
    }

    pub fn sub(p: *NodePool, a: *const Tensor, b: *const Tensor) *Tensor {
        return binop(p, .sub, a, b, Ndarray(f32).sub(p.allocator(), a.data, b.data));
    }

    pub fn dot(p: *NodePool, a: *const Tensor, b: *const Tensor) *Tensor {
        return binop(p, .dot, a, b, Ndarray(f32).dot(p.allocator(), a.data, b.data));
    }

    pub fn pow(p: *NodePool, a: *const Tensor, e: f32) *Tensor {
        return binop(p, .pow_imm, a, p.scalar(e), std.math.pow(f32, a.data, e));
    }

    pub fn transpose(p: *NodePool, a: *const Tensor) *Tensor {
        var permute: []const usize = &[_]usize{0};
        if (a.shape().len == 2) {
            permute = &[_]usize{ 1, 0 };
        } else if (a.shape().len == 3) {
            permute = &[_]usize{ 0, 2, 1 };
        } else {
            std.debug.assert(false); // TODO
        }
        return unop(p, .transpose2, a, a.data.transpose(p.allocator(), .{ .axes = permute }));
    }

    pub fn neg(p: *NodePool, a: *const Tensor) *Tensor {
        return unop(p, .neg, a, a.data.neg(p.allocator()));
    }

    pub fn relu(p: *NodePool, a: *const Tensor) *Tensor {
        return unop(p, .relu, a, a.data.clipMin(p.allocator(), 0));
    }

    pub fn exp(p: *NodePool, a: *const Tensor) *Tensor {
        return unop(p, .exp, a, a.data.exp(p.allocator()));
    }

    pub fn log(p: *NodePool, a: *const Tensor) *Tensor {
        return unop(p, .log, a, a.data.log(p.allocator()));
    }

    pub fn div(p: *NodePool, a: *const Tensor, b: *const Tensor) *Tensor {
        return mul(p, a, pow(p, b, -1));
    }

    pub fn gatherSlice(p: *NodePool, a: *const Tensor, idx: []usize) *Tensor {
        const mb_size = a.data.shape[0];
        std.debug.assert(mb_size == idx.len);
        const idx_copy = p.allocator().dupe(usize, idx) catch unreachable;
        var data = Ndarray(f32).init(p.allocator(), &[_]usize{mb_size});
        for (0..data.shape[0]) |i| {
            data.set(.{i}, a.data.get(&[_]usize{ i, idx[i] }).item());
        }
        return new(p, Expr{ .gather_slice = .{ .arg = @constCast(a), .idx = idx_copy } }, data);
    }

    // Hardcoded for axis=0
    pub fn stackSlice(p: *NodePool, arr: []const *Tensor) *Tensor {
        const new_dims = arr[0].data.shape.len + 1;
        var new_shape: [ndarray.max_dims]usize = undefined;
        new_shape[0] = arr.len;
        for (arr[0].data.shape[0..], 1..) |s, i| {
            new_shape[i] = s;
        }
        var v = Ndarray(f32).init(p.arena.allocator(), new_shape[0..new_dims]);
        for (0..arr.len) |i| {
            var dst = v.get(&[_]usize{i});
            dst.assign(arr[i].data);
        }
        return new(p, Expr{ .stack_slice = .{ .args = arr } }, v);
    }

    pub fn sum(p: *NodePool, a: *const Tensor, opts: Ndarray(f32).SumOpts) *Tensor {
        const s = a.data.sum(p.arena.allocator(), opts);
        return new(p, Expr{ .sum_axis = .{ .arg0 = @constCast(a), .opts = opts } }, s);
    }
};

pub const Random = struct {
    var random: std.rand.DefaultPrng = std.rand.DefaultPrng.init(0x1234);

    pub fn setSeed(seed: u64) void {
        random = std.rand.DefaultPrng.init(seed);
    }

    // return a random float in the [0, 1) range
    pub fn uniform() f32 {
        return random.random().float(f32);
    }

    // return a random float in the [0, 1) range
    pub fn shuffle(comptime T: type, buf: []T) void {
        random.random().shuffle(T, buf);
    }
};

test "scalars" {
    var gpa = std.heap.GeneralPurposeAllocator(.{ .stack_trace_frames = 12 }){};
    //var pool = NodePool.init(std.testing.allocator);
    var pool = NodePool.init(gpa.allocator());
    const p = &pool;

    var arr = Ndarray(f32).scalar(pool.arena.allocator(), 3);
    const a = p.tensor(arr);
    try std.testing.expect(a.data.get(&[_]usize{}).item() == 3);
    try std.testing.expect(p.add(p.scalar(1), p.scalar(2)).data.item() == 3);
    try std.testing.expect(p.mul(p.scalar(2), p.scalar(2)).data.item() == 4);
}

test "scalar bw" {
    var gpa = std.heap.GeneralPurposeAllocator(.{ .stack_trace_frames = 12 }){};

    var bw = Backward.init(gpa.allocator());
    defer bw.deinit();
    var h = NodePool.init(gpa.allocator());

    const a = h.scalar(-4);
    const b = h.scalar(2);
    var c = h.add(a, b);
    var d = h.mul(a, b);
    c = h.add(c, h.add(c, h.scalar(1)));
    c = h.add(c, h.add(h.add(h.scalar(1), c), h.neg(a)));
    const t0 = h.relu(h.add(b, a));
    d = h.add(d, h.add(h.mul(d, h.scalar(2)), t0));
    var g = d;

    bw.backward(gpa.allocator(), g);
    g.print(); // Value(data=-24.0, grad=1)
    a.print(); // Value(data=-4.0, grad=6.0)
    b.print(); // Value(data=2.0, grad=-12.0)

}

test "stack" {
    var h = NodePool.init(std.testing.allocator);
    defer h.deinit();

    var t2d = Ndarray(f32).init(h.arena.allocator(), &[_]usize{ 2, 4 });
    for (0..t2d.shape[0]) |yi| {
        for (0..t2d.shape[1]) |xi| {
            t2d.set(.{ yi, xi }, @floatFromInt(xi + yi * 10));
        }
    }
    var tt = h.tensor(t2d);

    const t = h.stackSlice(&[_]*Tensor{ tt, tt });
    _ = t;
}
