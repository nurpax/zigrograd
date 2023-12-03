const std = @import("std");
pub const PtrSet = @import("./ptrset.zig").PtrSet;
const time = @import("./time.zig");
const ndarray = @import("./ndarray.zig");
const Ndarray = ndarray.Ndarray;

const ExprKind = enum {
    imm,
    gather_slice,
    stack_slice,
    reshape,
    sum_axis,
    conv2d,
    avgpool2d,
    maxpool2d,
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
    reshape: struct {
        arg0: *Tensor,
        order: []usize,
    },
    sum_axis: struct {
        arg0: *Tensor,
        opts: Ndarray(f32).SumOpts,
    },
    conv2d: struct {
        arg_x: *Tensor,
        arg_w: *Tensor,
        opts: ndarray.Conv2dOpts,
    },
    avgpool2d: struct {
        arg0: *Tensor,
    },
    maxpool2d: struct {
        arg0: *Tensor,
        idx: []const u8, // max idx array for backwards pass
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
                        const prev_grad = self.grad;

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
            Expr.reshape => |expr| {
                expr.arg0.grad.add_(self.grad.reshape(alloc, expr.arg0.grad.shape));
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
                    const sh = expr.arg0.data.shape;
                    var new_shape = alloc.dupe(usize, sh) catch unreachable;
                    const axis = expr.opts.axis orelse unreachable;
                    new_shape[axis] = 1;
                    var gg = expr.arg0.data.zerosLike(alloc);
                    gg.add_(self.grad.reshape(alloc, new_shape));
                    expr.arg0.grad.add_(gg);
                }
            },
            Expr.conv2d => |expr| {
                const op_x = expr.arg_x.data;
                const op_w = expr.arg_w.data;

                const g = Ndarray(f32).conv2dBackwards(alloc, op_x, op_w, self.grad, expr.opts);
                expr.arg_x.grad.add_(g.dx);
                expr.arg_w.grad.add_(g.dw);
            },
            Expr.avgpool2d => |expr| {
                const g = Ndarray(f32).avgpool2dBackwards(alloc, expr.arg0.data, self.grad);
                expr.arg0.grad.add_(g);
            },
            Expr.maxpool2d => |expr| {
                const g = Ndarray(f32).maxpool2dBackwards(alloc, expr.arg0.data, expr.idx, self.grad);
                expr.arg0.grad.add_(g);
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
                Expr.reshape => |op| {
                    self.backward_rec(op.arg0);
                },
                Expr.stack_slice => |op| {
                    for (op.args) |a| {
                        self.backward_rec(a);
                    }
                },
                Expr.sum_axis => |op| {
                    self.backward_rec(op.arg0);
                },
                Expr.conv2d => |op| {
                    self.backward_rec(op.arg_x);
                    self.backward_rec(op.arg_w);
                },
                Expr.avgpool2d => |op| {
                    self.backward_rec(op.arg0);
                },
                Expr.maxpool2d => |op| {
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
        const n = self.arena.allocator().create(Tensor) catch unreachable;
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
        const arr = p.nd_scalar(v);
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

    pub fn reshape(p: *NodePool, a: *const Tensor, order: []const usize) *Tensor {
        const order_copy = p.allocator().dupe(usize, order) catch unreachable;
        return new(p, Expr{ .reshape = .{ .arg0 = @constCast(a), .order = order_copy } }, a.data.reshape(p.allocator(), order));
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

    pub fn conv2d(p: *NodePool, x: *const Tensor, w: *const Tensor, opts: ndarray.Conv2dOpts) *Tensor {
        const s = Ndarray(f32).conv2d(p.allocator(), x.data, w.data, opts);
        return new(p, Expr{ .conv2d = .{ .arg_x = @constCast(x), .arg_w = @constCast(w), .opts = opts } }, s);
    }

    pub fn avgpool2d(p: *NodePool, x: *const Tensor) *Tensor {
        const s = Ndarray(f32).avgpool2d(p.allocator(), x.data);
        return new(p, Expr{ .avgpool2d = .{ .arg0 = @constCast(x) } }, s);
    }

    pub fn maxpool2d(p: *NodePool, x: *const Tensor) *Tensor {
        const res = Ndarray(f32).maxpool2d(p.allocator(), x.data);
        return new(p, Expr{ .maxpool2d = .{ .arg0 = @constCast(x), .idx = res.idx } }, res.out);
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

    const arr = Ndarray(f32).scalar(pool.arena.allocator(), 3);
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
    const tt = h.tensor(t2d);
    const t = h.stackSlice(&[_]*Tensor{ tt, tt });
    _ = t;
}

test "conv2d" {
    var h = NodePool.init(std.testing.allocator);
    defer h.deinit();

    const N = 1;
    const C = 1;
    const C_out = 1;
    var x = Ndarray(f32).init(h.allocator(), &[_]usize{ N, C, 28, 28 });

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
    const ww = Ndarray(f32).initFromSlice2d(h.allocator(), &kernel);
    const w = Ndarray(f32).init(h.allocator(), &[_]usize{ C_out, C, 3, 3 });
    var wv = w.get(.{ 0, 0 });
    wv.assign(ww);

    const x_in = h.tensor(x);
    const w_in = h.tensor(w);
    const out = h.conv2d(x_in, w_in, .{});
    const loss = h.sum(out, .{});

    var bw = Backward.init(h.allocator());
    defer bw.deinit();
    bw.backward(h.allocator(), loss);
    var it = w_in.grad.get(.{ 0, 0 }).iterator(.{});
    std.debug.print("gradients:\n", .{});
    while (it.next()) |gv| {
        std.debug.print("{d:.4} ", .{gv.*});
    }
    std.debug.print("\n", .{});
}

test "avgpool2d" {
    var h = NodePool.init(std.testing.allocator);
    defer h.deinit();

    const N = 1;
    var x = Ndarray(f32).init(h.allocator(), &[_]usize{ N, 1, 8, 8 });

    for (0..8) |i| {
        for (0..8) |j| {
            const f: f32 = @floatFromInt(j);
            const fi: f32 = @floatFromInt(i);
            x.set(.{ 0, 0, i, j }, (f + 0.5) / 7.5 + fi);
        }
    }

    const x_in = h.tensor(x);
    const b = x.onesLike(h.allocator());
    const b_in = h.tensor(b);

    const xx = h.mul(x_in, b_in);
    const out = h.avgpool2d(xx);
    const loss = h.sum(out, .{});

    var bw = Backward.init(h.allocator());
    defer bw.deinit();

    bw.backward(h.allocator(), loss);
    {
        var it = out.data.get(.{ 0, 0 }).iterator(.{});
        std.debug.print("out:\n", .{});
        while (it.next()) |gv| {
            std.debug.print("{d:.4} ", .{gv.*});
        }
        std.debug.print("\n", .{});
    }
    {
        var it = b_in.grad.get(.{ 0, 0 }).iterator(.{});
        std.debug.print("b_in.grad:\n", .{});
        while (it.next()) |gv| {
            std.debug.print("{d:.4} ", .{gv.*});
        }
        std.debug.print("\n", .{});
    }
}

test "conv2d_advanced" {
    var h = NodePool.init(std.testing.allocator);
    defer h.deinit();

    const x_val = [_]f32{ -1.5514980018927877, -0.45355248136054743, -0.10798815571134068, 1.6813452250614485, 1.066550264988564, 0.6633564511909164, 2.4400674769315236, 0.48514782024716885, -0.18607428307276289, 0.13533005216388486, 0.8405814541371837, -0.6363887933413804, 0.08949976489880083, 0.06139981713174468, -0.7558171412790363, 1.096900289871768, 1.3296573084152972, -0.2592350511296529, 0.36668468789764763, -0.4875436220852375, 0.13994147184844374, 0.02590458263428963, -0.3674612209176302, 1.0197886240569474, 1.0381933647578485, -0.6334187958820615, 0.8410574480189749, 0.9686714599032803, 1.0818320225694251, -0.14385685333715065, 1.9475904002908915, -0.7438812668684192, -0.8213981990153552, 2.1283749635485374, 0.3628453296655981, 1.356865477188806, -0.05563420231993154, 0.445353842849581, -2.404217762007139, 0.02647354391777463, 1.4909254872399176, 0.807275213983085, -0.8139687441408764, 0.3533120270030588, -0.4664811890032297, 1.1550534110291013, -2.8495373078481543, -0.1695861972873907, 0.8584739064323185, 0.23393223551376857, 0.7321162330511874, 0.7185158278323177, -0.169565322925104, -0.09931585039817467, -0.9428846453921761, 0.7539051187660699, -0.26354505003680345, 0.1112711325922794, 1.9110960751238129, 0.44802362666038853, -1.584819943104402, -0.46910021180919687, 0.42372012970475037, -0.2613698887035046, 0.4315598794058523, 1.558981675612073, 0.19621113964238804, -0.6414177662645734, -1.079357301551414, -0.3626477128082682, -0.2745810517093195, 0.956239495637474, 1.1562063035588903, 0.03093844606101535, -0.6089857734754863, -0.41184118802170555, -0.06901377939234882, 1.871347810841448, -0.5669260289632926, -1.291871501424081, 1.029757543026265, -1.2491683484250402, 0.3243143860463973, 0.6874473210369401, 1.655925210431674, -1.5520979722926258, 1.5439221050973233, 0.49157376848612705, 0.5820705763162383, 1.0359185734204903, -0.5858012701427279, -0.22448143207385857, -0.44864274662416725, -0.0673826372872893, 0.459635637610304, -0.26044115253056704, -1.3094307097253624, 0.5438145305377742, -0.5330800372438476, 0.1463711495346702, -2.67083093681356, -2.6167407738980843, -1.2415795494276192, 0.7214214716442021, 0.8824838417728297, 0.17253111728250664, -0.33582642560975634, -1.392755962524067, -0.7557575829745099, 0.09849615755092847, -0.5223461081370866, 1.6991497755498752, 0.5632193662750183, 0.15197072777394735, -0.44204479551080084, 0.27871753218167405, 0.2720400698375866, 0.6156530146864546, -0.5476137312149718, 0.7172102387826412, -1.8370066792333306, 0.5994601338179392, -0.48860717725804803, -2.2260761023904143, -0.3281493265577712, -1.3916636378400484, -0.1336221208709417, -1.8316022544228212 };
    var x_arr = Ndarray(f32).initFromSlice1d(h.allocator(), &x_val);
    x_arr = x_arr.reshape(h.allocator(), &[_]usize{ 1, 2, 8, 8 });
    const w_val = [_]f32{ -0.524547430378453, 0.9747822972108419, -1.1231552756635068, -0.42633611007264227, -0.023012277306989598, 1.1356485517902213, -0.07104296420207049, -0.1265924860424059, 0.2149062104598703, 0.27382239792078567, 0.5006874922173139, -0.6422097474425572, -0.6112641382501514, -1.4046008478456051, 0.4095152877790629, 0.4580929832508549, 0.5913775275409168, 0.21839454812740458, 0.7102749641364988, -1.5294428464273317, -1.9272930157930424, -0.7996904701864265, 0.7390212108084837, -0.6840592077195161, 0.27866023443934756, 0.5828640070799599, -0.13430547400680254, -0.15438652733363561, 1.7187064932870728, -0.2823414519614258, 2.425027253111737, -0.4727796944872028, -0.617376975937011, -1.1316718811738435, 0.2935159692240746, 2.0829659465299826, -0.8938614958373575, -0.037827852536557016, 0.9092289225998712, 0.06123240612897065, 1.0325554768401786, 0.6288153875918607, 0.6524428263304702, 0.372061726152726, -0.4959131166885948, 0.8436640145192035, -0.1833303744523578, -1.0157391619388667, 2.41619981506533, 0.06444891665282051, -1.1115372959167424, -0.9778637076414575, 0.9976113053059511, -3.031635541418667, 1.6254311087447468, -0.49922492234781285, -0.9250769179645586, -0.4840877043763306, -2.516814417440328, -0.9195301345791288, -1.2787062352221015, 2.4207607635250867, -1.139439784500358, -0.7889888223263387, 0.8865675200088619, -0.6395951700565828, 0.3351138849386009, 0.3141576268160149, -0.8193305387394493, 1.0401322894822038, -0.6963300665894205, -0.15935121256322757 };
    var w_arr = Ndarray(f32).initFromSlice1d(h.allocator(), &w_val);
    w_arr = w_arr.reshape(h.allocator(), &[_]usize{ 4, 2, 3, 3 });
    const out_expected_val = [_]f32{ 1.1179402892351917, -1.313743412798906, 2.5863114632556643, 1.3612180652988832, -6.5848928602710775, 1.568303897393023, 2.060555071693625, 1.794967080850274, -2.411286296313418, -4.764255670140336, 3.6814352064578446, -1.0178722012206551, -3.965169944837345, 1.928576020908845, -1.0682796195652524, 0.43816737492906993, -2.2509018350318786, -4.521171808456034, -0.4709571507073285, 0.9980018800122471, -3.1489470903632304, 2.2028777433986835, -2.3916793144762276, 7.072296851833412, 0.17429507479229422, -2.7128381743290024, 4.340871640771361, 2.6142952838819182, -0.8435026327896313, -3.2552986674057567, 1.0056772325751835, -0.15269719525291792, -1.6847630671560647, -4.925702586066403, 1.7579497395327743, -4.105115616097189, 4.50992162074143, 1.5934672708555109, -4.891744851758855, -8.540206676668637, -4.560274450292534, 2.2535774576662955, -2.298468894020504, -5.321596460338643, -0.19037967731613187, 1.9291933645343975, 10.169630206994093, -6.73649829881786, -0.8595369496991665, 4.299341622207057, -4.351468535630868, -3.7138014200634086, -5.999518764148986, 6.031428528219786, 0.1671183930989947, -9.079986858003114, -3.6614451916095963, 1.6265311928004025, -4.6743006115058785, -7.170159671928981, -2.2599939523497112, -2.0773241423192297, 2.7137357486155778, -7.020484016524823, 0.551219066445568, -1.5099741614419022, 3.1724185908745484, -3.1792056742878345, -3.4355817749483726, -4.050008449832081, 4.153928300562048, 2.1606033760316317, 2.667101333952647, 4.508407451554638, -4.251529589620504, 1.5037062415911762, -6.7125632743428305, 5.3051739421803665, 6.8700540181665515, -5.378168942367546, -0.3381026536015668, 3.0916326453333896, 0.05867189627454256, 2.34597044856628, 5.500838145674587, 1.459907380552691, 8.617743212729458, 8.665617033087022, 4.283236144958287, -0.9670786780014757, 2.6001546565801092, 8.866841399794783, 4.33811194180489, 2.113063353528515, -2.361615978884995, -16.132176174313763, 4.475391460870337, -0.26605449579849577, 2.4852285578376563, -2.2044822188753046, -2.204047769950656, -11.770569302745397, 4.364537255759839, 8.739712339891643, 1.8087257513590664, 5.432381332974649, -3.8238862362784882, 4.285065429054989, -2.263068113544016, -4.183942380107256, -3.4129099576586013, 0.4432775793786885, 4.236901764343945, -3.6386884281181016, -6.126183006395039, 2.1224728462892286, 1.3047490309077339, 1.4470015996542962, -2.5462900212702455, 3.243865644086187, 5.276276565658076, -4.149586442002571, -1.0814410230270883, -2.1364106153808695, -2.4855444237704485, -10.643563076377383, -1.237903262118058, -9.60505594590474, 2.1175876792233423, -2.4176284147028637, 6.313854870473406, -3.924108802771716, -3.0722614306383695, 4.530171930922417, 2.5245512638483376, -1.0460921962400178, 3.6297099156758157, 5.540217263128639, -2.4555387551603878, 4.207782590725414, -2.3038621396304757, -5.790802008409669, 5.308350130135769, 3.86465451686201 };
    const dw_expected_val = [_]f32{ 11.812972755833123, 8.52379152457142, 8.513860437186025, 12.78891648306318, 3.706811221152672, 3.2781527154118613, 12.637494100571788, 4.812396876609107, 3.049527111864628, -4.510884695070311, -7.885407124954715, -7.662351262080504, -3.174658693862489, -6.9538732901217095, -5.562835736264205, -10.815353302895973, -11.168051008270627, -10.885265895168786, 11.812972755833123, 8.52379152457142, 8.513860437186025, 12.78891648306318, 3.706811221152672, 3.2781527154118613, 12.637494100571788, 4.812396876609107, 3.049527111864628, -4.510884695070311, -7.885407124954715, -7.662351262080504, -3.174658693862489, -6.9538732901217095, -5.562835736264205, -10.815353302895973, -11.168051008270627, -10.885265895168786, 11.812972755833123, 8.52379152457142, 8.513860437186025, 12.78891648306318, 3.706811221152672, 3.2781527154118613, 12.637494100571788, 4.812396876609107, 3.049527111864628, -4.510884695070311, -7.885407124954715, -7.662351262080504, -3.174658693862489, -6.9538732901217095, -5.562835736264205, -10.815353302895973, -11.168051008270627, -10.885265895168786, 11.812972755833123, 8.52379152457142, 8.513860437186025, 12.78891648306318, 3.706811221152672, 3.2781527154118613, 12.637494100571788, 4.812396876609107, 3.049527111864628, -4.510884695070311, -7.885407124954715, -7.662351262080504, -3.174658693862489, -6.9538732901217095, -5.562835736264205, -10.815353302895973, -11.168051008270627, -10.885265895168786 };
    const dx_expected_val = [_]f32{ 0.9172971466654352, -0.1744161774354247, -3.240712464256661, -3.240712464256661, -3.240712464256661, -3.240712464256661, -4.158009610922097, -3.0662962868212364, -0.7315847318409936, -2.591548063040509, -5.496969752778309, -5.496969752778309, -5.496969752778309, -5.496969752778309, -4.765385020937316, -2.9054216897377994, -1.150230870495348, 0.23889980902050345, -4.221274045453182, -4.221274045453182, -4.221274045453182, -4.221274045453182, -3.0710431749578344, -4.460173854473684, -1.150230870495348, 0.23889980902050345, -4.221274045453182, -4.221274045453182, -4.221274045453182, -4.221274045453182, -3.0710431749578344, -4.460173854473684, -1.150230870495348, 0.23889980902050345, -4.221274045453182, -4.221274045453182, -4.221274045453182, -4.221274045453182, -3.0710431749578344, -4.460173854473684, -1.150230870495348, 0.23889980902050345, -4.221274045453182, -4.221274045453182, -4.221274045453182, -4.221274045453182, -3.0710431749578326, -4.460173854473684, -2.067528017160783, 0.4133159864559284, -0.9805615811965196, -0.9805615811965196, -0.9805615811965196, -0.9805615811965196, 1.0869664359642632, -1.393877567652448, -0.4186461386543542, 2.8304478720610127, 1.2756957073251276, 1.2756957073251276, 1.2756957073251276, 1.2756957073251276, 1.6943418459794817, -1.554752164735885, 0.17411106278001476, 3.096742193840906, 0.5168566624414734, 0.5168566624414734, 0.5168566624414734, 0.5168566624414734, 0.34274559966145857, -2.5798855313994324, 4.739187877645532, 6.16304500984245, 1.4444299556288769, 1.4444299556288769, 1.4444299556288769, 1.4444299556288769, -3.294757922016654, -4.718615054213572, 4.12787756156329, 6.73790942924173, 1.1296681157036494, 1.1296681157036494, 1.1296681157036494, 1.1296681157036494, -2.9982094458596387, -5.6082413135380795, 4.12787756156329, 6.73790942924173, 1.1296681157036494, 1.1296681157036494, 1.1296681157036494, 1.1296681157036494, -2.9982094458596387, -5.6082413135380795, 4.12787756156329, 6.73790942924173, 1.1296681157036494, 1.1296681157036494, 1.1296681157036494, 1.1296681157036494, -2.9982094458596387, -5.6082413135380795, 4.12787756156329, 6.73790942924173, 1.1296681157036494, 1.1296681157036494, 1.1296681157036494, 1.1296681157036494, -2.9982094458596387, -5.6082413135380795, 3.9537664987832746, 3.6411672354008235, 0.6128114532621766, 0.612811453262177, 0.612811453262177, 0.612811453262177, -3.3409550455210977, -3.028355782138647, -0.6113103160822422, 0.5748644193992798, -0.31476183992522744, -0.31476183992522744, -0.3147618399252273, -0.3147618399252273, 0.29654847615701496, -0.889626259324507 };

    const x = h.tensor(x_arr);
    const w = h.tensor(w_arr);
    const out = h.conv2d(x, w, .{});
    const loss = h.sum(out, .{});

    const flat_out = out.data.reshape(h.allocator(), &[_]usize{Ndarray(f32).shapeProd(out.shape())});

    try std.testing.expect(flat_out.shape.len == 1 and flat_out.shape[0] == out_expected_val.len);

    for (out_expected_val, flat_out.getContigousSlice(.{})) |expected, actual| {
        try std.testing.expectApproxEqRel(expected, actual, 0.00001);
    }

    var bw = Backward.init(h.allocator());
    defer bw.deinit();
    bw.backward(h.allocator(), loss);

    const flat_dx = x.grad.reshape(h.allocator(), &[_]usize{Ndarray(f32).shapeProd(x.grad.shape)});
    for (dx_expected_val, flat_dx.getContigousSlice(.{})) |expected, actual| {
        try std.testing.expectApproxEqRel(expected, actual, 0.00001);
    }

    const flat_dw = w.grad.reshape(h.allocator(), &[_]usize{Ndarray(f32).shapeProd(w.grad.shape)});
    std.debug.print("\n", .{});
    for (dw_expected_val, flat_dw.getContigousSlice(.{})) |expected, actual| {
        try std.testing.expectApproxEqRel(expected, actual, 0.00001);
    }
}
