const std = @import("std");
pub const PtrSet = @import("./ptrset.zig").PtrSet;
const time = @import("./time.zig");

const ExprKind = enum {
    imm,
    unop,
    binop,
};

const UnopKind = enum {
    exp,
    log,
    relu,
};

const BinopKind = enum {
    add,
    sub,
    mul,
    pow_imm,
};

const Expr = union(ExprKind) {
    imm: f32,
    unop: struct {
        op: UnopKind,
        arg0: *Value,
    },
    binop: struct {
        op: BinopKind,
        arg0: *Value,
        arg1: *Value,
    },
};

pub const Value = struct {
    data: f32 = 0,
    grad: f32 = 0,
    expr: Expr,

    pub fn backward_node(self: *Value) void {
        switch (self.expr) {
            Expr.imm => {},
            Expr.unop => |expr| {
                switch (expr.op) {
                    UnopKind.exp => {
                        expr.arg0.grad += @exp(expr.arg0.data) * self.grad;
                    },
                    UnopKind.log => {
                        expr.arg0.grad += (1.0 / expr.arg0.data) * self.grad;
                    },
                    UnopKind.relu => {
                        const g = if (self.data > 0) self.grad else 0;
                        expr.arg0.grad += g;
                    },
                }
            },
            Expr.binop => |expr| {
                switch (expr.op) {
                    BinopKind.add => {
                        expr.arg0.grad += self.grad;
                        expr.arg1.grad += self.grad;
                    },
                    BinopKind.sub => {
                        expr.arg0.grad += self.grad;
                        expr.arg1.grad -= self.grad;
                    },
                    BinopKind.mul => {
                        expr.arg0.grad += expr.arg1.data * self.grad;
                        expr.arg1.grad += expr.arg0.data * self.grad;
                    },
                    BinopKind.pow_imm => {
                        std.debug.assert(expr.arg1.expr == .imm); // only immediate field supported for now
                        const e = expr.arg1.expr.imm;
                        expr.arg0.grad += (e * std.math.pow(f32, expr.arg0.data, (e - 1))) * self.grad;
                    },
                }
            },
        }
    }
    pub fn print(self: *const Value) void {
        std.debug.print("Value(data={d}, grad={d})\n", .{ self.data, self.grad });
    }
};

// Wrapper around AutoHashMap or PtrSet as they have a slightly different API
pub fn PointerSet(comptime V: type, comptime useStdHashMap: bool) type {
    return struct {
        pub const T = if (useStdHashMap) std.AutoHashMap(V, void) else PtrSet(V);
        hash: T,

        pub fn init(alloc: std.mem.Allocator) @This() {
            return @This(){ .hash = if (useStdHashMap) T.init(alloc) else T.init(alloc, 1024) };
        }
        pub fn deinit(self: *@This()) void {
            self.hash.deinit();
        }
        pub fn insert(self: *@This(), v: V) bool {
            if (useStdHashMap) {
                const res = self.hash.getOrPut(v) catch unreachable;
                return res.found_existing;
            }
            return self.hash.insert(v);
        }
        pub fn clearRetainingCapacity(self: *@This()) void {
            self.hash.clearRetainingCapacity();
        }
    };
}

pub const Backward = struct {
    const Set = PointerSet(*const Value, false);
    //const Set = PointerSet(*const Value, true);

    topo: std.ArrayList(*Value),
    visited: Set,

    pub fn init(alloc: std.mem.Allocator) Backward {
        return Backward{
            .topo = std.ArrayList(*Value).init(alloc),
            .visited = Set.init(alloc),
        };
    }

    pub fn deinit(self: *@This()) void {
        self.topo.deinit();
        self.visited.deinit();
    }

    fn backward_rec(self: *@This(), root: *Value) void {
        if (!self.visited.insert(root)) {
            switch (root.expr) {
                Expr.unop => |op| {
                    self.backward_rec(op.arg0);
                },
                Expr.binop => |op| {
                    self.backward_rec(op.arg0);
                    self.backward_rec(op.arg1);
                },
                Expr.imm => {},
            }
            self.topo.append(root) catch unreachable;
        }
    }

    pub fn backward(self: *@This(), root: *Value) void {
        //const t0 = time.now();

        // Compute topological sort.
        self.topo.clearRetainingCapacity();
        self.visited.clearRetainingCapacity();
        self.backward_rec(root);

        //std.debug.print("bw elapsed {d:.2} ns per node\n", .{time.to_ns(time.since(t0)) / @as(f64, @floatFromInt(self.topo.items.len))});

        // Backprop.
        root.grad = 1;
        var topo = self.topo.items;
        for (self.topo.items, 0..) |_, idx| {
            topo[topo.len - idx - 1].backward_node();
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

    pub fn new(self: *NodePool, expr: Expr, v: f32) *Value {
        var n = self.arena.allocator().create(Value) catch unreachable;
        n.* = Value{
            .data = v,
            .grad = 0,
            .expr = expr,
        };
        return n;
    }

    pub fn c(p: *NodePool, v: f32) *Value {
        return new(p, Expr{ .imm = v }, v);
    }

    fn unop(p: *NodePool, op: UnopKind, a: *const Value, data: f32) *Value {
        return new(p, Expr{ .unop = .{ .op = op, .arg0 = @constCast(a) } }, data);
    }

    fn binop(p: *NodePool, op: BinopKind, a: *const Value, b: *const Value, data: f32) *Value {
        return new(p, Expr{ .binop = .{ .op = op, .arg0 = @constCast(a), .arg1 = @constCast(b) } }, data);
    }

    pub fn add(p: *NodePool, a: *const Value, b: *const Value) *Value {
        return binop(p, .add, a, b, a.data + b.data);
    }

    pub fn mul(p: *NodePool, a: *const Value, b: *const Value) *Value {
        return binop(p, .mul, a, b, a.data * b.data);
    }

    pub fn sub(p: *NodePool, a: *const Value, b: *const Value) *Value {
        return binop(p, .sub, a, b, a.data - b.data);
    }

    pub fn pow(p: *NodePool, a: *const Value, e: f32) *Value {
        return binop(p, .pow_imm, a, c(p, e), std.math.pow(f32, a.data, e));
    }

    pub fn neg(p: *NodePool, a: *const Value) *Value {
        return mul(p, a, c(p, -1));
    }

    pub fn relu(p: *NodePool, a: *const Value) *Value {
        return unop(p, .relu, a, @max(0, a.data));
    }

    pub fn exp(p: *NodePool, a: *const Value) *Value {
        return unop(p, .exp, a, @exp(a.data));
    }

    pub fn log(p: *NodePool, a: *const Value) *Value {
        return unop(p, .log, a, @log(a.data));
    }

    pub fn div(p: *NodePool, a: *const Value, b: *const Value) *Value {
        return mul(p, a, pow(p, b, -1));
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

test "init" {
    var gpa = std.heap.GeneralPurposeAllocator(.{ .stack_trace_frames = 12 }){};
    //var pool = NodePool.init(std.testing.allocator);
    var pool = NodePool.init(gpa.allocator());
    const p = &pool;
    const a = p.c(3);
    try std.testing.expect(a.data == 3);
    try std.testing.expect(p.add(p.c(1), p.c(2)).data == 3);
    try std.testing.expect(p.mul(p.c(2), p.c(2)).data == 4);
    try std.testing.expect(p.div(p.c(8), p.c(4)).data == 2);
    const expected: f32 = 2.718;
    const ev: f32 = p.exp(p.c(1)).data;
    try std.testing.expectApproxEqAbs(expected, ev, 0.01);
    try std.testing.expect(p.log(p.c(1)).data == 0);
    try std.testing.expect(p.log(p.c(1)).data == 0);
}
