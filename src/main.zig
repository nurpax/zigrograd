const std = @import("std");

const zg = @import("./zigrograd.zig");
const nn = @import("./nn.zig");
const mnist = @import("./mnist.zig");
const time = @import("./time.zig");
const Ndarray = @import("./ndarray.zig").Ndarray;

var gpa = std.heap.GeneralPurposeAllocator(.{ .stack_trace_frames = 12 }){};

pub const Model = struct {
    linears: [3]nn.Layer,

    pub const ParamIterator = nn.NestedModuleIterator(nn.Layer);

    pub fn init(pool: *zg.NodePool) Model {
        return Model{
            .linears = .{
                nn.Layer.init(pool, 28 * 28, 50, true),
                nn.Layer.init(pool, 50, 50, true),
                nn.Layer.init(pool, 50, 10, false),
            },
        };
    }

    pub fn forward(self: *@This(), pool: *zg.NodePool, x: *const zg.Tensor) *zg.Tensor {
        const N = x.shape()[0];
        var tmp = self.linears[0].forward(pool, x);
        tmp = self.linears[1].forward(pool, tmp);
        tmp = self.linears[2].forward(pool, tmp);

        // logsoftmax = logits - log(reduce_sum(exp(logits), axis=1))
        const softmax_sum = pool.sum(pool.exp(tmp), .{ .axis = 1 });
        tmp = pool.sub(tmp, pool.reshape(pool.log(softmax_sum), .{ N, 1 }));
        return tmp;
    }

    pub fn parameters(self: *const @This()) ParamIterator {
        return ParamIterator.init(&self.linears);
    }
};

pub const ModelCnn = struct {
    convos: [2]nn.Conv2d,
    fc1: nn.Layer,
    iter_tmp: []*zg.Tensor,

    pub const ParamIterator = nn.SliceIterator(*zg.Tensor);

    pub fn init(pool: *zg.NodePool) ModelCnn {
        var self = ModelCnn{
            .convos = .{
                nn.Conv2d.init(pool, 1, 32, 3, .{ .bias = true }),
                nn.Conv2d.init(pool, 32, 64, 3, .{ .bias = true }),
            },
            .fc1 = nn.Layer.init(pool, 1600, 10, false),
            .iter_tmp = undefined,
        };
        var params = std.ArrayList(*zg.Tensor).init(pool.arena.allocator());
        for (self.convos) |t| {
            var it = t.parameters();
            while (it.next()) |v| {
                params.append(v) catch unreachable;
            }
        }

        var it = self.fc1.parameters();
        while (it.next()) |v| {
            params.append(v) catch unreachable;
        }
        self.iter_tmp = params.toOwnedSlice() catch unreachable;
        return self;
    }

    pub fn forward(self: *@This(), pool: *zg.NodePool, x: *const zg.Tensor) *zg.Tensor {
        const N = x.shape()[0];
        const xr = pool.reshape(x, .{ N, 1, 28, 28 });
        var tmp = pool.relu(self.convos[0].forward(pool, xr));

        tmp = pool.maxpool2d(tmp);

        const conv1s = time.now();
        tmp = pool.relu(self.convos[1].forward(pool, tmp));
        const conv1 = time.since(conv1s);
        _ = conv1;
        //std.debug.print("conv2d fwd {d} ms\n", .{time.to_ms(conv1)});

        tmp = pool.maxpool2d(tmp);
        tmp = pool.reshape(tmp, .{ N, -1 });

        tmp = self.fc1.forward(pool, tmp);

        //std.debug.print("conv1 {d} ms\n", .{time.to_ms(conv1)});

        const softmax_sum = pool.sum(pool.exp(tmp), .{ .axis = 1 });
        tmp = pool.sub(tmp, pool.reshape(pool.log(softmax_sum), .{ N, 1 }));
        return tmp;
    }

    pub fn parameters(self: *const @This()) ParamIterator {
        return ParamIterator.init(self.iter_tmp);
    }
};

fn samplesToBatch(pool: *zg.NodePool, samples: []mnist.MnistLoader.Sample) Ndarray(f32) {
    var pixin = Ndarray(f32).init(pool.arena.allocator(), &[_]usize{ samples.len, 28 * 28 });
    for (samples, 0..) |s, j| {
        for (s.pixels, 0..) |pix, i| {
            const fpix: f32 = @floatFromInt(pix);
            pixin.setItem(.{ j, i }, fpix / 127.5 - 1);
        }
    }
    return pixin;
}

fn validate(pool: *zg.NodePool, model: anytype, dataset: mnist.MnistLoader) void {
    const batch_size = 4;
    var correct_count: usize = 0;
    const max_samples = dataset.num_samples;

    std.debug.print("validating model accuracy: ", .{});
    std.debug.assert(max_samples & 3 == 0);
    const start = time.now();
    for (0..max_samples >> 2) |sample_idx| {
        pool.reset();

        var samples: [batch_size]mnist.MnistLoader.Sample = undefined;
        for (0..batch_size) |i| {
            samples[i] = dataset.sample(sample_idx * batch_size + i);
        }
        const batch = samplesToBatch(pool, &samples);
        const input = pool.tensor(batch);
        const logits = model.forward(pool, input);

        for (0..batch_size) |bidx| {
            const s = dataset.sample(sample_idx * batch_size + bidx);
            const label = logits.data.get(&[_]usize{bidx}).argmax();
            if (label == s.label) {
                correct_count += 1;
            }
        }
    }
    std.debug.print(" {d:.3}% elapsed {d:.2} ms\n", .{ @as(f32, @floatFromInt(correct_count)) / @as(f32, @floatFromInt(max_samples)) * 100.0, time.to_ms(time.since(start)) });
}

pub fn trainClassifier(init_pool: *zg.NodePool, fwd: *zg.NodePool) void {
    var model = ModelCnn.init(init_pool);
    //var model = Model.init(init_pool);

    const batch_size = 32;
    const num_epochs = 10;

    const mnist_path = "data/mnist/";
    var loader = mnist.MnistLoader.open(gpa.allocator(), mnist_path, .train) catch {
        std.debug.print("failed to open mnist. exiting", .{});
        return;
    };
    defer loader.deinit();
    var val_loader = mnist.MnistLoader.open(gpa.allocator(), mnist_path, .validate) catch {
        std.debug.print("failed to open mnist. exiting", .{});
        return;
    };
    defer val_loader.deinit();

    const init_arena = init_pool.arena.allocator();
    var backward = zg.Backward.init(init_arena);
    defer backward.deinit();

    var shuffle = init_arena.alloc(u64, loader.num_samples) catch unreachable;
    defer init_arena.free(shuffle);
    for (0..shuffle.len) |i| {
        shuffle[i] = i;
    }

    std.debug.print("model size: {d}\n", .{model.parameters().len()});

    const train_start: i64 = std.time.milliTimestamp();
    var tick_start_time: i64 = std.time.microTimestamp();
    for (0..num_epochs) |epoch_idx| {

        // Shuffle dataset order at the start of every epoch.
        zg.Random.shuffle(u64, shuffle);

        for (0..loader.num_samples / batch_size) |batch_idx| {
            // Clear forward pass memory pool.
            fwd.reset();

            // Zero gradients.
            var params_it = model.parameters();
            while (params_it.next()) |p| {
                p.grad.fill(0);
            }

            // Load a minibatch.
            var labels: [batch_size]usize = undefined;
            var samples: [batch_size]mnist.MnistLoader.Sample = undefined;
            for (0..batch_size) |idx| {
                const sample_idx = batch_idx * batch_size + idx;
                samples[idx] = loader.sample(shuffle[sample_idx]);
                labels[idx] = samples[idx].label;
            }

            // Predict logits and compute loss.
            const digits = fwd.tensor(samplesToBatch(fwd, &samples));
            const pred = model.forward(fwd, digits);
            var loss = fwd.sum(fwd.neg(fwd.gatherSlice(pred, &labels)), .{});
            const mb_loss = loss.data.item() / batch_size;
            backward.backward(fwd.arena.allocator(), loss);

            // Update weights.
            const lr = 0.01;
            const lr_mb = fwd.nd_scalar(lr / @as(f32, batch_size));
            params_it = model.parameters();
            while (params_it.next()) |p| {
                p.grad.mul_(lr_mb);
                p.data.sub_(p.grad);
            }

            const B = 25;
            if (batch_idx % B != 0) {
                continue;
            }

            const elapsed = (@as(f64, @floatFromInt(std.time.microTimestamp() - tick_start_time)) / 1e6);
            const sec_per_kimg = elapsed / (@as(f64, batch_size * B) / 1000);
            std.debug.print("epoch {d:<2} sample {d:<6} loss {d:.5} elapsed {d:.3} sec/kimg {d:.3}\n", .{ epoch_idx, batch_idx * batch_size, mb_loss, elapsed, sec_per_kimg });
            tick_start_time = std.time.microTimestamp();
        }
        validate(fwd, &model, val_loader);
    }

    const elapsed = std.time.milliTimestamp() - train_start;
    std.debug.print("training completed in {d} seconds\n", .{@as(f64, @floatFromInt(elapsed)) / 1e3});
}

pub fn main() !void {
    time.init() catch {
        std.debug.print("failed to initialize timer\n", .{});
        return;
    };

    var init_pool = zg.NodePool.init(gpa.allocator());
    defer init_pool.deinit();
    var fwd_pool = zg.NodePool.init(gpa.allocator());
    defer fwd_pool.deinit();
    trainClassifier(&init_pool, &fwd_pool);
}

test {
    @import("std").testing.refAllDecls(@This());
}
