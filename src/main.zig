const std = @import("std");

const zg = @import("./zigrograd.zig");
const nn = @import("./nn.zig");
const mnist = @import("./mnist.zig");
const time = @import("./time.zig");

var gpa = std.heap.GeneralPurposeAllocator(.{ .stack_trace_frames = 12 }){};

pub const Model = struct {
    linears: [3]nn.Layer,
    out: [10]*zg.Value,

    pub const ParamIterator = nn.NestedModuleIterator(nn.Layer);

    pub fn init(pool: *zg.NodePool) Model {
        return Model{
            .linears = .{
                nn.Layer.init(pool, 28 * 28, 50, true),
                nn.Layer.init(pool, 50, 50, true),
                nn.Layer.init(pool, 50, 10, false),
            },
            .out = .{undefined} ** 10,
        };
    }

    pub fn forward(self: *@This(), pool: *zg.NodePool, x: []*const zg.Value) []*zg.Value {
        var tmp = self.linears[0].forward(pool, x);
        tmp = self.linears[1].forward(pool, tmp);
        tmp = self.linears[2].forward(pool, tmp);

        @memcpy(&self.out, tmp);

        // logsoftmax = logits - log(reduce_sum(exp(logits), axis))
        var softmax_sum = pool.c(0);
        for (tmp) |tmp_i| {
            softmax_sum = pool.add(softmax_sum, pool.exp(tmp_i));
        }
        const sum_log = pool.log(softmax_sum);
        for (self.out, 0..) |out_i, i| {
            self.out[i] = pool.sub(out_i, sum_log);
        }
        return &self.out;
    }

    pub fn parameters(self: *const @This()) ParamIterator {
        return ParamIterator.init(&self.linears);
    }
};

pub fn argmax(arr: []*const zg.Value) usize {
    var m = arr[0].data;
    var max_idx: usize = 0;
    var idx: usize = 1;
    while (idx < arr.len) : (idx += 1) {
        const val = arr[idx].data;
        if (val > m) {
            m = val;
            max_idx = idx;
        }
    }
    return max_idx;
}

pub fn validate(pool: *zg.NodePool, model: *Model, dataset: mnist.MnistLoader) void {
    var correct_count: usize = 0;
    const max_samples = dataset.num_samples;

    std.debug.print("validating model accuracy: ", .{});
    for (0..max_samples) |sample_idx| {
        pool.reset();

        const s = dataset.sample(sample_idx);
        var input: [28 * 28]*zg.Value = undefined;
        for (s.pixels, 0..) |pix, i| {
            const fpix: f32 = @floatFromInt(pix);
            input[i] = pool.c(fpix / 127.5 - 1);
        }

        const logits = model.forward(pool, &input);
        const label = argmax(logits);
        if (label == s.label) {
            correct_count += 1;
        }
    }
    std.debug.print(" {d:.3}%\n", .{@as(f32, @floatFromInt(correct_count)) / @as(f32, @floatFromInt(max_samples)) * 100.0});
}

pub fn trainClassifier(init_pool: *zg.NodePool, fwd: *zg.NodePool) void {
    var model = Model.init(init_pool);

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
                p.grad = 0;
            }

            // Train one minibatch.
            var mb_loss: f32 = 0.0;
            for (0..batch_size) |idx| {
                const sample_idx = batch_idx * batch_size + idx;
                const sample = loader.sample(shuffle[sample_idx]);

                var input: [28 * 28]*zg.Value = undefined;
                for (sample.pixels, 0..) |pix, i| {
                    const fpix: f32 = @floatFromInt(pix);
                    input[i] = fwd.c(fpix / 127.5 - 1);
                }

                // Predict logits and compute loss.
                const pred = model.forward(fwd, &input);
                const loss = fwd.neg(pred[sample.label]); // TODO is the NLL loss correct?  Seems to work though..
                mb_loss += loss.data;

                backward.backward(loss);
            }
            mb_loss /= batch_size;

            // Update weights.
            const lr = 0.01;
            const lr_mb = lr / @as(f32, batch_size);
            params_it = model.parameters();
            while (params_it.next()) |p| {
                p.data -= p.grad * lr_mb;
            }

            if (batch_idx % 5 != 0) {
                continue;
            }

            const elapsed = (@as(f64, @floatFromInt(std.time.microTimestamp() - tick_start_time)) / 1e6);
            const sec_per_kimg = elapsed / (@as(f64, batch_size * 5) / 1000);
            std.debug.print("epoch {d:<2} sample {d:<6} loss {d:.5} elapsed {d:.3} sec/kimg {d:.3}\n", .{ epoch_idx, batch_idx * batch_size, mb_loss, elapsed, sec_per_kimg });

            if (batch_idx != 0 and batch_idx % 100 == 0) {
                validate(fwd, &model, val_loader);
            }
            tick_start_time = std.time.microTimestamp();
        }
    }
    validate(fwd, &model, val_loader);
}

pub fn main() !void {
    time.init();

    var init_pool = zg.NodePool.init(gpa.allocator());
    defer init_pool.deinit();
    var fwd_pool = zg.NodePool.init(gpa.allocator());
    defer fwd_pool.deinit();
    trainClassifier(&init_pool, &fwd_pool);
}

test {
    @import("std").testing.refAllDecls(@This());
}
