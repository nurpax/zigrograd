const std = @import("std");

pub const MnistLoader = struct {
    pub const Mode = enum { train, validate };
    alloc: std.mem.Allocator,
    num_samples: usize,
    img_data: []const u8,
    idx_data: []const u8,

    pub const Sample = struct {
        pixels: []const u8,
        label: u8,

        pub fn print(self: *const @This()) void {
            const asc = " .,:ilwW";
            var offs: usize = 0;
            std.debug.print("Sample label={d}\n", .{self.label});
            for (0..28) |_| {
                for (0..28) |_| {
                    const v = self.pixels[offs] >> 5;
                    offs += 1;
                    std.debug.print("{c}", .{asc[v]});
                }
                std.debug.print("\n", .{});
            }
        }
    };

    pub fn open(alloc: std.mem.Allocator, dir: []const u8, mode: Mode) !MnistLoader {
        const img_fname = try std.fs.path.join(alloc, &[_][]const u8{ dir, if (mode == .train) "train-images-idx3-ubyte" else "t10k-images-idx3-ubyte" });
        defer alloc.free(img_fname);
        const img_data = std.fs.cwd().readFileAlloc(alloc, img_fname, std.math.maxInt(u32)) catch |err| {
            std.log.err("failed to open file {s}", .{img_fname});
            return err;
        };
        errdefer alloc.free(img_data);

        const idx_fname = try std.fs.path.join(alloc, &[_][]const u8{ dir, if (mode == .train) "train-labels-idx1-ubyte" else "t10k-labels-idx1-ubyte" });
        defer alloc.free(idx_fname);
        const idx_data = std.fs.cwd().readFileAlloc(alloc, idx_fname, std.math.maxInt(u32)) catch |err| {
            std.log.err("failed to open file {s}", .{idx_fname});
            return err;
        };

        const num_img_samples = std.mem.bigToNative(u32, std.mem.bytesAsSlice(u32, img_data[4..8])[0]);
        const num_idx_samples = std.mem.bigToNative(u32, std.mem.bytesAsSlice(u32, idx_data[4..8])[0]);
        std.debug.assert(num_img_samples == num_idx_samples);
        return MnistLoader{
            .alloc = alloc,
            .num_samples = num_img_samples,
            .img_data = img_data,
            .idx_data = idx_data,
        };
    }

    pub fn sample(self: *const @This(), idx: usize) Sample {
        std.debug.assert(idx < self.num_samples);
        const offs = 16 + idx * 28 * 28;
        const img = self.img_data[offs .. offs + 784];
        return Sample{
            .pixels = img,
            .label = self.idx_data[8 + idx],
        };
    }

    pub fn deinit(self: *@This()) void {
        self.alloc.free(self.img_data);
        self.alloc.free(self.idx_data);
    }
};
