const std = @import("std");

fn hash(v: usize) usize {
    // TODO probably not a reliable hash function, could end up conflicting a lot
    return v >> 3;
}

// Quick and dirty "ptr set" that's used in the backward pass to store information
// about visited nodes.  This usage pattern does not need delete so the implementation
// is very simple.
//
// Implemented as a linear probing hash table.
pub fn PtrSet(comptime T: type) type {
    return struct {
        alloc: std.mem.Allocator,
        table: []usize,
        used: usize,

        pub fn init(alloc: std.mem.Allocator, init_size: usize) @This() {
            var table = alloc.alloc(usize, std.math.ceilPowerOfTwoAssert(usize, init_size)) catch unreachable;
            @memset(table, 0);
            return @This(){
                .alloc = alloc,
                .table = table,
                .used = 0,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.alloc.free(self.table);
        }

        fn insertIntptr(self: *@This(), ip: usize) bool {
            const hk = hash(ip);
            const len_mask = self.table.len - 1;
            for (0..self.table.len) |i| {
                const tbl_idx = (hk + i) & len_mask;
                if (self.table[tbl_idx] == 0) {
                    self.table[tbl_idx] = ip;
                    self.used += 1;
                    return false;
                }
                if (self.table[tbl_idx] == ip) {
                    return true; // existing value
                }
            }
            unreachable; // should not get here if there's room in the table
        }

        // return true if value was already contained in the set, false otherwise
        pub fn insert(self: *@This(), v: T) bool {
            const ip = @intFromPtr(v);
            const ret = self.insertIntptr(ip);

            if (self.used > self.table.len / 2) {
                self.resize(self.table.len * 2);
            }
            return ret;
        }

        fn resize(self: *@This(), new_size: usize) void {
            var old_table = self.table;
            var new_table = self.alloc.alloc(usize, new_size) catch unreachable;
            @memset(new_table, 0);

            self.used = 0;
            self.table = new_table;
            for (old_table) |e| {
                if (e != 0) {
                    _ = self.insertIntptr(e);
                }
            }
            self.alloc.free(old_table);
        }

        pub fn clearRetainingCapacity(self: *@This()) void {
            self.used = 0;
            @memset(self.table, 0);
        }
    };
}

test "simplest" {
    var random: std.rand.DefaultPrng = std.rand.DefaultPrng.init(0x1234);

    const N = 1000;
    const ITERS = 16;

    var set = PtrSet(*const i32).init(std.testing.allocator, 16);
    defer set.deinit();
    for (0..ITERS) |_| {
        var hmap = std.AutoHashMap(*const i32, bool).init(std.testing.allocator);
        defer hmap.deinit();
        for (0..N) |_| {
            var rndPtr = random.random().int(usize) & 0xfff0;
            if (rndPtr != 0) {
                var iptr: *const i32 = @ptrFromInt(rndPtr);
                const res = hmap.getOrPut(iptr) catch unreachable;
                var existing = set.insert(iptr);
                try std.testing.expect(existing == res.found_existing);
            }
        }
        set.clearRetainingCapacity();
    }
}
