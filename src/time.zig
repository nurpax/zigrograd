const std = @import("std");

var timer: std.time.Timer = undefined;

pub fn init() !void {
    timer = try std.time.Timer.start();
}

pub fn now() u64 {
    return timer.read();
}

pub fn since(prev: u64) u64 {
    return now() - prev;
}

pub fn to_sec(t: u64) f64 {
    return @as(f64, @floatFromInt(t)) / 1e9;
}

pub fn to_ms(t: u64) f64 {
    return @as(f64, @floatFromInt(t)) / 1e6;
}

pub fn to_us(t: u64) f64 {
    return @as(f64, @floatFromInt(t)) / 1e3;
}

pub fn to_ns(t: u64) f64 {
    return @as(f64, @floatFromInt(t));
}
