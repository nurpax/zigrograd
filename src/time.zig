pub extern fn stm_setup() void;
pub extern fn stm_now() u64;
pub extern fn stm_since(prev: u64) u64;
pub extern fn stm_sec(prev: u64) f64;
pub extern fn stm_ms(prev: u64) f64;
pub extern fn stm_us(prev: u64) f64;
pub extern fn stm_ns(prev: u64) f64;

pub fn init() void {
    stm_setup();
}

pub fn now() u64 {
    return stm_now();
}

pub fn since(prev: u64) u64 {
    return stm_since(prev);
}

pub fn to_sec(t: u64) f64 {
    return stm_sec(t);
}

pub fn to_ms(t: u64) f64 {
    return stm_ms(t);
}

pub fn to_us(t: u64) f64 {
    return stm_us(t);
}

pub fn to_ns(t: u64) f64 {
    return stm_ns(t);
}
