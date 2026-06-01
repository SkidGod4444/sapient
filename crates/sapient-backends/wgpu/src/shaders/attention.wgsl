// FlashDecoding: causal grouped-query attention with online softmax. Never
// materialises the seq_q×seq_k score matrix; O(head_dim) working set per query row.
// Matches the CPU `scaled_dot_product_attention` (flash_attn_row), causal-only.
//
//   q: [batch, n_heads,    seq_q, head_dim]
//   k: [batch, n_kv_heads, seq_k, head_dim]   (cached prefix = seq_k - seq_q)
//   v: [batch, n_kv_heads, seq_k, head_dim]
//   out: [batch, n_heads,  seq_q, head_dim]
//
// One workgroup per (batch, head, query-row). 128 lanes cooperate: they parallelise
// the q·k dot (tree reduction) and the O = corr*O + p*V accumulation over head_dim.
// Each lane owns channels {lane, lane+128, ...}; o_reg holds its running output.
// GQA: head h reads kv-head h / (n_heads/n_kv_heads). f32 throughout.

struct P {
    batch: u32,
    n_heads: u32,
    n_kv_heads: u32,
    seq_q: u32,
    seq_k: u32,        // valid (attended) length
    head_dim: u32,
    kv_offset: u32,    // seq_k - seq_q
    jcount: u32,       // ceil(head_dim / 128)
    kv_stride: u32,    // allocated positions per kv-head (cache capacity; ≥ seq_k)
    scale: f32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read>       q:   array<f32>;
@group(0) @binding(1) var<storage, read>       k:   array<f32>;
@group(0) @binding(2) var<storage, read>       v:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform>             p:   P;

const WG: u32 = 128u;
var<workgroup> partial: array<f32, 128>;
var<workgroup> sh_p: f32;
var<workgroup> sh_corr: f32;
var<workgroup> sh_m: f32;
var<workgroup> sh_l: f32;

@compute @workgroup_size(128)
fn cs_main(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = wg.x + wg.y * nwg.x;
    if (idx >= p.batch * p.n_heads * p.seq_q) { return; }

    let qi = idx % p.seq_q;
    let bh = idx / p.seq_q;
    let h = bh % p.n_heads;
    let b = bh / p.n_heads;
    let kv_rep = p.n_heads / p.n_kv_heads;
    let kvh = h / kv_rep;

    let hd = p.head_dim;
    let q_base = ((b * p.n_heads + h) * p.seq_q + qi) * hd;
    let kv_base = (b * p.n_kv_heads + kvh) * p.kv_stride * hd;
    let attend_len = qi + p.kv_offset + 1u;   // causal

    let tid = lid.x;

    // Cache this lane's q channels and zero its output accumulator.
    var q_reg: array<f32, 4>;
    var o_reg: array<f32, 4>;
    for (var j = 0u; j < p.jcount; j = j + 1u) {
        let c = tid + j * WG;
        if (c < hd) { q_reg[j] = q[q_base + c]; }
        o_reg[j] = 0.0;
    }
    if (tid == 0u) { sh_m = -3.0e38; sh_l = 0.0; }
    workgroupBarrier();

    for (var ki = 0u; ki < attend_len; ki = ki + 1u) {
        let krow = kv_base + ki * hd;
        // Partial dot over this lane's channels.
        var local = 0.0;
        for (var j = 0u; j < p.jcount; j = j + 1u) {
            let c = tid + j * WG;
            if (c < hd) { local = local + q_reg[j] * k[krow + c]; }
        }
        partial[tid] = local;
        workgroupBarrier();
        // Tree reduction → partial[0] = full q·k.
        var s = WG / 2u;
        loop {
            if (s == 0u) { break; }
            if (tid < s) { partial[tid] = partial[tid] + partial[tid + s]; }
            workgroupBarrier();
            s = s / 2u;
        }
        // Online-softmax update (lane 0), broadcast p & correction.
        if (tid == 0u) {
            let score = partial[0] * p.scale;
            let m_new = max(sh_m, score);
            sh_p = exp(score - m_new);
            sh_corr = exp(sh_m - m_new);
            sh_l = sh_corr * sh_l + sh_p;
            sh_m = m_new;
        }
        workgroupBarrier();
        // O = corr*O + p*V[ki].
        let pw = sh_p;
        let corr = sh_corr;
        let vrow = kv_base + ki * hd;
        for (var j = 0u; j < p.jcount; j = j + 1u) {
            let c = tid + j * WG;
            if (c < hd) { o_reg[j] = corr * o_reg[j] + pw * v[vrow + c]; }
        }
        workgroupBarrier();
    }

    let inv_l = select(1.0 / sh_l, 1.0 / 1.1754944e-38, sh_l == 0.0);
    for (var j = 0u; j < p.jcount; j = j + 1u) {
        let c = tid + j * WG;
        if (c < hd) { out[q_base + c] = o_reg[j] * inv_l; }
    }
}
