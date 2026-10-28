use super::ToRgb;
use super::icc::ICCProfile;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone)]
pub(crate) struct DeviceCmyk;

impl ToRgb for DeviceCmyk {
    fn convert(&self, input: &[u8], output: &mut [u8]) -> Option<()> {
        let pixels = input.len() / 4;
        if CONVERTED.fetch_add(pixels, Ordering::Relaxed) + pixels > TABLE_PIXELS {
            return CMYK_TRANSFORM.convert(input, output);
        }

        for (cmyk, rgb) in input
            .as_chunks::<4>()
            .0
            .iter()
            .zip(output.as_chunks_mut::<3>().0)
        {
            *rgb = cmyk_to_rgb(*cmyk);
        }

        Some(())
    }
}

/// Pixels converted with `TABLE` so far. moxcms takes about 7 ms to build its transform but
/// then converts a pixel in about 8 ns against the table's 24, so once this many have been
/// converted (12 ms or so) it's built and used instead. Both give the same values.
static CONVERTED: AtomicUsize = AtomicUsize::new(0);
const TABLE_PIXELS: usize = 1 << 19;

static CMYK_TRANSFORM: LazyLock<ICCProfile> = LazyLock::new(|| {
    ICCProfile::new(
        include_bytes!("../../assets/CGATS001Compat-v2-micro.icc"),
        4,
    )
    .unwrap()
});

const GRID: usize = 17;

/// The table moxcms 0.8.1 interpolates for 8-bit CMYK to sRGB from
/// `CGATS001Compat-v2-micro.icc`, built ahead of time: the profile sampled at 17 points along
/// each of C, M, Y and K (K varying slowest, then C, M and Y), three 8-bit levels each, and a
/// byte of padding so that each entry can be read as four. `matches_moxcms` checks it.
static TABLE: &[u8; GRID * GRID * GRID * GRID * 3 + 1] =
    include_bytes!("../../assets/cmyk-to-srgb.bin");

/// Where an 8-bit level falls between grid points: the lower point's index, the upper one's,
/// and the weight of the upper one in Q0.15, as moxcms computes them.
#[derive(Clone, Copy)]
struct Weight {
    x: usize,
    x_n: usize,
    w: i32,
}

const fn weights() -> [Weight; 256] {
    let mut weights = [Weight { x: 0, x_n: 0, w: 0 }; 256];
    let mut index = 0;
    while index < 256 {
        let x = index * (GRID - 1) / 255;
        let x_n = if x + 1 < GRID { x + 1 } else { GRID - 1 };
        let scale = (GRID - 1) as f32 * (1.0 / 255.0);
        let w = (index as f32 * scale - x as f32) * ((1 << 15) - 1) as f32;
        // Round half away from zero, as `f32::round` does; `w` isn't negative.
        weights[index] = Weight {
            x,
            x_n,
            w: (w + 0.5) as i32,
        };
        index += 1;
    }
    weights
}

static WEIGHTS: [Weight; 256] = weights();

/// `_mm_mulhrs_epi16`: the rounded high half of a Q0.15 product.
#[inline(always)]
fn mulhrs(a: i32, b: i32) -> i32 {
    (a * b + 0x4000) >> 15
}

/// What moxcms's fixed-point executor gives with its default (linear) interpolation:
/// trilinear over C, M and Y in the two K slices around K, then linear between them.
#[inline]
fn cmyk_to_rgb([c, m, y, k]: [u8; 4]) -> [u8; 3] {
    const G3: usize = GRID * GRID * GRID;
    let (wc, wm, wy, wk) = (
        WEIGHTS[c as usize],
        WEIGHTS[m as usize],
        WEIGHTS[y as usize],
        WEIGHTS[k as usize],
    );
    const Q: i32 = (1 << 15) - 1;
    let (rx, ry, rz) = (wc.w, wm.w, wy.w);
    let (sx, sy, sz) = (Q - rx, Q - ry, Q - rz);
    let base = (wc.x * GRID + wm.x) * GRID + wy.x;
    let dx = (wc.x_n - wc.x) * GRID * GRID;
    let dy = (wm.x_n - wm.x) * GRID;
    let dz = wy.x_n - wy.x;

    // Four lanes at a time (the fourth is the next entry's first level, and ignored), which
    // the compiler vectorizes.
    let slice = |k: usize| {
        let cube = &TABLE[(k * G3 + base) * 3..];
        let at = |offset: usize| -> [i32; 4] {
            let e = &cube[offset * 3..offset * 3 + 4];
            [e[0] as i32, e[1] as i32, e[2] as i32, e[3] as i32]
        };
        let c00 = lerp(at(0), at(dx), sx, rx);
        let c10 = lerp(at(dy), at(dx + dy), sx, rx);
        let c01 = lerp(at(dz), at(dx + dz), sx, rx);
        let c11 = lerp(at(dy + dz), at(dx + dy + dz), sx, rx);
        let c0 = lerp(c00, c10, sy, ry);
        let c1 = lerp(c01, c11, sy, ry);
        lerp(c0, c1, sz, rz)
    };
    let t = wk.w;
    let v = lerp(slice(wk.x), slice(wk.x_n), Q - t, t);

    [v[0], v[1], v[2]].map(|v| v.clamp(0, 255) as u8)
}

/// `a` and `b` weighted by `s` and `w` in Q0.15, each product rounded as `_mm_mulhrs_epi16`
/// rounds it.
#[inline(always)]
fn lerp(a: [i32; 4], b: [i32; 4], s: i32, w: i32) -> [i32; 4] {
    std::array::from_fn(|i| mulhrs(a[i], s) + mulhrs(b[i], w))
}

#[cfg(test)]
mod tests {
    use super::cmyk_to_rgb;
    use moxcms::{ColorProfile, Layout, TransformOptions};

    /// The table and its interpolation give what moxcms gives from the profile, over a grid
    /// of 49^4 colors that includes every grid point and both ends of each channel, and 2^20
    /// colors spread over all of them (every level of each channel about 4,000 times).
    #[test]
    fn matches_moxcms() {
        let profile = ColorProfile::new_from_slice(include_bytes!(
            "../../assets/CGATS001Compat-v2-micro.icc"
        ))
        .unwrap();
        let transform = profile
            .create_transform_8bit(
                Layout::Rgba,
                &ColorProfile::new_srgb(),
                Layout::Rgb,
                TransformOptions::default(),
            )
            .unwrap();
        let levels: Vec<u8> = (0..32)
            .map(|i| (i * 255 / 31) as u8)
            .chain((0..17).map(|i| (i * 255 / 16) as u8))
            .collect();
        let mut input = Vec::new();
        for &c in &levels {
            for &m in &levels {
                for &y in &levels {
                    for &k in &levels {
                        input.extend_from_slice(&[c, m, y, k]);
                    }
                }
            }
        }
        let mut state = 0x2545_f491_u32;
        for _ in 0..1 << 20 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            input.extend_from_slice(&state.to_le_bytes());
        }
        let mut expected = vec![0; input.len() / 4 * 3];
        transform.transform(&input, &mut expected).unwrap();
        for (cmyk, rgb) in input.chunks_exact(4).zip(expected.chunks_exact(3)) {
            assert_eq!(
                cmyk_to_rgb([cmyk[0], cmyk[1], cmyk[2], cmyk[3]]),
                [rgb[0], rgb[1], rgb[2]],
                "{cmyk:?}"
            );
        }
    }
}
