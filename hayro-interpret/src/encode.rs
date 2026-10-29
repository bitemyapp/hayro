//! Encoding shading patterns for easy sampling.

use crate::color::{AlphaColor, ColorComponents, ColorSpace};
use crate::function::MAX_COMPONENTS;
use crate::interpret::state::ActiveTransferFunction;
use crate::pattern::ShadingPattern;
use crate::shading::{ShadingFunction, ShadingType, Triangle};
use crate::util::x_y_advances;
use kurbo::{Affine, Point, Rect, Shape};
use rustc_hash::FxHashMap;
use std::ops::Range;
use std::sync::Arc;

/// A shading pattern that was encoded so it can be sampled.
#[derive(Clone, Debug)]
pub struct EncodedShadingPattern {
    /// The base transform of the shading pattern.
    pub base_transform: Affine,
    pub(crate) color_space: ColorSpace,
    pub(crate) background_color: AlphaColor,
    /// The encoded shading type.
    pub shading_type: EncodedShadingType,
    pub(crate) opacity: f32,
    pub(crate) transfer_function: Option<ActiveTransferFunction>,
}

/// The encoded shading type.
#[derive(Clone, Debug)]
pub enum EncodedShadingType {
    /// Function-based shading.
    FunctionBased(EncodedFunctionBasedShading),
    /// Axial or radial shading.
    RadialAxial(EncodedRadialAxialShading),
    /// Mesh-based shading sampled into device pixels.
    Sampled(EncodedSampledShading),
    /// Empty dummy shading.
    Dummy(EncodedDummyShading),
}

/// Encoded function-based shading.
#[derive(Clone, Debug)]
pub struct EncodedFunctionBasedShading {
    pub(crate) domain: Rect,
    pub(crate) function: ShadingFunction,
}

/// Encoded axial or radial shading.
#[derive(Clone, Debug)]
pub struct EncodedRadialAxialShading {
    pub(crate) function: ShadingFunction,
    pub(crate) params: RadialAxialParams,
    pub(crate) domain: [f32; 2],
    pub(crate) extend: [bool; 2],
}

/// Encoded sampled mesh shading.
#[derive(Clone, Debug)]
pub struct EncodedSampledShading {
    pub(crate) samples: Arc<MeshSamples>,
    pub(crate) function: Option<ShadingFunction>,
}

/// Encoded dummy shading.
#[derive(Clone, Debug)]
pub struct EncodedDummyShading;

/// A shading that was rasterized into a texture.
pub struct ShadingTexture {
    /// The straight (non-premultiplied) RGBA8 pixel data, row-major.
    pub data: Vec<u8>,
    /// The width of the texture in pixels.
    pub width: u32,
    /// The height of the texture in pixels.
    pub height: u32,
    /// The transform from texture space to the space of the rasterized
    /// bounding box.
    pub transform: Affine,
    /// Whether any pixel of the texture is not fully opaque.
    pub has_transparency: bool,
}

/// Return the texture dimensions [`EncodedShadingPattern::sample_texture`]
/// uses for the given bounding box and scale factor.
///
/// Exposed so that callers can bound the size of the resulting texture
/// before sampling.
pub fn texture_dimensions(bbox: Rect, scale: f32) -> (u32, u32) {
    (
        (bbox.width() as f32 * scale).max(1.0).ceil() as u32,
        (bbox.height() as f32 * scale).max(1.0).ceil() as u32,
    )
}

impl EncodedShadingPattern {
    /// Sample the shading at the given position.
    #[inline]
    pub fn sample(&self, pos: Point) -> [f32; 4] {
        self.finish(
            self.shading_type
                .eval(pos, self.background_color, &self.color_space),
        )
    }

    /// Apply the pattern's opacity and transfer function to an evaluated color.
    #[inline]
    fn finish(&self, color: Option<AlphaColor>) -> [f32; 4] {
        color
            .map(|v| {
                let mut components = v.components();
                components[3] *= self.opacity;

                if let Some(tf) = &self.transfer_function {
                    return tf.apply(&AlphaColor::new(components)).components();
                }

                components
            })
            .unwrap_or([0.0, 0.0, 0.0, 0.0])
    }

    /// Sample the shading on a pixel grid covering `bbox`, with `scale`
    /// pixels per unit of `bbox`.
    ///
    /// The callback is invoked once per pixel in row-major order with the
    /// straight (non-premultiplied) RGBA sample, letting callers apply their
    /// own conversion. Returns the dimensions of the grid (see
    /// [`texture_dimensions`]) and the transform from grid space to the space
    /// of `bbox`. No size cap is applied — callers that need to bound the
    /// texture size should check [`texture_dimensions`] first.
    pub fn sample_texture(
        &self,
        bbox: Rect,
        scale: f32,
        mut f: impl FnMut([f32; 4]),
    ) -> (u32, u32, Affine) {
        let mut lut = self.gradient_lut(bbox, scale, |color| color);

        self.for_each_texel(bbox, scale, |point| {
            f(match &mut lut {
                Some(lut) => lut.sample(point),
                None => self.sample(point),
            });
        })
    }

    /// Sample the shading as [`sample_texture`](Self::sample_texture) does, into texels of
    /// premultiplied RGBA8 (each sample premultiplied and rounded as
    /// `color::AlphaColor::premultiply` and `PremulColor::to_rgba8` do).
    pub fn sample_texture_premultiplied(&self, bbox: Rect, scale: f32) -> PremultipliedTexture {
        let (width, height) = texture_dimensions(bbox, scale);
        let mut data = Vec::with_capacity(width as usize * height as usize);

        let transform = if let Some(mut lut) = self.gradient_lut(bbox, scale, premultiplied_rgba8) {
            self.for_each_texel(bbox, scale, |point| data.push(lut.sample(point)))
                .2
        } else if let Some(texel) = self.opaque_mesh_texel() {
            self.mesh_texels(bbox, scale, &mut data, texel)
        } else if let Some(texel) = self.function_mesh_texel() {
            self.mesh_texels(bbox, scale, &mut data, texel)
        } else {
            self.for_each_texel(bbox, scale, |point| {
                data.push(premultiplied_rgba8(self.sample(point)));
            })
            .2
        };

        let may_have_transparency = data.iter().any(|texel| texel[3] != 255);

        PremultipliedTexture {
            data,
            width,
            height,
            transform,
            may_have_transparency,
        }
    }

    /// Call `f` with each point of the grid [`sample_texture`](Self::sample_texture)
    /// samples, in row-major order.
    #[inline(always)]
    fn for_each_texel(
        &self,
        bbox: Rect,
        scale: f32,
        mut f: impl FnMut(Point),
    ) -> (u32, u32, Affine) {
        let grid = self.texel_grid(bbox, scale);
        let mut start_point = grid.start;

        for _ in 0..grid.height {
            let mut point = start_point;

            for _ in 0..grid.width {
                f(point);
                point += grid.x_advance;
            }

            start_point += grid.y_advance;
        }

        (grid.width, grid.height, grid.transform)
    }

    fn texel_grid(&self, bbox: Rect, scale: f32) -> TexelGrid {
        let (width, height) = texture_dimensions(bbox, scale);
        let inv_scale = 1.0 / scale as f64;

        let (x_advance, y_advance) =
            x_y_advances(&(Affine::scale(inv_scale) * self.base_transform));

        let start = self.base_transform
            * Affine::translate((0.5 * inv_scale, 0.5 * inv_scale))
            * Point::new(bbox.x0, bbox.y0);

        TexelGrid {
            width,
            height,
            start,
            x_advance,
            y_advance,
            transform: Affine::translate((bbox.x0, bbox.y0)) * Affine::scale(inv_scale),
        }
    }

    /// For an axial or radial shading sampled at enough points, its colors along it,
    /// each converted with `convert`.
    fn gradient_lut<T: Copy>(
        &self,
        bbox: Rect,
        scale: f32,
        convert: fn([f32; 4]) -> T,
    ) -> Option<GradientLut<'_, T>> {
        let (width, height) = texture_dimensions(bbox, scale);
        match &self.shading_type {
            EncodedShadingType::RadialAxial(shading)
                if width as usize * height as usize > GRADIENT_STEPS / 4 =>
            {
                Some(GradientLut::new(self, shading, convert))
            }
            _ => None,
        }
    }

    /// For a mesh shading of opaque gray or RGB colors without a function, the texel of a
    /// sample, without converting its color to floats and back: the same as
    /// `premultiplied_rgba8(self.sample(point))` at its point.
    fn opaque_mesh_texel(&self) -> Option<impl FnMut(&[f32]) -> [u8; 4] + use<>> {
        let EncodedShadingType::Sampled(EncodedSampledShading { function: None, .. }) =
            &self.shading_type
        else {
            return None;
        };
        if self.opacity != 1.0 || self.transfer_function.is_some() {
            return None;
        }
        let gray = if self.color_space.is_device_gray() {
            true
        } else if self.color_space.is_device_rgb() {
            false
        } else {
            return None;
        };
        // As `ColorSpace::to_rgba` quantizes them.
        let level = |c: &[f32], i: usize| c.get(i).map_or(0, |c| (c * 255.0 + 0.5) as u8);

        Some(move |c: &[f32]| match gray {
            true => [level(c, 0), level(c, 0), level(c, 0), 255],
            false => [level(c, 0), level(c, 1), level(c, 2), 255],
        })
    }

    /// For a mesh shading whose colors are a function's values at one input, the texel of a
    /// sample from a table of the function at [`GRADIENT_STEPS`] evenly spaced inputs between
    /// the least and greatest sampled, each evaluated the first time it's needed: a sample's
    /// input is rounded to a step, which changes its texel by at most a level where the
    /// function crosses one within the step, as for axial and radial shadings.
    fn function_mesh_texel(&self) -> Option<impl FnMut(&[f32]) -> [u8; 4] + '_> {
        let EncodedShadingType::Sampled(EncodedSampledShading {
            samples,
            function: Some(function),
        }) = &self.shading_type
        else {
            return None;
        };
        let MeshSamples::Dense {
            stride: 1,
            components,
            covered,
            ..
        } = &**samples
        else {
            return None;
        };
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for (&t, _) in components
            .iter()
            .zip(covered)
            .filter(|(_, covered)| **covered)
        {
            (lo, hi) = (lo.min(t), hi.max(t));
        }
        if !(lo <= hi) {
            return None;
        }

        let span = hi - lo;
        let texel_at = move |t: f32| {
            let mut val = [0.0; MAX_COMPONENTS];
            let val = &mut val[..function.num_outputs()];
            let color = function
                .eval(&[t], val)
                .map(|()| self.color_space.to_rgba(val, 1.0));
            premultiplied_rgba8(self.finish(color))
        };
        let mut table = vec![None; GRADIENT_STEPS + 1];

        Some(move |c: &[f32]| {
            let t = c[0];
            if !(lo..=hi).contains(&t) {
                return texel_at(t);
            }
            let step = if span > 0.0 {
                ((t - lo) / span * GRADIENT_STEPS as f32).round() as usize
            } else {
                0
            };
            *table[step]
                .get_or_insert_with(|| texel_at(lo + span * step as f32 / GRADIENT_STEPS as f32))
        })
    }

    /// The texels of a mesh shading, each from its sample with `texel`.
    fn mesh_texels(
        &self,
        bbox: Rect,
        scale: f32,
        data: &mut Vec<[u8; 4]>,
        mut texel: impl FnMut(&[f32]) -> [u8; 4],
    ) -> Affine {
        let EncodedShadingType::Sampled(EncodedSampledShading { samples, .. }) = &self.shading_type
        else {
            unreachable!("only called for mesh shadings");
        };
        let background = premultiplied_rgba8(self.finish(Some(self.background_color)));
        let grid = self.texel_grid(bbox, scale);

        let MeshSamples::Dense {
            x0,
            y0,
            width,
            height,
            stride,
            components,
            covered,
        } = &**samples
        else {
            return self.mesh_texels_at_points(data, bbox, scale, samples, texel, background);
        };
        if grid.x_advance.y != 0.0 || grid.y_advance.x != 0.0 {
            return self.mesh_texels_at_points(data, bbox, scale, samples, texel, background);
        }

        // The grid steps across along rows and down along columns only, so the points of a
        // column share their x and those of a row their y (adding 0 changes neither): find
        // each once, stepping as `for_each_texel` does, as an index into the samples.
        let mut point = grid.start;
        let columns: Vec<Option<usize>> = (0..grid.width)
            .map(|_| {
                let x = (point.x as u16).checked_sub(*x0).map(usize::from);
                point += grid.x_advance;
                x.filter(|&x| x < *width)
            })
            .collect();
        // Where the columns with samples are one stretch of consecutive ones: its first
        // column and length.
        let consecutive = columns.iter().position(Option::is_some).and_then(|first| {
            let count = columns[first..].iter().take_while(|x| x.is_some()).count();
            let start = columns[first]?;
            let consecutive = columns[first..first + count]
                .iter()
                .enumerate()
                .all(|(i, x)| *x == Some(start + i));
            let rest_empty = columns[first + count..].iter().all(Option::is_none);
            (consecutive && rest_empty).then_some((first, count))
        });
        let mut point = grid.start;
        for _ in 0..grid.height {
            let row = (point.y as u16)
                .checked_sub(*y0)
                .map(usize::from)
                .filter(|&y| y < *height);
            point += grid.y_advance;

            let Some(row) = row else {
                data.extend(std::iter::repeat_n(background, grid.width as usize));
                continue;
            };
            // Usually the columns are consecutive samples: read them as a slice of the row.
            if let Some((first, count)) = consecutive
                && *stride > 0
            {
                let samples = row * width + columns[first].unwrap();
                let samples = samples..samples + count;
                data.extend(std::iter::repeat_n(background, first));
                data.extend(
                    components[samples.start * stride..samples.end * stride]
                        .chunks_exact(*stride)
                        .zip(&covered[samples])
                        .map(|(c, &covered)| if covered { texel(c) } else { background }),
                );
                data.extend(std::iter::repeat_n(
                    background,
                    grid.width as usize - first - count,
                ));
                continue;
            }
            data.extend(columns.iter().map(|x| match x {
                Some(x) if covered[row * width + x] => {
                    texel(&components[(row * width + x) * stride..][..*stride])
                }
                _ => background,
            }));
        }

        grid.transform
    }

    /// The texels of a mesh shading from its samples at each point of the grid.
    fn mesh_texels_at_points(
        &self,
        data: &mut Vec<[u8; 4]>,
        bbox: Rect,
        scale: f32,
        samples: &MeshSamples,
        mut texel: impl FnMut(&[f32]) -> [u8; 4],
        background: [u8; 4],
    ) -> Affine {
        self.for_each_texel(bbox, scale, |point| {
            let c = samples.get((point.x as u16, point.y as u16));
            data.push(c.map_or(background, &mut texel));
        })
        .2
    }

    /// Rasterize the shading into a straight (non-premultiplied) RGBA8
    /// texture covering `bbox`, with `scale` pixels per unit of `bbox`.
    ///
    /// See [`sample_texture`](Self::sample_texture) for the grid semantics.
    pub fn render_texture(&self, bbox: Rect, scale: f32) -> ShadingTexture {
        let (width, height) = texture_dimensions(bbox, scale);
        let mut data = Vec::with_capacity(width as usize * height as usize * 4);
        let mut has_transparency = false;

        let (_, _, transform) = self.sample_texture(bbox, scale, |sample| {
            let converted = [
                (sample[0] * 255.0 + 0.5) as u8,
                (sample[1] * 255.0 + 0.5) as u8,
                (sample[2] * 255.0 + 0.5) as u8,
                (sample[3] * 255.0 + 0.5) as u8,
            ];

            has_transparency |= converted[3] != 255;
            data.extend_from_slice(&converted);
        });

        ShadingTexture {
            data,
            width,
            height,
            transform,
            has_transparency,
        }
    }
}

impl ShadingPattern {
    /// Encode the shading pattern.
    pub fn encode(&self) -> EncodedShadingPattern {
        self.encode_within(Rect::new(0.0, 0.0, 65536.0, 65536.0))
    }

    /// Encode the shading pattern for drawing within `target`, in device pixels. Mesh
    /// shadings, which are sampled at device pixels, are only sampled within it.
    pub fn encode_within(&self, target: Rect) -> EncodedShadingPattern {
        let base_transform;

        let shading_type = match self.shading.shading_type.as_ref() {
            ShadingType::FunctionBased {
                domain,
                matrix,
                function,
            } => {
                base_transform = (self.matrix * *matrix).inverse();
                encode_function_shading(domain, function)
            }
            ShadingType::RadialAxial {
                coords,
                domain,
                function,
                extend,
                axial,
            } => {
                let (encoded, initial_transform) =
                    encode_axial_shading(*coords, *domain, function, *extend, *axial);

                base_transform = initial_transform * self.matrix.inverse();

                encoded
            }
            ShadingType::TriangleMesh {
                triangles,
                function,
            } => {
                let full_transform = self.matrix;
                let samples = sample_triangles(triangles, full_transform, target);

                base_transform = Affine::IDENTITY;

                EncodedShadingType::Sampled(EncodedSampledShading {
                    samples: Arc::new(samples),
                    function: function.clone(),
                })
            }
            ShadingType::CoonsPatchMesh { function, .. } => {
                let full_transform = self.matrix;
                let samples = self
                    .shading
                    .with_patch_triangles(full_transform, |triangles| {
                        sample_triangles(triangles, full_transform, target)
                    });

                base_transform = Affine::IDENTITY;

                EncodedShadingType::Sampled(EncodedSampledShading {
                    samples: Arc::new(samples),
                    function: function.clone(),
                })
            }
            ShadingType::TensorProductPatchMesh { function, .. } => {
                let full_transform = self.matrix;
                let samples = self
                    .shading
                    .with_patch_triangles(full_transform, |triangles| {
                        sample_triangles(triangles, full_transform, target)
                    });

                base_transform = Affine::IDENTITY;

                EncodedShadingType::Sampled(EncodedSampledShading {
                    samples: Arc::new(samples),
                    function: function.clone(),
                })
            }
            ShadingType::Dummy => {
                base_transform = Affine::IDENTITY;

                EncodedShadingType::Dummy(EncodedDummyShading)
            }
        };

        let color_space = self.shading.color_space.clone();

        let background_color = self
            .shading
            .background
            .as_ref()
            .map(|b| color_space.to_rgba(b, 1.0))
            .unwrap_or(AlphaColor::TRANSPARENT);

        EncodedShadingPattern {
            color_space,
            background_color,
            shading_type,
            base_transform,
            opacity: self.opacity,
            transfer_function: self.transfer_function.clone(),
        }
    }
}

fn encode_axial_shading(
    coords: [f32; 6],
    domain: [f32; 2],
    function: &ShadingFunction,
    extend: [bool; 2],
    is_axial: bool,
) -> (EncodedShadingType, Affine) {
    let initial_transform;

    let params = if is_axial {
        let [x_0, y_0, x_1, y_1, _, _] = coords;

        initial_transform = ts_from_line_to_line(
            Point::new(x_0 as f64, y_0 as f64),
            Point::new(x_1 as f64, y_1 as f64),
            Point::ZERO,
            Point::new(1.0, 0.0),
        );

        RadialAxialParams::Axial
    } else {
        let [x_0, y_0, r0, x_1, y_1, r_1] = coords;

        initial_transform = Affine::translate((-x_0 as f64, -y_0 as f64));
        let new_x1 = x_1 - x_0;
        let new_y1 = y_1 - y_0;

        let p1 = Point::new(new_x1 as f64, new_y1 as f64);
        let r = Point::new(r0 as f64, r_1 as f64);

        RadialAxialParams::Radial { p1, r }
    };

    (
        EncodedShadingType::RadialAxial(EncodedRadialAxialShading {
            function: function.clone(),
            params,
            domain,
            extend,
        }),
        initial_transform,
    )
}

/// Mesh shadings whose triangles span at most this many device pixels are sampled into a
/// grid; larger ones keep only the pixels their triangles cover.
const MAX_DENSE_MESH_PIXELS: usize = 1 << 25;

/// The colors of a mesh shading at the device pixels its triangles cover.
#[derive(Debug)]
pub(crate) enum MeshSamples {
    Dense {
        x0: u16,
        y0: u16,
        width: usize,
        height: usize,
        /// The number of components of each pixel.
        stride: usize,
        components: Vec<f32>,
        covered: Vec<bool>,
    },
    Sparse(FxHashMap<(u16, u16), ColorComponents>),
}

impl MeshSamples {
    #[inline]
    fn get(&self, (x, y): (u16, u16)) -> Option<&[f32]> {
        match self {
            Self::Dense {
                x0,
                y0,
                width,
                height,
                stride,
                components,
                covered,
            } => {
                let x = x.checked_sub(*x0)? as usize;
                let y = y.checked_sub(*y0)? as usize;
                if x >= *width || y >= *height {
                    return None;
                }

                let i = y * width + x;
                covered[i].then(|| &components[i * stride..][..*stride])
            }
            Self::Sparse(map) => map.get(&(x, y)).map(|c| c.as_slice()),
        }
    }
}

/// The device pixels a triangle's bounding box covers, as in [`sample_triangles`].
fn pixel_span(bbox: Rect) -> (Range<u16>, Range<u16>) {
    (
        floor_u16(bbox.x0)..ceil_u16(bbox.x1),
        floor_u16(bbox.y0)..ceil_u16(bbox.y1),
    )
}

/// `x.floor() as u16`, without the call to `floor` that x86-64 without SSE 4.1 makes.
#[inline(always)]
fn floor_u16(x: f64) -> u16 {
    // `as` saturates (and makes NaN 0) as the cast of the floor would.
    let i = x as i64;
    let i = if i as f64 > x { i - 1 } else { i };
    i.clamp(0, u16::MAX as i64) as u16
}

/// `x.ceil() as u16`, likewise.
#[inline(always)]
fn ceil_u16(x: f64) -> u16 {
    let i = x as i64;
    let i = if (i as f64) < x {
        i.saturating_add(1)
    } else {
        i
    };
    i.clamp(0, u16::MAX as i64) as u16
}

/// The part of a span of pixels within another.
fn clip(span: Range<u16>, within: &Range<u16>) -> Range<u16> {
    span.start.max(within.start)..span.end.min(within.end)
}

/// Pixels along a row that a triangle covers, and the weights of its vertices' colors at
/// each of them.
struct Run {
    y: u16,
    xs: Range<u16>,
    /// The weights of the second and third vertices at the first pixel, and how they change
    /// from pixel to pixel (stepping them differs from computing them at each pixel only in
    /// the last bits).
    v: f64,
    w: f64,
    dv: f64,
    dw: f64,
}

impl Run {
    #[inline(always)]
    fn weights(&self) -> impl Iterator<Item = [f32; 3]> {
        let (mut v, mut w, dv, dw) = (self.v, self.w, self.dv, self.dw);
        self.xs.clone().map(move |_| {
            let weights = [(1.0 - v - w) as f32, v as f32, w as f32];
            (v, w) = (v + dv, w + dw);
            weights
        })
    }
}

/// Call `f` with each run of pixels within `xs` by `ys` that the triangle, mapped by
/// `transform`, covers.
fn for_each_run(
    t: &Triangle,
    transform: Affine,
    xs: &Range<u16>,
    ys: &Range<u16>,
    mut f: impl FnMut(Run),
) {
    let [a, b, c] = [t.p0.point, t.p1.point, t.p2.point].map(|p| transform * p);
    let triangle = kurbo::Triangle::new(a, b, c);
    let (t_xs, t_ys) = pixel_span(triangle.bounding_box());
    let (t_xs, t_ys) = (clip(t_xs, xs), clip(t_ys, ys));
    if t_xs.is_empty() || t_ys.is_empty() {
        return;
    }

    // As in `Triangle::interpolate`.
    let (v0, v1) = (b - a, c - a);
    let (d00, d01, d11) = (v0.dot(v0), v0.dot(v1), v1.dot(v1));
    let denom = d00 * d11 - d01 * d01;

    // The weights of `b` and `c` at a pixel, and how they change along a row.
    let weights = |x: u16, y: u16| {
        let v2 = Point::new(x as f64, y as f64) - a;
        let (d20, d21) = (v2.dot(v0), v2.dot(v1));
        let v = (d11 * d20 - d01 * d21) / denom;
        let w = (d00 * d21 - d01 * d20) / denom;
        (v, w)
    };
    let (dv, dw) = (
        (d11 * v0.x - d01 * v1.x) / denom,
        (d00 * v1.x - d01 * v0.x) / denom,
    );
    let mut run = |xs: Range<u16>, y: u16| {
        if !xs.is_empty() {
            let (v, w) = weights(xs.start, y);
            f(Run {
                y,
                xs,
                v,
                w,
                dv,
                dw,
            });
        }
    };

    // Small or degenerate triangles have every pixel tested, as rounding near their corners
    // could make the side tests disagree in ways `covered_run` doesn't expect.
    if t_xs.len() <= 8 || v0.cross(v1).abs() < 1.0 {
        for y in t_ys {
            let covered = |x: &u16| triangle.winding(Point::new(*x as f64, y as f64)) != 0;
            let mut xs = t_xs.clone();
            while let Some(start) = xs.find(covered) {
                let end = xs.find(|x| !covered(x)).unwrap_or(t_xs.end);
                run(start..end, y);
            }
        }
    } else {
        let edges = Edges::new(&triangle);
        for y in t_ys {
            run(covered_run(&triangle, &edges, y, &t_xs), y);
        }
    }
}

/// Where a triangle's sloped edges cross a row: each as a point on it, its run per unit of
/// rise, and whether the triangle lies left of it.
struct Edges([Option<(Point, f64, bool)>; 3]);

impl Edges {
    fn new(triangle: &kurbo::Triangle) -> Self {
        let [a, b, c] = [triangle.a, triangle.b, triangle.c];
        let orientation = (b - a).cross(c - a);
        Self([(a, b), (b, c), (c, a)].map(|(p, q)| {
            let (dx, dy) = (q.x - p.x, q.y - p.y);
            // The side test decreases along a row where `dy` is positive.
            (dy != 0.0).then(|| (p, dx / dy, (dy > 0.0) == (orientation > 0.0)))
        }))
    }

    /// Roughly where the triangle begins and ends along a row.
    #[inline]
    fn span(&self, row: f64, xs: &Range<u16>) -> (f64, f64) {
        let (mut lo, mut hi) = (xs.start as f64, xs.end as f64);
        for (p, run, left) in self.0.iter().flatten() {
            let x = p.x + run * (row - p.y);
            if *left {
                hi = hi.min(x);
            } else {
                lo = lo.max(x);
            }
        }
        (lo, hi)
    }
}

/// The pixels within `xs` of row `y` where `triangle.winding` isn't 0, which are one run:
/// along a row, each of the three side tests it makes is computed as a constant minus a
/// multiple of `x - a.x`, which is monotonic in `x`, so each changes sign at most once.
/// Where the edges cross the row gives the run's ends roughly, and the test itself the exact
/// ends, so that the pixels are those testing every pixel would give.
fn covered_run(triangle: &kurbo::Triangle, edges: &Edges, y: u16, xs: &Range<u16>) -> Range<u16> {
    let covered = |x: u16| triangle.winding(Point::new(x as f64, y as f64)) != 0;
    let (lo, hi) = edges.span(y as f64, xs);
    let clamp = |x: f64| (x.max(xs.start as f64).min(xs.end as f64 - 1.0)) as u16;

    // Walk from the estimated first pixel to the exact one.
    let mut start = clamp(lo.ceil());
    if !covered(start) {
        match (start + 1..xs.end)
            .find(|&x| covered(x))
            .or_else(|| (xs.start..start).rev().find(|&x| covered(x)))
        {
            Some(x) => start = x,
            None => return 0..0,
        }
    }
    while start > xs.start && covered(start - 1) {
        start -= 1;
    }

    // And from the estimated last one; `start` is covered, so walking left stops there at the
    // latest.
    let mut end = clamp(hi.floor()).max(start);
    if covered(end) {
        while end + 1 < xs.end && covered(end + 1) {
            end += 1;
        }
    } else {
        while !covered(end) {
            end -= 1;
        }
    }

    start..end + 1
}

/// The color at a point of a triangle, from its vertices' colors and their weights there.
#[inline]
fn interpolate([c0, c1, c2]: [&[f32]; 3], [u, v, w]: [f32; 3], out: &mut [f32]) {
    for (out, ((c0, c1), c2)) in out.iter_mut().zip(c0.iter().zip(c1).zip(c2)) {
        *out = u * c0 + v * c1 + w * c2;
    }
}

/// [`interpolate`] for `N` components.
#[inline(always)]
fn interpolate_n<const N: usize>(colors: [&[f32]; 3], [u, v, w]: [f32; 3], out: &mut [f32]) {
    let [c0, c1, c2] = colors.map(|c| &c[..N]);
    for (i, out) in out[..N].iter_mut().enumerate() {
        *out = u * c0[i] + v * c1[i] + w * c2[i];
    }
}

/// [`interpolate_n`] at each pixel of a run, into its pixels' components.
#[inline(always)]
fn fill_run<const N: usize>(colors: [&[f32]; 3], run: &Run, out: &mut [f32]) {
    for (out, weights) in out.as_chunks_mut::<N>().0.iter_mut().zip(run.weights()) {
        interpolate_n::<N>(colors, weights, out);
    }
}

/// A triangle's vertex colors.
#[inline]
fn colors(t: &Triangle) -> [&[f32]; 3] {
    [&t.p0.colors, &t.p1.colors, &t.p2.colors].map(|c| c.as_slice())
}

fn sample_triangles(triangles: &[Triangle], transform: Affine, target: Rect) -> MeshSamples {
    // The transformed bounds of all the vertices contain every triangle's pixels.
    let stride = triangles.first().map_or(0, |t| t.p0.colors.len());
    let mut same_stride = true;
    let (mut min, mut max) = (
        Point::new(f64::MAX, f64::MAX),
        Point::new(f64::MIN, f64::MIN),
    );
    for vertex in triangles.iter().flat_map(|t| [&t.p0, &t.p1, &t.p2]) {
        let p = vertex.point;
        (min.x, min.y, max.x, max.y) = (
            min.x.min(p.x),
            min.y.min(p.y),
            max.x.max(p.x),
            max.y.max(p.y),
        );
        same_stride &= vertex.colors.len() == stride;
    }
    let bounds = transform.transform_rect_bbox(Rect::from_points(min, max));
    let (target_xs, target_ys) = pixel_span(target);
    let (xs, ys) = pixel_span(bounds);
    let (xs, ys) = (clip(xs, &target_xs), clip(ys, &target_ys));
    let (x0, y0) = (xs.start, ys.start);
    let (width, height) = (xs.len(), ys.len());

    if !same_stride || width * height > MAX_DENSE_MESH_PIXELS {
        return MeshSamples::Sparse(sample_triangles_sparse(triangles, transform, target));
    }

    let mut components = vec![0.0; width * height * stride];
    let mut covered = vec![false; width * height];

    for t in triangles {
        let colors = colors(t);
        for_each_run(t, transform, &xs, &ys, |run| {
            let start = (run.y - y0) as usize * width + (run.xs.start - x0) as usize;
            let pixels = start..start + run.xs.len();
            covered[pixels.clone()].fill(true);
            let out = &mut components[pixels.start * stride..pixels.end * stride];
            // The usual numbers of components, unrolled.
            match stride {
                1 => fill_run::<1>(colors, &run, out),
                3 => fill_run::<3>(colors, &run, out),
                4 => fill_run::<4>(colors, &run, out),
                // Vertices without colors have none to interpolate.
                0 => {}
                _ => {
                    for (out, weights) in out.chunks_exact_mut(stride).zip(run.weights()) {
                        interpolate(colors, weights, out);
                    }
                }
            }
        });
    }

    MeshSamples::Dense {
        x0,
        y0,
        width,
        height,
        stride,
        components,
        covered,
    }
}

fn sample_triangles_sparse(
    triangles: &[Triangle],
    transform: Affine,
    target: Rect,
) -> FxHashMap<(u16, u16), ColorComponents> {
    let mut map = FxHashMap::default();
    let (xs, ys) = pixel_span(target);

    for t in triangles {
        for_each_run(t, transform, &xs, &ys, |run| {
            for (x, weights) in run.xs.clone().zip(run.weights()) {
                let mut color = ColorComponents::from_elem(0.0, t.p0.colors.len());
                interpolate(colors(t), weights, &mut color);
                map.insert((x, run.y), color);
            }
        });
    }

    map
}

fn encode_function_shading(domain: &[f32; 4], function: &ShadingFunction) -> EncodedShadingType {
    let domain = Rect::new(
        domain[0] as f64,
        domain[2] as f64,
        domain[1] as f64,
        domain[3] as f64,
    );

    EncodedShadingType::FunctionBased(EncodedFunctionBasedShading {
        domain,
        function: function.clone(),
    })
}

/// Axial and radial shadings drawn into a texture of more than a quarter this many pixels are
/// evaluated at most at this many evenly spaced positions along them, each once, rather than at
/// every pixel.
const GRADIENT_STEPS: usize = 1 << 16;

/// The colors of an axial or radial shading at evenly spaced positions along it, each
/// evaluated (and converted) the first time it's needed.
struct GradientLut<'a, T> {
    pattern: &'a EncodedShadingPattern,
    shading: &'a EncodedRadialAxialShading,
    convert: fn([f32; 4]) -> T,
    /// `None` until evaluated.
    colors: Vec<Option<T>>,
    background: T,
}

impl<'a, T: Copy> GradientLut<'a, T> {
    fn new(
        pattern: &'a EncodedShadingPattern,
        shading: &'a EncodedRadialAxialShading,
        convert: fn([f32; 4]) -> T,
    ) -> Self {
        Self {
            pattern,
            shading,
            convert,
            colors: vec![None; GRADIENT_STEPS + 1],
            background: convert(pattern.finish(Some(pattern.background_color))),
        }
    }

    #[inline]
    fn sample(&mut self, pos: Point) -> T {
        let Some(t) = self.shading.position(pos) else {
            return self.background;
        };

        let i = (t * GRADIENT_STEPS as f32).round() as usize;
        *self.colors[i].get_or_insert_with(|| {
            let t = i as f32 / GRADIENT_STEPS as f32;
            (self.convert)(
                self.pattern
                    .finish(self.shading.color_at(t, &self.pattern.color_space)),
            )
        })
    }
}

/// The points [`EncodedShadingPattern::sample_texture`] samples: `width` by `height` of them
/// from `start`, `x_advance` apart along a row and `y_advance` between rows.
struct TexelGrid {
    width: u32,
    height: u32,
    start: Point,
    x_advance: kurbo::Vec2,
    y_advance: kurbo::Vec2,
    /// From the grid to the space of the sampled bounding box.
    transform: Affine,
}

/// A shading sampled into texels of premultiplied RGBA8.
pub struct PremultipliedTexture {
    /// The texels, row-major.
    pub data: Vec<[u8; 4]>,
    /// The width of the texture in texels.
    pub width: u32,
    /// The height of the texture in texels.
    pub height: u32,
    /// The transform from texture space to the space of the sampled bounding box.
    pub transform: Affine,
    /// Whether any texel isn't fully opaque.
    pub may_have_transparency: bool,
}

/// A straight RGBA color premultiplied and rounded to 8 bits, as `color::AlphaColor`'s
/// `premultiply` and `PremulColor::to_rgba8` do it.
#[inline(always)]
fn premultiplied_rgba8([r, g, b, a]: [f32; 4]) -> [u8; 4] {
    [r * a, g * a, b * a, a].map(|c| (c * 255.0 + 0.5) as u8)
}

impl EncodedRadialAxialShading {
    /// Where a point falls along the shading, from 0 to 1, or `None` where the
    /// background shows.
    #[inline]
    fn position(&self, pos: Point) -> Option<f32> {
        let t = match &self.params {
            RadialAxialParams::Axial => pos.x as f32,
            RadialAxialParams::Radial { p1, r } => {
                radial_pos(&pos, p1, *r, self.extend[0], self.extend[1])?
            }
        };

        if t < 0.0 {
            self.extend[0].then_some(0.0)
        } else if t > 1.0 {
            self.extend[1].then_some(1.0)
        } else {
            Some(t)
        }
    }

    /// The color at a position along the shading, from 0 to 1.
    fn color_at(&self, t: f32, color_space: &ColorSpace) -> Option<AlphaColor> {
        let (t0, t1) = (self.domain[0], self.domain[1]);
        let t = t0 + (t1 - t0) * t;

        let mut val = [0.0; MAX_COMPONENTS];
        let val = &mut val[..self.function.num_outputs()];
        self.function.eval(&[t], val)?;

        Some(color_space.to_rgba(val, 1.0))
    }
}

#[derive(Clone, Debug)]
pub(crate) enum RadialAxialParams {
    Axial,
    Radial { p1: Point, r: Point },
}

impl EncodedShadingType {
    pub(crate) fn eval(
        &self,
        pos: Point,
        bg_color: AlphaColor,
        color_space: &ColorSpace,
    ) -> Option<AlphaColor> {
        match self {
            Self::FunctionBased(EncodedFunctionBasedShading { domain, function }) => {
                if !domain.contains(pos) {
                    Some(bg_color)
                } else {
                    let mut out = [0.0; MAX_COMPONENTS];
                    let out = &mut out[..function.num_outputs()];
                    function.eval(&[pos.x as f32, pos.y as f32], out)?;
                    // TODO: Clamp out-of-range values.
                    Some(color_space.to_rgba(out, 1.0))
                }
            }
            Self::RadialAxial(shading) => match shading.position(pos) {
                Some(t) => shading.color_at(t, color_space),
                None => Some(bg_color),
            },
            Self::Sampled(EncodedSampledShading { samples, function }) => {
                let sample_point = (pos.x as u16, pos.y as u16);

                if let Some(color) = samples.get(sample_point) {
                    if let Some(function) = function {
                        let mut val = [0.0; MAX_COMPONENTS];
                        let val = &mut val[..function.num_outputs()];
                        function.eval(color, val)?;
                        Some(color_space.to_rgba(val, 1.0))
                    } else {
                        Some(color_space.to_rgba(color, 1.0))
                    }
                } else {
                    Some(bg_color)
                }
            }
            Self::Dummy(_) => Some(AlphaColor::TRANSPARENT),
        }
    }
}

fn ts_from_line_to_line(src1: Point, src2: Point, dst1: Point, dst2: Point) -> Affine {
    let unit_to_line1 = unit_to_line(src1, src2);
    let line1_to_unit = unit_to_line1.inverse();
    let unit_to_line2 = unit_to_line(dst1, dst2);

    unit_to_line2 * line1_to_unit
}

fn unit_to_line(p0: Point, p1: Point) -> Affine {
    Affine::new([
        p1.y - p0.y,
        p0.x - p1.x,
        p1.x - p0.x,
        p1.y - p0.y,
        p0.x,
        p0.y,
    ])
}

fn radial_pos(
    pos: &Point,
    p1: &Point,
    r: Point,
    min_extend: bool,
    max_extend: bool,
) -> Option<f32> {
    let r0 = r.x as f32;
    let dx = p1.x as f32;
    let dy = p1.y as f32;
    let dr = r.y as f32 - r0;

    let px = pos.x as f32;
    let py = pos.y as f32;

    let a = dx * dx + dy * dy - dr * dr;
    let b = -2.0 * (px * dx + py * dy + r0 * dr);
    let c = px * px + py * py - r0 * r0;

    let discriminant = b * b - 4.0 * a * c;

    // No solution available.
    if discriminant < 0.0 {
        return None;
    }

    if a.abs() < 1e-6 {
        if b.abs() < 1e-6 {
            return None;
        }

        let t = -c / b;

        if (!min_extend && t < 0.0) || (!max_extend && t > 1.0) {
            return None;
        }

        let r_t = r0 + dr * t;
        if r_t < 0.0 {
            return None;
        }

        return Some(t);
    }

    let sqrt_d = discriminant.sqrt();
    let t1 = (-b - sqrt_d) / (2.0 * a);
    let t2 = (-b + sqrt_d) / (2.0 * a);

    let max = t1.max(t2);
    let mut take_max = Some(max);
    let min = t1.min(t2);
    let mut take_min = Some(min);

    if (!min_extend && min < 0.0) || r0 + dr * min < 0.0 {
        take_min = None;
    }

    if (!max_extend && max > 1.0) || r0 + dr * max < 0.0 {
        take_max = None;
    }

    match (take_min, take_max) {
        (Some(_), Some(max)) => Some(max),
        (Some(min), None) => Some(min),
        (None, Some(max)) => Some(max),
        (None, None) => None,
    }
}
