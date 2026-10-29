use crate::Renderer;
use hayro_interpret::Paint;
use hayro_interpret::encode::EncodedShadingType;
use hayro_interpret::gradient::SvgGradientKind;
use hayro_interpret::pattern::Pattern;
use hayro_interpret::util::x_y_advances;
use kurbo::{Affine, BezPath, Rect};
use std::sync::Arc;
use vello_cpu::color::{AlphaColor, DynamicColor, Srgb};
use vello_cpu::peniko::{ColorStop, Gradient, ImageQuality, ImageSampler};
use vello_cpu::{Image, ImageSource, PaintType, PixelMetadata, Pixmap, peniko};

impl Renderer<'_> {
    /// Set the paint, and return the clip path to push for it, or `None` if the paint can't
    /// change any pixel (every color of it is fully transparent and drawn over the target),
    /// so that drawing can be skipped.
    #[must_use]
    pub(super) fn set_paint(
        &mut self,
        paint: &Paint<'_>,
        path_bbox: impl Fn() -> Rect,
        is_stroke: bool,
    ) -> Option<Option<BezPath>> {
        let mut paint_transform = Affine::IDENTITY;
        let mut clip_path = None;
        let mut transparent = false;

        let paint: PaintType = match paint.clone() {
            Paint::Color(c) => {
                let c = c.to_rgba().to_rgba8();
                transparent = c[3] == 0;
                AlphaColor::from_rgba8(c[0], c[1], c[2], c[3]).into()
            }
            Paint::Pattern(p) => {
                let path_transform = self.ctx.transform();
                let bbox = {
                    let mut bbox = path_transform.transform_rect_bbox(path_bbox());

                    if is_stroke {
                        // Try to account for stroke in bbox.
                        let (a1, a2) = x_y_advances(path_transform);
                        let factor = a1.length().max(a2.length()) * self.ctx.stroke().width;
                        bbox = bbox.inflate(factor, factor);
                    }

                    bbox.intersect(Rect::new(
                        0.0,
                        0.0,
                        self.ctx.width() as f64,
                        self.ctx.height() as f64,
                    ))
                };
                // Textures only need to cover what the clips let through.
                let visible = trim(bbox, self.visible());

                match *p {
                    Pattern::Shading(s) => {
                        const NATIVE_GRADIENT_TOLERANCE: f32 = 0.01;

                        clip_path = s.shading.clip_path.clone();
                        let encoded = s.encode_within(visible.inflate(1.0, 1.0));

                        if let EncodedShadingType::RadialAxial(gradient) = &encoded.shading_type
                            && let Some(native) =
                                gradient.as_svg_gradient(&encoded, bbox, NATIVE_GRADIENT_TOLERANCE)
                        {
                            paint_transform = path_transform.inverse()
                                * Affine::translate((-0.5, -0.5))
                                * native.transform;

                            transparent = native.stops.iter().all(|stop| stop.color[3] == 0.0);
                            let stops = native
                                .stops
                                .iter()
                                .map(|stop| ColorStop {
                                    offset: stop.offset,
                                    color: DynamicColor::from_alpha_color(AlphaColor::<Srgb>::new(
                                        stop.color,
                                    )),
                                })
                                .collect::<Vec<_>>();

                            let gradient = match native.kind {
                                SvgGradientKind::Linear { start, end } => {
                                    Gradient::new_linear(start, end)
                                }
                                SvgGradientKind::Radial {
                                    start_center,
                                    start_radius,
                                    end_center,
                                    end_radius,
                                } => Gradient::new_two_point_radial(
                                    start_center,
                                    start_radius,
                                    end_center,
                                    end_radius,
                                ),
                            }
                            .with_extend(peniko::Extend::Pad)
                            .with_stops(stops.as_slice());

                            PaintType::Gradient(gradient)
                        } else {
                            let texture = encoded.sample_texture_premultiplied(visible, 1.0);
                            transparent = texture.data.iter().all(|texel| texel[3] == 0);
                            paint_transform = path_transform.inverse() * texture.transform;

                            let pixmap = Pixmap::from_parts(
                                bytemuck::cast_vec(texture.data),
                                texture.width as u16,
                                texture.height as u16,
                                PixelMetadata {
                                    may_have_transparency: texture.may_have_transparency,
                                    ..Default::default()
                                },
                            );

                            // A texture on whole device pixels has a texel at each pixel's
                            // center, which filtering would only give back.
                            let [a, b, c, d, e, f] = texture.transform.as_coeffs();
                            let on_pixels = [a, b, c, d] == [1.0, 0.0, 0.0, 1.0]
                                && e.fract() == 0.0
                                && f.fract() == 0.0;
                            let image = Image {
                                image: ImageSource::Pixmap(Arc::new(pixmap)),
                                sampler: ImageSampler {
                                    x_extend: peniko::Extend::Repeat,
                                    y_extend: peniko::Extend::Repeat,
                                    quality: if on_pixels {
                                        ImageQuality::Low
                                    } else {
                                        ImageQuality::Medium
                                    },
                                    alpha: 1.0,
                                },
                            };

                            PaintType::Image(image)
                        }
                    }
                    Pattern::Tiling(t) => {
                        let (image, transform) = self.tiling_paint(&t, bbox, visible, is_stroke);
                        paint_transform = path_transform.inverse() * transform;

                        PaintType::Image(image)
                    }
                }
            }
        };

        if transparent && self.ctx.blend_mode().compose == peniko::Compose::SrcOver {
            return None;
        }

        self.ctx.set_paint_transform(paint_transform);
        self.ctx.set_paint(paint);

        Some(clip_path)
    }
}

/// `bbox` without the whole pixels along each side that lie outside `visible`, so that what's
/// left keeps its place on the pixel grid.
pub(crate) fn trim(bbox: Rect, visible: Rect) -> Rect {
    let outside = |by: f64| by.max(0.0).floor();
    let x0 = bbox.x0 + outside(visible.x0 - bbox.x0);
    let y0 = bbox.y0 + outside(visible.y0 - bbox.y0);
    let x1 = bbox.x1 - outside(bbox.x1 - visible.x1);
    let y1 = bbox.y1 - outside(bbox.y1 - visible.y1);

    Rect::new(x0, y0, x1.max(x0), y1.max(y0))
}
