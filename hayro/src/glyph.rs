use crate::{Renderer, convert_fill_rule};
use hayro_interpret::color::Color;
use hayro_interpret::font::{Glyph, GlyphRun, OutlineGlyph};
use hayro_interpret::{CacheKey, DrawMode, DrawProps, FillRule, Paint, StrokeProps};
use kurbo::{Affine, BezPath, Rect, Shape};
use rustc_hash::{FxHashMap, FxHashSet};
use std::rc::Rc;
use std::sync::Arc;
use vello_cpu::color::palette::css::BLACK;
use vello_cpu::color::{AlphaColor, Srgb};
use vello_cpu::peniko::{Extend, Fill, ImageQuality, ImageSampler};
use vello_cpu::{Image, ImageSource, Pixmap, Tint, TintMode};

/// Glyph origins are placed to this fraction of a pixel when drawn from the cache.
const SUBPIXEL: f64 = 4.0;
/// Glyphs whose em is larger than this many pixels are filled as paths.
const MAX_CACHED_EM: f64 = 192.0;
/// The cache is emptied when its coverage grows past this.
const MAX_CACHED_BYTES: usize = 64 << 20;
/// Glyphs drawn once are forgotten when there are more than this many.
const MAX_SEEN_GLYPHS: usize = 1 << 18;

/// The coverage of outline glyphs as drawn at one size, orientation and subpixel
/// position, so that drawing a glyph again is copying its coverage in the paint's
/// color instead of filling its outline again. A glyph is cached the second time it's drawn,
/// so glyphs drawn once cost no more than filling them.
#[derive(Default)]
pub(crate) struct GlyphCache {
    glyphs: FxHashMap<GlyphKey, Option<CachedGlyph>>,
    seen: FxHashSet<GlyphKey>,
    bytes: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct GlyphKey {
    id: u128,
    /// The linear part of the glyph's transform, for an em of 1, in 1/65536 pixels.
    linear: [i32; 4],
    /// Where the origin falls within its pixel, in steps of `1 / SUBPIXEL`.
    phase: [u8; 2],
}

#[derive(Clone)]
struct CachedGlyph {
    pixmap: Arc<Pixmap>,
    /// The pixmap's top left, relative to the pixel the glyph origin falls in.
    offset: (f64, f64),
}

fn solid(color: &Color) -> AlphaColor<Srgb> {
    let c = color.to_rgba().to_rgba8();
    AlphaColor::from_rgba8(c[0], c[1], c[2], c[3])
}

impl Renderer<'_> {
    /// The cached coverage of the glyph under the transform and where to draw it, or
    /// `None` if the glyph should be filled as a path. A cached glyph without coverage
    /// (a space) draws nothing.
    fn cached_coverage(
        &self,
        glyph: &OutlineGlyph,
        transform: Affine,
    ) -> Option<Option<(CachedGlyph, f64, f64)>> {
        let [a, b, c, d, e, f] = transform.as_coeffs();
        let em = (a * a + b * b).max(c * c + d * d).sqrt() * 1000.0;
        if em.is_nan() || em > MAX_CACHED_EM || !e.is_finite() || !f.is_finite() {
            return None;
        }

        let quantize = |v: f64| (v * 1000.0 * 65536.0).round() as i32;
        let (x, y) = ((e * SUBPIXEL).round(), (f * SUBPIXEL).round());
        let (whole_x, whole_y) = (x.div_euclid(SUBPIXEL), y.div_euclid(SUBPIXEL));
        let phase = [x.rem_euclid(SUBPIXEL), y.rem_euclid(SUBPIXEL)];
        let key = GlyphKey {
            id: glyph.identifier().cache_key(),
            linear: [quantize(a), quantize(b), quantize(c), quantize(d)],
            phase: [phase[0] as u8, phase[1] as u8],
        };

        let placed = |cached: Option<CachedGlyph>| {
            cached.map(|cached| {
                let (x, y) = (whole_x + cached.offset.0, whole_y + cached.offset.1);
                (cached, x, y)
            })
        };

        {
            let mut cache = self.global.glyph_cache.borrow_mut();
            if let Some(cached) = cache.glyphs.get(&key) {
                return Some(placed(cached.clone()));
            }
            if cache.seen.len() >= MAX_SEEN_GLYPHS {
                cache.seen.clear();
            }
            if cache.seen.insert(key) {
                return None;
            }
        }

        let outline = self.cached_outline(glyph);
        let at = Affine::new([a, b, c, d, phase[0] / SUBPIXEL, phase[1] / SUBPIXEL]);
        let bbox = at.transform_rect_bbox(outline.control_box());
        let cached = if outline.elements().is_empty() || bbox.area() == 0.0 {
            None
        } else {
            let (x0, y0) = (bbox.x0.floor(), bbox.y0.floor());
            let width = (bbox.x1.ceil() - x0) as u16;
            let height = (bbox.y1.ceil() - y0) as u16;
            let mut ctx = self.child_context(width, height);
            ctx.set_transform(Affine::translate((-x0, -y0)) * at);
            ctx.set_fill_rule(Fill::NonZero);
            ctx.set_paint(BLACK);
            ctx.fill_path(outline.as_ref());
            ctx.flush();
            let mut pixmap = Pixmap::new(width, height);
            ctx.render(&mut pixmap, &mut vello_cpu::Resources::default());
            Some(CachedGlyph {
                pixmap: Arc::new(pixmap),
                offset: (x0, y0),
            })
        };

        let mut cache = self.global.glyph_cache.borrow_mut();
        let bytes = cached.as_ref().map_or(0, |c| {
            c.pixmap.width() as usize * c.pixmap.height() as usize * 4
        });
        if cache.bytes + bytes > MAX_CACHED_BYTES {
            cache.glyphs.clear();
            cache.seen.clear();
            cache.bytes = 0;
        }
        cache.bytes += bytes;
        cache.glyphs.insert(key, cached.clone());

        Some(placed(cached))
    }

    fn fill_cached_glyphs(
        &mut self,
        glyph_run: &GlyphRun<'_, '_>,
        props: &DrawProps<'_>,
        color: &Color,
    ) {
        let clip_path = self.begin_fill_run(props, Rect::ZERO);
        let color = solid(color);
        let mut tinted = false;

        for glyph in glyph_run.glyphs() {
            let Glyph::Outline(outline) = &**glyph else {
                continue;
            };

            let transform = props.transform * glyph.transform();
            match self.cached_coverage(outline, transform) {
                Some(Some((cached, x, y))) => {
                    let (width, height) = (cached.pixmap.width(), cached.pixmap.height());
                    let rect = Rect::new(x, y, x + width as f64, y + height as f64);
                    if self.misses_target(rect) {
                        continue;
                    }

                    if !tinted {
                        self.ctx.set_tint(Some(Tint {
                            color,
                            mode: TintMode::AlphaMask,
                        }));
                        tinted = true;
                    }

                    self.ctx.set_transform(Affine::translate((x, y)));
                    self.ctx.set_paint_transform(Affine::IDENTITY);
                    self.ctx.set_paint(Image {
                        image: ImageSource::Pixmap(cached.pixmap),
                        sampler: ImageSampler {
                            x_extend: Extend::Pad,
                            y_extend: Extend::Pad,
                            quality: ImageQuality::Low,
                            alpha: 1.0,
                        },
                    });
                    self.ctx
                        .fill_rect(&Rect::new(0.0, 0.0, width as f64, height as f64));
                }
                Some(None) => {}
                None => {
                    if tinted {
                        self.ctx.reset_tint();
                        tinted = false;
                    }

                    self.ctx.set_paint(color);
                    self.ctx.set_transform(transform);
                    self.ctx.fill_path(self.cached_outline(outline).as_ref());
                }
            }
        }

        if tinted {
            self.ctx.reset_tint();
        }

        self.finish_run(clip_path);
    }

    fn fill_glyph<'a>(&mut self, glyph: &Glyph<'a>, props: DrawProps<'a>, transform: Affine) {
        match glyph {
            Glyph::Outline(glyph) => {
                let outline = self.cached_outline(glyph);
                let props = DrawProps {
                    transform: props.transform * transform,
                    ..props
                };

                self.fill_path(outline.as_ref(), props, FillRule::NonZero);
            }
            Glyph::Type3(glyph) => {
                self.in_type3_glyph = true;
                glyph.interpret(self, props.transform, transform, &props.paint);
                self.in_type3_glyph = false;
            }
        }
    }

    fn stroke_glyph<'a>(
        &mut self,
        glyph: &Glyph<'a>,
        props: DrawProps<'a>,
        transform: Affine,
        stroke_props: &StrokeProps,
    ) {
        match glyph {
            Glyph::Outline(glyph) => {
                let outline = self.cached_outline(glyph);
                self.stroke_path(
                    &(transform * outline.as_ref().clone()),
                    props,
                    stroke_props,
                    true,
                );
            }
            Glyph::Type3(glyph) => {
                glyph.interpret(self, props.transform, transform, &props.paint);
            }
        }
    }

    fn cached_outline(&self, glyph: &OutlineGlyph) -> Rc<BezPath> {
        let id = glyph.identifier().cache_key();

        if let Some(path) = self.global.outline_cache.borrow().get(&id) {
            return path.clone();
        }

        let path = Rc::new(glyph.outline());
        self.global
            .outline_cache
            .borrow_mut()
            .insert(id, path.clone());
        path
    }

    fn transformed_outlines(&self, glyph_run: &GlyphRun<'_, '_>) -> Vec<BezPath> {
        glyph_run
            .glyphs()
            .iter()
            .filter_map(|glyph| {
                let Glyph::Outline(outline) = &**glyph else {
                    return None;
                };

                Some(glyph.transform() * self.cached_outline(outline).as_ref().clone())
            })
            .collect()
    }

    fn outline_bbox(outlines: &[BezPath]) -> Rect {
        outlines
            .iter()
            .filter(|outline| !outline.elements().is_empty())
            .map(Shape::bounding_box)
            .reduce(|bbox, glyph_bbox| bbox.union(glyph_bbox))
            .unwrap_or(Rect::ZERO)
    }

    fn begin_fill_run(&mut self, props: &DrawProps<'_>, bbox: Rect) -> Option<BezPath> {
        self.ctx.set_fill_rule(convert_fill_rule(FillRule::NonZero));
        self.apply_draw_props(props);

        let clip_path = self.set_paint(&props.paint, || bbox, false);
        if let Some(clip_path) = clip_path.as_ref() {
            self.push_clip_path_inner(clip_path, FillRule::NonZero);
        }

        clip_path
    }

    fn begin_stroke_run(
        &mut self,
        props: &DrawProps<'_>,
        stroke_props: &StrokeProps,
        bbox: Rect,
    ) -> Option<BezPath> {
        self.apply_draw_props(props);
        self.set_stroke_properties(stroke_props, true);

        let clip_path = self.set_paint(&props.paint, || bbox, true);
        if let Some(clip_path) = clip_path.as_ref() {
            self.push_clip_path_inner(clip_path, FillRule::NonZero);
        }

        clip_path
    }

    fn finish_run(&mut self, clip_path: Option<BezPath>) {
        if clip_path.is_some() {
            self.ctx.pop_clip();
        }
    }

    fn fill_outline_run(&mut self, glyph_run: &GlyphRun<'_, '_>, props: &DrawProps<'_>) {
        match &props.paint {
            Paint::Color(color) => self.fill_cached_glyphs(glyph_run, props, color),
            Paint::Pattern(_) => {
                let outlines = self.transformed_outlines(glyph_run);
                let clip_path = self.begin_fill_run(props, Self::outline_bbox(&outlines));
                for outline in &outlines {
                    self.ctx.fill_path(outline);
                }
                self.finish_run(clip_path);
            }
        }
    }

    fn stroke_outline_run(
        &mut self,
        glyph_run: &GlyphRun<'_, '_>,
        props: &DrawProps<'_>,
        stroke_props: &StrokeProps,
    ) {
        match &props.paint {
            Paint::Color(_) => {
                let clip_path = self.begin_stroke_run(props, stroke_props, Rect::ZERO);
                for glyph in glyph_run.glyphs() {
                    let Glyph::Outline(outline) = &**glyph else {
                        continue;
                    };

                    let outline = glyph.transform() * self.cached_outline(outline).as_ref().clone();
                    self.ctx.stroke_path(&outline);
                }
                self.finish_run(clip_path);
            }
            Paint::Pattern(_) => {
                let outlines = self.transformed_outlines(glyph_run);
                let clip_path =
                    self.begin_stroke_run(props, stroke_props, Self::outline_bbox(&outlines));
                for outline in &outlines {
                    self.ctx.stroke_path(outline);
                }
                self.finish_run(clip_path);
            }
        }
    }

    fn draw_glyphs_individually<'a>(
        &mut self,
        glyph_run: &GlyphRun<'_, 'a>,
        props: DrawProps<'a>,
        draw_mode: &DrawMode,
    ) {
        for glyph in glyph_run.glyphs() {
            let transform = glyph.transform();
            match draw_mode {
                DrawMode::Fill(_) => {
                    Self::fill_glyph(self, glyph, props.clone(), transform);
                }
                DrawMode::Stroke(stroke) => {
                    Self::stroke_glyph(self, glyph, props.clone(), transform, stroke);
                }
                DrawMode::FillAndStroke(_, stroke) => {
                    Self::fill_glyph(self, glyph, props.clone(), transform);
                    Self::stroke_glyph(self, glyph, props.clone(), transform, stroke);
                }
                DrawMode::Invisible => {}
            }
        }
    }

    pub(super) fn draw_glyph_run<'a>(
        &mut self,
        glyph_run: &GlyphRun<'_, 'a>,
        props: DrawProps<'a>,
        draw_mode: &DrawMode,
    ) {
        let outlines_only = glyph_run
            .glyphs()
            .iter()
            .all(|glyph| matches!(&**glyph, Glyph::Outline(_)));

        if !outlines_only {
            self.draw_glyphs_individually(glyph_run, props, draw_mode);

            return;
        }

        match draw_mode {
            DrawMode::Fill(_) => self.fill_outline_run(glyph_run, &props),
            DrawMode::Stroke(stroke) => self.stroke_outline_run(glyph_run, &props, stroke),
            DrawMode::FillAndStroke(_, stroke) => {
                self.fill_outline_run(glyph_run, &props);
                self.stroke_outline_run(glyph_run, &props, stroke);
            }
            DrawMode::Invisible => {}
        }
    }
}
