use crate::Renderer;
use crate::paint::trim;
use hayro_interpret::pattern::{Pattern, TilingPattern};
use hayro_interpret::util::x_y_advances;
use hayro_interpret::{BlendMode, ClipPath, DrawProps, FillRule, Paint};
use kurbo::{Affine, BezPath, Rect, Shape};
use std::ops::{Range, RangeInclusive};
use std::sync::Arc;
use vello_cpu::peniko::{ImageQuality, ImageSampler};
use vello_cpu::{Image, ImageSource, Pixmap, peniko};

/// Fills showing at most this many copies of a pattern's cell draw them directly: more would
/// interpret the cell again for each, where a texture interprets it once.
const MAX_DIRECT_COPIES: usize = 1;

impl Renderer<'_> {
    /// Fill `path`, in the space of `props.transform`, with a tiling pattern by drawing the
    /// copies of its cell within it directly, rather than through a texture, when only a few
    /// can be seen and nothing applies to the fill as a whole. Returns whether it did. Drawn at
    /// device resolution, the cell's lines get the minimum width other lines get.
    pub(crate) fn fill_tiling_directly(
        &mut self,
        path: &BezPath,
        props: &DrawProps<'_>,
        fill_rule: FillRule,
    ) -> bool {
        let Paint::Pattern(pattern) = &props.paint else {
            return false;
        };
        let Pattern::Tiling(t) = pattern.as_ref() else {
            return false;
        };
        if props.soft_mask.is_some() || props.blend_mode != BlendMode::Normal {
            return false;
        }

        let path = props.transform * path.clone();
        let target = Rect::new(0.0, 0.0, self.ctx.width() as f64, self.ctx.height() as f64);
        let area = path.bounding_box().intersect(target);
        let (x, y) = self.tile_axes(t, area, trim(area, self.visible()));
        let copies = x.copies.clone().count() * y.copies.clone().count();
        if x.extend != peniko::Extend::Pad
            || y.extend != peniko::Extend::Pad
            || copies > MAX_DIRECT_COPIES
        {
            return false;
        }

        self.push_clip_path(&ClipPath {
            path,
            fill: fill_rule,
        });
        if t.opacity < 1.0 {
            self.ctx.push_layer(None, None, Some(t.opacity), None, None);
        }

        for j in y.copies.clone() {
            for i in x.copies.clone() {
                let offset = Affine::translate((f64::from(i) * x.step, f64::from(j) * y.step));
                t.interpret(self, t.matrix * offset, false);
            }
        }

        if t.opacity < 1.0 {
            self.ctx.pop_layer();
        }
        self.pop_clip();

        true
    }

    /// The pattern's axes, for filling `area` of which `visible` can be seen (device space).
    fn tile_axes(&self, t: &TilingPattern<'_>, area: Rect, visible: Rect) -> (TileAxis, TileAxis) {
        let area = t.matrix.inverse().transform_rect_bbox(area);
        let visible = t.matrix.inverse().transform_rect_bbox(visible);
        let (x_advance, y_advance) = x_y_advances(&t.matrix);
        let x = TileAxis::new(
            x_advance.length(),
            t.bbox.x0..t.bbox.x1,
            t.x_step,
            area.x0..area.x1,
            visible.x0..visible.x1,
        );
        let y = TileAxis::new(
            y_advance.length(),
            t.bbox.y0..t.bbox.y1,
            t.y_step,
            area.y0..area.y1,
            visible.y0..visible.y1,
        );

        (x, y)
    }

    pub(crate) fn tiling_paint(
        &self,
        t: &TilingPattern<'_>,
        area: Rect,
        visible: Rect,
        is_stroke: bool,
    ) -> (Image, Affine) {
        let (x, y) = self.tile_axes(t, area, visible);

        let image = Image {
            image: ImageSource::Pixmap(Arc::new(self.render_tiling(t, &x, &y, is_stroke))),
            sampler: ImageSampler {
                x_extend: x.extend,
                y_extend: y.extend,
                quality: ImageQuality::Medium,
                alpha: t.opacity,
            },
        };
        let transform = t.matrix
            * Affine::translate((x.start, y.start))
            * Affine::scale_non_uniform(1.0 / x.scale, 1.0 / y.scale);

        (image, transform)
    }

    fn render_tiling(
        &self,
        t: &TilingPattern<'_>,
        x: &TileAxis,
        y: &TileAxis,
        is_stroke: bool,
    ) -> Pixmap {
        let to_texture =
            Affine::scale_non_uniform(x.scale, y.scale) * Affine::translate((-x.start, -y.start));
        let mut ctx = self.child_context(x.size, y.size);
        let mut renderer = Renderer::new(&mut ctx, self.global);
        renderer.inside_pattern = true;

        for j in y.copies.clone() {
            for i in x.copies.clone() {
                let offset = Affine::translate((f64::from(i) * x.step, f64::from(j) * y.step));
                t.interpret(&mut renderer, to_texture * offset, is_stroke);
            }
        }

        let mut pixmap = Pixmap::new(x.size, y.size);
        ctx.flush();
        ctx.render(&mut pixmap, &mut vello_cpu::Resources::default());

        pixmap
    }
}

struct TileAxis {
    start: f64,
    scale: f64,
    size: u16,
    step: f64,
    copies: RangeInclusive<i32>,
    extend: peniko::Extend,
}

impl TileAxis {
    /// `area` is what the pattern fills, `seen` the part of it that can be seen.
    fn new(scale: f64, cell: Range<f64>, step: f32, area: Range<f64>, seen: Range<f64>) -> Self {
        let step = f64::from(step.abs());
        let width = cell.end - cell.start;
        let visible = (area.end - area.start).max(0.0);
        let repeat = visible >= step;

        let scale = (step * scale).round().max(1.0) / step;

        if repeat {
            Self {
                start: cell.start,
                scale,
                size: (step * scale).round() as u16,
                step,
                copies: (-width / step).floor() as i32 + 1..=0,
                extend: peniko::Extend::Repeat,
            }
        } else {
            // Without repeating, the texture only covers what can be seen.
            let start = seen.start.max(area.start);
            let area = start..seen.end.min(area.end).max(start);
            let start = cell.start + ((area.start - cell.start) * scale).floor() / scale;

            Self {
                start,
                scale,
                size: ((area.end - start) * scale).ceil().max(1.0) as u16,
                step,
                copies: ((area.start - cell.end) / step).ceil() as i32
                    ..=((area.end - cell.start) / step).floor() as i32,
                extend: peniko::Extend::Pad,
            }
        }
    }
}
