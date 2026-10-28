use crate::{Renderer, convert_fill_rule};
use hayro_interpret::{ClipPath, FillRule};
use kurbo::{Affine, BezPath, Rect, Shape};

impl Renderer<'_> {
    pub(super) fn push_clip_path_inner(&mut self, clip_path: &BezPath, fill: FillRule) {
        let old_transform = *self.ctx.transform();

        self.ctx.set_fill_rule(convert_fill_rule(fill));
        self.ctx.set_transform(Affine::IDENTITY);
        self.ctx.push_clip_path(clip_path);

        self.ctx.set_transform(old_transform);
    }

    pub(super) fn push_clip_path(&mut self, clip_path: &ClipPath) {
        self.push_clip_bounds(clip_path.path.bounding_box());
        self.push_clip_path_inner(&clip_path.path, clip_path.fill);
    }

    pub(super) fn push_clip_rect(&mut self, rect: &Rect) {
        self.push_clip_bounds(*rect);
        self.ctx.set_fill_rule(convert_fill_rule(FillRule::NonZero));
        self.ctx.set_transform(Affine::IDENTITY);
        self.ctx.push_clip_rect(rect);
    }

    pub(super) fn pop_clip(&mut self) {
        self.clip_bounds.pop();
        self.ctx.pop_clip();
    }

    /// Note a clip with the given device-space bounds, which nothing outside of can be seen.
    pub(super) fn push_clip_bounds(&mut self, bounds: Rect) {
        let visible = self.visible();
        // Whole pixels, and one more: a pixel the clip only partly covers can still show a
        // drawing that only partly covers it.
        let bounds = bounds.expand().inflate(1.0, 1.0);
        self.clip_bounds.push(if bounds.is_finite() {
            visible.intersect(bounds)
        } else {
            visible
        });
    }

    /// The part of the render target, in device pixels, that the active clips let through.
    pub(super) fn visible(&self) -> Rect {
        self.clip_bounds.last().copied().unwrap_or_else(|| {
            Rect::new(0.0, 0.0, self.ctx.width() as f64, self.ctx.height() as f64)
        })
    }
}
