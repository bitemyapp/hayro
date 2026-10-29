use crate::{Renderer, convert_fill_rule};
use hayro_interpret::{ClipPath, FillRule};
use kurbo::{Affine, BezPath, PathEl, Rect, Shape};

/// A clip as the renderer sees it.
#[derive(Clone, Copy)]
pub(crate) struct Clip {
    /// The device-space bounds of the render target within this and the enclosing clips.
    visible: Rect,
    /// The same for the whole page, past the render target where that's a tile of it.
    on_page: Rect,
    /// Whether the clip was pushed to the render context, rather than skipped for clipping
    /// nothing that could be seen.
    pushed: bool,
}

impl Renderer<'_> {
    pub(super) fn push_clip_path_inner(&mut self, clip_path: &BezPath, fill: FillRule) {
        let old_transform = *self.ctx.transform();

        self.ctx.set_fill_rule(convert_fill_rule(fill));
        self.ctx.set_transform(Affine::IDENTITY);
        self.ctx.push_clip_path(clip_path);

        self.ctx.set_transform(old_transform);
    }

    pub(super) fn push_clip_path(&mut self, clip_path: &ClipPath) {
        let bounds = clip_path.path.bounding_box();
        if is_rect(&clip_path.path) && self.covers_visible(bounds) {
            self.push_clip_bounds(bounds, false);
            return;
        }

        self.push_clip_bounds(bounds, true);
        self.push_clip_path_inner(&clip_path.path, clip_path.fill);
    }

    pub(super) fn push_clip_rect(&mut self, rect: &Rect) {
        if self.covers_visible(*rect) {
            self.push_clip_bounds(*rect, false);
            return;
        }

        self.push_clip_bounds(*rect, true);
        self.ctx.set_fill_rule(convert_fill_rule(FillRule::NonZero));
        self.ctx.set_transform(Affine::IDENTITY);
        self.ctx.push_clip_rect(rect);
    }

    pub(super) fn pop_clip(&mut self) {
        if self.clips.pop().is_none_or(|clip| clip.pushed) {
            self.ctx.pop_clip();
        }
    }

    /// Note a clip with the given device-space bounds, which nothing outside of can be seen.
    pub(super) fn push_clip_bounds(&mut self, bounds: Rect, pushed: bool) {
        // Whole pixels, and one more: a pixel the clip only partly covers can still show a
        // drawing that only partly covers it.
        let bounds = bounds.expand().inflate(1.0, 1.0);
        let within = |rect: Rect| {
            if bounds.is_finite() {
                rect.intersect(bounds)
            } else {
                rect
            }
        };
        let (visible, on_page) = (within(self.visible()), within(self.on_page()));

        self.clips.push(Clip {
            visible,
            on_page,
            pushed,
        });
    }

    /// The part of the render target, in device pixels, that the active clips let through.
    pub(super) fn visible(&self) -> Rect {
        self.clips.last().map_or_else(
            || Rect::new(0.0, 0.0, self.ctx.width() as f64, self.ctx.height() as f64),
            |clip| clip.visible,
        )
    }

    /// The part of the page, in device pixels, that the active clips let through: the same for
    /// every tile of it.
    pub(super) fn on_page(&self) -> Rect {
        self.clips.last().map_or(self.page, |clip| clip.on_page)
    }

    /// Whether a rectangle clips nothing that can be seen: it covers everything the render
    /// target and the active clips let through, to within 1/256 of a pixel.
    fn covers_visible(&self, rect: Rect) -> bool {
        rect.inflate(1.0 / 256.0, 1.0 / 256.0)
            .contains_rect(self.visible())
    }
}

/// Whether a device-space path is a rectangle with sides along the axes.
fn is_rect(path: &BezPath) -> bool {
    let mut points = [kurbo::Point::ZERO; 5];
    let mut count = 0;
    for (i, el) in path.elements().iter().enumerate() {
        match (i, el) {
            (0, PathEl::MoveTo(p)) | (1..=4, PathEl::LineTo(p)) => {
                points[count] = *p;
                count += 1;
            }
            (4 | 5, PathEl::ClosePath) if i == count => {}
            _ => return false,
        }
    }

    let corners = match count {
        4 => &points[..4],
        5 if points[4] == points[0] => &points[..4],
        _ => return false,
    };

    (0..4).all(|i| {
        let (a, b) = (corners[i], corners[(i + 1) % 4]);
        (a.x - b.x).abs() < 1e-9 || (a.y - b.y).abs() < 1e-9
    })
}

impl Renderer<'_> {
    /// A clip for a layer whose contents lie within `bounds` (device space): the whole pixels
    /// around them that the active clips let through, or `None` if that's all of the render
    /// target. Compositing the layer then only touches those pixels. `vello_cpu` composites a
    /// clipped layer's partly covered pixels slightly differently from an unclipped one's, so
    /// this is only worth it for layers that would otherwise be composited across the whole
    /// target (destructive blend modes, as for images with a soft mask of another size).
    pub(crate) fn layer_clip(&self, bounds: Rect) -> Option<BezPath> {
        let region = bounds.expand().inflate(1.0, 1.0);
        let region = if region.is_finite() {
            region.intersect(self.visible())
        } else {
            self.visible()
        };
        let target = Rect::new(0.0, 0.0, self.ctx.width() as f64, self.ctx.height() as f64);

        (!region.contains_rect(target)).then(|| region.to_path(0.1))
    }

    /// Push a layer, clipped to `clip` in device space.
    pub(crate) fn push_layer_within(
        &mut self,
        clip: Option<&BezPath>,
        blend_mode: Option<vello_cpu::peniko::BlendMode>,
        opacity: Option<f32>,
        mask: Option<vello_cpu::Mask>,
    ) {
        let transform = *self.ctx.transform();
        self.ctx.set_transform(Affine::IDENTITY);
        self.ctx.push_layer(clip, blend_mode, opacity, mask, None);
        self.ctx.set_transform(transform);
    }
}
