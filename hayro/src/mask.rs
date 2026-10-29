use crate::{GlobalState, Renderer, convert_blend_mode, derive_settings};
use hayro_interpret::font::GlyphRun;
use hayro_interpret::util::x_y_advances;
use hayro_interpret::{
    BlendMode, CacheKey, ClipPath, Device, DrawMode, DrawProps, Image, ImageDrawProps, MaskType,
    SoftMask,
};
use kurbo::{BezPath, Rect, Vec2};
use vello_cpu::color::palette::css::BLACK;
use vello_cpu::color::{AlphaColor, Srgb};
use vello_cpu::{Mask, Pixmap, RenderSettings};

impl Renderer<'_> {
    pub(super) fn apply_soft_mask(&mut self, mask: Option<&SoftMask<'_>>) {
        let mask = mask.and_then(|m| self.soft_mask(m));

        if let Some(mask) = mask {
            self.ctx.set_mask(mask);
        } else {
            self.ctx.reset_mask();
        }
    }

    pub(super) fn push_transparency_group(
        &mut self,
        opacity: f32,
        mask: Option<SoftMask<'_>>,
        blend_mode: BlendMode,
    ) {
        let mask = mask.and_then(|m| self.soft_mask(&m));
        self.ctx.push_layer(
            None,
            Some(convert_blend_mode(blend_mode)),
            Some(opacity),
            mask,
            None,
        );
    }

    pub(super) fn pop_transparency_group(&mut self) {
        self.ctx.pop_layer();
    }

    /// The mask, drawn where it can matter: within what the active clips let through. `None`
    /// for a mask that lets everything through.
    fn soft_mask(&mut self, mask: &SoftMask<'_>) -> Option<Mask> {
        let settings = *self.ctx.render_settings();
        let (width, height) = (self.ctx.width(), self.ctx.height());
        let visible = self.visible();
        let region = [visible.x0, visible.y0, visible.x1, visible.y1].map(|v| v as u16);
        let global = self.global;

        self.soft_mask_cache
            .entry((mask.cache_key(), region))
            .or_insert_with(|| draw_soft_mask(mask, settings, width, height, global, visible))
            .clone()
    }
}

fn draw_soft_mask(
    mask: &SoftMask<'_>,
    settings: RenderSettings,
    width: u16,
    height: u16,
    global: &GlobalState,
    visible: Rect,
) -> Option<Mask> {
    // Outside the bounds of what its group draws, the mask is its background, and only what
    // the clips let through matters, so only the rest is drawn. Both are whole pixels.
    let region = mask.bbox().expand().inflate(1.0, 1.0);
    let region = if region.is_finite() {
        region.intersect(visible)
    } else {
        visible
    };
    // The group's bounds often cover the page where it draws a little, or nothing.
    let region = match drawn_bounds(mask, region) {
        Some(drawn) => region.intersect(drawn.expand().inflate(2.0, 2.0)),
        None => region,
    };
    let mut values = draw_region(mask, settings, region, global);
    let mut outside = [background(mask, settings)];

    if let Some(transfer_function) = mask.transfer_function() {
        transfer_function.apply_to(&mut values);
        transfer_function.apply_to(&mut outside);
    }

    // A mask that's opaque everywhere changes nothing.
    if outside[0] == 255
        && values
            .chunks(4096)
            .all(|chunk| chunk.iter().fold(255, |all, &v| all & v) == 255)
    {
        return None;
    }

    Some(placed(values, region, outside[0], width, height))
}

/// The bounds, within `region`, of what the mask's group draws (an empty rectangle if
/// nothing), found by interpreting it without drawing: the bounds of its paths (and how far
/// their strokes reach) and images, or `None` where it shows text, whose bounds aren't worth
/// finding here.
fn drawn_bounds(mask: &SoftMask<'_>, region: Rect) -> Option<Rect> {
    let mut bounds = DrawnBounds {
        drawn: Rect::ZERO,
        any: false,
        text: false,
    };
    mask.interpret_translated(&mut bounds, Vec2::ZERO);
    if bounds.text {
        return None;
    }
    Some(if bounds.any {
        bounds.drawn
    } else {
        Rect::new(region.x0, region.y0, region.x0, region.y0)
    })
}

/// A device collecting the bounds of what's drawn; see [`drawn_bounds`]. Clips and groups
/// only make what's drawn smaller, so they're ignored.
struct DrawnBounds {
    drawn: Rect,
    any: bool,
    text: bool,
}

impl DrawnBounds {
    fn add(&mut self, bounds: Rect) {
        self.drawn = if self.any {
            self.drawn.union(bounds)
        } else {
            bounds
        };
        self.any = true;
    }
}

impl<'a> Device<'a> for DrawnBounds {
    fn draw_path(&mut self, path: &BezPath, props: DrawProps<'a>, draw_mode: &DrawMode) {
        let bounds = props.transform.transform_rect_bbox(path.control_box());
        let reach = match draw_mode {
            DrawMode::Stroke(stroke) | DrawMode::FillAndStroke(_, stroke) => {
                let (x, y) = x_y_advances(&props.transform);
                let width = stroke.line_width.max(1.0) as f64 * x.length().max(y.length());
                width * (stroke.miter_limit as f64).max(std::f64::consts::SQRT_2)
            }
            DrawMode::Fill(_) => 0.0,
            DrawMode::Invisible => return,
        };
        self.add(bounds.inflate(reach, reach));
    }

    fn push_clip_path(&mut self, _: &ClipPath) {}

    fn push_transparency_group(&mut self, _: f32, _: Option<SoftMask<'a>>, _: BlendMode) {}

    fn draw_glyph_run(&mut self, _: &GlyphRun<'_, 'a>, _: DrawProps<'a>, _: &DrawMode) {
        self.text = true;
    }

    fn draw_image(&mut self, image: Image<'a, '_>, props: ImageDrawProps<'a>) {
        let local = Rect::new(0.0, 0.0, image.width() as f64, image.height() as f64);
        self.add(props.transform.transform_rect_bbox(local));
    }

    fn pop_clip(&mut self) {}

    fn pop_transparency_group(&mut self) {}
}

/// A mask of `width` by `height` with `values` in `region` (whole pixels within it, row by
/// row) and `outside` elsewhere.
pub(crate) fn placed(values: Vec<u8>, region: Rect, outside: u8, width: u16, height: u16) -> Mask {
    if values.len() == usize::from(width) * usize::from(height) {
        return Mask::from_parts(values, width, height);
    }

    let (x0, y0) = (region.x0 as usize, region.y0 as usize);
    let region_width = region.width() as usize;
    let mut all = vec![outside; usize::from(width) * usize::from(height)];
    if region_width > 0 {
        for (row, values) in values.chunks_exact(region_width).enumerate() {
            let start = (y0 + row) * usize::from(width) + x0;
            all[start..][..region_width].copy_from_slice(values);
        }
    }

    Mask::from_parts(all, width, height)
}

/// The mask's values within `region` of the render target, row by row.
fn draw_region(
    mask: &SoftMask<'_>,
    settings: RenderSettings,
    region: Rect,
    global: &GlobalState,
) -> Vec<u8> {
    let (width, height) = (region.width() as u16, region.height() as u16);
    if width == 0 || height == 0 {
        return Vec::new();
    }

    let mut ctx = vello_cpu::RenderContext::new_with(width, height, derive_settings(&settings));
    let mut renderer = Renderer::new(&mut ctx, global);

    let bg_color = mask.background_color().to_rgba();
    let apply_bg = bg_color.to_rgba8() != BLACK.to_rgba8().to_u8_array();

    if apply_bg {
        renderer
            .ctx
            .set_paint(AlphaColor::<Srgb>::new(bg_color.components()));
        renderer
            .ctx
            .fill_rect(&Rect::new(0.0, 0.0, width as f64, height as f64));
        renderer.ctx.push_layer(None, None, None, None, None);
    }

    mask.interpret_translated(&mut renderer, Vec2::new(-region.x0, -region.y0));

    if apply_bg {
        renderer.ctx.pop_layer();
    }

    let mut pix = crate::zeroed_pixmap(width, height);
    renderer.ctx.flush();
    let mut resources = vello_cpu::Resources::default();
    renderer.ctx.render(&mut pix, &mut resources);

    match mask.mask_type() {
        MaskType::Luminosity => luminance(&pix),
        MaskType::Alpha => pix.data().iter().map(|pixel| pixel.a).collect(),
    }
}

/// The mask's value where its group draws nothing: its background, filled as `draw_region`
/// fills it.
fn background(mask: &SoftMask<'_>, settings: RenderSettings) -> u8 {
    let bg_color = mask.background_color().to_rgba();
    if bg_color.to_rgba8() == BLACK.to_rgba8().to_u8_array() {
        return 0;
    }

    let mut ctx = vello_cpu::RenderContext::new_with(1, 1, derive_settings(&settings));
    ctx.set_paint(AlphaColor::<Srgb>::new(bg_color.components()));
    ctx.fill_rect(&Rect::new(0.0, 0.0, 1.0, 1.0));
    let mut pix = Pixmap::new(1, 1);
    ctx.flush();
    ctx.render(&mut pix, &mut vello_cpu::Resources::default());

    match mask.mask_type() {
        MaskType::Luminosity => luminance(&pix)[0],
        MaskType::Alpha => pix.data()[0].a,
    }
}

/// The luminance of each pixel of a premultiplied pixmap, as `Mask::new_luminance` computes it
/// (its weighted channels added in the same order, so the result is the same), with each
/// channel's weighted values looked up rather than divided and multiplied for every pixel.
pub(crate) fn luminance(pixmap: &Pixmap) -> Vec<u8> {
    let [r, g, b] = [0.2126_f32, 0.7152, 0.0722]
        .map(|weight| std::array::from_fn::<f32, 256, _>(|v| (v as f32 / 255.0) * weight));

    pixmap
        .data()
        .iter()
        .map(|pixel| {
            let luma = r[usize::from(pixel.r)] + g[usize::from(pixel.g)] + b[usize::from(pixel.b)];
            (luma * 255.0 + 0.5) as u8
        })
        .collect()
}
