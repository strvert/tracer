use crate::renderer::RenderCtx;
use crate::math::{Color, Ray};

pub trait Integrator: Send + Sync {
    // li: 光線 r に対する放射輝度 L_i を計算
    fn li(&self, ctx: &RenderCtx, ray: &Ray) -> Color;
}

mod direct_lighting;
pub use direct_lighting::DirectLighting;
