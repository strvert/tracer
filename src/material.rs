//! Simple material interface, dot-product material, and a registry for materials.

use crate::hit::HitRecord;
use crate::math::{Color, Ray};
use crate::types::MaterialId;

pub trait Material {
    /// Shade using incoming ray and hit information. Return linear color.
    fn shade(&self, ray_in: &Ray, rec: &HitRecord) -> Color;

    /// Base color used for simple Lambertian direct lighting.
    fn albedo(&self) -> Color { Color::ONE }
}

#[derive(Clone, Debug)]
pub struct Lambertian {
    pub albedo: Color,
}

impl Material for Lambertian {
    fn shade(&self, ray_in: &Ray, rec: &HitRecord) -> Color {
    // Lambert: f = ρ/π, ここでは shade = (ρ/π)·max(n·ω_i, 0)
    // 既存の約束に合わせ、ray_in.direction は「入射方向の逆」を与えることで i= -ray_in.dir → ω_i
    let i = (-ray_in.direction).normalized(); // ω_i
    let ndoti = rec.normal.dot(i).max(0.0);
    self.albedo * ndoti * core::f32::consts::FRAC_1_PI
    }

    fn albedo(&self) -> Color { self.albedo }
}

#[derive(Default)]
pub struct MaterialRegistry {
    materials: Vec<Box<dyn Material + Send + Sync>>, // allow sharing later
}

impl MaterialRegistry {
    pub fn new() -> Self { Self { materials: Vec::new() } }

    pub fn add<M: Material + Send + Sync + 'static>(&mut self, mat: M) -> MaterialId {
        let id = self.materials.len();
        self.materials.push(Box::new(mat));
        id
    }

    pub fn get(&self, id: MaterialId) -> &dyn Material {
        &*self.materials[id]
    }
}
