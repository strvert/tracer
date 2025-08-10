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
pub struct DotShading {
    pub albedo: Color,
}

impl Material for DotShading {
    fn shade(&self, ray_in: &Ray, rec: &HitRecord) -> Color {
        // 入射方向はレイの逆ベクトル。単位化して法線との類似度（コサイン）を取る。
        let i = (-ray_in.direction).normalized();
        let ndoti = rec.normal.dot(i).max(0.0);
        self.albedo * ndoti
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
