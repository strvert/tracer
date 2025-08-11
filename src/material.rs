//! Simple material interface and a registry for materials.

use crate::math::Color;
use crate::types::MaterialId;

pub trait Material {
    /// 直接照明のシェーディング係数を返す（線形色空間）。
    /// 入力ベクトルはすべて単位ベクトル:
    /// - wi: 入射（点→光）方向
    /// - wo: 出射（点→カメラ）方向
    /// - n:  シェーディング法線
    fn shade(&self, wi: crate::math::Vec3, wo: crate::math::Vec3, n: crate::math::Vec3) -> Color;
}

// 古典的アルゴリズム系のマテリアルは submodule に集約
pub mod classic;
pub use classic::{Lambertian, Phong, BlinnPhong, NormalizedBlinnPhong};

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
