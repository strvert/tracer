use crate::math::{Color, Vec3};
use super::Material;

#[derive(Clone, Debug)]
pub struct Lambertian {
    pub albedo: Color,
}

impl Material for Lambertian {
    fn shade(&self, wi: Vec3, _wo: Vec3, n: Vec3) -> Color {
        // Lambert: f = ρ/π。ここでは shade = (ρ/π)·max(n·wi, 0)
        let ndotl = n.dot(wi).max(0.0);
        self.albedo * ndotl * core::f32::consts::FRAC_1_PI
    }
}

/// 非正規化の古典 Phong マテリアル。
/// 係数: diffuse·max(n·l,0) + specular·max(r·v,0)^exponent
/// 非物理（正規化なし）の見た目重視モデル。エネルギー保存は保証しません。
#[derive(Clone, Debug)]
pub struct Phong {
    /// 拡散係数（色）。Lambert 項の重み。
    pub diffuse: Color,
    /// 鏡面係数（色）。ハイライトの色・強さ。
    pub specular: Color,
    /// Phong 指数（通称 shininess）。大きいほどハイライトが鋭い。
    pub shininess: f32,
}

impl Material for Phong {
    fn shade(&self, wi: Vec3, wo: Vec3, n: Vec3) -> Color {
        let ndotl = n.dot(wi).max(0.0);
        if ndotl <= 0.0 { return Color::ZERO; }
        // 反射ベクトル r = reflect(-wi, n) = -wi + 2(n·wi)n
        let r = -wi + 2.0 * ndotl * n;
        let rv = r.dot(wo).max(0.0);
        let spec = rv.powf(self.shininess.max(0.0));
        self.diffuse * ndotl + self.specular * spec
    }
}
