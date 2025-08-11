use crate::math::{Color, Vec3};
use super::Material;

#[derive(Clone, Debug)]
pub struct Lambertian {
    pub albedo: Color,
}

impl Material for Lambertian {
    fn eval(&self, wi: Vec3, _wo: Vec3, n: Vec3) -> Color {
        // Lambert BRDF: f_d = ρ/π（cosθ はインテグレータ側で掛ける）。
        // 反対半球からの入射は 0。
        if n.dot(wi) <= 0.0 { return Color::BLACK; }
        self.albedo * core::f32::consts::FRAC_1_PI
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
    fn eval(&self, wi: Vec3, wo: Vec3, n: Vec3) -> Color {
        // 反対半球からの入射は 0。
        let ndotl = n.dot(wi);
        if ndotl <= 0.0 { return Color::BLACK; }
        // 反射方向ベクトル r
        let r = -wi + 2.0 * ndotl * n;
        // 反射方向と視線の整合度（R·V）
        let rv = r.dot(wo).max(0.0);
        // 非正規化の見た目モデル: f ≈ ρ_d/π + ρ_s·(R·V)^n
        let diffuse_f = self.diffuse * core::f32::consts::FRAC_1_PI;
        let specular_f = self.specular * rv.powf(self.shininess.max(0.0));
        diffuse_f + specular_f
    }
}

/// 非正規化 Blinn–Phong（N·H モデル）。
/// 係数: diffuse·max(N·L,0) + specular·max(N·H,0)^shininess
#[derive(Clone, Debug)]
pub struct BlinnPhong {
    /// 拡散係数（色）
    pub diffuse: Color,
    /// 鏡面係数（色）
    pub specular: Color,
    /// 指数（大きいほどハイライトが鋭い）
    pub shininess: f32,
}

impl Material for BlinnPhong {
    fn eval(&self, wi: Vec3, wo: Vec3, n: Vec3) -> Color {
        if n.dot(wi) <= 0.0 { return Color::BLACK; }
        // ハーフベクトル H = normalize(wi + wo)
        let h = (wi + wo).normalized();
        let ndoth = n.dot(h).max(0.0);
        // BRDF 近似: f ≈ ρ_d/π + ρ_s·(N·H)^m（非正規化）
        let diffuse_f = self.diffuse * core::f32::consts::FRAC_1_PI;
        let specular_f = self.specular * ndoth.powf(self.shininess.max(0.0));
        diffuse_f + specular_f
    }
}

/// 正規化 Blinn–Phong（エネルギー整合のための係数）。
/// BRDF 近似: f_s ≈ ρ_s · (m+8)/(8π) · max(N·H,0)^m
#[derive(Clone, Debug)]
pub struct NormalizedBlinnPhong {
    /// 拡散反射率（色） ρ_d
    pub diffuse: Color,
    /// 鏡面反射率（色） ρ_s（0〜1 程度を推奨）
    pub specular: Color,
    /// 指数 m（大きいほどローブが鋭い）
    pub shininess: f32,
}

impl Material for NormalizedBlinnPhong {
    fn eval(&self, wi: Vec3, wo: Vec3, n: Vec3) -> Color {
        if n.dot(wi) <= 0.0 { return Color::BLACK; }
        // ハーフベクトル H = normalize(wi + wo)
        let h = (wi + wo).normalized();
        let ndoth = n.dot(h).max(0.0);

        // f_d = ρ_d / π
        let diffuse_f = self.diffuse * core::f32::consts::FRAC_1_PI;

        // 正規化係数: (m+8)/(8π)
        let m = self.shininess.max(0.0);
        let norm_coeff = (m + 8.0) * (core::f32::consts::FRAC_1_PI / 8.0);
        // f_s ≈ ρ_s · norm_coeff · (N·H)^m
        let specular_f = self.specular * (norm_coeff * ndoth.powf(m));

        diffuse_f + specular_f
    }
}

/// 正規化 Phong（エネルギー一貫性のある係数）
/// BRDF: f_s = ρ_s · (n+2)/(2π) · max(R·V,0)^n
/// このレンダラの API では shade が L_o += shade · L_i に使われるため、
/// 余弦項 N·L は内部で掛け合わせる（diffuse/鏡面ともに ndotl を乗算）。
#[derive(Clone, Debug)]
pub struct NormalizedPhong {
    /// 拡散反射率（色）。Lambert の ρ_d に相当。
    pub diffuse: Color,
    /// 鏡面反射率（色）。スペキュラの ρ_s に相当（0〜1 程度を推奨）。
    pub specular: Color,
    /// Phong 指数 n（大きいほどローブが鋭い）。
    pub shininess: f32,
}

impl Material for NormalizedPhong {
    fn eval(&self, wi: Vec3, wo: Vec3, n: Vec3) -> Color {
        // 反対半球は寄与しない
        let ndotl = n.dot(wi);
        if ndotl <= 0.0 { return Color::BLACK; }
        // 反射ベクトル
        let r = -wi + 2.0 * ndotl * n;
        let rv = r.dot(wo).max(0.0);
        // f_d = ρ_d/π
        let diffuse_f = self.diffuse * core::f32::consts::FRAC_1_PI;
        // f_s = ρ_s * (n+2)/(2π) * (R·V)^n
        let n_clamped = self.shininess.max(0.0);
        let norm_coeff = 0.5 * (n_clamped + 2.0) * core::f32::consts::FRAC_1_PI;
        let specular_f = self.specular * (norm_coeff * rv.powf(n_clamped));
        diffuse_f + specular_f
    }
}
