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
        // 拡散項: Lambertian と同様に計算
        let ndotl = n.dot(wi).max(0.0);
        // バックフェースは寄与しない
        if ndotl <= 0.0 { return Color::ZERO; }
        // 反射方向ベクトル r
        let r = -wi + 2.0 * ndotl * n;
        // 反射方向と視線の整合度
        let rv = r.dot(wo).max(0.0);
        // スペキュラ項
        let spec = rv.powf(self.shininess.max(0.0));
        // 拡散項とスペキュラ項を合成
        self.diffuse * ndotl + self.specular * spec
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
    fn shade(&self, wi: Vec3, wo: Vec3, n: Vec3) -> Color {
        // cosθ = max(N·L, 0)
        let ndotl = n.dot(wi).max(0.0); // 入射と法線の余弦
        if ndotl <= 0.0 { return Color::ZERO; } // バックフェースは寄与なし

        // 反射ベクトル R = reflect(-wi, n) = -wi + 2(n·wi)n
        let r = -wi + 2.0 * ndotl * n; // 鏡面反射方向

        // ハイライトの鋭さ用ドット（R·V）を 0 以上にクランプ
        let rv = r.dot(wo).max(0.0); // 反射方向と視線の整合度

        // Lambert の BRDF: f_d = ρ_d/π
        // 本APIでは shadeが直接 Li 係数となるため、ここで ndotl を掛ける
        let diffuse_term = self.diffuse * (core::f32::consts::FRAC_1_PI * ndotl);

        // 正規化 Phong のスペキュラ BRDF 係数: (n+2)/(2π)
        let n_clamped = self.shininess.max(0.0); // 負の指数を防止
        let norm_coeff = 0.5 * (n_clamped + 2.0) * core::f32::consts::FRAC_1_PI; // (n+2)/(2π)

        // スペキュラ BRDF: f_s = ρ_s * norm_coeff * (R·V)^n
        let fs = self.specular * (norm_coeff * rv.powf(n_clamped));

        // レンダリング方程式の cosθ（N·L）はここでまとめて乗算する
        let specular_term = fs * ndotl;

        // 拡散 + 鏡面（どちらも cosθ を内部で掛け済み）
        diffuse_term + specular_term
    }
}
