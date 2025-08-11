//! 環境光（Environment）インターフェースと、空のグラデーション実装。
//! これにより「背景」と「アンビエント風の項」を統一的に扱います。

use crate::math::{Color, Vec3};
use crate::math::{lerp, remap};

/// Environment は、与えられた方向に対する放射輝度（環境放射）を返します。
/// - radiance(dir): L_env(ω) をリニア色で返す。dir は正規化想定（内部で防御的に正規化）。
/// - sample/pdf: 将来のモンテカルロ積分用のフック。デフォルトは「球面一様」。
pub trait Environment: Send + Sync {
    /// 方向 dir からの環境放射 L_env(dir) を返す（リニア空間）。
    fn radiance(&self, dir: Vec3) -> Color;

    /// 球面一様サンプリングで方向を生成し、pdf = 1/(4π) を返します。
    /// u,v は [0,1) の乱数。
    fn sample(&self, u: f32, v: f32) -> (Vec3, f32) {
        let z = 1.0 - 2.0 * u;
        let phi = 2.0 * core::f32::consts::PI * v;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let dir = Vec3::new(r * phi.cos(), r * phi.sin(), z);
        (dir, core::f32::consts::FRAC_1_PI * 0.25) // 1/(4π)
    }

    /// 球面一様サンプリング時の確率密度関数（pdf）。
    fn pdf(&self, _dir: Vec3) -> f32 { core::f32::consts::FRAC_1_PI * 0.25 }
}

/// 空のグラデーション: 下（y=-1）で白、上（y=+1）で空色に補間。
#[derive(Clone, Copy, Debug)]
pub struct SkyGradient {
    pub bottom: Color, // 地平線・地面付近の色
    pub top: Color,    // 天頂の色
}

impl Default for SkyGradient {
    fn default() -> Self {
        Self { bottom: Color::splat(1.0), top: Color::new(0.5, 0.7, 1.0) }
    }
}

impl SkyGradient {
    /// 下端色と上端色を指定して作成。
    pub fn new(bottom: Color, top: Color) -> Self { Self { bottom, top } }
}

impl Environment for SkyGradient {
    fn radiance(&self, dir: Vec3) -> Color {
        let unit = dir.normalized();
        let t = remap(unit.y, -1.0, 1.0, 0.0, 1.0);
        lerp(self.bottom, self.top, t)
    }
}
