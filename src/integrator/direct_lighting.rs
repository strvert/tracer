use super::Integrator;
use crate::hit::Hittable;
use crate::math::{Color, Ray, Vec3};
use crate::renderer::RenderCtx;
use crate::types::MaterialId;

/// 直接照明 + 疑似環境寄与のみのインテグレータ。
#[derive(Default, Clone, Copy, Debug)]
pub struct DirectLighting;

impl Integrator for DirectLighting {
    fn li(&self, ctx: &RenderCtx, r: &Ray) -> Color {
        if let Some(rec) = ctx.world.hit(r, 1e-3, f32::INFINITY) {
            let wo = (-r.direction).normalized();
            let direct = shade_direct(ctx, rec.p, rec.normal, wo, rec.material_id);
            let ambient = shade_ambient(ctx, rec.normal, wo, rec.material_id);
            return direct + ambient;
        }
        ctx.env.radiance(r.direction)
    }
}

#[inline]
fn visible_to_light(rec_p: Vec3, light_pos: Vec3, world: &dyn Hittable) -> bool {
    let to_light = light_pos - rec_p;
    let dist = to_light.length();
    let dir = to_light / dist;
    let shadow_ray = Ray::new(rec_p + 1e-3 * dir, dir);
    world.hit(&shadow_ray, 1e-3, dist - 1e-3).is_none()
}

#[inline]
fn shade_direct(ctx: &RenderCtx, rec_p: Vec3, rec_n: Vec3, wo: Vec3, mat_id: MaterialId) -> Color {
    let mat = ctx.mats.get(mat_id);
    let mut sum = Color::ZERO;
    // ライト影響を累積的に計算
    for light in ctx.lights.iter() {
        // 光源からの方向を計算
        let to_light = light.position - rec_p;
        // 光源までの距離の二乗
        let d2 = to_light.length_squared();
        // 光源までの距離がゼロならスキップ
        if d2 == 0.0 {
            continue;
        }
        // 光源までの単位ベクトル
        let wi = to_light / d2.sqrt();
        // 入射方向が法線方向と逆向きならスキップ
        let cos_theta = rec_n.dot(wi);
        if cos_theta <= 0.0 {
            continue;
        }
        // 光源が遮蔽されているならスキップ
        if !visible_to_light(rec_p, light.position, ctx.world) {
            continue;
        }
        // BRDF を評価（cosθ は別で掛ける）
        let f = mat.eval(wi, wo, rec_n);
        // 光源の放射輝度を計算
        let li = light.color * (light.intensity / d2);
        // f · Li · cosθ
        sum += f * li * cos_theta.max(0.0);
    }
    sum
}

/// 環境寄与（簡易）
/// 単一点サンプルの近似: wi = n の方向で環境を 1 サンプルし、f(wi,wo) · L_i(wi) · max(N·wi,0)
/// 注意: 積分の一貫性のため BRDF は cosθ を含めない（Integrate 側で掛ける）。
#[inline]
fn shade_ambient(ctx: &RenderCtx, n: Vec3, wo: Vec3, mat_id: MaterialId) -> Color {
    let mat = ctx.mats.get(mat_id);
    let wi = n; // 代表方向としてシェーディング法線方向
    let cos_theta = n.dot(wi).max(0.0);
    if cos_theta <= 0.0 {
        return Color::ZERO;
    }
    let f = mat.eval(wi, wo, n);
    let li = ctx.env.radiance(wi);
    f * li * cos_theta
}
