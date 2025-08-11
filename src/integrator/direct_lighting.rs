use crate::hit::Hittable;
use crate::math::{Color, Ray, Vec3};
use crate::renderer::RenderCtx;
use crate::types::MaterialId;
use super::Integrator;

/// 直接照明 + 疑似環境寄与のみのインテグレータ。
#[derive(Default, Clone, Copy, Debug)]
pub struct DirectLighting;

impl Integrator for DirectLighting {
    fn li(&self, ctx: &RenderCtx, r: &Ray) -> Color {
        if let Some(rec) = ctx.world.hit(r, 1e-3, f32::INFINITY) {
            let wo = (-r.direction).normalized();
            let direct = shade_direct(ctx, rec.p, rec.normal, wo, rec.material_id);
            // 環境寄与: 既存挙動維持（albedo/π · L_env(n)）を、材質の shade を n 向き入射として近似
            let mat = ctx.mats.get(rec.material_id);
            let f_env = mat.shade(rec.normal, wo, rec.normal); // ω_i ≈ n で近似
            let env_term = f_env * ctx.env.radiance(rec.normal);
            return direct + env_term;
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
        if d2 == 0.0 { continue; }
        // 光源までの単位ベクトル
        let wi = to_light / d2.sqrt();
        // 入射方向が法線方向と逆向きならスキップ
        if rec_n.dot(wi) <= 0.0 { continue; }
        // 光源が遮蔽されているならスキップ
        if !visible_to_light(rec_p, light.position, ctx.world) { continue; }
        // Material で BRDF を計算
        let f_sum = mat.shade(wi, wo, rec_n);
        // 光源の放射輝度を計算
        let li = light.color * (light.intensity / d2);
        // BRDF と放射輝度を掛け合わせて加算
        sum += f_sum * li;
    }
    sum
}
