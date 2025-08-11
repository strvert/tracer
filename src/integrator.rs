use crate::hit::{Hittable, HitRecord};
use crate::math::{Color, Ray, Vec3};
use crate::renderer::RenderCtx;
use crate::types::MaterialId;

pub trait Integrator: Send + Sync {
    // li: 光線 r に対する放射輝度 L_i を計算
    fn li(&self, ctx: &RenderCtx, ray: &Ray) -> Color;
}

/// 直接照明 + 疑似環境寄与のみのインテグレータ。
#[derive(Default, Clone, Copy, Debug)]
pub struct DirectLighting;

impl Integrator for DirectLighting {
    fn li(&self, ctx: &RenderCtx, r: &Ray) -> Color {
        if let Some(rec) = ctx.world.hit(r, 1e-3, f32::INFINITY) {
            let direct = shade_direct(ctx, rec.p, rec.normal, rec.material_id);
            // 環境寄与: 既存挙動維持（albedo/π · L_env(n)）を、材質の shade を n 向き入射として近似
            let mat = ctx.mats.get(rec.material_id);
            let probe_ray = Ray::new(rec.p, -rec.normal); // ω_i = n で近似
            let rec_stub = HitRecord { t: rec.t, p: rec.p, normal: rec.normal, front_face: rec.front_face, material_id: rec.material_id };
            let f_env = mat.shade(&probe_ray, &rec_stub); // ≈ ρ/π
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
fn shade_direct(ctx: &RenderCtx, rec_p: Vec3, rec_n: Vec3, mat_id: MaterialId) -> Color {
    let mat = ctx.mats.get(mat_id);
    let mut sum = Color::ZERO;
    for light in ctx.lights.iter() {
        let to_light = light.position - rec_p;
        let d2 = to_light.length_squared();
        if d2 == 0.0 { continue; }
        let wi = to_light / d2.sqrt();
        if rec_n.dot(wi) <= 0.0 { continue; }
        if !visible_to_light(rec_p, light.position, ctx.world) { continue; }
        let probe_ray = Ray::new(rec_p, -wi);
        let rec = HitRecord { t: 0.0, p: rec_p, normal: rec_n, front_face: true, material_id: mat_id };
        let f = mat.shade(&probe_ray, &rec);
        let li = light.color * (light.intensity / d2);
        sum += f * li;
    }
    sum
}
