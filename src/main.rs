use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

mod math;
use math::{Color, Vec3, Ray, Mat3};
mod hit;
use hit::{Hittable, HittableList, BvhStats, begin_primary_bvh_stats, end_bvh_stats};
mod camera;
use camera::Camera;
mod material;
use material::{Lambertian, MaterialRegistry};
mod light;
pub use light::{PointLight, LightList};
mod types;
use types::MaterialId;
mod environment;
use environment::{Environment, SkyGradient};
mod sampler;
use sampler::{Sampler, MsaaRgGeneric};
mod output;
use output::{ImageBackend, PngBackend};
mod gltf_loader;
use rayon::prelude::*;

const WIDTH: u32 = 1280; // 720p 横幅
const HEIGHT: u32 = 720; // 720p 縦幅


fn visible_to_light(rec_p: Vec3, light_pos: Vec3, world: &dyn Hittable) -> bool {
    // シャドウレイ r(t) = p + t·d, ここで d = normalize(x_L - p), 距離 d_len = ||x_L - p||。
    //       遮蔽判定のため t ∈ [ε, d_len-ε] に限定し、自己交差を避けるため p を ε·d だけ押し出す。
    let to_light = light_pos - rec_p;
    let dist = to_light.length();
    let dir = to_light / dist;
    let shadow_ray = Ray::new(rec_p + 1e-3 * dir, dir);
    world.hit(&shadow_ray, 1e-3, dist - 1e-3).is_none()
}

fn direct_lighting(rec_p: Vec3, rec_n: Vec3, world: &dyn Hittable, mats: &MaterialRegistry, mat_id: MaterialId, lights: &LightList) -> Color {
    // マテリアルの shade() を使って cos項とρ/πを包含した寄与を計算
    let mat = mats.get(mat_id);
    let mut sum = Color::ZERO;
    for light in lights.iter() {
        let to_light = light.position - rec_p;
        let d2 = to_light.length_squared();
        if d2 == 0.0 { continue; }
        let wi = to_light / d2.sqrt();
        // 可視チェック
        if rec_n.dot(wi) <= 0.0 { continue; }
        if !visible_to_light(rec_p, light.position, world) { continue; }
        // マテリアルのシェーディング（入射方向ベクトルの逆を ray.direction として与える）
        let probe_ray = Ray::new(rec_p, -wi);
        let rec = crate::hit::HitRecord { t: 0.0, p: rec_p, normal: rec_n, front_face: true, material_id: mat_id };
        let f = mat.shade(&probe_ray, &rec);
        let li = light.color * (light.intensity / d2);
        sum += f * li;
    }
    sum
}

// 単一レンダラー: 常にカラーを返し、AABB テスト数も常に返す（bvh-stats 無効時は 0）
fn render_scene(width: u32, height: u32, cam: &Camera, world: &(dyn Hittable + Send + Sync), mats: &MaterialRegistry, lights: &LightList, env: &dyn Environment, sampler: &dyn Sampler) -> (Vec<u8>, Vec<u32>) {
    let offsets = sampler.samples();
    let inv_gamma = 1.0 / 2.2_f32;
    let rows: Vec<(Vec<u8>, Vec<u32>)> = (0..height)
        .into_par_iter()
        .rev()
        .map(|y| {
            let mut row_rgb = Vec::with_capacity((width as usize) * 3);
            let mut row_counts: Vec<u32> = Vec::with_capacity(width as usize);
            for x in 0..width {
                let mut color = Color::ZERO;
                let mut aabb_sum: u32 = 0;
                for (du, dv) in offsets.iter().copied() {
                    let u = (x as f32 + du) / (width - 1) as f32;
                    let v = (y as f32 + dv) / (height - 1) as f32;
                    let r = cam.get_ray(u, v);
                    // 主レイの BVH 統計を収集（無効時は no-op）
                    let mut stats = BvhStats::default();
                    begin_primary_bvh_stats(&mut stats);
                    let c = ray_color(&r, world, mats, lights, env);
                    end_bvh_stats();
                    color += c;
                    aabb_sum = aabb_sum.saturating_add(stats.aabb_tests);
                }
                color /= offsets.len() as f32;
                color = Color::new(color.x.powf(inv_gamma), color.y.powf(inv_gamma), color.z.powf(inv_gamma));
                row_rgb.extend_from_slice(&color.to_rgb8());
                row_counts.push(aabb_sum);
            }
            (row_rgb, row_counts)
        })
        .collect();
    let mut rgb = Vec::with_capacity((width as usize) * (height as usize) * 3);
    let mut counts = Vec::with_capacity((width as usize) * (height as usize));
    for (row_rgb, row_counts) in rows.into_iter() { rgb.extend_from_slice(&row_rgb); counts.extend_from_slice(&row_counts); }
    (rgb, counts)
}

#[cfg(feature = "bvh-stats")]
fn counts_to_heatmap_rgb(counts: &[u32], width: u32, height: u32) -> Vec<u8> {
    let mut max_c: u32 = 0;
    for &c in counts { if c > max_c { max_c = c; } }
    let maxf = (max_c as f32).max(1.0);
    let mut out = Vec::with_capacity((width as usize) * (height as usize) * 3);
    for &c in counts {
        // 対数スケールでコントラストを確保
        let v = (c as f32 + 1.0).ln() / (maxf + 1.0).ln();
        let g = (255.0 * v.clamp(0.0, 1.0)) as u8;
        out.extend_from_slice(&[g, g, g]);
    }
    out
}


fn main() -> std::io::Result<()> {
    // マテリアル登録
    let mut mats = MaterialRegistry::new();
    let orange: MaterialId = mats.add(Lambertian { albedo: Color::new(0.9, 0.6, 0.2) });
    let gray: MaterialId = mats.add(Lambertian { albedo: Color::new(0.7, 0.7, 0.7) });
    let green: MaterialId = mats.add(Lambertian { albedo: Color::new(0.2, 0.8, 0.3) });

    // シーン構築: 小球 + 地面の大球 + 単一三角形メッシュ（各オブジェクトにマテリアルIDを割当）
    let mut world = HittableList::new();
    // world.add(Box::new(Sphere { center: Vec3::new(0.0, 0.0, -1.0), radius: 0.5, material_id: orange }));
    // world.add(Box::new(Sphere { center: Vec3::new(0.0, -100.5, -1.0), radius: 100.0, material_id: gray }));

    if let Ok(mesh) = gltf_loader::load_gltf_mesh_with_transform(
        "assets/sphere.glb",
        gray,
        Vec3::new(0.0, 0.0, -3.0),
        Mat3::from_euler_y(0.0) * Mat3::from_scale(1.0, 1.0, 1.0)
    ) {
        world.add(Box::new(mesh));
    }
    if let Ok(mesh) = gltf_loader::load_gltf_mesh_with_transform(
        "assets/fox.glb",
        orange,
        Vec3::new(-0.5, -0.5, -3.0),
        Mat3::from_euler_y(0.5) * Mat3::from_scale(0.01, 0.01, 0.01),
    ) {
        world.add(Box::new(mesh));
    }
    if let Ok(mesh) = gltf_loader::load_gltf_mesh_with_transform(
        "assets/fox.glb",
        green,
        Vec3::new(0.5, -0.5, -3.0),
        Mat3::from_euler_y(0_f32) * Mat3::from_scale(0.01, 0.01, 0.01),
    ) {
        world.add(Box::new(mesh));
    }

    // 点光源
    let mut lights = LightList::new();
    lights.add(PointLight::new(Vec3::new(2.0, 2.0, 0.0), Color::new(1.0, 1.0, 1.0), 20.0));

    // 環境（空のグラデーション）
    let sky = SkyGradient::default();

    // アスペクト比は解像度から算出。カメラは FOV からビューポート構築
    let aspect = WIDTH as f32 / HEIGHT as f32;
    let camera = Camera::new(Vec3::ZERO, 60.0, aspect);

    // Unix timestamp を使用したファイル名生成
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs();
    let backend = PngBackend::default();
    let filename = format!("output/{}.{}", timestamp, backend.file_extension());
    let out_path = Path::new(&filename);
    
    // 使用するサンプラーを選択：汎用回転グリッド RGSS。AA 無しなら `NoAa::default()` を使う。
    let sampler = MsaaRgGeneric::new(8);
    let (pixels, counts) = render_scene(WIDTH, HEIGHT, &camera, &world, &mats, &lights, &sky, &sampler);
    backend.write(out_path, WIDTH, HEIGHT, &pixels)?;
    #[cfg(feature = "bvh-stats")]
    {
        let heat = counts_to_heatmap_rgb(&counts, WIDTH, HEIGHT);
        let heat_name = format!("output/{}_heat_aabb.{}", timestamp, backend.file_extension());
        let heat_path = Path::new(&heat_name);
        backend.write(heat_path, WIDTH, HEIGHT, &heat)?;
        eprintln!("wrote {} and {}", out_path.display(), heat_path.display());
    }
    #[cfg(not(feature = "bvh-stats"))]
    {
        let _ = &counts; // 未使用抑止
        eprintln!("wrote {}", out_path.display());
    }
    Ok(())
}

fn ray_color(r: &Ray, world: &dyn Hittable, mats: &MaterialRegistry, lights: &LightList, env: &dyn Environment) -> Color {
    if let Some(rec) = world.hit(r, 1e-3, f32::INFINITY) {
    let direct = direct_lighting(rec.p, rec.normal, world, mats, rec.material_id, lights);
    // 環境寄与: 既存挙動維持（albedo/π · L_env(n)）を、材質の shade を n 向き入射として近似
    let mat = mats.get(rec.material_id);
    let probe_ray = Ray::new(rec.p, -rec.normal); // ω_i = n で近似
    let rec_stub = crate::hit::HitRecord { t: rec.t, p: rec.p, normal: rec.normal, front_face: rec.front_face, material_id: rec.material_id };
    let f_env = mat.shade(&probe_ray, &rec_stub); // ≈ ρ/π
    let env_term = f_env * env.radiance(rec.normal);
    return direct + env_term;
    }
    env.radiance(r.direction)
}
