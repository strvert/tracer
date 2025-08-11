// use std::fs; // 現在は未使用
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

mod math;
use math::{Color, Vec3, Ray};
mod hit;
use hit::{Hittable, Sphere, HittableList, Mesh};
mod camera;
use camera::Camera;
mod material;
use material::{DotShading, MaterialRegistry};
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

fn direct_lighting(rec_p: Vec3, n: Vec3, world: &dyn Hittable, mats: &MaterialRegistry, mat_id: MaterialId, lights: &LightList) -> Color {
    // Lambertian の BRDF は f_r = ρ/π。点光源からの出射は Lo = (ρ/π)·(n·ω_i)_+·L_i。
    // 点光源の放射輝度は L_i = (I / d^2)·color（逆二乗）。
    let albedo = mats.get(mat_id).albedo();
    let mut sum = Color::ZERO;
    let inv_pi = core::f32::consts::FRAC_1_PI;

    for light in lights.iter() {
        // d_len^2 = ||x_L - p||^2, ω_i = (x_L - p)/d_len, (n·l) = n·ω_i。
        let to_light = light.position - rec_p;
        let d2 = to_light.length_squared();
        if d2 == 0.0 { continue; }
        let wi = to_light / d2.sqrt();
        let ndotl = n.dot(wi).max(0.0);
        if ndotl <= 0.0 { continue; }
        if !visible_to_light(rec_p, light.position, world) { continue; }

        let li = light.color * (light.intensity / d2); // L_i = color·I/d^2
        sum += albedo * inv_pi * ndotl * li;           // Lo += (ρ/π)·(n·l)·L_i
    }

    sum
}

fn ray_color(r: &Ray, world: &dyn Hittable, mats: &MaterialRegistry, lights: &LightList, env: &dyn Environment) -> Color {
    if let Some(rec) = world.hit(r, 1e-3, f32::INFINITY) {
        let direct = direct_lighting(rec.p, rec.normal, world, mats, rec.material_id, lights);
        // 本来は ∫_{Ω+} (ρ/π)·L_env(ω)·(n·ω)_+ dω を評価。
        // ここでは近似として Lo_env ≈ (ρ/π)·L_env(n) を足し合わせる。
        let albedo = mats.get(rec.material_id).albedo();
        let env_term = albedo * core::f32::consts::FRAC_1_PI * env.radiance(rec.normal);
        return direct + env_term;
    }

    // レイがシーンに当たらない場合は、環境（背景）をそのまま返す
    // L_background(view_dir) = L_env(view_dir)
    env.radiance(r.direction)
}

fn render_scene_rgb(width: u32, height: u32, cam: &Camera, world: &dyn Hittable, mats: &MaterialRegistry, lights: &LightList, env: &dyn Environment, sampler: &dyn Sampler) -> Vec<u8> {
    let mut buf = Vec::with_capacity((width as usize) * (height as usize) * 3);

    // サブピクセルのオフセットはサンプラー実装が提供する。
    let offsets = sampler.samples();

    // 出力時ガンマ補正（sRGB近似）。計算はリニア空間のまま。
    let inv_gamma = 1.0 / 2.2_f32; // 2.2 の逆数

    for y in (0..height).rev() {
        for x in 0..width {
            let mut color = Color::ZERO;
            for (du, dv) in offsets.iter().copied() {
                let u = (x as f32 + du) / (width - 1) as f32;
                let v = (y as f32 + dv) / (height - 1) as f32;
                let r = cam.get_ray(u, v);
                color += ray_color(&r, world, mats, lights, env);
            }
            color /= offsets.len() as f32; // C = (1/N)·Σ_i C_i（単純平均）

            // ガンマ補正（出力直前）
            // C_out = C_lin^{1/γ}（ここでは γ ≈ 2.2）
            color = Color::new(
                color.x.powf(inv_gamma),
                color.y.powf(inv_gamma),
                color.z.powf(inv_gamma),
            );

            buf.extend_from_slice(&color.to_rgb8());
        }
    }

    buf
}

// PPM(P6) を書き出す。pixels は RGB の連続バイト列を想定
// 出力は ImageBackend 経由に統一（PPM 以外の追加も容易にする）

fn main() -> std::io::Result<()> {
    // マテリアル登録
    let mut mats = MaterialRegistry::new();
    let _orange: MaterialId = mats.add(DotShading { albedo: Color::new(0.9, 0.6, 0.2) });
    let gray: MaterialId = mats.add(DotShading { albedo: Color::new(0.7, 0.7, 0.7) });
    let green: MaterialId = mats.add(DotShading { albedo: Color::new(0.2, 0.8, 0.3) });

    // シーン構築: 小球 + 地面の大球 + 単一三角形メッシュ（各オブジェクトにマテリアルIDを割当）
    let mut world = HittableList::new();
    // world.add(Box::new(Sphere { center: Vec3::new(0.0, 0.0, -1.0), radius: 0.5, material_id: orange }));
    world.add(Box::new(Sphere { center: Vec3::new(0.0, -100.5, -1.0), radius: 100.0, material_id: gray }));

    // 単一三角形メッシュ（カメラ前方に配置）
    let vertices = vec![
        Vec3::new(-0.8, -0.2, -1.2),
        Vec3::new( 0.8, -0.2, -1.2),
        Vec3::new( 0.0,  0.7, -1.2),
    ];
    let indices = vec![[0u32, 1u32, 2u32]];
    let tri_mesh = Mesh::new(vertices, indices, green);
    world.add(Box::new(tri_mesh));

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
    let pixels = render_scene_rgb(WIDTH, HEIGHT, &camera, &world, &mats, &lights, &sky, &sampler);
    backend.write(out_path, WIDTH, HEIGHT, &pixels)?;
    eprintln!("wrote {}", out_path.display());
    Ok(())
}
