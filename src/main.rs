use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

mod math;
use math::{Color, Vec3, Mat3};
mod hit;
use hit::HittableList;
mod camera;
use camera::Camera;
mod material;
use material::{Lambertian, MaterialRegistry};
mod light;
pub use light::{PointLight, LightList};
mod types;
use types::MaterialId;
mod environment;
use environment::SkyGradient;
mod sampler;
use sampler::MsaaRgGeneric;
mod rng;
mod integrator;
use integrator::DirectLighting;
mod renderer;
use renderer::{RenderSettings, Renderer};
mod output;
use output::{ImageBackend, PngBackend};
mod gltf_loader;

const WIDTH: u32 = 1280; // 720p 横幅
const HEIGHT: u32 = 720; // 720p 縦幅

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

    // シーン構築
    let mut world = HittableList::new();

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

    // カメラ
    let aspect = WIDTH as f32 / HEIGHT as f32;
    let camera = Camera::new(Vec3::ZERO, 60.0, aspect);

    // 出力ファイル名
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs();
    let backend = PngBackend::default();
    let filename = format!("output/{}.{}", timestamp, backend.file_extension());
    let out_path = Path::new(&filename);

    // サンプラーとインテグレータ
    let sampler = MsaaRgGeneric::new(8);
    let integrator = DirectLighting::default();
    let settings = RenderSettings { width: WIDTH, height: HEIGHT, gamma: 2.2, frame: 0 };
    let renderer = Renderer { camera: &camera, world: &world, mats: &mats, lights: &lights, env: &sky, sampler: &sampler, integrator, settings };

    // レンダリング
    let out = renderer.render();
    backend.write(out_path, WIDTH, HEIGHT, &out.pixels)?;

    #[cfg(feature = "bvh-stats")]
    {
        let heat = counts_to_heatmap_rgb(&out.counts, WIDTH, HEIGHT);
        let heat_name = format!("output/{}_heat_aabb.{}", timestamp, backend.file_extension());
        let heat_path = Path::new(&heat_name);
        backend.write(heat_path, WIDTH, HEIGHT, &heat)?;
        eprintln!("wrote {} and {}", out_path.display(), heat_path.display());
    }
    #[cfg(not(feature = "bvh-stats"))]
    {
        eprintln!("wrote {}", out_path.display());
    }
    Ok(())
}

