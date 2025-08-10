use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

mod math;
use math::{Color, Vec3, Ray, lerp, remap};
mod hit;
use hit::{Hittable, Sphere, HittableList};
mod camera;
use camera::Camera;
mod material;
use material::{Material, DotShading, MaterialRegistry};
mod types;
use types::MaterialId;

const WIDTH: u32 = 1280; // 720p 横幅
const HEIGHT: u32 = 720; // 720p 縦幅

// 背景の空グラ: unit(dir).y を [0,1] にマップして白→空色を線形補間
fn background(unit_dir: Vec3) -> Color {
    let t = remap(unit_dir.y, -1.0, 1.0, 0.0, 1.0);
    lerp(Color::splat(1.0), Color::new(0.5, 0.7, 1.0), t)
}

// ヒットがあれば内積シェーディング、なければ空グラ
fn ray_color(r: &Ray, world: &dyn Hittable, mats: &MaterialRegistry) -> Color {
    if let Some(rec) = world.hit(r, 1e-3, f32::INFINITY) {
        let mat = mats.get(rec.material_id);
        return mat.shade(r, &rec);
    }

    let unit_dir = r.direction.normalized();
    background(unit_dir)
}

// カメラとシーンからレンダリング
fn render_scene_rgb(width: u32, height: u32, cam: &Camera, world: &dyn Hittable, mats: &MaterialRegistry) -> Vec<u8> {
    let mut buf = Vec::with_capacity((width as usize) * (height as usize) * 3);

    // 4x MSAA: 2x2 固定サブピクセルオフセット
    let offsets: [(f32, f32); 4] = [
        (0.25, 0.25),
        (0.75, 0.25),
        (0.25, 0.75),
        (0.75, 0.75),
    ];

    // 出力時ガンマ補正（sRGB近似）。計算はリニア空間のまま。
    let inv_gamma = 1.0 / 2.2_f32; // 2.2 の逆数

    for y in (0..height).rev() {
        for x in 0..width {
            let mut color = Color::ZERO;
            for (du, dv) in offsets {
                let u = (x as f32 + du) / (width - 1) as f32;
                let v = (y as f32 + dv) / (height - 1) as f32;
                let r = cam.get_ray(u, v);
                color += ray_color(&r, world, mats);
            }
            color /= offsets.len() as f32; // 平均化（リニア）

            // ガンマ補正（出力直前）
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

// PPM(P6)を書き出す。pixels はRGBの連続バイト列を想定
fn write_ppm_binary(path: &Path, width: u32, height: u32, pixels: &[u8]) -> std::io::Result<()> {
    // 出力ディレクトリ作成（必要なら）
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    // 入力検証
    let expected = (width as usize) * (height as usize) * 3;
    assert!(pixels.len() == expected, "pixel buffer size mismatch: {} != {}", pixels.len(), expected);

    // 上書きで作成（初回は新規作成）
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // PPM (P6) ヘッダ: マジック、サイズ、最大値
    writer.write_all(format!("P6\n{} {}\n255\n", width, height).as_bytes())?;

    // ピクセル本体
    writer.write_all(pixels)?;

    writer.flush()?;
    Ok(())
}

fn main() -> std::io::Result<()> {
    // マテリアル登録
    let mut mats = MaterialRegistry::new();
    let orange: MaterialId = mats.add(DotShading { albedo: Color::new(0.9, 0.6, 0.2) });
    let gray: MaterialId = mats.add(DotShading { albedo: Color::new(0.7, 0.7, 0.7) });

    // シーン構築: 小球 + 地面の大球（各オブジェクトにマテリアルIDを割当）
    let mut world = HittableList::new();
    world.add(Box::new(Sphere { center: Vec3::new(0.0, 0.0, -1.0), radius: 0.5, material_id: orange }));
    world.add(Box::new(Sphere { center: Vec3::new(0.0, -100.5, -1.0), radius: 100.0, material_id: gray }));

    // アスペクト比は解像度から算出。カメラはFOVからビューポート構築
    let aspect = WIDTH as f32 / HEIGHT as f32;
    let camera = Camera::new(Vec3::ZERO, 60.0, aspect);

    let out_path = Path::new("output/step10.ppm");
    let pixels = render_scene_rgb(WIDTH, HEIGHT, &camera, &world, &mats);
    write_ppm_binary(out_path, WIDTH, HEIGHT, &pixels)?;
    eprintln!("wrote {}", out_path.display());
    Ok(())
}
