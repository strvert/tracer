//! 画像出力バックエンドの抽象化と実装。

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use image::{ImageBuffer, Rgb};

/// 画像出力バックエンドの共通トレイト。
/// ピクセルは RGB の連続バイト列（行は上から下へ、左→右）を想定。
pub trait ImageBackend {
    /// 画像を書き出す。
    fn write(&self, path: &Path, width: u32, height: u32, pixels: &[u8]) -> std::io::Result<()>;

    /// ファイル拡張子（例: "ppm", "png"）。ドットなしの小文字を想定。
    fn file_extension(&self) -> &'static str;
}

/// PPM(P6) バックエンド。最小・高速なバイナリ出力。
#[derive(Default, Clone, Copy, Debug)]
pub struct PpmBackend;

impl ImageBackend for PpmBackend {
    fn write(&self, path: &Path, width: u32, height: u32, pixels: &[u8]) -> std::io::Result<()> {
        // 出力ディレクトリを作成
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

    fn file_extension(&self) -> &'static str { "ppm" }
}

/// PNG バックエンド。`image` クレートを使用してエンコード。
#[derive(Default, Clone, Copy, Debug)]
pub struct PngBackend;

impl ImageBackend for PngBackend {
    fn write(&self, path: &Path, width: u32, height: u32, pixels: &[u8]) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let expected = (width as usize) * (height as usize) * 3;
        assert!(pixels.len() == expected, "pixel buffer size mismatch: {} != {}", pixels.len(), expected);

        // image crate の ImageBuffer に詰め替え（RGB8）
        let img: ImageBuffer<Rgb<u8>, _> = ImageBuffer::from_raw(width, height, pixels.to_vec())
            .expect("invalid buffer size for ImageBuffer");

        img.save(path).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    fn file_extension(&self) -> &'static str { "png" }
}
