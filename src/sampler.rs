//! ピクセル内スーパーサンプリング用のサンプリング戦略。

/// サンプラーの簡易インターフェイス。[0,1]^2 内のサブピクセルオフセットを返す。
/// 各ピクセルで同じオフセットを使い回し（ジッターなし）。
pub trait Sampler: Send + Sync {
    /// ピクセル内のサブピクセル座標（u,v）。範囲は [0,1]。
    /// 返り値は自身が所有するバッファへの参照でも、静的配列でもよい。
    fn samples(&self) -> &[(f32, f32)];
}

/// ピクセル中心 1 サンプル（アンチエイリアスなし）。
#[derive(Default, Clone, Copy, Debug)]
pub struct NoAa;

impl Sampler for NoAa {
    fn samples(&self) -> &[(f32, f32)] {
        // ピクセル中心
        static NO_AA_SAMPLES: [(f32, f32); 1] = [(0.5, 0.5)];
        &NO_AA_SAMPLES
    }
}

/// 2x2 グリッド（4x MSAA）。1/4 オフセットの固定サンプル。
#[derive(Default, Clone, Copy, Debug)]
pub struct Msaa2x2;

impl Sampler for Msaa2x2 {
    fn samples(&self) -> &[(f32, f32)] {
        static MSAA_2X2_SAMPLES: [(f32, f32); 4] = [
            (0.25, 0.25),
            (0.75, 0.25),
            (0.25, 0.75),
            (0.75, 0.75),
        ];
        &MSAA_2X2_SAMPLES
    }
}

/// 2x2 回転グリッド（RGSS 4x）。格子を45度相当に回転させ、格子縞の相関を下げる。
#[derive(Default, Clone, Copy, Debug)]
pub struct Msaa2x2Rg;

impl Sampler for Msaa2x2Rg {
    fn samples(&self) -> &[(f32, f32)] {
        // ピクセル中心 (0.5,0.5) 周りに回転配置（範囲は [0,1] 内）。
        // 中心相対のオフセット: (-0.125,-0.375), (0.375,-0.125), (-0.375,0.125), (0.125,0.375)
        // を (0.5,0.5) に足した座標。
        static MSAA_2X2_RG_SAMPLES: [(f32, f32); 4] = [
            (0.375, 0.125),
            (0.875, 0.375),
            (0.125, 0.625),
            (0.625, 0.875),
        ];
        &MSAA_2X2_RG_SAMPLES
    }
}

/// 任意個数の回転格子（RGSS風）MSAA サンプラー。
/// m×k の格子点を生成し、行ごとに位相をずらすスタガード（斜交）配置で格子相関を軽減する。
/// n が平方数でなくても近い m,k を自動で選択し、先頭 n サンプルを用いる。
#[derive(Clone, Debug)]
pub struct MsaaRgGeneric {
    samples: Vec<(f32, f32)>,
}

impl MsaaRgGeneric {
    /// サンプル数 n（例: 4, 8, 16 など）を指定して生成。
    pub fn new(n: usize) -> Self {
        let n = n.max(1);
        // できるだけ正方に近い格子に分解
        let mut cols = (n as f32).sqrt().floor() as usize;
        if cols == 0 { cols = 1; }
        let mut rows = (n + cols - 1) / cols;
        while rows * cols < n { cols += 1; rows = (n + cols - 1) / cols; }

        let mut samples = Vec::with_capacity(n);
        for j in 0..rows {
            for i in 0..cols {
                if samples.len() == n { break; }
                // 基本格子（セル中心）
                let u = (i as f32 + 0.5) / cols as f32;
                let v = (j as f32 + 0.5) / rows as f32;
                // 行インデックスに応じて u をずらす（スタガード）。
                // ずらし量は 0.5・(j+0.5)/rows に比例。範囲外は [0,1) に折り返す。
                let shift = 0.5 * (j as f32 + 0.5) / rows as f32;
                let mut ur = u + shift;
                if ur >= 1.0 { ur -= 1.0; }
                samples.push((ur, v));
            }
        }

        Self { samples }
    }
}

impl Sampler for MsaaRgGeneric {
    fn samples(&self) -> &[(f32, f32)] { &self.samples }
}
