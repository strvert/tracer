// 診断（可視化）ユーティリティ。トレーサの集計結果を受け取り、画像化などを行う。

/// 統計データのビュー（必要に応じてフィールドを feature で増減）。
pub struct StatsView<'a> {
    pub width: u32,
    pub height: u32,
    #[cfg(feature = "stats")]
    pub aabb_tests: &'a [u32],
    #[cfg(not(feature = "stats"))]
    _phantom: std::marker::PhantomData<&'a ()>,
}

/// AABB テスト回数をグレースケールのヒートマップに変換（対数スケール）。
#[cfg(feature = "stats")]
pub fn heatmap_from_aabb_tests(aabb: &[u32], width: u32, height: u32) -> Vec<u8> {
    let mut max_c: u32 = 0;
    for &c in aabb { if c > max_c { max_c = c; } }
    let maxf = (max_c as f32).max(1.0);
    let mut out = Vec::with_capacity((width as usize) * (height as usize) * 3);
    for &c in aabb {
        let v = (c as f32 + 1.0).ln() / (maxf + 1.0).ln();
        let g = (255.0 * v.clamp(0.0, 1.0)) as u8;
        out.extend_from_slice(&[g, g, g]);
    }
    out
}
