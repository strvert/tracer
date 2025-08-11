//! Hit utilities: sphere intersection and hit record.

use crate::math::{Point3, Ray, Vec3, Mat3, Mat4};
use crate::types::MaterialId;

// 数値安定用の小さな閾値（Möller–Trumbore の det 判定などに使用）
const MT_EPSILON: f32 = 1e-8_f32;

// --- BVH 計測（M1: AABB/ノード/葉/三角カウント; 主レイのみ有効化） -----------------
#[derive(Clone, Copy, Debug, Default)]
pub struct BvhStats {
    pub aabb_tests: u32,
    pub node_visits: u32,
    pub leaf_visits: u32,
    pub tri_tests: u32,
}

// 実計測は feature("stats") でのみ有効化。未指定時はゼロコスト no-op。
#[cfg(feature = "stats")]
mod bvh_stats_impl {
    use super::BvhStats;

    thread_local! {
        static BVH_STATS_ENABLED: core::cell::Cell<bool> = core::cell::Cell::new(false);
        static BVH_STATS_PTR: core::cell::Cell<usize> = core::cell::Cell::new(0);
    }

    pub fn begin_primary_bvh_stats(stats: &mut BvhStats) {
        let ptr = stats as *mut BvhStats as usize;
        BVH_STATS_PTR.with(|c| c.set(ptr));
        BVH_STATS_ENABLED.with(|f| f.set(true));
    }

    pub fn end_bvh_stats() {
        BVH_STATS_ENABLED.with(|f| f.set(false));
        BVH_STATS_PTR.with(|c| c.set(0));
    }

    #[inline(always)]
    fn stats_with<F: FnOnce(&mut BvhStats)>(f: F) {
        BVH_STATS_ENABLED.with(|en| {
            if en.get() {
                BVH_STATS_PTR.with(|p| {
                    let addr = p.get();
                    if addr != 0 {
                        // 安全性: ptr は同スレッド内の短命バッファ（呼び出し側で生存管理）
                        let s = unsafe { &mut *(addr as *mut BvhStats) };
                        f(s);
                    }
                });
            }
        });
    }

    #[inline(always)] pub fn stats_inc_aabb() { stats_with(|s| s.aabb_tests = s.aabb_tests.wrapping_add(1)); }
    #[inline(always)] pub fn stats_inc_node() { stats_with(|s| s.node_visits = s.node_visits.wrapping_add(1)); }
    #[inline(always)] pub fn stats_inc_leaf() { stats_with(|s| s.leaf_visits = s.leaf_visits.wrapping_add(1)); }
    #[inline(always)] pub fn stats_inc_tri()  { stats_with(|s| s.tri_tests  = s.tri_tests .wrapping_add(1)); }
}

#[cfg(not(feature = "stats"))]
mod bvh_stats_impl {
    use super::BvhStats;
    pub fn begin_primary_bvh_stats(_stats: &mut BvhStats) {}
    pub fn end_bvh_stats() {}
    #[inline(always)] pub fn stats_inc_aabb() {}
    #[inline(always)] pub fn stats_inc_node() {}
    #[inline(always)] pub fn stats_inc_leaf() {}
    #[inline(always)] pub fn stats_inc_tri() {}
}

#[allow(unused_imports)]
pub use bvh_stats_impl::{begin_primary_bvh_stats, end_bvh_stats, stats_inc_aabb, stats_inc_leaf, stats_inc_node, stats_inc_tri};

#[derive(Clone, Copy, Debug)]
pub struct HitRecord {
    pub t: f32,
    pub p: Point3,
    pub normal: Vec3,
    pub front_face: bool,
    pub material_id: MaterialId,
}

impl HitRecord {
    pub fn new(p: Point3, t: f32, outward_normal: Vec3, ray_dir: Vec3, material_id: MaterialId) -> Self {
        let front_face = ray_dir.dot(outward_normal) < 0.0;
        let normal = if front_face { outward_normal } else { -outward_normal };
        Self { t, p, normal, front_face, material_id }
    }
}

// Hittable トレイトと Sphere 実装
pub trait Hittable: Send + Sync {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord>;
    /// ワールド空間での軸平行境界箱（AABB）を返す。
    /// TLAS（シーンBVH）の構築に用いる。可逆な線形変換（回転・スケール）と平行移動のみを想定。
    fn bounds(&self) -> Aabb;
}

#[derive(Clone, Copy, Debug)]
pub struct Sphere {
    pub center: Point3,
    pub radius: f32,
    pub material_id: MaterialId,
}

impl Hittable for Sphere {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let oc = r.origin - self.center;
        let a = r.direction.dot(r.direction);
        let half_b = oc.dot(r.direction);
        let c = oc.dot(oc) - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;
        if discriminant < 0.0 {
            return None;
        }
        let sqrtd = discriminant.sqrt();

        // 範囲内の最小根を選択
        let mut root = (-half_b - sqrtd) / a;
        if root < t_min || t_max < root {
            root = (-half_b + sqrtd) / a;
            if root < t_min || t_max < root {
                return None;
            }
        }

        let t = root;
        let p = r.at(t);
        let outward_normal = (p - self.center) / self.radius;
        Some(HitRecord::new(p, t, outward_normal, r.direction, self.material_id))
    }

    fn bounds(&self) -> Aabb {
        let r = Vec3::splat(self.radius);
    Aabb::new(self.center - r, self.center + r)
    }
}

// Hittable のコレクション（シーン）
#[derive(Default)]
pub struct HittableList {
    pub objects: Vec<Box<dyn Hittable + Send + Sync>>,
    // TLAS（シーンBVH）用の構造: オブジェクトごとの AABB をもとに中央値分割 BVH を構築
    obj_index: Vec<u32>,
    tlas_nodes: Vec<BvhNode>,
    tlas_root: usize,
}

impl HittableList {
    pub fn new() -> Self { Self { objects: Vec::new(), obj_index: Vec::new(), tlas_nodes: Vec::new(), tlas_root: 0 } }
    pub fn clear(&mut self) {
        self.objects.clear();
        self.obj_index.clear();
        self.tlas_nodes.clear();
        self.tlas_root = 0;
    }
    pub fn add(&mut self, object: Box<dyn Hittable + Send + Sync>) {
        self.objects.push(object);
        self.build_tlas();
    }

    /// TLAS 構築（中央値分割）。objects の順序は保持し、インデックス配列だけを並べ替える。
    fn build_tlas(&mut self) {
        let n = self.objects.len();
        self.obj_index.clear();
        self.tlas_nodes.clear();
        if n == 0 { self.tlas_root = 0; return; }
        self.obj_index.reserve(n);
        for i in 0..n { self.obj_index.push(i as u32); }

        // 各オブジェクトの AABB を一時的に取得
        let mut obj_bounds: Vec<Aabb> = Vec::with_capacity(n);
        for i in 0..n { obj_bounds.push(self.objects[i].bounds()); }

        let root = build_bvh_nodes(&mut self.tlas_nodes, &mut self.obj_index[..], &obj_bounds[..], 4);
        self.tlas_root = root;
    }
}

impl Hittable for HittableList {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let inv_dir = Vec3::new(1.0 / r.direction[0], 1.0 / r.direction[1], 1.0 / r.direction[2]);
        let o = r.origin;
        traverse_bvh(
            &self.tlas_nodes,
            self.tlas_root,
            t_max,
            |idx, closest| self.tlas_nodes[idx].bounds.hit(o, inv_dir, t_min, closest),
            |start, count, closest| {
                let end = start + count;
                let mut best: Option<HitRecord> = None;
                let mut cs = closest;
                for &oid in &self.obj_index[start..end] {
                    let obj = &self.objects[oid as usize];
                    if let Some(rec) = obj.hit(r, t_min, cs) {
                        cs = rec.t; best = Some(rec);
                    }
                }
                best
            },
        )
    }

    fn bounds(&self) -> Aabb {
        // ルートの AABB（無ければ空に近い AABB を返す）
        if self.tlas_nodes.is_empty() {
            // 線形に統合
            let mut acc = Aabb::new(Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
            for obj in &self.objects { acc = acc.union(obj.bounds()); }
            return acc;
        }
        self.tlas_nodes[self.tlas_root].bounds
    }
}

// 三角形（Möller–Trumbore）。属性は最小構成：頂点とマテリアルのみ。
//
// 数学的設定（レイ r(t)=o+t d, 三角形 v0,v1,v2）
// - 辺: e1 = v1 - v0, e2 = v2 - v0
// - 交点のバリセントリック: p = v0 + u e1 + v e2, かつ u>=0, v>=0, u+v<=1
// - 便宜: T = o - v0, P = d × e2, Q = T × e1, det = e1 · P
//
// Cramer/三重積で導出される解:
//   u = (T · P) / det,
//   v = (d · Q) / det,
//   t = (e2 · Q) / det.
// 判定:
//   |det| < ε → 平行/退化で不採用
//   u∉[0,1] または v<0 または u+v>1 → 三角形外
//   t∉[t_min,t_max] → パラメータ範囲外
// 法線:
//   幾何法線 n_g = normalize(e1 × e2)。front_face 補正で外向きに揃える。
#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    pub v0: Point3,
    pub v1: Point3,
    pub v2: Point3,
    pub material_id: MaterialId,
}

// --- 共通ヘルパー: 三角形交差（Möller–Trumbore）と幾何法線 -------------------
/// Möller–Trumbore 法によるレイと三角形の交差判定。
/// 成功時に (t, u, v) を返す。バリセントリックは p = v0 + u*(v1-v0) + v*(v2-v0)。
#[inline(always)]
fn intersect_triangle_moller_trumbore(r: &Ray, v0: Vec3, v1: Vec3, v2: Vec3, t_min: f32, t_max: f32) -> Option<(f32, f32, f32)> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let pvec = r.direction.cross(e2);
    let det = e1.dot(pvec);
    if det.abs() < MT_EPSILON { return None; }
    let inv_det = 1.0 / det;
    let tvec = r.origin - v0;
    let u = tvec.dot(pvec) * inv_det;
    if u < 0.0 || u > 1.0 { return None; }
    let qvec = tvec.cross(e1);
    let v = r.direction.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 { return None; }
    let t = e2.dot(qvec) * inv_det;
    if t < t_min || t > t_max { return None; }
    Some((t, u, v))
}

/// 三角形の幾何法線（非補間）。
#[inline(always)]
fn triangle_geometric_normal(v0: Vec3, v1: Vec3, v2: Vec3) -> Vec3 {
    (v1 - v0).cross(v2 - v0).normalized()
}

/// 前計算済み e1/e2 を使う高速 Möller–Trumbore。
#[inline(always)]
fn intersect_triangle_precomputed(r: &Ray, v0: Vec3, e1: Vec3, e2: Vec3, t_min: f32, t_max: f32) -> Option<(f32, f32, f32)> {
    let pvec = r.direction.cross(e2);
    let det = e1.dot(pvec);
    if det.abs() < MT_EPSILON { return None; }
    let inv_det = 1.0 / det;
    let tvec = r.origin - v0;
    let u = tvec.dot(pvec) * inv_det;
    if u < 0.0 || u > 1.0 { return None; }
    let qvec = tvec.cross(e1);
    let v = r.direction.dot(qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 { return None; }
    let t = e2.dot(qvec) * inv_det;
    if t < t_min || t > t_max { return None; }
    Some((t, u, v))
}

impl Hittable for Triangle {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        if let Some((t, _u, _v)) = intersect_triangle_moller_trumbore(r, self.v0, self.v1, self.v2, t_min, t_max) {
            let p = r.at(t);
            let outward_normal = triangle_geometric_normal(self.v0, self.v1, self.v2);
            Some(HitRecord::new(p, t, outward_normal, r.direction, self.material_id))
        } else {
            None
        }
    }

    fn bounds(&self) -> Aabb {
        Aabb::from_triangle(self.v0, self.v1, self.v2)
    }
}

// --- BLAS（メッシュ内 BVH）: AABB と BVH ノード -------------------------------
//
// アルゴリズム概要
// - 各三角形にローカル空間での AABB を付与（min/max）。
// - 三角形重心の最大エクステント軸で中央値分割（簡易版）。
// - 葉ノードには tri_index の連続範囲 [start, start+count) を保持。
// - トラバースはスラブ法で AABB を切り詰め、葉でのみ三角交差。
// - メッシュの平行移動/線形変換はレイをローカルへ逆変換してから BVH を使うため、
//   BVH は「メッシュローカル空間」固定のままで再構築不要。

#[derive(Clone, Copy, Debug, Default)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }

    fn from_triangle(v0: Vec3, v1: Vec3, v2: Vec3) -> Self {
        let min = Vec3::new(
            v0[0].min(v1[0]).min(v2[0]),
            v0[1].min(v1[1]).min(v2[1]),
            v0[2].min(v1[2]).min(v2[2]),
        );
        let max = Vec3::new(
            v0[0].max(v1[0]).max(v2[0]),
            v0[1].max(v1[1]).max(v2[1]),
            v0[2].max(v1[2]).max(v2[2]),
        );
        Self { min, max }
    }

    fn union(self, other: Self) -> Self {
        Self::new(
            Vec3::new(
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ),
            Vec3::new(
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ),
        )
    }

    fn centroid(self) -> Vec3 { (self.min + self.max) * 0.5 }

    /// スラブ法（ブランチレス寄り）: 事前計算した inv_dir を受け取り、[t_near,t_far] を返す。
    #[inline(always)]
    pub fn hit(&self, o: Vec3, inv_d: Vec3, mut t_min: f32, mut t_max: f32) -> Option<(f32, f32)> {
        // X
    let tx1 = (self.min[0] - o[0]) * inv_d[0];
    let tx2 = (self.max[0] - o[0]) * inv_d[0];
        t_min = t_min.max(tx1.min(tx2));
        t_max = t_max.min(tx1.max(tx2));
        if t_min > t_max { return None; }
        // Y
    let ty1 = (self.min[1] - o[1]) * inv_d[1];
    let ty2 = (self.max[1] - o[1]) * inv_d[1];
        t_min = t_min.max(ty1.min(ty2));
        t_max = t_max.min(ty1.max(ty2));
        if t_min > t_max { return None; }
        // Z
    let tz1 = (self.min[2] - o[2]) * inv_d[2];
    let tz2 = (self.max[2] - o[2]) * inv_d[2];
        t_min = t_min.max(tz1.min(tz2));
        t_max = t_max.min(tz1.max(tz2));
        if t_min > t_max { return None; }
        Some((t_min, t_max))
    }

    /// ローカル AABB を 4x4 アフィン変換でワールドへ送り、ワールド AABB を返す。
    /// 任意の回転・非一様スケールでも、8 つの頂点を変換して再 AABB 化すれば安全。
    pub fn transform_affine(self, m: Mat4) -> Self {
        let corners = [
            Vec3::new(self.min[0], self.min[1], self.min[2]),
            Vec3::new(self.max[0], self.min[1], self.min[2]),
            Vec3::new(self.min[0], self.max[1], self.min[2]),
            Vec3::new(self.min[0], self.min[1], self.max[2]),
            Vec3::new(self.max[0], self.max[1], self.min[2]),
            Vec3::new(self.max[0], self.min[1], self.max[2]),
            Vec3::new(self.min[0], self.max[1], self.max[2]),
            Vec3::new(self.max[0], self.max[1], self.max[2]),
        ];
        let mut wmin = Vec3::splat(f32::INFINITY);
        let mut wmax = Vec3::splat(f32::NEG_INFINITY);
        for c in corners.iter() {
            let w = m.mul_point(*c);
            wmin = Vec3::new(wmin[0].min(w[0]), wmin[1].min(w[1]), wmin[2].min(w[2]));
            wmax = Vec3::new(wmax[0].max(w[0]), wmax[1].max(w[1]), wmax[2].max(w[2]));
        }
        Aabb::new(wmin, wmax)
    }
}

#[derive(Clone, Debug)]
struct BvhNode {
    bounds: Aabb,
    left: i32,   // <0 なら葉
    right: i32,  // <0 なら葉
    start: u32,  // 葉: インデックス範囲の開始（内部でも範囲メタとして保持）
    count: u32,  // 葉: 要素数（内部でも範囲メタとして保持）
}

// 共通 BVH ビルダー（中央値分割）
// - bounds: 各要素の AABB 配列
// - index: 並べ替え可能なインデックス配列（この順序に従って葉に格納）
// - leaf_threshold: 葉の最大要素数
// 戻り値: ルートノードのインデックス
fn build_bvh_nodes(nodes: &mut Vec<BvhNode>, index: &mut [u32], bounds: &[Aabb], leaf_threshold: usize) -> usize {
    fn range_bounds(idx: &[u32], bounds: &[Aabb], start: usize, end: usize) -> Aabb {
        let mut b = Aabb::new(Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for i in start..end { b = b.union(bounds[idx[i] as usize]); }
        b
    }
    fn centroid_bounds(idx: &[u32], bounds: &[Aabb], start: usize, end: usize) -> (Vec3, Vec3) {
        let mut cmin = Vec3::splat(f32::INFINITY);
        let mut cmax = Vec3::splat(f32::NEG_INFINITY);
        for i in start..end {
            let c = bounds[idx[i] as usize].centroid();
            cmin = Vec3::new(cmin[0].min(c[0]), cmin[1].min(c[1]), cmin[2].min(c[2]));
            cmax = Vec3::new(cmax[0].max(c[0]), cmax[1].max(c[1]), cmax[2].max(c[2]));
        }
        (cmin, cmax)
    }
    fn build_range(nodes: &mut Vec<BvhNode>, idx: &mut [u32], bounds: &[Aabb], start: usize, end: usize, leaf_threshold: usize) -> usize {
        let b = range_bounds(idx, bounds, start, end);
        let count = end - start;
        if count <= leaf_threshold {
            let i = nodes.len();
            nodes.push(BvhNode { bounds: b, left: -1, right: -1, start: start as u32, count: count as u32 });
            return i;
        }
        let (cmin, cmax) = centroid_bounds(idx, bounds, start, end);
    let ext = cmax - cmin;
    let mut axis = 0; if ext[1] > ext[0] { axis = 1; } if ext[2] > ext[axis] { axis = 2; }
        let mid = (start + end) / 2;
        idx[start..end].sort_by(|&a, &b| {
            let ca = bounds[a as usize].centroid();
            let cb = bounds[b as usize].centroid();
            let ka = ca[axis];
            let kb = cb[axis];
            ka.partial_cmp(&kb).unwrap_or(core::cmp::Ordering::Equal)
        });
        let l = build_range(nodes, idx, bounds, start, mid, leaf_threshold);
        let r = build_range(nodes, idx, bounds, mid, end, leaf_threshold);
        let i = nodes.len();
        nodes.push(BvhNode { bounds: b, left: l as i32, right: r as i32, start: start as u32, count: count as u32 });
        i
    }

    nodes.clear();
    build_range(nodes, index, bounds, 0, index.len(), leaf_threshold)
}

// 共通 BVH トラバース（近い順探索 + 枝刈り）
// - aabb_hit: ノードインデックスと現closestを受け取り、(t_near, t_far) を返す（なければ None）。
// - visit_leaf: 葉ノードの [start,count) を処理し、より良いヒットがあれば返す（closest は参照用）。
fn traverse_bvh<AH, VL>(
    nodes: &[BvhNode],
    root: usize,
    t_max: f32,
    mut aabb_hit: AH,
    mut visit_leaf: VL,
) -> Option<HitRecord>
where
    AH: FnMut(usize, f32) -> Option<(f32, f32)>,
    VL: FnMut(usize, usize, f32) -> Option<HitRecord>,
{
    if nodes.is_empty() { return None; }
    let mut closest_so_far = t_max;
    let mut best: Option<HitRecord> = None;
    let mut stack: [usize; 128] = [0; 128];
    let mut sp: usize = 0;
    stack[sp] = root; sp += 1;

    while sp > 0 {
        sp -= 1;
        let node_idx = stack[sp];
        stats_inc_node();
        stats_inc_aabb();
        if aabb_hit(node_idx, closest_so_far).is_none() { continue; }
        let node = &nodes[node_idx];
        if node.left < 0 { // 葉
            stats_inc_leaf();
            if let Some(rec) = visit_leaf(node.start as usize, node.count as usize, closest_so_far) {
                if rec.t < closest_so_far { closest_so_far = rec.t; }
                best = Some(rec);
            }
        } else {
            // 近い順探索（遠い方を先push）
            let lidx = node.left as usize;
            let ridx = node.right as usize;
            stats_inc_aabb();
            let lhit = aabb_hit(lidx, closest_so_far).map(|(tn, _)| tn);
            stats_inc_aabb();
            let rhit = aabb_hit(ridx, closest_so_far).map(|(tn, _)| tn);
            match (lhit, rhit) {
                (Some(ln), Some(rn)) => {
                    if ln <= rn { debug_assert!(sp + 2 <= stack.len()); stack[sp] = ridx; sp += 1; stack[sp] = lidx; sp += 1; }
                    else          { debug_assert!(sp + 2 <= stack.len()); stack[sp] = lidx; sp += 1; stack[sp] = ridx; sp += 1; }
                }
                (Some(_), None) => { debug_assert!(sp + 1 <= stack.len()); stack[sp] = lidx; sp += 1; }
                (None, Some(_)) => { debug_assert!(sp + 1 <= stack.len()); stack[sp] = ridx; sp += 1; }
                (None, None) => {}
            }
        }
    }

    best
}

// （重複定義削除）

// メッシュ。
// 各フェースで上の Möller–Trumbore を実行し、もっとも近いヒットを採用する。
// 注意: 端点の包含規則（u,v の境界不等号）は隣接三角と整合するよう統一すること。
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<Point3>,
    pub indices: Vec<[u32; 3]>,
    pub material_id: MaterialId,
    pub xform: Mat4,          // ワールドへのアフィン変換
    pub inv_xform: Mat4,      // その逆（アフィン）
    pub normal_linear: Mat3,  // 法線変換用 3x3（inv_linear.transpose()）
    // BLAS: ローカル BVH（AABB と三角の並び替え）
    tri_bounds: Vec<Aabb>,
    tri_index: Vec<u32>,
    tri_accel: Vec<TriAccel>,
    nodes: Vec<BvhNode>,
    root_idx: usize,
}

#[derive(Clone, Copy, Debug)]
struct TriAccel {
    v0: Vec3,
    e1: Vec3,
    e2: Vec3,
    n_g: Vec3,
}

impl Mesh {
    pub fn new(vertices: Vec<Point3>, indices: Vec<[u32; 3]>, material_id: MaterialId) -> Self {
        let mut m = Self {
            vertices,
            indices,
            material_id,
            xform: Mat4::identity(),
            inv_xform: Mat4::identity(),
            normal_linear: Mat3::identity(),
            tri_bounds: Vec::new(),
            tri_index: Vec::new(),
            tri_accel: Vec::new(),
            nodes: Vec::new(),
            root_idx: 0,
        };
        m.build_bvh();
        m
    }

    pub fn with_transform(
        vertices: Vec<Point3>,
        indices: Vec<[u32; 3]>,
        material_id: MaterialId,
        translate: Vec3,
        linear: Mat3,
    ) -> Self {
        let inv_linear = linear.inverse().unwrap_or(Mat3::identity());
        let normal_linear = inv_linear.transpose();
        let xform = Mat4::from_trs(translate, linear);
        let inv_xform = xform.inverse_affine().unwrap_or(Mat4::identity());
        let mut m = Self {
            vertices,
            indices,
            material_id,
            xform,
            inv_xform,
            normal_linear,
            tri_bounds: Vec::new(),
            tri_index: Vec::new(),
            tri_accel: Vec::new(),
            nodes: Vec::new(),
            root_idx: 0,
        };
        m.build_bvh();
        m
    }

    /// 4x4 アフィン行列で直接ワールド変換を与えるコンストラクタ
    pub fn with_xform(
        vertices: Vec<Point3>,
        indices: Vec<[u32; 3]>,
        material_id: MaterialId,
        xform: Mat4,
    ) -> Self {
        let l = xform.upper3x3();
        let inv_l = l.inverse().unwrap_or(Mat3::identity());
        let normal_linear = inv_l.transpose();
        let inv_xform = xform.inverse_affine().unwrap_or(Mat4::identity());
        let mut m = Self {
            vertices,
            indices,
            material_id,
            xform,
            inv_xform,
            normal_linear,
            tri_bounds: Vec::new(),
            tri_index: Vec::new(),
            tri_accel: Vec::new(),
            nodes: Vec::new(),
            root_idx: 0,
        };
        m.build_bvh();
        m
    }

    pub fn add_translate(&mut self, t: Vec3) {
        // xform = T(t) * xform
        let m = Mat4::from_trs(t, Mat3::identity());
        self.xform = m * self.xform;
        self.inv_xform = self.xform.inverse_affine().unwrap_or(Mat4::identity());
    }
    pub fn set_translate(&mut self, t: Vec3) {
        let l = self.xform.upper3x3();
        self.xform = Mat4::from_trs(t, l);
        self.inv_xform = self.xform.inverse_affine().unwrap_or(Mat4::identity());
    }
    pub fn set_linear(&mut self, m: Mat3) {
        let t = self.xform.translation();
        self.xform = Mat4::from_trs(t, m);
        let inv_linear = m.inverse().unwrap_or(Mat3::identity());
        self.normal_linear = inv_linear.transpose();
        self.inv_xform = self.xform.inverse_affine().unwrap_or(Mat4::identity());
    }

    // BVH 構築（中央値分割）。ローカル空間内での固定構造。
    fn build_bvh(&mut self) {
        let n = self.indices.len();
        self.tri_bounds.clear();
        self.tri_index.clear();
        self.tri_accel.clear();
        self.nodes.clear();
        self.tri_bounds.reserve(n);
        self.tri_index.reserve(n);
        self.tri_accel.reserve(n);
        for (ti, idx) in self.indices.iter().enumerate() {
            let v0 = self.vertices[idx[0] as usize];
            let v1 = self.vertices[idx[1] as usize];
            let v2 = self.vertices[idx[2] as usize];
            self.tri_bounds.push(Aabb::from_triangle(v0, v1, v2));
            self.tri_index.push(ti as u32);
            let e1 = v1 - v0;
            let e2 = v2 - v0;
            let n_g = e1.cross(e2).normalized();
            self.tri_accel.push(TriAccel { v0, e1, e2, n_g });
        }
    if n == 0 { self.root_idx = 0; return; }
    let root = build_bvh_nodes(&mut self.nodes, &mut self.tri_index[..], &self.tri_bounds[..], 4);
        self.root_idx = root;
    }

    /// メッシュ全体のローカル AABB（BLAS ルートの境界）。
    fn local_bounds(&self) -> Option<Aabb> {
        if self.nodes.is_empty() { None } else { Some(self.nodes[self.root_idx].bounds) }
    }
}

impl Hittable for Mesh {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
    // レイをローカル空間に一度だけ変換（アフィンの逆）
    let o_local = self.inv_xform.mul_point(r.origin);
    let d_local = self.inv_xform.mul_dir(r.direction);
        let r_local = Ray::new(o_local, d_local);
    let inv_dir_l = Vec3::new(1.0 / d_local[0], 1.0 / d_local[1], 1.0 / d_local[2]);

        traverse_bvh(
            &self.nodes,
            self.root_idx,
            t_max,
            |idx, closest| self.nodes[idx].bounds.hit(o_local, inv_dir_l, t_min, closest),
            |start, count, closest| {
                let end = start + count;
                let mut best: Option<HitRecord> = None;
                let mut cs = closest;
                for &tri_id in &self.tri_index[start..end] {
                    let acc = self.tri_accel[tri_id as usize];
                    stats_inc_tri();
                    if let Some((t_local, _u, _v)) = intersect_triangle_precomputed(&r_local, acc.v0, acc.e1, acc.e2, t_min, cs) {
                        // 交差点をローカル→ワールドへ
                        let p_local = r_local.at(t_local);
                        let p_world = self.xform.mul_point(p_local);
                        // ワールド空間での t を再計算（射影）
                        let num = (p_world - r.origin).dot(r.direction);
                        let denom = r.direction.dot(r.direction);
                        let t_world = if denom != 0.0 { num / denom } else { t_local };
                        if t_world < t_min || t_world > cs { continue; }
                        let n_world = self.normal_linear.mul_vec3(acc.n_g).normalized();
                        let rec = HitRecord::new(p_world, t_world, n_world, r.direction, self.material_id);
                        cs = t_world; best = Some(rec);
                    }
                }
                best
            },
        )
    }

    fn bounds(&self) -> Aabb {
        // BLAS ルートのローカル AABB をワールドへ変換（4x4）
        if let Some(b) = self.local_bounds() {
            return b.transform_affine(self.xform);
        }
        // 退避: 頂点から直接計算（空でなければ）
        if !self.vertices.is_empty() {
            let mut b = Aabb::new(Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
            for v in &self.vertices { b = b.union(Aabb::new(*v, *v)); }
            return b.transform_affine(self.xform);
        }
        Aabb::new(Vec3::splat(0.0), Vec3::splat(0.0))
    }
}
