//! Hit utilities: sphere intersection and hit record.

use crate::math::{Point3, Ray, Vec3};
use crate::types::MaterialId;

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
}

// 新規: Hittable のコレクション（シーン）
#[derive(Default)]
pub struct HittableList {
    pub objects: Vec<Box<dyn Hittable + Send + Sync>>,
}

impl HittableList {
    pub fn new() -> Self { Self { objects: Vec::new() } }
    pub fn clear(&mut self) { self.objects.clear(); }
    pub fn add(&mut self, object: Box<dyn Hittable + Send + Sync>) { self.objects.push(object); }
}

impl Hittable for HittableList {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let mut closest_so_far = t_max;
        let mut hit_rec: Option<HitRecord> = None;

        for obj in &self.objects {
            if let Some(rec) = obj.hit(r, t_min, closest_so_far) {
                closest_so_far = rec.t;
                hit_rec = Some(rec);
            }
        }

        hit_rec
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

impl Hittable for Triangle {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
    // Möller–Trumbore 本体
    // e1 = v1 - v0, e2 = v2 - v0
    let e1 = self.v1 - self.v0;               // 辺 e1
    let e2 = self.v2 - self.v0;               // 辺 e2

    // P = d × e2, det = e1 · P
    // det は行列式（= スカラー三重積）。|det| が小さいときは平行/退化。
    let pvec = r.direction.cross(e2);         // P
    let det = e1.dot(pvec);                   // det
        let eps = 1e-8_f32;
        if det.abs() < eps {
            return None; // 平行 or 面積極小
        }
    let inv_det = 1.0 / det;                  // inv_det = 1/det

    // T = o - v0, u = (T · P) / det
    let tvec = r.origin - self.v0;            // T
    let u = tvec.dot(pvec) * inv_det;         // u
        if u < 0.0 || u > 1.0 { return None; }

    // Q = T × e1, v = (d · Q) / det
    let qvec = tvec.cross(e1);                // Q
    let v = r.direction.dot(qvec) * inv_det;  // v
        if v < 0.0 || u + v > 1.0 { return None; }

    // t = (e2 · Q) / det
    let t = e2.dot(qvec) * inv_det;           // t
        if t < t_min || t > t_max { return None; }

    // 交点と法線（幾何法線）
    let p = r.at(t);
    let outward_normal = e1.cross(e2).normalized(); // n_g
        Some(HitRecord::new(p, t, outward_normal, r.direction, self.material_id))
    }
}

// メッシュ（線形探索）。全フェース共通のマテリアルID。
// 各フェースで上の Möller–Trumbore を実行し、もっとも近いヒットを採用する。
// 注意: 端点の包含規則（u,v の境界不等号）は隣接三角と整合するよう統一すること。
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<Point3>,
    pub indices: Vec<[u32; 3]>,
    pub material_id: MaterialId,
    // 追加: ローカル→ワールドの簡易変換（ワールド位置と各軸スケール）
    pub position: Point3,
    pub scale: Vec3,
}

impl Mesh {
    /// 既定の変換（position=0, scale=1）でメッシュを生成。
    pub fn new(vertices: Vec<Point3>, indices: Vec<[u32; 3]>, material_id: MaterialId) -> Self {
        Self { vertices, indices, material_id, position: Point3::ZERO, scale: Vec3::ONE }
    }

    /// 変換付きでメッシュを生成。
    pub fn with_transform(
        vertices: Vec<Point3>,
        indices: Vec<[u32; 3]>,
        material_id: MaterialId,
        position: Point3,
        scale: Vec3,
    ) -> Self {
        Self { vertices, indices, material_id, position, scale }
    }

    /// 位置を再設定。
    pub fn set_position(&mut self, position: Point3) { self.position = position; }

    /// スケールを再設定（非一様スケール可）。
    pub fn set_scale(&mut self, scale: Vec3) { self.scale = scale; }

    pub fn set_scale_uniform(&mut self, scale: f32) { self.scale = Vec3::new(scale, scale, scale); }
}

impl Hittable for Mesh {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let mut closest_so_far = t_max;
        let mut best: Option<HitRecord> = None;

        for idx in &self.indices {
            let i0 = idx[0] as usize;
            let i1 = idx[1] as usize;
            let i2 = idx[2] as usize;
            // 安全のため境界チェック（外れていればスキップ）
            if i0 >= self.vertices.len() || i1 >= self.vertices.len() || i2 >= self.vertices.len() {
                continue;
            }
            // ローカル頂点にスケール・平行移動を適用してワールド座標へ
            // v_w = position + (scale ◦ v_local)
            let v0 = self.position + (self.vertices[i0] * self.scale);
            let v1 = self.position + (self.vertices[i1] * self.scale);
            let v2 = self.position + (self.vertices[i2] * self.scale);

            // Möller–Trumbore（Triangle と同じ手順）
            // e1=v1-v0, e2=v2-v0, P=d×e2, det=e1·P, T=o-v0, Q=T×e1
            let e1 = v1 - v0;
            let e2 = v2 - v0;
            let pvec = r.direction.cross(e2);
            let det = e1.dot(pvec);
            let eps = 1e-8_f32;
            if det.abs() < eps { continue; }
            let inv_det = 1.0 / det;
            let tvec = r.origin - v0;
            let u = tvec.dot(pvec) * inv_det;
            if u < 0.0 || u > 1.0 { continue; }
            let qvec = tvec.cross(e1);
            let v = r.direction.dot(qvec) * inv_det;
            if v < 0.0 || u + v > 1.0 { continue; }
            let t = e2.dot(qvec) * inv_det;
            if t < t_min || t > closest_so_far { continue; }

            let p = r.at(t);
            // 非一様スケールでも、変換後の辺から計算するクロスは det(S)·S^{-T} n に比例し、方向は正しい。
            let outward_normal = e1.cross(e2).normalized();
            let rec = HitRecord::new(p, t, outward_normal, r.direction, self.material_id);
            closest_so_far = t;
            best = Some(rec);
        }

        best
    }
}
