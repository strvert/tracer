//! Basic math primitives for the ray tracer (Rust 2024).
//! - Vec3: 3D vector with common operations
//! - Color: RGB color (alias of Vec3)
//! - Point3: 3D point (alias of Vec3)
//! - Ray: origin + t * direction

use core::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// Generic linear interpolation.
/// Returns a + (b - a) * t, where t is in [0,1] for standard lerp.
/// Works for any T that supports +, -, and scalar (f32) multiplication.
pub fn lerp<T>(a: T, b: T, t: f32) -> T
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<f32, Output = T>,
{
    a + (b - a) * t
}

/// Remap a value x from [from_min, from_max] to [to_min, to_max].
/// x can be a scalar (f32) or a vector (e.g. Vec3). For vectors, the operation is component-wise.
pub fn remap<T>(x: T, from_min: f32, from_max: f32, to_min: f32, to_max: f32) -> T
where
    T: Copy
        + Sub<f32, Output = T>
        + Div<f32, Output = T>
        + Mul<f32, Output = T>
        + Add<f32, Output = T>,
{
    let t = (x - from_min) / (from_max - from_min);
    t * (to_max - to_min) + to_min
}

pub type Vec3 = VecN<3>;
pub type Color = Vec3;
pub type Point3 = Vec3;

impl Color {
    pub const BLACK: Self = Self {
        data: [0.0, 0.0, 0.0],
    };
    pub const WHITE: Self = Self {
        data: [1.0, 1.0, 1.0],
    };
    pub const RED: Self = Self {
        data: [1.0, 0.0, 0.0],
    };
    pub const GREEN: Self = Self {
        data: [0.0, 1.0, 0.0],
    };
    pub const BLUE: Self = Self {
        data: [0.0, 0.0, 1.0],
    };
}

// --------------------------- 汎用 N 次元ベクトル ---------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VecN<const N: usize> {
    pub data: [f32; N],
}

impl<const N: usize> Default for VecN<N> {
    fn default() -> Self {
        Self { data: [0.0; N] }
    }
}

impl<const N: usize> VecN<N> {
    pub const fn from_array(data: [f32; N]) -> Self {
        Self { data }
    }
    pub fn splat(v: f32) -> Self {
        Self { data: [v; N] }
    }
    pub fn zero() -> Self {
        Self { data: [0.0; N] }
    }
    pub fn one() -> Self {
        Self { data: [1.0; N] }
    }

    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    pub fn length_squared(self) -> f32 {
        let mut acc = 0.0;
        let d = self.data;
        let mut i = 0;
        while i < N {
            acc += d[i] * d[i];
            i += 1;
        }
        acc
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len == 0.0 { self } else { self / len }
    }

    pub fn dot(self, rhs: Self) -> f32 {
        let mut acc = 0.0;
        let a = self.data;
        let b = rhs.data;
        let mut i = 0;
        while i < N {
            acc += a[i] * b[i];
            i += 1;
        }
        acc
    }

    pub fn clamp(self, min: f32, max: f32) -> Self {
        debug_assert!(min <= max);
        let mut out = [0.0; N];
        let a = self.data;
        let mut i = 0;
        while i < N {
            out[i] = a[i].clamp(min, max);
            i += 1;
        }
        Self { data: out }
    }
}

impl VecN<3> {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { data: [x, y, z] }
    }

    /// 外積. 外積は内積と違って 3 次元ベクトルに対して定義される。
    pub fn cross(self, rhs: Self) -> Self {
        let a = self.data;
        let b = rhs.data;
        Self::new(
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )
    }

    pub fn to_rgb8(self) -> [u8; 3] {
        let c = self.clamp(0.0, 1.0);
        [
            (255.999 * c.data[0]) as u8,
            (255.999 * c.data[1]) as u8,
            (255.999 * c.data[2]) as u8,
        ]
    }
}

impl<const N: usize> core::ops::Index<usize> for VecN<N> {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<const N: usize> core::ops::IndexMut<usize> for VecN<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

// ベクトル演算（要素ごと）
impl<const N: usize> Add for VecN<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = [0.0; N];
        let a = self.data;
        let b = rhs.data;
        let mut i = 0;
        while i < N {
            out[i] = a[i] + b[i];
            i += 1;
        }
        Self { data: out }
    }
}

impl<const N: usize> Sub for VecN<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = [0.0; N];
        let a = self.data;
        let b = rhs.data;
        let mut i = 0;
        while i < N {
            out[i] = a[i] - b[i];
            i += 1;
        }
        Self { data: out }
    }
}

impl<const N: usize> Mul for VecN<N> {
    // Hadamard
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut out = [0.0; N];
        let a = self.data;
        let b = rhs.data;
        let mut i = 0;
        while i < N {
            out[i] = a[i] * b[i];
            i += 1;
        }
        Self { data: out }
    }
}

impl<const N: usize> Div for VecN<N> {
    // Hadamard
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        let mut out = [0.0; N];
        let a = self.data;
        let b = rhs.data;
        let mut i = 0;
        while i < N {
            out[i] = a[i] / b[i];
            i += 1;
        }
        Self { data: out }
    }
}

impl<const N: usize> Mul<f32> for VecN<N> {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        let mut out = [0.0; N];
        let a = self.data;
        let mut i = 0;
        while i < N {
            out[i] = a[i] * rhs;
            i += 1;
        }
        Self { data: out }
    }
}

impl<const N: usize> Div<f32> for VecN<N> {
    type Output = Self;
    fn div(self, rhs: f32) -> Self::Output {
        let mut out = [0.0; N];
        let a = self.data;
        let mut i = 0;
        while i < N {
            out[i] = a[i] / rhs;
            i += 1;
        }
        Self { data: out }
    }
}

impl<const N: usize> Add<f32> for VecN<N> {
    type Output = Self;
    fn add(self, rhs: f32) -> Self::Output {
        let mut out = [0.0; N];
        let a = self.data;
        let mut i = 0;
        while i < N {
            out[i] = a[i] + rhs;
            i += 1;
        }
        Self { data: out }
    }
}

impl<const N: usize> Sub<f32> for VecN<N> {
    type Output = Self;
    fn sub(self, rhs: f32) -> Self::Output {
        let mut out = [0.0; N];
        let a = self.data;
        let mut i = 0;
        while i < N {
            out[i] = a[i] - rhs;
            i += 1;
        }
        Self { data: out }
    }
}

impl<const N: usize> Neg for VecN<N> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let mut out = [0.0; N];
        let a = self.data;
        let mut i = 0;
        while i < N {
            out[i] = -a[i];
            i += 1;
        }
        Self { data: out }
    }
}

impl<const N: usize> AddAssign for VecN<N> {
    fn add_assign(&mut self, rhs: Self) {
        let b = rhs.data;
        let a = &mut self.data;
        let mut i = 0;
        while i < N {
            a[i] += b[i];
            i += 1;
        }
    }
}

impl<const N: usize> SubAssign for VecN<N> {
    fn sub_assign(&mut self, rhs: Self) {
        let b = rhs.data;
        let a = &mut self.data;
        let mut i = 0;
        while i < N {
            a[i] -= b[i];
            i += 1;
        }
    }
}

impl<const N: usize> MulAssign<f32> for VecN<N> {
    fn mul_assign(&mut self, rhs: f32) {
        let a = &mut self.data;
        let mut i = 0;
        while i < N {
            a[i] *= rhs;
            i += 1;
        }
    }
}

impl<const N: usize> DivAssign<f32> for VecN<N> {
    fn div_assign(&mut self, rhs: f32) {
        let a = &mut self.data;
        let mut i = 0;
        while i < N {
            a[i] /= rhs;
            i += 1;
        }
    }
}

impl<const N: usize> Mul<VecN<N>> for f32 {
    type Output = VecN<N>;
    fn mul(self, rhs: VecN<N>) -> Self::Output {
        rhs * self
    }
}

pub type Vec2 = VecN<2>;
pub type Vec4 = VecN<4>;

// --------------------------- 汎用 N x N 行列（列ベクトル・列優先） ---------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat<const N: usize> {
    pub cols: [[f32; N]; N],
}

impl<const N: usize> Default for Mat<N> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<const N: usize> Mat<N> {
    pub fn zero() -> Self {
        Self {
            cols: [[0.0; N]; N],
        }
    }

    pub fn identity() -> Self {
        let mut cols = [[0.0; N]; N];
        for i in 0..N {
            cols[i][i] = 1.0;
        }
        Self { cols }
    }
    pub fn from_cols_array(cols: [[f32; N]; N]) -> Self {
        Self { cols }
    }

    /// 列優先のまま転置行列を返す
    pub fn transpose(self) -> Self {
        let mut cols_t = [[0.0; N]; N];
        for i in 0..N {
            // 新しい列 i
            for j in 0..N {
                cols_t[i][j] = self.cols[j][i];
            }
        }
        Mat { cols: cols_t }
    }

    /// 対角行列を作成（diag を対角成分に配置）
    pub fn from_diagonal(diag: [f32; N]) -> Self {
        let mut cols = [[0.0; N]; N];
        for i in 0..N {
            cols[i][i] = diag[i];
        }
        Mat { cols }
    }

    /// i 番目の列を配列で返す
    #[inline(always)]
    pub fn col_array(&self, i: usize) -> [f32; N] {
        self.cols[i]
    }

    #[inline(always)]
    pub fn mul_vec(&self, v: [f32; N]) -> [f32; N] {
        // 列優先: M*v = sum_i col_i * v_i
        let mut r = [0.0; N];
        for i in 0..N {
            // 列 index
            let vi = v[i];
            for rj in 0..N {
                r[rj] += self.cols[i][rj] * vi;
            }
        }
        r
    }

    pub fn mul(self, rhs: Mat<N>) -> Mat<N> {
        // (M*N).col_i = M * (N.col_i)
        let mut cols = [[0.0; N]; N];
        for i in 0..N {
            // 生成する列 i
            let mut c = [0.0; N];
            let bcol = rhs.cols[i];
            for k in 0..N {
                let bk = bcol[k];
                for rj in 0..N {
                    c[rj] += self.cols[k][rj] * bk;
                }
            }
            cols[i] = c;
        }
        Mat { cols }
    }

    #[inline(always)]
    pub fn mul_vecn(&self, v: VecN<N>) -> VecN<N> {
        VecN::from_array(self.mul_vec(v.data))
    }
}

impl<const N: usize> Mul<Mat<N>> for Mat<N> {
    type Output = Mat<N>;
    fn mul(self, rhs: Mat<N>) -> Mat<N> {
        self.mul(rhs)
    }
}

// 4x4 向けのユーティリティ
pub type Mat4 = Mat<4>;

impl Mat<4> {
    /// Affine 4x4 from translate (t) and linear (R*S as 3x3)
    pub fn from_trs(t: Vec3, l: Mat3) -> Self {
        let c0 = l.col(0);
        let c1 = l.col(1);
        let c2 = l.col(2);
        Mat::from_cols_array([
            [c0[0], c0[1], c0[2], 0.0],
            [c1[0], c1[1], c1[2], 0.0],
            [c2[0], c2[1], c2[2], 0.0],
            [t[0], t[1], t[2], 1.0],
        ])
    }

    #[inline(always)]
    pub fn upper3x3(self) -> Mat3 {
        Mat3::from_cols(
            Vec3::new(self.cols[0][0], self.cols[0][1], self.cols[0][2]),
            Vec3::new(self.cols[1][0], self.cols[1][1], self.cols[1][2]),
            Vec3::new(self.cols[2][0], self.cols[2][1], self.cols[2][2]),
        )
    }

    #[inline(always)]
    pub fn translation(self) -> Vec3 {
        Vec3::new(self.cols[3][0], self.cols[3][1], self.cols[3][2])
    }

    /// アフィン逆行列（下段は [0 0 0 1] を仮定）
    pub fn inverse_affine(self) -> Option<Mat4> {
        let l = self.upper3x3();
        let inv_l = l.inverse()?;
        let t = self.translation();
        let inv_t = -(inv_l.mul_vec3(t));
        Some(Mat4::from_trs(inv_t, inv_l))
    }

    /// 点の変換（w=1）
    pub fn mul_point(self, p: Vec3) -> Vec3 {
        let v = [p[0], p[1], p[2], 1.0];
        let mut r = [0.0; 4];
        for i in 0..4 {
            // 列
            let vi = v[i];
            for rj in 0..4 {
                r[rj] += self.cols[i][rj] * vi;
            }
        }
        if r[3] != 0.0 {
            Vec3::new(r[0] / r[3], r[1] / r[3], r[2] / r[3])
        } else {
            Vec3::new(r[0], r[1], r[2])
        }
    }

    /// 方向ベクトルの変換（w=0）
    pub fn mul_dir(self, v3: Vec3) -> Vec3 {
        let v = [v3[0], v3[1], v3[2], 0.0];
        let mut r = [0.0; 4];
        for i in 0..4 {
            // 列
            let vi = v[i];
            for rj in 0..4 {
                r[rj] += self.cols[i][rj] * vi;
            }
        }
        Vec3::new(r[0], r[1], r[2])
    }
}

// Mat3 は Mat<3> の特殊化として提供する
pub type Mat3 = Mat<3>;

impl Mat<3> {
    pub fn from_cols(c0: Vec3, c1: Vec3, c2: Vec3) -> Self {
        Mat::<3>::from_cols_array([
            [c0[0], c0[1], c0[2]],
            [c1[0], c1[1], c1[2]],
            [c2[0], c2[1], c2[2]],
        ])
    }

    #[inline(always)]
    pub fn col(&self, i: usize) -> Vec3 {
        Vec3::new(self.cols[i][0], self.cols[i][1], self.cols[i][2])
    }

    pub fn from_scale(sx: f32, sy: f32, sz: f32) -> Self {
        Mat::<3>::from_diagonal([sx, sy, sz])
    }

    pub fn mul_vec3(self, v: Vec3) -> Vec3 {
        Vec3::from(self.mul_vecn(VecN::<3>::from(v)))
    }

    pub fn det(self) -> f32 {
        let a = self.col(0);
        let b = self.col(1);
        let c = self.col(2);
        a.dot(b.cross(c))
    }

    pub fn inverse(self) -> Option<Self> {
        let a = self.col(0);
        let b = self.col(1);
        let c = self.col(2);
        let r0 = b.cross(c);
        let r1 = c.cross(a);
        let r2 = a.cross(b);
        let det = a.dot(r0);
        if det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Mat3::from_cols(r0 * inv_det, r1 * inv_det, r2 * inv_det))
    }

    // 回転（軸別）
    pub fn from_euler_x(theta: f32) -> Self {
        let (s, c) = theta.sin_cos();
        let m00 = 1.0;
        let m01 = 0.0;
        let m02 = 0.0;
        let m10 = 0.0;
        let m11 = c;
        let m12 = -s;
        let m20 = 0.0;
        let m21 = s;
        let m22 = c;
        Mat3::from_cols(
            Vec3::new(m00, m10, m20),
            Vec3::new(m01, m11, m21),
            Vec3::new(m02, m12, m22),
        )
    }

    pub fn from_euler_y(theta: f32) -> Self {
        let (s, c) = theta.sin_cos();
        let m00 = c;
        let m01 = 0.0;
        let m02 = s;
        let m10 = 0.0;
        let m11 = 1.0;
        let m12 = 0.0;
        let m20 = -s;
        let m21 = 0.0;
        let m22 = c;
        Mat3::from_cols(
            Vec3::new(m00, m10, m20),
            Vec3::new(m01, m11, m21),
            Vec3::new(m02, m12, m22),
        )
    }

    pub fn from_euler_z(theta: f32) -> Self {
        let (s, c) = theta.sin_cos();
        let m00 = c;
        let m01 = -s;
        let m02 = 0.0;
        let m10 = s;
        let m11 = c;
        let m12 = 0.0;
        let m20 = 0.0;
        let m21 = 0.0;
        let m22 = 1.0;
        Mat3::from_cols(
            Vec3::new(m00, m10, m20),
            Vec3::new(m01, m11, m21),
            Vec3::new(m02, m12, m22),
        )
    }

    // 任意軸回転（Rodrigues の回転公式）。axis は正規化される。
    pub fn from_axis_angle(axis: Vec3, theta: f32) -> Self {
        let a = axis.normalized();
        let (s, c) = theta.sin_cos();
        let t = 1.0 - c;
        let (x, y, z) = (a[0], a[1], a[2]);
        let m00 = t * x * x + c;
        let m01 = t * x * y - s * z;
        let m02 = t * x * z + s * y;
        let m10 = t * x * y + s * z;
        let m11 = t * y * y + c;
        let m12 = t * y * z - s * x;
        let m20 = t * x * z - s * y;
        let m21 = t * y * z + s * x;
        let m22 = t * z * z + c;
        Mat3::from_cols(
            Vec3::new(m00, m10, m20),
            Vec3::new(m01, m11, m21),
            Vec3::new(m02, m12, m22),
        )
    }

    // Euler (XYZ): Rx * Ry * Rz の順で合成
    pub fn from_euler_xyz(rx: f32, ry: f32, rz: f32) -> Self {
        let rxm = Mat3::from_euler_x(rx);
        let rym = Mat3::from_euler_y(ry);
        let rzm = Mat3::from_euler_z(rz);
        rxm * (rym * rzm)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ray {
    pub origin: Point3,
    pub direction: Vec3,
}

impl Ray {
    pub const fn new(origin: Point3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    pub fn at(&self, t: f32) -> Point3 {
        self.origin + self.direction * t
    }
}
