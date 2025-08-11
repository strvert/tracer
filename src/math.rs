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

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);
    pub const ONE: Self = Self::new(1.0, 1.0, 1.0);

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub const fn splat(v: f32) -> Self {
        Self::new(v, v, v)
    }

    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    pub const fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len == 0.0 {
            Self::ZERO
        } else {
            self / len
        }
    }

    pub const fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub const fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    pub fn clamp(self, min: f32, max: f32) -> Self {
        debug_assert!(min <= max);
        Self::new(
            self.x.clamp(min, max),
            self.y.clamp(min, max),
            self.z.clamp(min, max),
        )
    }

    pub fn to_rgb8(self) -> [u8; 3] {
        // assume 0..1 range; clamp to be safe
        let c = self.clamp(0.0, 1.0);
        [
            (255.999 * c.x) as u8,
            (255.999 * c.y) as u8,
            (255.999 * c.z) as u8,
        ]
    }
}

// Operators
impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul for Vec3 {
    type Output = Self; // Hadamard product
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl Div for Vec3 {
    type Output = Self; // Hadamard division
    fn div(self, rhs: Self) -> Self::Output {
        Self::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, rhs: Vec3) -> Self::Output {
        rhs * self
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;
    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

// Add/Sub with scalar (component-wise) to support generic remap over vectors
impl Add<f32> for Vec3 {
    type Output = Self;
    fn add(self, rhs: f32) -> Self::Output {
        Self::new(self.x + rhs, self.y + rhs, self.z + rhs)
    }
}

impl Sub<f32> for Vec3 {
    type Output = Self;
    fn sub(self, rhs: f32) -> Self::Output {
        Self::new(self.x - rhs, self.y - rhs, self.z - rhs)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, rhs: f32) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl DivAssign<f32> for Vec3 {
    fn div_assign(&mut self, rhs: f32) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl Index<usize> for Vec3 {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("index out of bounds: {}", index),
        }
    }
}

impl IndexMut<usize> for Vec3 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("index out of bounds: {}", index),
        }
    }
}

pub type Color = Vec3;
pub type Point3 = Vec3;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Mat3 {
    // 列ベクトル表現: [a b c]（a,b,c は各列）
    pub cols: [Vec3; 3],
}

impl Mat3 {
    pub const fn identity() -> Self {
        Self { cols: [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ] }
    }

    pub const fn from_cols(c0: Vec3, c1: Vec3, c2: Vec3) -> Self { Self { cols: [c0, c1, c2] } }

    pub fn from_scale(sx: f32, sy: f32, sz: f32) -> Self {
        Self::from_cols(
            Vec3::new(sx, 0.0, 0.0),
            Vec3::new(0.0, sy, 0.0),
            Vec3::new(0.0, 0.0, sz),
        )
    }

    pub fn transpose(self) -> Self {
        let a = self.cols[0];
        let b = self.cols[1];
        let c = self.cols[2];
        // 行ベクトルを列に並べる
        Self::from_cols(
            Vec3::new(a.x, b.x, c.x),
            Vec3::new(a.y, b.y, c.y),
            Vec3::new(a.z, b.z, c.z),
        )
    }

    pub fn mul_vec3(self, v: Vec3) -> Vec3 {
        // 列優先: M v = a*v.x + b*v.y + c*v.z
        self.cols[0] * v.x + self.cols[1] * v.y + self.cols[2] * v.z
    }

    pub fn det(self) -> f32 {
        let a = self.cols[0];
        let b = self.cols[1];
        let c = self.cols[2];
        a.dot(b.cross(c))
    }

    pub fn inverse(self) -> Option<Self> {
        let a = self.cols[0];
        let b = self.cols[1];
        let c = self.cols[2];
        // 逆行列の列は以下で与えられる（adjugate 列 / det）
        let r0 = b.cross(c);
        let r1 = c.cross(a);
        let r2 = a.cross(b);
        let det = a.dot(r0);
        if det.abs() < 1e-12 { return None; }
        let inv_det = 1.0 / det;
        Some(Self::from_cols(r0 * inv_det, r1 * inv_det, r2 * inv_det))
    }

    // 回転（軸別）
    pub fn from_euler_x(theta: f32) -> Self {
        let (s, c) = theta.sin_cos();
        // 行列（行優先）
        // [ 1  0  0 ]
        // [ 0  c -s ]
        // [ 0  s  c ]
        let m00 = 1.0; let m01 = 0.0; let m02 = 0.0;
        let m10 = 0.0; let m11 = c;   let m12 = -s;
        let m20 = 0.0; let m21 = s;   let m22 = c;
        Self::from_cols(
            Vec3::new(m00, m10, m20),
            Vec3::new(m01, m11, m21),
            Vec3::new(m02, m12, m22),
        )
    }

    pub fn from_euler_y(theta: f32) -> Self {
        let (s, c) = theta.sin_cos();
        // [  c  0  s ]
        // [  0  1  0 ]
        // [ -s  0  c ]
        let m00 = c;   let m01 = 0.0; let m02 = s;
        let m10 = 0.0; let m11 = 1.0; let m12 = 0.0;
        let m20 = -s;  let m21 = 0.0; let m22 = c;
        Self::from_cols(
            Vec3::new(m00, m10, m20),
            Vec3::new(m01, m11, m21),
            Vec3::new(m02, m12, m22),
        )
    }

    pub fn from_euler_z(theta: f32) -> Self {
        let (s, c) = theta.sin_cos();
        // [  c -s  0 ]
        // [  s  c  0 ]
        // [  0  0  1 ]
        let m00 = c;   let m01 = -s;  let m02 = 0.0;
        let m10 = s;   let m11 = c;   let m12 = 0.0;
        let m20 = 0.0; let m21 = 0.0; let m22 = 1.0;
        Self::from_cols(
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
        let (x, y, z) = (a.x, a.y, a.z);
        // 行優先 m_{ij}
        let m00 = t*x*x + c;
        let m01 = t*x*y - s*z;
        let m02 = t*x*z + s*y;
        let m10 = t*x*y + s*z;
        let m11 = t*y*y + c;
        let m12 = t*y*z - s*x;
        let m20 = t*x*z - s*y;
        let m21 = t*y*z + s*x;
        let m22 = t*z*z + c;
        Self::from_cols(
            Vec3::new(m00, m10, m20),
            Vec3::new(m01, m11, m21),
            Vec3::new(m02, m12, m22),
        )
    }

    // Euler (XYZ): R = Rz(rz) * Ry(ry) * Rx(rx) ではなく、ここでは Rx * Ry * Rz の順で合成。
    pub fn from_euler_xyz(rx: f32, ry: f32, rz: f32) -> Self {
        let rxm = Self::from_euler_x(rx);
        let rym = Self::from_euler_y(ry);
        let rzm = Self::from_euler_z(rz);
        // v' = (Rx * Ry * Rz) v （右から適用: まず Z, 次に Y, 最後に X）
        rxm * (rym * rzm)
    }
}

impl Mul<Mat3> for Mat3 {
    type Output = Mat3;
    fn mul(self, rhs: Mat3) -> Mat3 {
        // 列ごとに掛ける: (M*N).col_i = M * (N.col_i)
        let c0 = self.mul_vec3(rhs.cols[0]);
        let c1 = self.mul_vec3(rhs.cols[1]);
        let c2 = self.mul_vec3(rhs.cols[2]);
        Mat3::from_cols(c0, c1, c2)
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
