//! Simple pinhole camera. Viewport is built from camera params (v_fov) and aspect ratio.

use crate::math::{Point3, Ray, Vec3};

#[derive(Clone, Debug)]
pub struct Camera {
    origin: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    lower_left_corner: Point3,
}

impl Camera {
    /// Create a camera looking along -Z with given vertical FOV (degrees) and aspect ratio.
    pub fn new(origin: Point3, v_fov_deg: f32, aspect_ratio: f32) -> Self {
        let theta = v_fov_deg.to_radians();
        let h = (theta * 0.5).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;
        let focal_length = 1.0_f32;

        let horizontal = Vec3::new(viewport_width, 0.0, 0.0);
        let vertical = Vec3::new(0.0, viewport_height, 0.0);
        let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - Vec3::new(0.0, 0.0, focal_length);

        Self { origin, horizontal, vertical, lower_left_corner }
    }

    /// Generate a ray for screen coordinates u,v in [0,1].
    pub fn get_ray(&self, u: f32, v: f32) -> Ray {
        let dir = self.lower_left_corner + u * self.horizontal + v * self.vertical - self.origin;
        Ray::new(self.origin, dir)
    }
}
