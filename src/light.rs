//! Point light and light list. Color and intensity are separated.

use crate::math::{Color, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    pub position: Vec3,
    pub color: Color,     // spectral color (linear)
    pub intensity: f32,   // scalar intensity (e.g., radiometric scale)
}

impl PointLight {
    pub const fn new(position: Vec3, color: Color, intensity: f32) -> Self {
        Self { position, color, intensity }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LightList {
    pub lights: Vec<PointLight>,
}

impl LightList {
    pub fn new() -> Self { Self { lights: Vec::new() } }
    pub fn clear(&mut self) { self.lights.clear(); }
    pub fn add(&mut self, light: PointLight) { self.lights.push(light); }
    pub fn len(&self) -> usize { self.lights.len() }
    pub fn is_empty(&self) -> bool { self.lights.is_empty() }
    pub fn iter(&self) -> impl Iterator<Item = &PointLight> { self.lights.iter() }
}
