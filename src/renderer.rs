use rayon::prelude::*;

use crate::camera::Camera;
use crate::environment::Environment;
use crate::hit::{begin_primary_bvh_stats, end_bvh_stats, BvhStats, Hittable};
use crate::integrator::Integrator;
use crate::light::LightList;
use crate::material::MaterialRegistry;
use crate::rng;
use crate::sampler::Sampler;

pub struct RenderSettings {
    pub width: u32,
    pub height: u32,
    pub gamma: f32,
    #[allow(dead_code)]
    pub frame: u64,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self { width: 1280, height: 720, gamma: 2.2, frame: 0 }
    }
}

pub struct RenderOutput {
    pub pixels: Vec<u8>,
    #[cfg(feature = "stats")]
    pub stats: StatsOutput,
}

#[cfg(feature = "stats")]
pub struct StatsOutput {
    pub aabb_tests: Vec<u32>,
}

pub struct RenderCtx<'a> {
    pub world: &'a (dyn Hittable + Send + Sync),
    pub mats: &'a MaterialRegistry,
    pub lights: &'a LightList,
    pub env: &'a dyn Environment,
}

pub struct Renderer<'a, I: Integrator, S: Sampler> {
    pub camera: &'a Camera,
    pub world: &'a (dyn Hittable + Send + Sync),
    pub mats: &'a MaterialRegistry,
    pub lights: &'a LightList,
    pub env: &'a dyn Environment,
    pub sampler: &'a S,
    pub integrator: I,
    pub settings: RenderSettings,
}

impl<'a, I: Integrator, S: Sampler> Renderer<'a, I, S> {
    pub fn render(&self) -> RenderOutput {
        let w = self.settings.width;
        let h = self.settings.height;
        let inv_gamma = 1.0 / self.settings.gamma;
        let rows: Vec<(Vec<u8>, Vec<u32>)> = (0..h)
            .into_par_iter()
            .rev()
            .map(|y| {
                let ctx = RenderCtx { world: self.world, mats: self.mats, lights: self.lights, env: self.env };
                let mut row_rgb = Vec::with_capacity((w as usize) * 3);
                let mut row_counts: Vec<u32> = Vec::with_capacity(w as usize);
                for x in 0..w {
                    let seed = rng::splitmix64(((y as u64) << 32) | (x as u64));
                    let mut rng = rng::Rng::from_seed(seed);
                    let mut offsets_buf: Vec<(f32, f32)> = Vec::with_capacity(16);
                    self.sampler.sample_offsets(&mut rng, &mut offsets_buf);
                    let mut color = crate::math::Color::ZERO;
                    let mut aabb_sum: u32 = 0;
                    for (du, dv) in offsets_buf.iter().copied() {
                        let u = (x as f32 + du) / (w - 1) as f32;
                        let v = (y as f32 + dv) / (h - 1) as f32;
                        let r = self.camera.get_ray(u, v);
                        let mut stats = BvhStats::default();
                        begin_primary_bvh_stats(&mut stats);
                        let c = self.integrator.li(&ctx, &r);
                        end_bvh_stats();
                        color += c;
                        aabb_sum = aabb_sum.saturating_add(stats.aabb_tests);
                    }
                    color /= offsets_buf.len() as f32;
                    color = crate::math::Color::new(color.x.powf(inv_gamma), color.y.powf(inv_gamma), color.z.powf(inv_gamma));
                    row_rgb.extend_from_slice(&color.to_rgb8());
                    row_counts.push(aabb_sum);
                }
                (row_rgb, row_counts)
            })
            .collect();

        let mut pixels = Vec::with_capacity((w as usize) * (h as usize) * 3);
        let mut counts = Vec::with_capacity((w as usize) * (h as usize));
        for (row_rgb, row_counts) in rows.into_iter() {
            pixels.extend_from_slice(&row_rgb);
            counts.extend_from_slice(&row_counts);
        }
        RenderOutput {
            pixels,
            #[cfg(feature = "stats")]
            stats: StatsOutput { aabb_tests: counts },
        }
    }
}
