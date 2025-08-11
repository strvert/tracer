//! glTF から Mesh を構築する簡易ローダ。
//! - 対応: TRIANGLES と TRIANGLE_STRIP のプリミティブ、position(必須) と indices（なければ順序どおり）
//! - 変換: ノード変換は未適用（今後の拡張余地）。Mesh には translate と linear(3x3) を適用可能。

use crate::hit::Mesh;
use crate::math::{Vec3, Mat3};
use crate::types::MaterialId;

/// glTF/GLB から最初のメッシュ（全プリミティブ連結）を読み込み、
/// 指定の position/scale を適用した Mesh を構築する。
pub fn load_gltf_mesh_with_transform(
    path: &str,
    material_id: MaterialId,
    translate: Vec3,
    linear: Mat3,
) -> Result<Mesh, Box<dyn std::error::Error>> {
    let (doc, buffers, _images) = gltf::import(path)?;

    let mut vertices: Vec<Vec3> = Vec::new();
    let mut indices: Vec<[u32; 3]> = Vec::new();

    for mesh in doc.meshes() {
        for prim in mesh.primitives() {
            use gltf::mesh::Mode;

            // 対応モードのみ処理（その他はスキップ）
            match prim.mode() {
                Mode::Triangles | Mode::TriangleStrip => {}
                _ => continue,
            }

            let reader = prim.reader(|buffer| Some(&buffers[buffer.index()].0));

            let base_index = vertices.len() as u32;

            if let Some(pos_iter) = reader.read_positions() {
                for p in pos_iter {
                    vertices.push(Vec3::new(p[0], p[1], p[2]));
                }
            } else {
                // position がないメッシュはスキップ
                continue;
            }

            // ローカル関数: 三頂点ローカルインデックス a,b,c を CCW として push（base_index を加算）
            let mut push_tri = |a: u32, b: u32, c: u32| {
                // 退化（インデックスが重複）をスキップ
                if a == b || b == c || a == c {
                    return;
                }
                indices.push([base_index + a, base_index + b, base_index + c]);
            };

            match prim.mode() {
                Mode::Triangles => {
                    if let Some(read_indices) = reader.read_indices() {
                        let tmp: Vec<u32> = read_indices.into_u32().collect();
                        for tri in tmp.chunks_exact(3) {
                            push_tri(tri[0], tri[1], tri[2]);
                        }
                    } else {
                        // 非インデックス: 連続 3 つで 1 面
                        let count_this = (vertices.len() as u32) - base_index;
                        let mut i = 0u32;
                        while i + 2 < count_this {
                            push_tri(i, i + 1, i + 2);
                            i += 3;
                        }
                    }
                }
                Mode::TriangleStrip => {
                    if let Some(read_indices) = reader.read_indices() {
                        let iv: Vec<u32> = read_indices.into_u32().collect();
                        if iv.len() >= 3 {
                            for k in 0..(iv.len() - 2) {
                                // 偶奇で向きを反転させ CCW を維持
                                if k % 2 == 0 {
                                    push_tri(iv[k], iv[k + 1], iv[k + 2]);
                                } else {
                                    push_tri(iv[k + 1], iv[k], iv[k + 2]);
                                }
                            }
                        }
                    } else {
                        // 非インデックス: 頂点配列そのものが strip の順序
                        let count_this = (vertices.len() as u32) - base_index;
                        if count_this >= 3 {
                            let mut k = 0u32;
                            while k + 2 < count_this {
                                if k % 2 == 0 {
                                    push_tri(k, k + 1, k + 2);
                                } else {
                                    push_tri(k + 1, k, k + 2);
                                }
                                k += 1;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    Ok(Mesh::with_transform(vertices, indices, material_id, translate, linear))
}

pub fn load_gltf_mesh(path: &str, material_id: MaterialId) -> Result<Mesh, Box<dyn std::error::Error>> {
    load_gltf_mesh_with_transform(path, material_id, Vec3::new(0.0, 0.0, 0.0), Mat3::identity())
}
