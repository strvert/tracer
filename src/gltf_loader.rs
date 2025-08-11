//! glTF から Mesh を構築する簡易ローダ。
//! - 対応: TRIANGLES と TRIANGLE_STRIP のプリミティブ、position(必須) と indices（なければ順序どおり）
//! - 変換: ノード変換（シーングラフ）に対応。ノードに Mesh がぶら下がっている場合、親子の TRS を合成して
//!   Mesh::with_transform(translate, linear) に反映する。linear は R*S（回転*スケール）の 3x3。

use log::{debug, warn};

use crate::hit::Mesh;
use crate::math::{Mat3, Mat4, Vec3};
use crate::types::MaterialId;

/// glTF/GLB から最初のメッシュ（全プリミティブ連結）を読み込み、
/// 指定の position/scale を適用した Mesh を構築する。
// ----------------------------- 共通ヘルパ -----------------------------

/// クォータニオン(x,y,z,w) → 回転（3x3）。
fn quat_to_mat3(x: f32, y: f32, z: f32, w: f32) -> Mat3 {
    let len2 = x * x + y * y + z * z + w * w;
    let (x, y, z, w) = if len2 > 0.0 {
        let inv = 1.0 / len2.sqrt();
        (x * inv, y * inv, z * inv, w * inv)
    } else {
        (0.0, 0.0, 0.0, 1.0)
    };
    let m00 = 1.0 - 2.0 * (y * y + z * z);
    let m01 = 2.0 * (x * y - z * w);
    let m02 = 2.0 * (x * z + y * w);
    let m10 = 2.0 * (x * y + z * w);
    let m11 = 1.0 - 2.0 * (x * x + z * z);
    let m12 = 2.0 * (y * z - x * w);
    let m20 = 2.0 * (x * z - y * w);
    let m21 = 2.0 * (y * z + x * w);
    let m22 = 1.0 - 2.0 * (x * x + y * y);
    Mat3::from_cols(
        Vec3::new(m00, m10, m20),
        Vec3::new(m01, m11, m21),
        Vec3::new(m02, m12, m22),
    )
}

/// glTF Mesh の全プリミティブを読み取り、vertices/indices に追記する（CCW）。
fn append_gltf_mesh_data(
    gmesh: gltf::Mesh,
    buffers: &Vec<gltf::buffer::Data>,
    vertices: &mut Vec<Vec3>,
    indices: &mut Vec<[u32; 3]>,
) {
    use gltf::mesh::Mode;
    for (prim_idx, prim) in gmesh.primitives().enumerate() {
        let mode = prim.mode();
        match mode {
            Mode::Triangles | Mode::TriangleStrip | Mode::TriangleFan => {}
            _ => {
                warn!("[gltf_loader] skip primitive {}: unsupported mode {:?}", prim_idx, mode);
                continue;
            }
        }
        let reader = prim.reader(|buffer| Some(&buffers[buffer.index()].0));
        let base_index = vertices.len() as u32;

        let mut pos_count = 0;
        if let Some(pos_iter) = reader.read_positions() {
            for p in pos_iter {
                vertices.push(Vec3::new(p[0], p[1], p[2]));
                pos_count += 1;
            }
        } else {
            warn!("[gltf_loader] primitive {}: missing POSITION attribute, skipping", prim_idx);
            continue;
        }

        let mut tri_count = 0;
        let mut push_tri = |a: u32, b: u32, c: u32| {
            if a == b || b == c || a == c {
                debug!("[gltf_loader] primitive {}: degenerate triangle skipped: {} {} {}", prim_idx, a, b, c);
                return;
            }
            indices.push([base_index + a, base_index + b, base_index + c]);
            tri_count += 1;
        };

        match mode {
            Mode::Triangles => {
                if let Some(read_indices) = reader.read_indices() {
                    let tmp: Vec<u32> = read_indices.into_u32().collect();
                    for tri in tmp.chunks_exact(3) {
                        push_tri(tri[0], tri[1], tri[2]);
                    }
                    debug!("[gltf_loader] primitive {}: TRIANGLES indexed, verts={}, tris={}", prim_idx, pos_count, tri_count);
                } else {
                    let count_this = (vertices.len() as u32) - base_index;
                    let mut i = 0u32;
                    while i + 2 < count_this {
                        push_tri(i, i + 1, i + 2);
                        i += 3;
                    }
                    debug!("[gltf_loader] primitive {}: TRIANGLES non-indexed, verts={}, tris={}", prim_idx, pos_count, tri_count);
                }
            }
            Mode::TriangleStrip => {
                if let Some(read_indices) = reader.read_indices() {
                    let iv: Vec<u32> = read_indices.into_u32().collect();
                    if iv.len() >= 3 {
                        for k in 0..(iv.len() - 2) {
                            if k % 2 == 0 {
                                push_tri(iv[k], iv[k + 1], iv[k + 2]);
                            } else {
                                push_tri(iv[k + 1], iv[k], iv[k + 2]);
                            }
                        }
                    }
                    debug!("[gltf_loader] primitive {}: TRIANGLE_STRIP indexed, verts={}, tris={}", prim_idx, pos_count, tri_count);
                } else {
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
                    debug!("[gltf_loader] primitive {}: TRIANGLE_STRIP non-indexed, verts={}, tris={}", prim_idx, pos_count, tri_count);
                }
            }
            Mode::TriangleFan => {
                if let Some(read_indices) = reader.read_indices() {
                    let iv: Vec<u32> = read_indices.into_u32().collect();
                    if iv.len() >= 3 {
                        for k in 1..(iv.len() - 1) {
                            push_tri(iv[0], iv[k], iv[k + 1]);
                        }
                    }
                    debug!("[gltf_loader] primitive {}: TRIANGLE_FAN indexed, verts={}, tris={}", prim_idx, pos_count, tri_count);
                } else {
                    let count_this = (vertices.len() as u32) - base_index;
                    if count_this >= 3 {
                        for k in 1..(count_this - 1) {
                            push_tri(0, k, k + 1);
                        }
                    }
                    debug!("[gltf_loader] primitive {}: TRIANGLE_FAN non-indexed, verts={}, tris={}", prim_idx, pos_count, tri_count);
                }
            }
            _ => {}
        }
    }
}

// ----------------------------- メッシュ単体読み込み -----------------------------

pub fn load_gltf_mesh_with_transform(
    path: &str,
    material_id: MaterialId,
    translate: Vec3,
    linear: Mat3,
    mesh_index: usize,
) -> Result<Mesh, Box<dyn std::error::Error>> {
    let (doc, buffers, _images) = gltf::import(path)?;
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut indices: Vec<[u32; 3]> = Vec::new();
    let mut found = false;
    for mesh in doc.meshes() {
        if mesh.index() == mesh_index {
            append_gltf_mesh_data(mesh, &buffers, &mut vertices, &mut indices);
            found = true;
            break;
        }
    }
    if !found {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("glTF mesh index {} not found", mesh_index),
        )));
    }

    Ok(Mesh::with_transform(
        vertices,
        indices,
        material_id,
        translate,
        linear,
    ))
}

pub fn load_gltf_mesh(
    path: &str,
    material_id: MaterialId,
    mesh_index: usize,
) -> Result<Mesh, Box<dyn std::error::Error>> {
    load_gltf_mesh_with_transform(
        path,
        material_id,
        Vec3::new(0.0, 0.0, 0.0),
        Mat3::identity(),
        mesh_index,
    )
}

/// glTF/GLB から「指定インデックスのメッシュ」だけを読み込み、変換を適用した Mesh を返す。
/// - `mesh_index` は glTF ドキュメントにおける Mesh の index() 値（0 始まり）のこと。
// ----------------------------- シーングラフ対応 API -----------------------------

/// glTF の Scene を走査し、ノードの変換を適用した Mesh を列挙して返す。
/// - 各ノードに Mesh があれば、その Mesh の全プリミティブを結合して 1 つの Mesh として生成。
/// - ノードの TRS を親子合成して、Mesh::with_transform の translate と linear に埋め込む。
/// - glTF のマテリアルは未解釈（既存設計を尊重）。呼び出し元から渡された `material_id` を全ポリゴンに適用する。
pub fn load_gltf_scene_meshes(
    path: &str,
    material_id: MaterialId,
) -> Result<Vec<Mesh>, Box<dyn std::error::Error>> {
    let (doc, buffers, _images) = gltf::import(path)?;
    let mut out: Vec<Mesh> = Vec::new();

    // ノードを DFS で辿ってインスタンス化
    fn traverse_node(
        node: gltf::Node,
        parent_m: Mat4,
        buffers: &Vec<gltf::buffer::Data>,
        material_id: MaterialId,
        out: &mut Vec<Mesh>,
    ) {
        let (t, r, s) = node.transform().decomposed();
        let t_local = Vec3::new(t[0], t[1], t[2]);
        let r_local = quat_to_mat3(r[0], r[1], r[2], r[3]);
        let s_local = Mat3::from_scale(s[0], s[1], s[2]);
        let l_local = r_local * s_local; // linear = R * S
        let m_local = Mat4::from_trs(t_local, l_local); // T * R * S（列優先）
        let m_world = parent_m * m_local;

        if let Some(gmesh) = node.mesh() {
            let mut vertices: Vec<Vec3> = Vec::new();
            let mut indices: Vec<[u32; 3]> = Vec::new();
            append_gltf_mesh_data(gmesh, buffers, &mut vertices, &mut indices);
            if !vertices.is_empty() && !indices.is_empty() {
                out.push(Mesh::with_xform(
                    vertices,
                    indices,
                    material_id,
                    m_world,
                ));
            }
        }
        for child in node.children() {
            traverse_node(child, m_world, buffers, material_id, out);
        }
    }

    if let Some(scene) = doc.default_scene() {
        for node in scene.nodes() {
            traverse_node(
                node,
                Mat4::identity(),
                &buffers,
                material_id,
                &mut out,
            );
        }
    } else {
        // 既定シーンが無い場合は全シーンを順次解釈
        for scene in doc.scenes() {
            for node in scene.nodes() {
                traverse_node(
                    node,
                    Mat4::identity(),
                    &buffers,
                    material_id,
                    &mut out,
                );
            }
        }
    }

    Ok(out)
}
