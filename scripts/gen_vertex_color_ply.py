"""Export a mesh to binary PLY with per-vertex error-map colors."""
import argparse
import struct
from pathlib import Path

import numpy as np
import trimesh


def load_mesh(mesh_path):
    """Load mesh geometry and preserve normals/UVs when present."""
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f'No geometry found in mesh scene: {mesh_path}')
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    normals = None
    if getattr(mesh, 'vertex_normals', None) is not None and len(mesh.vertex_normals) == len(vertices):
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

    uv = None
    visual = getattr(mesh, 'visual', None)
    if visual is not None:
        visual_uv = getattr(visual, 'uv', None)
        if visual_uv is not None and len(visual_uv) == len(vertices):
            uv = np.asarray(visual_uv, dtype=np.float32)
            # Mitsuba's OBJ and PLY loaders use opposite conventions for the
            # vertical texture axis. Flip t here so exported PLY matches OBJ.
            uv[:, 1] = 1.0 - uv[:, 1]

    return vertices, faces, normals, uv


def blue_white_red_colormap(values):
    """Map normalized values in [0, 1] to a blue-white-red ramp."""
    values = np.clip(values, 0.0, 1.0).astype(np.float32)
    colors = np.zeros((values.shape[0], 3), dtype=np.float32)

    lower_mask = values <= 0.5
    upper_mask = ~lower_mask

    lower_t = values[lower_mask] / 0.5
    colors[lower_mask, 0] = lower_t
    colors[lower_mask, 1] = lower_t
    colors[lower_mask, 2] = 1.0

    upper_t = (values[upper_mask] - 0.5) / 0.5
    colors[upper_mask, 0] = 1.0
    colors[upper_mask, 1] = 1.0 - upper_t
    colors[upper_mask, 2] = 1.0 - upper_t

    return colors


def generate_vertex_colors(vertices):
    """Generate per-vertex colors from normalized z values."""
    z_values = vertices[:, 2]
    z_min = float(z_values.min())
    z_max = float(z_values.max())
    if np.isclose(z_min, z_max):
        normalized = np.full_like(z_values, 0.5, dtype=np.float32)
    else:
        normalized = (z_values - z_min) / (z_max - z_min)
    return blue_white_red_colormap(normalized)


def write_binary_ply(output_path, vertices, colors, faces, normals=None, uv=None):
    """Write a binary little-endian PLY mesh with preserved attributes."""
    header_parts = [
        'ply\n',
        'format binary_little_endian 1.0\n',
        f'element vertex {len(vertices)}\n',
        'property float x\n',
        'property float y\n',
        'property float z\n',
    ]
    if normals is not None:
        header_parts.extend([
            'property float nx\n',
            'property float ny\n',
            'property float nz\n',
        ])
    if uv is not None:
        header_parts.extend([
            'property float s\n',
            'property float t\n',
        ])
    header_parts.extend([
        'property float red\n',
        'property float green\n',
        'property float blue\n',
        f'element face {len(faces)}\n',
        'property list uchar int vertex_indices\n',
        'end_header\n',
    ])
    header = ''.join(header_parts).encode('ascii')

    with open(output_path, 'wb') as file:
        file.write(header)
        for index, (vertex, color) in enumerate(zip(vertices, colors)):
            fields = [vertex[0], vertex[1], vertex[2]]
            if normals is not None:
                fields.extend(normals[index].tolist())
            if uv is not None:
                fields.extend(uv[index].tolist())
            fields.extend(color.tolist())
            file.write(struct.pack(f'<{len(fields)}f', *fields))

        for face in faces:
            file.write(struct.pack('<B', len(face)))
            file.write(struct.pack(f'<{len(face)}i', *face.tolist()))


def main():
    parser = argparse.ArgumentParser(description='Export mesh to vertex-colored PLY')
    parser.add_argument('--mesh', required=True, help='Input mesh path')
    parser.add_argument('--output', required=True, help='Output PLY path')
    args = parser.parse_args()

    vertices, faces, normals, uv = load_mesh(args.mesh)
    colors = generate_vertex_colors(vertices)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_binary_ply(output_path, vertices, colors, faces, normals=normals, uv=uv)
    print(f'Saved {len(vertices)} vertices and {len(faces)} faces to {output_path}')


if __name__ == '__main__':
    main()
