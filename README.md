# RenderForge

RenderForge is a YAML-driven rendering framework built on [Mitsuba 3](https://github.com/mitsuba-renderer/mitsuba3) for generating synthetic data, result visualizations, and demo videos.

Research in computer graphics and other fields frequently relies on synthetic data. However, even with mature renderers like Mitsuba or Blender, producing synthetic datasets is far from plug-and-play -- it typically involves substantial boilerplate scripting for scene setup, camera placement, and data export.

RenderForge wraps Mitsuba 3 into a higher-level framework where scenes, rendering tasks, and output formats are all defined through YAML configuration files, enabling researchers to generate experimental data, result visualizations, and demo videos quickly and conveniently.

The framework has been used internally in our group to support the synthetic data pipeline of multiple publications. We have further cleaned up and standardized the codebase to make continued development easier, and we hope it can be useful to other researchers as well.

## Installation

```bash
conda create -n renderer python=3.10
conda activate renderer
pip install mitsuba imageio opencv-python numpy omegaconf tqdm
```

Mitsuba 3 requires a GPU with CUDA support for the `cuda_ad_rgb` variant. CPU rendering is available via `llvm_ad_rgb`.

## Quick Start

```bash
python main.py --conf configs/examples/show_white.yaml
```

You can also run `scripts/test_all.sh` to execute all example use cases in one pass.

**Command-line options:**

| Flag | Description |
|------|-------------|
| `--conf` | Path to YAML config file (default: `configs/examples/show_white.yaml`) |
| `--device` | Rendering backend: `gpu` (default) or `cpu` |
| `--preview` | Enable preview mode (lower quality, faster iteration) |
| `--project_name` | Override the output folder name |

## Example Configs

All examples are in `configs/examples/`. Each config specifies a **Project** type, a **Scene** with elements, and a **CamPose** for camera placement.

### Single-View Rendering

**`show_white.yaml`** -- Renders a single view of a white mesh (cow.ply) with studio-style area lights and a ground plane. We use this setup to render result figures for our transparent object reconstruction papers.

**`show_white_raw.yaml`** -- Same scene as above, but defined using raw Mitsuba dictionary payloads instead of high-level element types. Useful when you need direct control over Mitsuba scene parameters.

**`show_errormap.yaml`** -- Renders a mesh with per-vertex colors (e.g., error heatmaps from reconstruction evaluation). Uses `vertex_color_mesh` element type.

**`planar_calibration_demo.yaml`** -- Renders a textured checkerboard plane for camera calibration purposes. Outputs camera intrinsics/extrinsics alongside the rendered image. We use it to verify calibration programs in our SVBRDF acquisition research under planar lighting.

### Multi-View Rendering

**`multiview_pbr.yaml`** -- Multi-view rendering with full PBR (Disney Principled) materials. Exports albedo, roughness, and metallic maps alongside RGB. See [Gamma Handling](#pbr-texture-gamma-handling) for important details on texture I/O.

**`multiview_transparent.yaml`** -- Multi-view rendering of transparent objects (dielectric BSDF with configurable IoR). We use it to generate training data for transparent object reconstruction.

### Video Generation

**`video_demo.yaml`** -- Generates a 360-degree turntable video by rendering 60 frames on a spherical path and encoding to MP4 at 60 FPS. We use it to produce a hero shot in our demo video.

**`dynamic_light_SVBRDF.yaml`** -- Renders an SVBRDF material under a moving point light across 48 frames, producing a video that shows material appearance under varying illumination. We use it to produce SVBRDF demo video.

## PBR Texture Gamma Handling

When working with PBR materials, textures are handled differently depending on their type:

| Texture | Data space | Load (`raw`) | Save method |
|---------|-----------|-------------|-------------|
| Albedo / base_color | sRGB | `raw=False` (degamma on load) | `mi.util.write_bitmap` (gamma on save) |
| Roughness | Linear | `raw=True` | `imageio.imwrite` (no gamma) |
| Metallic | Linear | `raw=True` | `imageio.imwrite` (no gamma) |
| Normal | Linear | `raw=True` | `imageio.imwrite` (no gamma) |
| Depth | Linear | N/A (computed) | `imageio.imwrite` (no gamma) |

**Why this matters:** Mitsuba's `bitmap` texture loader applies inverse sRGB gamma by default (`raw=False`), which is correct for color textures (albedo) but incorrect for non-color data (roughness, metallic, normal). Similarly, `mi.util.write_bitmap` applies sRGB gamma when saving to PNG/JPG. Non-color maps must bypass this to preserve their linear values.

When adding custom PBR materials, always set `raw=True` in `make_bitmap_texture()` for non-color textures.

**SVBRDF note:** The SVBRDF pipeline (`svbrdf_rectangle`) handles gamma differently from the PBR pipeline above. All four SVBRDF channels (normal, diffuse, roughness, specular) are read directly via `imageio` and loaded into Mitsuba with `raw=True`, bypassing any gamma conversion entirely. This is because packed SVBRDF maps (e.g., from neural network predictions) are typically already in linear space, including the diffuse channel -- unlike standard PBR albedo textures which are usually stored in sRGB.

## Output Formats

Multi-view projects automatically export camera parameters in two formats:

- **`cameras_sphere.npz`** -- NeuS/DTU format. Contains world-to-camera projection matrices (`world_mat`), rotation matrices, camera origins, intrinsics, and scale matrices.
- **`transforms.json`** -- NeRF format. Contains camera-to-world (c2w) 4x4 matrices and field of view.

Preview outputs (GIF or MP4) are generated alongside the rendered images.

## Architecture Overview

The framework is organized around four core concepts, all driven by YAML configuration:

```
YAML Config
  |
  v
Project          -- Defines the rendering workflow
  |
  +-- Scene      -- Container for scene elements
  |     |
  |     +-- Element (shape, light, ...)
  |
  +-- CamPose    -- Camera path generator
```

- **Project** controls the overall rendering loop. Built-in types include `MultiView`, `MovingPointLight`, and `PlanarCalibration`.
- **Scene** assembles Mitsuba scene objects from a list of **Elements**.
- **Element** represents a scene component: meshes (`white_mesh`, `pbr_mesh`, `transparent_mesh`, ...), lights (`envmap`, `area_light`, `point_light`, ...), or raw Mitsuba nodes.
- **CamPose** generates camera positions. Built-in types include `SinglePose`, `SphereGrid`, `Trajectory`, and `Composite`.

All components are registered via a lightweight **Registry** system. To extend the framework, implement your own Project, Element, or CamPose and register it with the corresponding registry decorator.

For a detailed architecture reference and development guide, see [DESIGN.md](DESIGN.md). You can feed it directly to your AI coding assistant for context when extending the framework.
