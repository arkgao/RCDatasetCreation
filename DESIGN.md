# Architecture & Design Guide

This document describes the architecture of the RenderForge framework. It is intended as a reference for developers extending the framework, and can be fed directly to AI coding assistants for context.

## Overview

The framework follows a config-driven architecture. A single YAML file defines everything needed for a rendering task:

```text
YAML Config
  |
  +-- Project (str)     → which rendering workflow to run
  +-- Scene (dict)      → scene elements (meshes, lights, ...)
  +-- Camera (dict)     → sensor parameters (FOV, resolution, sampler)
  +-- CamPose (dict)    → camera path / placement strategy
```

Entry point is `main.py`: parse args → set Mitsuba variant → load config → build project → run.

## Module Responsibilities

### `main.py`

Entry point. Handles CLI argument parsing (`--conf`, `--device`, `--preview`, `--project_name`), sets the Mitsuba variant (`cuda_ad_rgb` or `llvm_ad_rgb`), loads the YAML config via `utils/config_utils.py`, builds the project, and calls `project.run()`.

### `projects/`

Each file defines a rendering workflow as a class inheriting from `BaseProject`.

- **`base.py`** — `BaseProject` (ABC). Handles shared setup: builds the Scene and CamPose from config, loads the Mitsuba camera sensor, applies preview-mode overrides (halved resolution, quartered spp), and manages output paths. Subclasses implement `run()`.
- **`MultiView.py`** — Multi-view rendering. Iterates over camera poses, renders RGB + optional AOV channels (normal, depth, albedo), exports material property maps (roughness, metallic) via base_color substitution, records camera parameters, and generates preview GIF/video.
- **`MovingPointLight.py`** / **`MovingItem.py`** — Animated rendering with moving light sources or objects across frames.
- **`PlanarCalibration.py`** — Single-view rendering with camera parameter export for calibration verification.

### `scene_builder/`

Responsible for assembling Mitsuba scene objects from config.

- **`scene.py`** — `Scene` base class and registered subclasses:
  - `CustomScene`: builds scene from a list of Element configs. Iterates element configs, resolves each via `ELEMENT_REGISTRY`, collects returned node dicts, attaches integrator, and calls `mi.load_dict` to produce the final Mitsuba scene.
  - `XMLScene`: loads a scene directly from a Mitsuba XML file.

- **`elements/`** — High-level element functions that translate config dicts into Mitsuba scene nodes. Each function reads config parameters, resolves file paths via `scene.resolve_source_path()`, calls `mitsuba_utils` to create Mitsuba objects, and returns a `{node_name: mitsuba_object}` dict.
  - `shape.py`: mesh and plane elements (`white_mesh`, `pbr_mesh`, `transparent_mesh`, `texture_mesh`, `vertex_color_mesh`, `svbrdf_rectangle`, `textured_rectangle`, `diffuse_rectangle`)
  - `light.py`: light elements (`envmap_light`, `constant_environment_light`, `point_light`, `directional_light`, `spot_light`, `mesh_area_light`, `rectangle_area_light`)
  - `builtin.py`: `raw` element for injecting arbitrary Mitsuba dicts directly

- **`mitsuba_utils.py`** — **All Mitsuba API calls are isolated here.** See [Element vs mitsuba_utils Separation](#element-vs-mitsuba_utils-separation) below.

- **`build_utils.py`** — Factory functions `build_scene()` and `build_pose_generator()` that look up the Registry by config `type` field and instantiate the corresponding class.

### `camera_poses/`

Camera path generators. Each class is registered with `CAMPOSE_REGISTRY` and exposes two attributes:
- `pose_list`: list of `{origin, target, up}` dicts
- `img_num`: number of poses

Built-in types:
- `SinglePose` — single camera pose from explicit coordinates or spherical angles
- `SphericalGridPose` — uniform grid on a sphere (theta × phi)
- `SphericalRandomPose` — random sampling on a spherical cap
- `SphericalFibonacciPose` — Fibonacci lattice sampling for near-uniform coverage
- `Trajectory` — arbitrary path from a list of waypoints
- `Composite` — chains multiple CamPose generators

### `utils/`

Shared utilities with no Mitsuba dependency (except `camera_utils.py` which is pure NumPy):
- `config_utils.py` — YAML config loader with `includes` support for template composition
- `registry.py` — lightweight `Registry` class, plus the four global registries (`PROJECT_REGISTRY`, `ELEMENT_REGISTRY`, `CAMPOSE_REGISTRY`, `SCENE_REGISTRY`)
- `logger.py` — `RenderLogger` for recording camera intrinsics/extrinsics and exporting to NeuS (`cameras_sphere.npz`) and NeRF (`transforms.json`) formats, plus preview generation
- `video_utils.py` — MP4/GIF generation from image sequences
- `camera_utils.py` — pure NumPy camera math (FOV → intrinsic matrix, look-at → extrinsic matrix)
- `tool_utils.py` — misc helpers (config conversion, random seed, spherical coordinates, color space conversion)

## Element vs mitsuba_utils Separation

This is a core design principle of the `scene_builder` module.

**Element functions** (`elements/shape.py`, `elements/light.py`) are the user-facing layer. They:
- Read parameters from the YAML config dict
- Resolve file paths via `scene.resolve_source_path()`
- Delegate all Mitsuba object creation to `mitsuba_utils` (imported as `lib`)
- Return a `{node_name: mitsuba_object}` dict
- **Never call `mi.*` directly**

**`mitsuba_utils.py`** is the Mitsuba isolation layer. It contains:
- All `mi.load_dict()` calls (materials, lights, geometry, integrators)
- All `mi.traverse()` calls (parameter manipulation)
- Transform construction (`look_at`, `scale`, `translate`, `rotate`)
- Material builders (`diffuse_color`, `principled_bsdf`, `svbrdf_bsdf`, ...)
- Light builders (`point_light`, `make_environment_map_light`, ...)
- Geometry builders (`make_mesh_shape`, `make_rectangle_shape`, ...)
- AOV integrator wrapping (`wrap_with_aov`)
- Runtime scene manipulation (`set_camera_pose`, `swap_base_color`, ...)

**Why this separation matters:**
- Element functions remain clean config-to-object mappings, easy to read and add
- All Mitsuba API specifics (dict schemas, parameter paths, type names) are in one place
- If Mitsuba's API changes across versions, only `mitsuba_utils.py` needs updating
- New elements can be added by composing existing `mitsuba_utils` functions without learning Mitsuba internals

## Registry System

The framework uses a lightweight registry pattern (`utils/registry.py`) for component discovery.

```python
class Registry:
    def register(self, obj=None, name=None):  # usable as @decorator or direct call
    def get(self, name):                       # lookup by name
```

Four global registries exist:
- `PROJECT_REGISTRY` — rendering workflow classes
- `ELEMENT_REGISTRY` — scene element functions
- `CAMPOSE_REGISTRY` — camera pose generator classes
- `SCENE_REGISTRY` — scene builder classes

### Important: Project registration differs from other components

**Project** uses lazy import in `projects/__init__.py`:

```python
def build_project(conf):
    name = conf['Project']
    importlib.import_module(f'.{name}', package=__name__)
    project_cls = PROJECT_REGISTRY.get(name)
    return project_cls(conf)
```

This means the **file name must exactly match the class name** (e.g., class `MultiView` must be in `MultiView.py`). Adding a new Project always requires creating a new file.

**All other components** (Element, CamPose, Scene) are auto-discovered via directory traversal (`camera_poses/__init__.py` globs all `.py` files) or explicit imports. They can be added to existing files or new files freely.

## Config System

Configs are YAML files loaded by `utils/config_utils.py`. The loader supports an `includes` key for template composition:

```yaml
includes:
  - ../templates/camera_default.yaml
  - ../templates/integrator_path.yaml

Project: MultiView

Scene:
  type: CustomScene
  source_path: ./resources
  element:
    - type: white_mesh
      mesh_filename: cow.ply
      reflectance: 0.6

Camera:
  type: perspective
  fov: 50
  # ...

CamPose:
  type: SphericalGridPose
  theta_range: [45, 65]
  phi_num: 8
  # ...
```

Four top-level keys:
- **`Project`** (str) — looked up in `PROJECT_REGISTRY`
- **`Scene`** (dict) — `type` selects Scene class; `element` is a list of element configs, each with a `type` field looked up in `ELEMENT_REGISTRY`
- **`Camera`** (dict) — passed directly to `mi.load_dict()` as a Mitsuba sensor definition
- **`CamPose`** (dict) — `type` selects CamPose class from `CAMPOSE_REGISTRY`

Templates in `configs/templates/` provide reusable fragments (camera defaults, integrator settings, etc.) that can be included and overridden.

## Adding New Components

### New Element

1. Add a function in `scene_builder/elements/shape.py` (or `light.py`, or a new file)
2. Decorate with `@ELEMENT_REGISTRY.register()`
3. Function signature: `def my_element(scene, conf):`
4. Use `scene.resolve_source_path()` for file paths
5. Call `mitsuba_utils` (imported as `lib`) to build Mitsuba objects — do not call `mi.*` directly
6. Return `{node_name: mitsuba_object}`

Elements can also be defined directly inside a Project file. If a project requires a highly specialized element that is not reusable by other projects, you can register it in the same file:

```python
# projects/MyProject.py
from utils.registry import ELEMENT_REGISTRY, PROJECT_REGISTRY

@ELEMENT_REGISTRY.register()
def my_special_element(scene, conf):
    # project-specific element logic
    ...

@PROJECT_REGISTRY.register()
class MyProject(BaseProject):
    def run(self):
        ...
```

This keeps the element co-located with the only project that uses it, avoiding clutter in the shared `elements/` directory.

### New CamPose

1. Add a class in `camera_poses/` (existing file or new `.py` file — auto-discovered)
2. Decorate with `@CAMPOSE_REGISTRY.register()`
3. In `__init__(self, conf)`: populate `self.pose_list` (list of `{origin, target, up}` dicts) and `self.img_num`

### New Project

1. **Create a new file** in `projects/` — file name must match class name (e.g., `MyProject.py` for class `MyProject`)
2. Decorate with `@PROJECT_REGISTRY.register()`
3. Inherit from `BaseProject`
4. Implement `run()` method with your rendering logic
5. Set `Project: MyProject` in your YAML config
