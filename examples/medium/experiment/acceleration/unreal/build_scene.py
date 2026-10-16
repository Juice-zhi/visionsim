"""Build the acceleration study's fog scenes in an Unreal Engine project, from ``scene.json`` (``export_scene.py``).

Run inside the editor, e.g.::

    UnrealEditor-Cmd.exe FogBench.uproject -run=pythonscript -script="build_scene.py" -unattended -nosplash

with the environment variable ``FOG_SCENE`` set to the folder of ``scene.json`` and ``sky.hdr``. Creates, under
``/Game/Fog``: materials, the sky's cubemap, the levels ``Day`` (sun and sky) and ``Night`` (the point light behind
the green cube), each with Unreal's exponential height fog and its volumetric fog matched to visionsim's fog, a level
sequence moving a cine camera along the frames of the animation, and Movie Render Queue presets rendering linear EXR
frames with the volumetric fog at Unreal's High, Epic and Cinematic quality, and without fog.
"""

import json
import os

import unreal

FOLDER = os.environ["FOG_SCENE"]
SCENE = json.load(open(os.path.join(FOLDER, "scene.json")))
ROOT = "/Game/Fog"
tools = unreal.AssetToolsHelpers.get_asset_tools()
library = unreal.EditorAssetLibrary
materials = unreal.MaterialEditingLibrary
levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
QUALITY = {  # Unreal's scalability levels of volumetric fog (BaseScalability.ini)
    "high": {
        "r.VolumetricFog.GridPixelSize": 16,
        "r.VolumetricFog.GridSizeZ": 64,
        "r.VolumetricFog.HistoryMissSupersampleCount": 4,
    },
    "epic": {
        "r.VolumetricFog.GridPixelSize": 8,
        "r.VolumetricFog.GridSizeZ": 128,
        "r.VolumetricFog.HistoryMissSupersampleCount": 4,
    },
    "cinematic": {
        "r.VolumetricFog.GridPixelSize": 4,
        "r.VolumetricFog.GridSizeZ": 128,
        "r.VolumetricFog.HistoryMissSupersampleCount": 16,
    },
}


def log(*args):
    unreal.log("FOGBENCH " + " ".join(str(a) for a in args))


def fresh(name: str, cls, factory):
    path = f"{ROOT}/{name}"
    if library.does_asset_exist(path):
        library.delete_asset(path)
    return tools.create_asset(name, ROOT, cls, factory)


def expression(material, cls, x, y, **properties):
    node = materials.create_material_expression(material, cls, x, y)
    for key, value in properties.items():
        node.set_editor_property(key, value)
    return node


def build_materials() -> dict:
    lit = fresh("M_Lit", unreal.Material, unreal.MaterialFactoryNew())
    color = expression(lit, unreal.MaterialExpressionVectorParameter, -400, 0, parameter_name="Color")
    materials.connect_material_property(color, "", unreal.MaterialProperty.MP_BASE_COLOR)
    rough = expression(lit, unreal.MaterialExpressionConstant, -400, 200, r=0.8)
    materials.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
    materials.recompile_material(lit)

    # Checker of 1 m squares, from a 2x2 texture of linear values tiled once per 2 m, without filtering
    task = unreal.AssetImportTask()
    for key, value in {
        "filename": os.path.join(FOLDER, "checker.png"),
        "destination_path": ROOT,
        "destination_name": "T_Checker",
        "automated": True,
        "replace_existing": True,
        "save": True,
    }.items():
        task.set_editor_property(key, value)
    tools.import_asset_tasks([task])
    checker = library.load_asset(f"{ROOT}/T_Checker")
    checker.set_editor_property("srgb", False)
    checker.set_editor_property("filter", unreal.TextureFilter.TF_NEAREST)
    checker.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_VECTOR_DISPLACEMENTMAP)
    checker.set_editor_property("mip_gen_settings", unreal.TextureMipGenSettings.TMGS_NO_MIPMAPS)
    library.save_loaded_asset(checker, only_if_is_dirty=False)
    ground = fresh("M_Ground", unreal.Material, unreal.MaterialFactoryNew())
    tiles = SCENE["ground"]["scale"] / 2
    coordinates = expression(ground, unreal.MaterialExpressionTextureCoordinate, -600, 0, u_tiling=tiles, v_tiling=tiles)
    sample = expression(
        ground,
        unreal.MaterialExpressionTextureSample,
        -300,
        0,
        texture=checker,
        sampler_type=unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR,
    )
    materials.connect_material_expressions(coordinates, "", sample, "UVs")
    materials.connect_material_property(sample, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)
    rough = expression(ground, unreal.MaterialExpressionConstant, 0, 300, r=0.8)
    materials.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
    materials.recompile_material(ground)

    sky = fresh("M_Sky", unreal.Material, unreal.MaterialFactoryNew())
    sky.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    sky.set_editor_property("two_sided", True)
    emissive = expression(sky, unreal.MaterialExpressionVectorParameter, -400, 0, parameter_name="Color")
    materials.connect_material_property(emissive, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    materials.recompile_material(sky)
    for material in (lit, ground, sky):
        library.save_loaded_asset(material, only_if_is_dirty=False)

    result = {"ground": ground}
    for k, obj in enumerate(SCENE["objects"]):
        key = tuple(obj["color"])
        if key not in result:
            instance = fresh(
                f"MI_Color{len(result)}", unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew()
            )
            materials.set_material_instance_parent(instance, lit)
            materials.set_material_instance_vector_parameter_value(instance, "Color", unreal.LinearColor(*key, 1.0))
            library.save_loaded_asset(instance, only_if_is_dirty=False)
            result[key] = instance
    # The sky is black at night, as Cycles' night scenes turn their world off
    for name, color in (("Day", SCENE["sky"]["color"]), ("Night", (0.0, 0.0, 0.0))):
        instance = fresh(f"MI_Sky{name}", unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
        materials.set_material_instance_parent(instance, sky)
        materials.set_material_instance_vector_parameter_value(instance, "Color", unreal.LinearColor(*color, 1.0))
        library.save_loaded_asset(instance, only_if_is_dirty=False)
        result[f"sky_{name}"] = instance
    return result


def import_sky():
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", os.path.join(FOLDER, "sky.hdr"))
    task.set_editor_property("destination_path", ROOT)
    task.set_editor_property("destination_name", "T_Sky")
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", True)
    tools.import_asset_tasks([task])
    texture = library.load_asset(f"{ROOT}/T_Sky")
    log("sky texture", type(texture).__name__)
    return texture


def spawn(cls, location=(0.0, 0.0, 0.0)):
    return actors.spawn_actor_from_class(cls, unreal.Vector(*location), unreal.Rotator(0.0, 0.0, 0.0))


def mesh(name: str, location, scale, material, shadows=True):
    actor = spawn(unreal.StaticMeshActor, location)
    component = actor.static_mesh_component
    component.set_static_mesh(library.load_asset(f"/Engine/BasicShapes/{name}.{name}"))
    component.set_material(0, material)
    component.set_editor_property("cast_shadow", shadows)
    actor.set_actor_scale3d(unreal.Vector(*scale))
    return actor


def rotation(forward, up=None):
    f = unreal.Vector(*forward)
    if up is None:
        return unreal.MathLibrary.make_rot_from_x(f)
    return unreal.MathLibrary.make_rot_from_xz(f, unreal.Vector(*up))


def build_level(name: str, mats: dict, night: bool):
    path = f"{ROOT}/{name}"
    if library.does_asset_exist(path):
        library.delete_asset(path)
    levels.new_level(path)
    mesh("Plane", (0.0, 0.0, 0.0), (SCENE["ground"]["scale"],) * 2 + (1.0,), mats["ground"])
    for obj in SCENE["objects"]:
        mesh(obj["mesh"], obj["location"], obj["scale"], mats[tuple(obj["color"])])
    sphere = mesh("Sphere", (0.0, 0.0, 0.0), (SCENE["sky_sphere_cm"] / 50.0,) * 3, mats[f"sky_{name}"], shadows=False)
    sphere.static_mesh_component.set_editor_property("affect_dynamic_indirect_lighting", False)
    sphere.static_mesh_component.set_editor_property("affect_distance_field_lighting", False)

    if night:
        lamp = spawn(unreal.PointLight, SCENE["point"]["location"])
        component = lamp.point_light_component
        component.set_editor_property("intensity_units", unreal.LightUnits.CANDELAS)
        component.set_editor_property("intensity", SCENE["point"]["candela"])
        component.set_editor_property("source_radius", SCENE["point"]["radius_cm"])
        component.set_editor_property("attenuation_radius", 20000.0)
        component.set_editor_property("use_inverse_squared_falloff", True)
        component.set_editor_property("cast_volumetric_shadow", True)
        component.set_editor_property("volumetric_scattering_intensity", 1.0)
    else:
        sun = spawn(unreal.DirectionalLight)
        sun.set_actor_rotation(rotation([-c for c in SCENE["sun"]["towards"]]), False)
        component = sun.light_component
        component.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
        component.set_editor_property("intensity", SCENE["sun"]["lux"])
        component.set_editor_property("light_source_angle", SCENE["sun"]["angle_deg"])
        component.set_editor_property("cast_volumetric_shadow", True)
        component.set_editor_property("volumetric_scattering_intensity", 1.0)
        sky = spawn(unreal.SkyLight)
        component = sky.light_component
        component.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
        component.set_editor_property("source_type", unreal.SkyLightSourceType.SLS_SPECIFIED_CUBEMAP)
        component.set_editor_property("cubemap", mats["cubemap"])
        component.set_editor_property("lower_hemisphere_is_black", True)
        component.set_editor_property("intensity", 1.0)
        component.set_editor_property("volumetric_scattering_intensity", 1.0)
        component.recapture_sky()

    fog = spawn(unreal.ExponentialHeightFog, (0.0, 0.0, 0.0))
    component = fog.get_editor_property("component")
    for key, value in {
        "fog_density": SCENE["fog"]["density"],
        "fog_height_falloff": SCENE["fog"]["falloff"],
        "fog_max_opacity": 1.0,
        "start_distance": 0.0,
        "fog_inscattering_luminance": unreal.LinearColor(0, 0, 0, 1),
        "directional_inscattering_luminance": unreal.LinearColor(0, 0, 0, 1),
        "enable_volumetric_fog": True,
        "volumetric_fog_scattering_distribution": SCENE["fog"]["g"],
        "volumetric_fog_albedo": unreal.Color(255, 255, 255, 255),
        "volumetric_fog_extinction_scale": 1.0,
        "volumetric_fog_distance": SCENE["fog"]["distance_cm"],
        "volumetric_fog_start_distance": 0.0,
    }.items():
        try:
            component.set_editor_property(key, value)
        except Exception as error:  # report properties that differ in this version of Unreal
            log("fog property", key, "failed:", error)

    volume = spawn(unreal.PostProcessVolume)
    volume.set_editor_property("unbound", True)
    settings = volume.get_editor_property("settings")
    for key, value in {
        "auto_exposure_method": unreal.AutoExposureMethod.AEM_MANUAL,
        "auto_exposure_bias": 0.0,
        "auto_exposure_apply_physical_camera_exposure": False,
        "bloom_intensity": 0.0,
        "vignette_intensity": 0.0,
        "film_grain_intensity": 0.0,
        "lens_flare_intensity": 0.0,
        "motion_blur_amount": 0.0,
        "scene_fringe_intensity": 0.0,
        "local_exposure_highlight_contrast_scale": 1.0,
        "local_exposure_shadow_contrast_scale": 1.0,
    }.items():
        try:
            settings.set_editor_property("override_" + key, True)
            settings.set_editor_property(key, value)
        except Exception as error:
            log("post process", key, "failed:", error)
    volume.set_editor_property("settings", settings)

    # A cine camera keyed along the animation, and a camera at frame 345 that the game views through, for timing
    camera = spawn(unreal.CineCameraActor)
    component = camera.get_cine_camera_component()
    filmback = component.get_editor_property("filmback")
    filmback.set_editor_property("sensor_width", SCENE["sensor_mm"][0])
    filmback.set_editor_property("sensor_height", SCENE["sensor_mm"][1])
    component.set_editor_property("filmback", filmback)
    component.set_editor_property("current_focal_length", SCENE["focal_mm"])
    focus = component.get_editor_property("focus_settings")
    focus.set_editor_property("focus_method", unreal.CameraFocusMethod.DISABLE)
    component.set_editor_property("focus_settings", focus)
    still = SCENE["frames"]["345"]
    game_camera = spawn(unreal.CineCameraActor, still["location"])
    game_camera.set_actor_rotation(rotation(still["forward"], still["up"]), False)
    game_component = game_camera.get_cine_camera_component()
    game_component.set_editor_property("filmback", filmback)
    game_component.set_editor_property("current_focal_length", SCENE["focal_mm"])
    game_component.set_editor_property("focus_settings", focus)
    game_camera.set_editor_property("auto_activate_for_player", unreal.AutoReceiveInput.PLAYER0)
    build_sequence(f"Seq_{name}", camera)
    levels.save_current_level()
    log("level", name, "saved")


def build_sequence(name: str, camera):
    sequence = fresh(name, unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
    frames = sorted(int(f) for f in SCENE["frames"])
    sequence.set_display_rate(unreal.FrameRate(25, 1))
    sequence.set_playback_start(frames[0])
    sequence.set_playback_end(frames[-1] + 1)
    binding = sequence.add_possessable(camera)
    track = binding.add_track(unreal.MovieScene3DTransformTrack)
    section = track.add_section()
    section.set_range(frames[0], frames[-1] + 1)
    channels = section.get_all_channels()
    for frame in frames:
        pose = SCENE["frames"][str(frame)]
        rot = rotation(pose["forward"], pose["up"])
        values = list(pose["location"]) + [rot.roll, rot.pitch, rot.yaw]
        for channel, value in zip(channels[:6], values):
            channel.add_key(unreal.FrameNumber(frame), value, interpolation=unreal.MovieSceneKeyInterpolation.LINEAR)
    cuts = sequence.add_track(unreal.MovieSceneCameraCutTrack)
    cut = cuts.add_section()
    cut.set_range(frames[0], frames[-1] + 1)
    cut.set_camera_binding_id(sequence.get_binding_id(binding))
    library.save_loaded_asset(sequence)
    log("sequence", name, "saved with", len(frames), "frames")
    return sequence


def build_preset(name: str, output: str, cvars: dict, resolution, temporal_samples: int = 1):
    preset = fresh(name, unreal.MoviePipelinePrimaryConfig, unreal.MoviePipelinePrimaryConfigFactory())
    out = preset.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    out.set_editor_property("output_resolution", unreal.IntPoint(*resolution))
    out.set_editor_property("output_directory", unreal.DirectoryPath(output))
    out.set_editor_property("file_name_format", "{frame_number}")
    out.set_editor_property("zero_pad_frame_numbers", 4)
    preset.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    preset.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_EXR)
    color = preset.find_or_add_setting_by_class(unreal.MoviePipelineColorSetting)
    color.set_editor_property("disable_tone_curve", True)
    aa = preset.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa.set_editor_property("temporal_sample_count", temporal_samples)
    aa.set_editor_property("spatial_sample_count", 1)
    aa.set_editor_property("engine_warm_up_count", 16)
    console = preset.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
    for key, value in cvars.items():
        console.add_or_update_console_variable(key, float(value))
    library.save_loaded_asset(preset)
    log("preset", name, "saved")


def main():
    if not library.does_directory_exist(ROOT):
        library.make_directory(ROOT)
    mats = build_materials()
    mats["cubemap"] = import_sky()
    for name in ("Day", "Night"):
        build_level(name, mats, night=name == "Night")
    renders = os.path.join(FOLDER, "renders")
    for quality, cvars in QUALITY.items():
        cvars = {"r.VolumetricFog": 1, **cvars}
        build_preset(f"P_{quality}", os.path.join(renders, "{level_name}_" + quality), cvars, SCENE["resolution"])
    build_preset(
        "P_nofog", os.path.join(renders, "{level_name}_nofog"), {"r.Fog": 0, "r.VolumetricFog": 0}, SCENE["resolution"]
    )
    # Materials, their instances and every other asset created above
    library.save_directory(ROOT, only_if_is_dirty=False, recursive=True)
    log("done", sorted(library.list_assets(ROOT, recursive=True)))


main()
