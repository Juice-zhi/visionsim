import functools
import itertools
import json

import numpy as np
import pytest
import torch

from tests.test_occlusion import CELLS, EDGES, FOG, SKY, box, lamp_map, occlusion_of, rectangle, render_maps
from visionsim.medium import (
    EFFECT_GIVEN,
    EFFECT_GROUPS,
    EFFECT_INPUTS,
    EFFECT_PRESETS,
    Effects,
    Lighting,
    Medium,
    PointLight,
    Sun,
    apply_medium,
    apply_medium_upsampled,
    trace_shadows,
)
from visionsim.medium import halos as halos_module
from visionsim.medium import occlusion as occlusion_module
from visionsim.medium import reflections as reflections_module
from visionsim.medium import render as render_module
from visionsim.medium import surfaces as surfaces_module
from visionsim.medium.halos import HaloGrid
from visionsim.medium.model import HeightFog, Homogeneous
from visionsim.medium.occlusion import sky_cell_directions
from visionsim.medium.render import camera_rays

NAMES = list(Effects.model_fields)
MULTIPLE = FOG.model_copy(update={"multiple_scattering": True})
BOUNDS = [[-2.0, 10.0, 0.0], [2.0, 14.0, 4.0]]
LAMP = PointLight(position=(3.5, 8.0, 1.5), power=(800.0,), radius=0.05)
SUN = Sun(direction=(0.6, -0.3, 0.5), irradiance=(4.0,))
LIGHTING = Lighting(sky=SKY, suns=[SUN], points=[LAMP], ground_albedo=(0.3,) * 3, object_albedo=(0.5,) * 3)
# Halos that also scatter the light that surfaces reflect more than once, which takes three orders, and the light of
# surfaces lit by the glow of the third order, which takes four
GRID = HaloGrid(radii=10, extent=(0.05, 12.0), directions=(6, 8), angles=(12, 6), gathered=(10, 10), orders=4)


def T(x):
    return torch.as_tensor(np.asarray(x, dtype=float), dtype=torch.float64)


def test_switches_declare_their_terms():
    """Every switch is a boolean, on by default, which describes its term, and declares its group and prerequisites
    among the other switches, without cycles, and the conditions of the vocabulary, as its JSON schema exposes them."""
    schema = Effects.model_json_schema()["properties"]
    declared = Effects.prerequisites()
    assert set(NAMES) == set(schema) == set(declared) and not set(NAMES) & set(EFFECT_INPUTS)
    assert set(EFFECT_GIVEN) <= set(EFFECT_INPUTS)
    for name, field in Effects.model_fields.items():
        assert field.annotation in (bool, "bool") and field.default is True
        assert field.description and field.description.endswith(".") and len(field.description) < 400
        assert declared[name]["group"] in EFFECT_GROUPS
        assert set(declared[name]["requires"]) <= set(NAMES) - {name}
        assert set(declared[name]["requires_inputs"]) <= set(EFFECT_INPUTS)
        properties = schema[name]
        assert properties["type"] == "boolean" and properties["default"] is True
        assert properties["description"] == field.description
        for key in ("group", "requires", "requires_inputs"):
            assert properties[key] == declared[name][key]
    json.dumps(Effects.model_json_schema())

    def visit(name, path):
        assert name not in path, f"Cycle of prerequisites: {path + (name,)}"
        for required in declared[name]["requires"]:
            visit(required, path + (name,))

    for name in NAMES:
        visit(name, ())
    assert all(Effects().effective(MULTIPLE, LIGHTING, occlusion=True, normals=True, albedo=True).values())
    assert Effects.model_validate_json('{"halos": false}') == Effects(halos=False)
    with pytest.raises(ValueError):
        Effects(light_shafts=False)


def test_conditions_of_the_inputs():
    """Conditions of the vocabulary, as found from the medium, the lighting and the inputs given."""
    assert list(Effects.conditions(FOG, Lighting())) == list(EFFECT_INPUTS)
    # Inputs left out aren't given
    nothing = Effects.conditions(FOG, Lighting(), **dict.fromkeys(EFFECT_GIVEN, False))
    assert nothing == Effects.conditions(FOG, Lighting()) and not any(nothing[name] for name in EFFECT_GIVEN)
    night = Effects.conditions(FOG, Lighting(points=[LAMP], suns=[Sun(direction=(0, 0.2, -0.1), irradiance=(5.0,))]))
    assert night["lamps"] and night["direct_light"] and not night["suns"] and not night["daylight"]
    assert not night["received"] and not night["reflectors"] and night["scattering"]
    sky = Effects.conditions(FOG, Lighting(sky=SKY, ground_albedo=(0.3,)), occlusion=True)
    assert sky["daylight"] and not sky["suns"] and not sky["direct_light"] and not sky["normals"] and not sky["lamps"]
    # Shadows are traced from shadow maps, which some terms need themselves
    assert sky["occlusion"] and sky["shadows"]
    traced = Effects.conditions(FOG, LIGHTING, shadows=True)
    assert traced["shadows"] and not traced["occlusion"]
    # Objects have the ground's albedo unless they have their own
    assert sky["ground_albedo"] and sky["object_albedo"] and sky["received"] and sky["reflectors"]
    dark = Effects.conditions(FOG, Lighting(ground_albedo=(0.3,), object_albedo=(0.0,)), normals=True)
    assert dark["ground_albedo"] and not dark["object_albedo"] and dark["received"] and dark["normals"]
    assert not dark["measured"] and not dark["albedo"] and not dark["irradiance"]
    # The albedo or the irradiance of each pixel measure the light that surfaces received, but don't tell the albedo with
    # which the ground and objects reflect light
    for given in ("albedo", "irradiance"):
        black = Effects.conditions(FOG, Lighting(points=[LAMP]), normals=True, **{given: True})
        assert black[given] and black["measured"] and black["received"] and not black["reflectors"]
        assert not black["ground_albedo"] and not black["object_albedo"]
        effective = Effects().effective(FOG, Lighting(points=[LAMP]), normals=True, **{given: True})
        assert effective["received_irradiance"] and effective["measured_indirect"]
    assert not Effects().effective(FOG, Lighting(points=[LAMP]), normals=True)["received_irradiance"]
    # Given the mean albedo of the ground and objects only, the light received is estimated, not measured
    estimated = Effects().effective(FOG, LIGHTING, normals=True)
    assert estimated["received_irradiance"] and not estimated["measured_indirect"]
    # Objects that reflect the light of lamps are those that their maps record
    objects = Lighting(points=[LAMP], ground_albedo=(0.0,), object_albedo=(0.5,))
    assert not Effects.conditions(FOG, objects, shadows=True)["reflectors"]
    assert Effects.conditions(FOG, objects, occlusion=True)["reflectors"]
    # Surfaces have the specular layer of a Principled BSDF but with an index of refraction of one
    assert sky["specular"] and not Effects.conditions(FOG, Lighting(surface_ior=1.0))["specular"]
    plain = Effects.conditions(Medium(extinction=0.1, components=[Homogeneous()]), LIGHTING)
    assert not plain["sun_attenuation"] and not plain["multiple_scattering"] and plain["scattering"]
    assert Effects.conditions(MULTIPLE, LIGHTING)["multiple_scattering"]
    assert not Effects.conditions(FOG.model_copy(update={"albedo": 0.0}), LIGHTING)["scattering"]
    with pytest.raises(ValueError, match="Unknown inputs"):
        Effects.conditions(FOG, LIGHTING, suns=True)
    with pytest.raises(ValueError, match="Unknown inputs"):
        Effects().effective(FOG, LIGHTING, maps=True)


# What is given with frames: nothing, shadows traced beforehand, or shadow maps, from which shadows are traced
GIVEN = ({}, {"shadows": True}, {"occlusion": True})
# What measures the light that the surface of each pixel received: nothing, its albedo or its irradiance
MEASURED = ({}, {"albedo": True}, {"irradiance": True})


def scenarios():
    """Media and lightings that hold every combination of the conditions on the inputs, with what is given."""
    plain = Medium(extinction=0.08, components=[HeightFog(falloff=2.5)])
    media = (plain, FOG, MULTIPLE, MULTIPLE.model_copy(update={"albedo": 0.0}))
    suns, skies, lamps = ((), (SUN,)), ((0.0,) * 3, SKY), ((), (LAMP,))
    albedos = (((0.0,), None), ((0.3,), None), ((0.0,), (0.5,)), ((0.3,), (0.0,)))
    for medium, sun, sky, lamp, (ground, objects), given, normals, measured in itertools.product(
        media, suns, skies, lamps, albedos, GIVEN, (False, True), MEASURED
    ):
        lighting = Lighting(sky=sky, suns=list(sun), points=list(lamp), ground_albedo=ground, object_albedo=objects)
        yield medium, lighting, given | measured | {"normals": normals}


def met(name, effects, holds):
    """Whether the prerequisites of a switch are met, by definition: the conditions it requires hold, and the switches it
    requires are on, with their own prerequisites met."""
    declared = Effects.prerequisites()[name]
    inputs = all(holds[condition] for condition in declared["requires_inputs"])
    return inputs and all(getattr(effects, r) and met(r, effects, holds) for r in declared["requires"])


def test_effective_switches():
    """A switch is effective if it is on, the conditions it requires hold, and the switches it requires are effective,
    and what each is missing are the switches it requires that are off and the conditions that don't hold, followed
    recursively, for any switches and inputs."""
    rng = np.random.default_rng(0)
    checked = 0
    for medium, lighting, given in scenarios():
        holds = Effects.conditions(medium, lighting, **given)
        for _ in range(3):
            effects = Effects(**{name: bool(on) for name, on in zip(NAMES, rng.random(len(NAMES)) < 0.8)})
            effective = effects.effective(medium, lighting, **given)
            missing = effects.missing(medium, lighting, **given)
            for name in NAMES:
                assert effective[name] == (getattr(effects, name) and met(name, effects, holds)), name
                assert (not missing[name]) == met(name, effects, holds), name
                assert len(set(missing[name])) == len(missing[name])
                for item in missing[name]:
                    assert (item in EFFECT_INPUTS and not holds[item]) or (item in NAMES and not getattr(effects, item))
            assert effects.resolve(medium, lighting, **given) == Effects(**effective)
            checked += 1
    assert checked > 2000


def test_presets_keep_what_their_switches_require():
    """Presets are described, and every switch on in a preset has the switches it requires on too, so that none is
    silently ineffective."""
    assert set(EFFECT_PRESETS) >= {"complete", "fast"}
    assert all(getattr(EFFECT_PRESETS["complete"][0], name) for name in NAMES)
    required = {name: declared["requires"] for name, declared in Effects.prerequisites().items()}
    for name, (effects, description) in EFFECT_PRESETS.items():
        assert isinstance(effects, Effects) and description, name
        for switch in NAMES:
            if getattr(effects, switch):
                assert all(getattr(effects, needed) for needed in required[switch]), (name, switch)
    assert sum(not getattr(EFFECT_PRESETS["fast"][0], name) for name in NAMES) == 10


def test_missing_prerequisites_are_explained():
    """What each switch is missing names the switches that are off and the conditions that don't hold, in order."""
    everything = {"occlusion": True, "normals": True, "albedo": True}
    assert not any(Effects().missing(MULTIPLE, LIGHTING, **everything).values())
    # The light measured for each pixel only replaces the share of the frame's direct light when it is used at all
    assert Effects().missing(MULTIPLE, LIGHTING, occlusion=True, normals=True)["measured_indirect"] == ["measured"]
    missing = Effects(received_irradiance=False).missing(MULTIPLE, Lighting(points=[LAMP]), normals=True)
    assert missing["measured_indirect"] == ["measured", "received_irradiance", "received"]
    missing = Effects(sky_occlusion=False, glow_shadows=False).missing(MULTIPLE, LIGHTING, **everything)
    assert missing["reflected_daylight"] == missing["ground_shadows"] == missing["scattering_occlusion"]
    assert missing["reflected_daylight"] == ["sky_occlusion"] and missing["glow_shadow_edges"] == ["glow_shadows"]
    assert missing["azimuthal_occlusion"] == missing["scattered_daylight_reflections"] == ["sky_occlusion"]
    assert missing["ground_sky_occlusion"] == ["sky_occlusion"]
    assert missing["sky_occlusion"] == [] and missing["sun_shadows"] == []
    # Conditions are followed through the switches required, whether these are on or off
    night = Lighting(points=[LAMP], ground_albedo=(0.3,))
    missing = Effects(daylight_glow=False).missing(FOG, night)
    assert missing["glow_shadow_edges"] == ["occlusion", "suns", "daylight_glow", "normals", "daylight"]
    assert missing["lamp_glow"] == ["normals", "multiple_scattering"] and missing["reflection_bounces"] == ["occlusion"]
    assert missing["reflection_occlusion"] == missing["reflection_shadows"] == ["occlusion"]
    assert Effects(halos=False).missing(FOG, night)["lamp_glow"] == ["normals", "halos", "multiple_scattering"]
    # Every surface and light bouncing more than once need light to bounce once, which needs the shadow maps, and a
    # specular layer an index of refraction other than one
    missing = Effects(reflection_bounces=False).missing(MULTIPLE, night.model_copy(update={"surface_ior": 1.0}))
    assert missing["bounce_surfaces"] == missing["interreflections"] == ["reflection_bounces", "occlusion"]
    assert missing["specular_reflections"] == ["specular"]
    assert Effects(relit_reflections=False).missing(MULTIPLE, night)["relit_scattering"] == ["relit_reflections"]
    # Shadows traced beforehand serve the shadows of suns and the sky, but not the terms that need the maps themselves
    missing = Effects().missing(MULTIPLE, LIGHTING, shadows=True, normals=True)
    assert not missing["sun_shadows"] and not missing["ground_shadows"] and not missing["lamp_reflections"]
    assert missing["reflected_daylight"] == missing["lamp_shadows"] == missing["surface_shadows"] == ["occlusion"]
    assert missing["scattered_daylight_reflections"] == ["occlusion"] and not missing["azimuthal_occlusion"]
    assert not missing["ground_sky_occlusion"]
    # In a medium that scatters no light, only the terms of surfaces that don't light them through it are left
    absorbing = Effects().effective(MULTIPLE.model_copy(update={"albedo": 0.0}), LIGHTING, **everything)
    assert [name for name, on in absorbing.items() if on] == [
        "surface_shadows",
        "surface_sky_occlusion",
        "received_irradiance",
        "measured_indirect",
    ]


@functools.cache
def scene(width=24, height=16):
    """A box on the ground in front of a camera looking along +y, beside a lamp whose light it shadows, lit by a low sun
    and the sky, with shadow maps of the sun, the sky and the lamp, and the depth and normals of the box and the
    ground."""
    sun = np.asarray(SUN.direction) / np.linalg.norm(SUN.direction)
    first_hit = box(*BOUNDS)
    occlusion = occlusion_of(
        render_maps([sun], BOUNDS, first_hit, 64),
        [sun],
        render_maps(sky_cell_directions(EDGES, CELLS), BOUNDS, first_hit, 64),
        BOUNDS,
    )._replace(lamp_maps=lamp_map(LAMP.position, first_hit))
    camera = {"w": width, "h": height, "fl_x": width / 2, "fl_y": width / 2, "cx": width / 2, "cy": height / 2}
    pose = np.eye(4)
    pose[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    pose[:3, 3] = [0.0, 0.0, 1.6]
    origin, directions, scale = (x.numpy() for x in camera_rays(camera, pose))
    starts = np.broadcast_to(origin, directions.shape)
    on_box = first_hit(starts, directions)
    with np.errstate(divide="ignore"):
        on_ground = np.where(directions[..., 2] < 0, -origin[2] / directions[..., 2], np.inf)
    distance = np.minimum(on_box, on_ground)
    depth = np.where(distance < 1e9, distance / scale, 1e10)
    normals = np.where((on_box < on_ground)[..., None], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0])
    radiance = np.full((height, width, 3), 0.2)
    return occlusion, camera, pose, depth, normals, radiance


@functools.cache
def albedo_of(width=24, height=16):
    """Albedo of the surface seen by each pixel of :func:`scene`: a checkerboard on the ground, around the ground's albedo
    in ``LIGHTING``, and that of objects on the box."""
    _, _, _, depth, normals, _ = scene(width, height)
    checker = np.where((np.arange(width)[None, :] + np.arange(height)[:, None]) % 2 == 0, 0.2, 0.4)
    albedo = np.where(depth < 1e9, np.where(normals[..., 1] < 0, 0.5, checker), 0.0)
    return np.repeat(albedo[..., None], 3, axis=-1)


@pytest.fixture(scope="module")
def small_halos():
    """Halos of a coarse grid, which are much faster to compute."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("visionsim.medium.halos.cached_lamp_halo.__defaults__", (0.0, None, GRID, 0.0))
        yield


def run(lighting=LIGHTING, effects=None, **kwargs):
    occlusion, camera, pose, depth, normals, radiance = scene()
    options = {"occlusion": occlusion, "normals": normals, "albedo": albedo_of()} | kwargs
    return apply_medium(radiance, depth, camera, pose, MULTIPLE, lighting, effects=effects, **options)


# Functions that compute each term, which are spied on: they are all called with every switch on, and the term's are
# never called without its switch, as a switch that is off costs nothing. Surfaces that reflect the light of lamps are
# only relit by halos when lit with the full extinction, i.e. given their albedos
RELIT = (halos_module, "relit_reflection", lambda reflection, *args, **kwargs: reflection.albedos is not None)

WORK = {
    "sun_shadows": [(occlusion_module, "_surely_lit")],
    "sky_occlusion": [(occlusion_module, "_sky_cells")],
    "scattering_occlusion": [(occlusion_module, "_arriving")],
    "azimuthal_occlusion": [(render_module, "cell_profile")],
    "ground_shadows": [(occlusion_module, "_ground_shadowed")],
    "ground_sky_occlusion": [(occlusion_module, "ground_sky"), (occlusion_module, "_ground_unseen")],
    "reflected_daylight": [(render_module, "object_reflections")],
    "scattered_daylight_reflections": [(occlusion_module, "similarity_extinction")],
    "lamp_shadows": [(render_module, "lamp_shadow")],
    "halos": [(render_module, "cached_lamp_halo"), (render_module, "halo_inscatter")],
    "halo_occlusion": [(halos_module, "occupancy")],
    "halo_shadow_edges": [(halos_module, "_refined_axes")],
    "lamp_reflections": [(render_module, "reflection_inscatter"), (halos_module, "lamp_reflection")],
    "reflection_shadows": [(reflections_module, "lamp_visibility")],
    "reflection_occlusion": [(reflections_module, "anchor_maps")],
    "reflection_bounces": [(reflections_module, "_bounced")],
    "bounce_surfaces": [(render_module, "scene_surfels"), (reflections_module, "surface_receivers")],
    "interreflections": [(reflections_module, "_energy")],
    "scattered_reflections": [(halos_module, "arriving_light")],
    "relit_reflections": [RELIT],
    "relit_scattering": [(halos_module, "dark")],
    "specular_reflections": [(reflections_module, "specular_factors")],
    "surface_shadows": [(surfaces_module, "lamp_visibility")],
    "surface_sky_occlusion": [(surfaces_module, "_sky_seen")],
    "daylight_glow": [(surfaces_module, "_sun_glow_integral")],
    "glow_shadows": [(surfaces_module, "_glow_visibility")],
    "glow_shadow_edges": [(surfaces_module, "_glow_visibility")],
    "lamp_glow": [(render_module, "halo_glow")],
    "received_irradiance": [(render_module, "received_irradiance")],
    "measured_indirect": [
        (render_module, "surface_attenuation", lambda *args, **kwargs: kwargs.get("measured") is not None)
    ],
}


def spy(patch, called):
    """Record the calls of the functions of ``WORK`` that do work, by module and name."""
    for module, name, *working in {target for targets in WORK.values() for target in targets}:
        original = getattr(module, name)

        def wrapper(*args, original=original, key=(module.__name__, name), working=working, **kwargs):
            if not working or working[0](*args, **kwargs):
                called.add(key)
            return original(*args, **kwargs)

        patch.setattr(module, name, wrapper)


def spied(name):
    return {(module.__name__, function) for module, function, *_ in WORK[name]}


def clear_caches():
    """Forget what is kept from one frame to the next, so that every term is computed again."""
    halos_module._HALOS.clear()
    reflections_module._cached_lamp_reflection.cache_clear()
    reflections_module.scene_surfels.cache_clear()
    surfaces_module._cached_surface_tables.cache_clear()


@pytest.fixture(scope="module")
def complete(small_halos):
    """The scene with every switch on, and the functions it calls."""
    called: set = set()
    clear_caches()
    with pytest.MonkeyPatch.context() as patch:
        spy(patch, called)
        result = run()
    return result, called


def test_every_term_is_computed_by_default(complete):
    """With every switch on, as by default, every term is computed: switches of new terms should add the functions that
    compute them to ``WORK``, which are then checked not to be called without them."""
    result, called = complete
    assert set().union(*(spied(name) for name in WORK)) <= called
    assert set(WORK) == set(NAMES)
    for effects in (Effects(), Effects().resolve(MULTIPLE, LIGHTING, occlusion=True, normals=True, albedo=True)):
        again = run(effects=effects)
        for field in result._fields:
            assert torch.equal(getattr(again, field), getattr(result, field)), field


@pytest.mark.parametrize("name", NAMES)
def test_each_switch_leaves_out_its_term(name, complete, monkeypatch):
    """Each switch off changes what its term models, and costs nothing: switches of the surfaces leave the medium's light
    alone, those of daylight the light of surfaces, and none changes the transmittance."""
    reference, _ = complete
    called: set = set()
    spy(monkeypatch, called)
    result = run(effects=Effects(**{name: False}))
    assert not spied(name) & called
    assert torch.equal(result.transmittance, reference.transmittance)
    assert result.illumination is not None and reference.illumination is not None
    group = Effects.prerequisites()[name]["group"]
    assert not torch.equal(result.radiance, reference.radiance)
    if group == "surfaces":
        assert torch.equal(result.inscatter, reference.inscatter)
        assert not torch.equal(result.illumination, reference.illumination)
    else:
        assert not torch.equal(result.inscatter, reference.inscatter)
    if group == "daylight":
        assert torch.equal(result.illumination, reference.illumination)


@pytest.mark.parametrize("name", [name for name in NAMES if Effects.prerequisites()[name]["group"] != "surfaces"])
def test_switches_only_change_their_light(name, small_halos):
    """Switches of the light of lamps change nothing by day without lamps, and those of daylight nothing at night (those
    of surfaces, which are lit by both, may change either)."""
    group = Effects.prerequisites()[name]["group"]
    lighting = (
        LIGHTING.model_copy(update={"points": []})
        if group == "lamps"
        else LIGHTING.model_copy(update={"suns": [], "sky": (0.0, 0.0, 0.0)})
    )
    expected, result = run(lighting), run(lighting, Effects(**{name: False}))
    for field in expected._fields:
        assert torch.equal(getattr(result, field), getattr(expected, field)), field


def test_switches_whose_prerequisites_are_off_are_off(small_halos):
    """A switch whose prerequisites aren't met is treated exactly as a switch that is off, and switches of terms whose
    inputs are missing change nothing."""
    for effects, off in (
        (
            Effects(sky_occlusion=False),
            (
                "reflected_daylight",
                "scattered_daylight_reflections",
                "ground_shadows",
                "ground_sky_occlusion",
                "scattering_occlusion",
                "azimuthal_occlusion",
            ),
        ),
        (Effects(scattering_occlusion=False), ("azimuthal_occlusion",)),
        (Effects(reflected_daylight=False), ("scattered_daylight_reflections",)),
        (
            Effects(halos=False),
            ("halo_occlusion", "halo_shadow_edges", "scattered_reflections", "relit_reflections", "lamp_glow"),
        ),
        (Effects(lamp_shadows=False), ("halo_shadow_edges",)),
        (Effects(glow_shadows=False), ("glow_shadow_edges",)),
    ):
        ours = run(effects=effects)
        theirs = run(effects=effects.model_copy(update={name: False for name in off}))
        for field in ours._fields:
            assert torch.equal(getattr(ours, field), getattr(theirs, field)), field
    # Without normals, the switches of surfaces change nothing
    surfaces = {name: False for name, d in Effects.prerequisites().items() if d["group"] == "surfaces"}
    ours, theirs = run(normals=None), run(effects=Effects(**surfaces), normals=None)
    for field in ("radiance", "inscatter", "transmittance"):
        assert torch.equal(getattr(ours, field), getattr(theirs, field)), field


def equal(a, b):
    """Whether two optional tensors are both missing or equal."""
    return a is b or (a is not None and b is not None and torch.equal(a, b))


def inputs_of(case):
    """Medium, lighting and inputs of the scene of the cases of :func:`test_switches_are_effective_where_they_matter`."""
    occlusion, camera, pose, depth, normals, _ = scene()
    medium, lighting, options = MULTIPLE, LIGHTING, {"occlusion": occlusion, "normals": normals}
    if case == "shadows":
        options = {"shadows": trace_shadows(occlusion, depth, camera, pose, MULTIPLE, LIGHTING), "normals": normals}
    elif case == "nothing":
        options = {"normals": normals}
    elif case == "no normals":
        options = {"occlusion": occlusion}
    elif case == "objects":
        # Only objects reflect, which the maps of lamps record, without maps
        lighting, options = LIGHTING.model_copy(update={"ground_albedo": (0.0,), "object_albedo": (0.5,)}), {}
    elif case == "sky":
        lighting = LIGHTING.model_copy(update={"suns": [], "points": []})
    elif case == "absorbing":
        medium = MULTIPLE.model_copy(update={"albedo": 0.0})
    elif case in ("albedo", "irradiance"):
        # Neither the ground nor objects reflect light into the medium, while the light that each pixel's surface
        # received is measured, by its albedo, or its irradiance, less than its direct light where the sun lights it
        lighting = LIGHTING.model_copy(update={"ground_albedo": (0.0,), "object_albedo": None})
        measured = {"albedo": albedo_of(), "irradiance": np.full((*depth.shape, 3), 1.0)}
        options = {"occlusion": occlusion, "normals": normals, case: measured[case]}
    return medium, lighting, options


@pytest.mark.parametrize(
    "case", ["shadows", "nothing", "no normals", "objects", "sky", "absorbing", "albedo", "irradiance"]
)
def test_switches_are_effective_where_they_matter(case, small_halos):
    """The switches that are effective for given inputs each change the frame, and those that aren't change nothing,
    e.g. with shadows traced beforehand but no shadow maps, which some terms need themselves, so that the switches that
    an interface greys out are exactly those that do nothing."""
    _, camera, pose, depth, _, radiance = scene()
    medium, lighting, options = inputs_of(case)
    given = {
        "shadows": "occlusion" in options or "shadows" in options,
        "occlusion": "occlusion" in options,
        "normals": "normals" in options,
        "albedo": "albedo" in options,
        "irradiance": "irradiance" in options,
    }
    effective = Effects().effective(medium, lighting, **given)
    # Surfaces bright enough to have received more light than the share of the frame's direct light it replaces, see
    # `received_irradiance`
    args = (5 * radiance, depth, camera, pose, medium, lighting)
    reference = apply_medium(*args, **options)
    for name in NAMES:
        result = apply_medium(*args, effects=Effects(**{name: False}), **options)
        same = {field: equal(getattr(result, field), getattr(reference, field)) for field in result._fields}
        if effective[name]:
            assert not (same["radiance"] and same["illumination"]), f"{name} is effective but changes nothing"
        else:
            assert all(same.values()), f"{name} isn't effective but changes {[f for f, s in same.items() if not s]}"
    expected = {
        "shadows": {
            "reflected_daylight",
            "scattered_daylight_reflections",
            "lamp_shadows",
            "halo_occlusion",
            "halo_shadow_edges",
            "reflection_shadows",
            "reflection_occlusion",
            "reflection_bounces",
            "bounce_surfaces",
            "interreflections",
            "surface_shadows",
            "surface_sky_occlusion",
            "glow_shadows",
            "glow_shadow_edges",
        },
        "objects": {name for name, d in Effects.prerequisites().items() if d["group"] == "surfaces"},
        # Without suns, nor lamps, nothing casts shadows but onto the light of the sky
        "sky": {name for name, d in Effects.prerequisites().items() if d["group"] == "lamps"}
        | {
            "sun_shadows",
            "ground_shadows",
            "azimuthal_occlusion",
            "surface_shadows",
            "glow_shadows",
            "glow_shadow_edges",
        },
    }
    if case in expected:
        assert expected[case] <= {name for name, on in effective.items() if not on}
    if case == "objects":
        assert not effective["lamp_reflections"]
    if case in ("albedo", "irradiance"):
        assert effective["received_irradiance"] and effective["measured_indirect"] and not effective["lamp_reflections"]
    else:
        assert not effective["measured_indirect"]


def test_upsampled_frames_follow_the_switches(small_halos):
    """Frames upsampled from a lower resolution leave out the same terms, and are those of every pixel without it."""
    occlusion, camera, pose, depth, normals, radiance = scene()
    args = (radiance, depth, camera, pose, MULTIPLE, LIGHTING)
    options = {"occlusion": occlusion, "normals": normals, "albedo": albedo_of()}
    for effects in (Effects(halos=False, reflected_daylight=False), Effects(measured_indirect=False)):
        upsampled = apply_medium_upsampled(*args, factor=2, effects=effects, **options)
        default = apply_medium_upsampled(*args, factor=2, **options)
        assert not torch.equal(upsampled.radiance, default.radiance)
        every = apply_medium_upsampled(*args, factor=1, effects=effects, **options)
        assert torch.equal(every.radiance, run(effects=effects).radiance)


def test_reused_shadows_follow_the_switches():
    """Shadows traced with every term serve frames with fewer, and shadows traced without a term leave it out."""
    occlusion, camera, pose, depth, _, radiance = scene()
    lighting = LIGHTING.model_copy(update={"points": []})
    shadows = trace_shadows(occlusion, depth, camera, pose, MULTIPLE, lighting)
    for effects in (
        Effects(sun_shadows=False),
        Effects(sky_occlusion=False),
        Effects(ground_shadows=False),
        Effects(ground_sky_occlusion=False),
    ):
        traced = apply_medium(radiance, depth, camera, pose, MULTIPLE, lighting, occlusion=occlusion, effects=effects)
        options = {"occlusion": occlusion, "shadows": shadows, "effects": effects}
        reused = apply_medium(radiance, depth, camera, pose, MULTIPLE, lighting, **options)
        assert torch.allclose(reused.radiance, traced.radiance, rtol=1e-10, atol=1e-12)
        lean = trace_shadows(occlusion, depth, camera, pose, MULTIPLE, lighting, effects=effects)
        options["shadows"] = lean
        borrowed = apply_medium(radiance, depth, camera, pose, MULTIPLE, lighting, **options)
        assert torch.allclose(borrowed.radiance, traced.radiance, rtol=1e-10, atol=1e-12)
    lean = trace_shadows(occlusion, depth, camera, pose, MULTIPLE, lighting, effects=Effects(sky_occlusion=False))
    assert lean.middles.numel() == 0 and len(lean.suns) == 1 and len(lean.suns[0].rays)
    lean = trace_shadows(occlusion, depth, camera, pose, MULTIPLE, lighting, effects=Effects(sun_shadows=False))
    assert lean.middles.numel() and len(lean.suns) == 1 and not len(lean.suns[0].rays)
    assert lean.shadowed_weights_below is not None and lean.unseen_weights_below is not None
    lean = trace_shadows(occlusion, depth, camera, pose, MULTIPLE, lighting, effects=Effects(ground_sky_occlusion=False))
    assert lean.unseen_weights_below is None and lean.shadowed_weights_below is not None


def test_daylight_glow_on_surfaces_can_be_left_out(monkeypatch):
    """Without the glow, surfaces receive the light of suns, the sky and the ground attenuated by the fog, and the glow's
    tables aren't computed."""
    beta = T([FOG.extinction] * 3)
    lighting = LIGHTING.model_copy(update={"points": []})
    points, normals = T([[0.0, 0.0, 0.0], [0.0, 5.0, 2.0]]), T([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    clear, glowing = surfaces_module.surface_irradiance(MULTIPLE, lighting, beta, points, normals)
    called: set = set()
    surfaces_module._cached_surface_tables.cache_clear()
    spy(monkeypatch, called)
    effects = Effects(daylight_glow=False)
    unlit, dark = surfaces_module.surface_irradiance(MULTIPLE, lighting, beta, points, normals, effects=effects)
    assert not spied("daylight_glow") & called
    assert torch.equal(unlit, clear) and torch.all(dark < glowing)


def test_cli_apply_with_effects(tmp_path):
    """``medium.apply`` reads the switches from a JSON file, and saves them with the frames."""
    from tests.test_medium import make_render
    from tests.test_occlusion import save_occlusion
    from visionsim.cli.medium import apply
    from visionsim.dataset import Dataset
    from visionsim.medium.occlusion import load_occlusion

    lighting = make_render(tmp_path / "render", n=1)
    sun = np.array(lighting.suns[0].direction) / np.linalg.norm(lighting.suns[0].direction)
    bounds = [[-50.0, -50.0, 6.0], [50.0, 50.0, 6.0]]
    roof = rectangle(6.0, (-50, 50), (-50, 50))
    sky = render_maps(sky_cell_directions(EDGES, CELLS), bounds, roof, 16)
    save_occlusion(tmp_path / "render" / "occlusion.npz", render_maps([sun], bounds, roof, 64), [sun], sky, bounds)
    medium = Medium(extinction=0.1, components=[HeightFog(density=1.0, falloff=5.0)], sun_attenuation=True)
    (tmp_path / "medium.json").write_text(medium.model_dump_json())
    effects = Effects(sun_shadows=False)
    (tmp_path / "effects.json").write_text(effects.model_dump_json(exclude_defaults=True))
    apply(
        tmp_path / "render", tmp_path / "fog", tmp_path / "medium.json", effects=tmp_path / "effects.json", device="cpu"
    )

    (radiance, transform), (depth, _) = (
        Dataset.from_path(tmp_path / "render" / name)[0] for name in ("frames", "depths")
    )
    occlusion = load_occlusion(tmp_path / "render" / "occlusion.npz", dtype=torch.float64)
    pose = transform["transform_matrix"]
    args = (radiance, depth, transform, pose, medium, lighting)
    expected = apply_medium(*args, occlusion=occlusion, effects=effects)
    shafts = apply_medium(*args, occlusion=occlusion)
    frame, _ = Dataset.from_path(tmp_path / "fog" / "frames")[0]
    assert np.allclose(frame, expected.radiance.numpy(), rtol=1e-6, atol=1e-7)
    assert not np.allclose(frame, shafts.radiance.numpy(), rtol=1e-3)
    assert Effects.model_validate_json((tmp_path / "fog" / "effects.json").read_text()) == effects
