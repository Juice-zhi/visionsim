import math

import numpy as np
import pytest
import torch

from visionsim.medium import Lighting, Medium, apply_medium
from visionsim.medium.model import AreaLight, HeightFog, Homogeneous, PointLight, SpotLight
from visionsim.medium.occlusion import ShadowMaps
from visionsim.medium.optics import density, henyey_greenstein, optical_depth, sky_quadrature
from visionsim.medium.reflections import (
    MIN_ROUGHNESS,
    REFLECTION_GRID,
    Patches,
    SpecularLayer,
    _bounced,
    _cell_density,
    _fresnel,
    _same_channels,
    _texture_cells,
    anchor_maps,
    anchor_visibility,
    dark,
    gather,
    ground_nodes,
    ground_texture,
    lamp_reflection,
    lit_surfaces,
    map_points,
    phase_directions,
    reflection_inscatter,
    relit_reflection,
    scene_surfaces,
    scene_surfels,
    specular_albedo,
    specular_factors,
    surface_patches,
    surface_receivers,
)
from visionsim.medium.surfaces import lamp_irradiance

FOG = Medium(extinction=0.06, anisotropy=0.7, albedo=0.9, components=[Homogeneous()])
LAMP = PointLight(position=(0.0, 0.0, 2.0), power=(1000.0,), radius=0.05)
UP = (0.0, 0.0, 1.0)


def T(x):
    return torch.as_tensor(np.asarray(x, dtype=float), dtype=torch.float64)


def box(low, high):
    """First hit of rays on an axis-aligned box, or a huge distance if they miss it."""
    low, high = T(low), T(high)

    def first_hit(start, directions):
        safe = torch.where(directions == 0, torch.full_like(directions, 1e-12), directions)
        t1, t2 = (low - start) / safe, (high - start) / safe
        near, far = torch.minimum(t1, t2).amax(dim=-1), torch.maximum(t1, t2).amin(dim=-1)
        return torch.where((far >= near) & (far > 0), near.clamp_min(0), torch.full_like(near, 1e10))

    return first_hit


def equirectangular(origin, first_hit, height=256, width=512):
    """Map of a lamp at a position, laid out as `lamp_visibility` reads it, of the first hit in every direction."""
    v = (torch.arange(height, dtype=torch.float64) + 0.5) / height
    u = (torch.arange(width, dtype=torch.float64) + 0.5) / width
    z = torch.cos((1 - v) * math.pi)[:, None].expand(height, width)
    phi = (math.pi - 2 * math.pi * u)[None, :].expand(height, width)
    s = (1 - z * z).sqrt()
    directions = torch.stack([s * torch.cos(phi), s * torch.sin(phi), z], dim=-1)
    depths = first_hit(T(origin).expand_as(directions), directions)
    return ShadowMaps(
        depths=depths.reshape(-1),
        offsets=torch.zeros(1, dtype=torch.long),
        shapes=torch.tensor([[height, width]]),
        origins=T(origin)[None],
        axes=torch.eye(3, dtype=torch.float64)[None],
        texels=T([2 * math.pi / width]),
    )


def fibonacci(count):
    i = torch.arange(count, dtype=torch.float64) + 0.5
    z = 1 - 2 * i / count
    phi = math.pi * (3 - math.sqrt(5)) * i
    s = (1 - z * z).sqrt()
    return torch.stack([s * torch.cos(phi), s * torch.sin(phi), z], dim=-1)


@pytest.mark.parametrize("below", [False, True])
def test_phase_directions(below):
    views = T([[0.0, 1.0, 0.0], [0.6, 0.0, -0.8], [0.0, 0.6, 0.8], [0.0, 0.0, 1.0]])
    directions, weights = phase_directions(views, 0.8, 8, 12, below=below)
    assert torch.allclose(directions.norm(dim=-1), torch.ones_like(weights))
    # Above the horizon, the nodes are those of the sky's quadrature, below it, those of the mirrored views
    elevations, sky_weights = sky_quadrature(views * T([1.0, 1.0, -1.0 if below else 1.0]), 0.8, 8, 12)
    used = weights > 0
    assert torch.allclose(((-1 if below else 1) * directions[..., 2])[used], elevations[used], atol=1e-9)
    assert torch.allclose(weights, sky_weights)
    # Integrals of a function of the direction, against many directions of the sphere
    many = fibonacci(400000)
    many = many[(many[:, 2] < 0) if below else (many[:, 2] > 0)]

    def f(u):
        return 1 + u[..., 0] - 0.5 * u[..., 1] * u[..., 2]

    for view, nodes, w in zip(views, directions, weights):
        expected = (henyey_greenstein(many @ view, 0.8) * f(many)).sum() * 4 * math.pi / 400000
        assert float((w * f(nodes)).sum()) == pytest.approx(float(expected), rel=2e-3, abs=1e-4)


def test_ground_nodes_integrate_the_ground():
    beta = T([FOG.extinction])
    center = T([0.0, 0.0, 0.0])
    texture = ground_texture(FOG, LAMP, beta, center, T([0.5]))
    nodes, cells = ground_nodes(texture, center, 2.0)
    assert len(nodes.areas) == REFLECTION_GRID.nodes[0] * REFLECTION_GRID.nodes[1]
    # The density of every node is that of its cell, against which other quadratures are weighed
    offset = nodes.positions[:, :2] - center[:2]
    found = _cell_density(cells, offset.norm(dim=-1), torch.atan2(offset[:, 1], offset[:, 0]), REFLECTION_GRID.extent)
    assert torch.allclose(found, nodes.density, rtol=1e-9)
    # The ground's radiance integrated over a fine polar grid, and against a smooth weight
    radii = torch.logspace(-3, math.log10(40), 4000, dtype=torch.float64)
    azimuths = torch.arange(256, dtype=torch.float64) / 256 * 2 * math.pi
    r, a = torch.meshgrid(radii, azimuths, indexing="ij")
    points = torch.stack([r * torch.cos(a), r * torch.sin(a), torch.zeros_like(r)], dim=-1).reshape(-1, 3)
    radiance = 0.5 / math.pi * lamp_irradiance(FOG, LAMP, beta, points, T(UP))[1][:, 0].reshape(r.shape)
    areas = (torch.diff(radii, prepend=radii.new_zeros(1)) * radii)[:, None] * 2 * math.pi / 256
    exact = 0.5 / math.pi * lamp_irradiance(FOG, LAMP, beta, nodes.positions, T(UP))[1][:, 0]
    assert float((nodes.areas * exact).sum()) == pytest.approx(float((radiance * areas).sum()), rel=0.02)

    def weight(x):
        return (1 + 0.5 * torch.tanh(x[..., 0])) / (1 + (x[..., :2].norm(dim=-1) / 3) ** 2)

    ours = (nodes.areas * exact * weight(nodes.positions)).sum()
    expected = (radiance * areas * weight(points).reshape(r.shape)).sum()
    assert float(ours) == pytest.approx(float(expected), rel=0.03)


def brute_force(medium, lamp, beta, albedo, origin, directions, distance, steps=120, n_mu=64, n_phi=128, specular=None):
    """Light of a lamp reflected by a Lambertian ground, or with a specular layer of given roughness and index of
    refraction, and scattered once towards the origin of rays, integrated with many nodes along rays, and many
    directions from each of them."""
    t, w = (T(a) for a in np.polynomial.legendre.leggauss(n_mu))
    t, w = (t + 1) / 2, w / 2
    mu, mu_weights = t**2, 2 * t * w
    phi = (torch.arange(n_phi, dtype=torch.float64) + 0.5) / n_phi * 2 * math.pi
    m, p = torch.meshgrid(mu, phi, indexing="ij")
    s = (1 - m * m).sqrt()
    down = torch.stack([s * torch.cos(p), s * torch.sin(p), -m], dim=-1).reshape(-1, 3)
    solid = (mu_weights[:, None] * torch.full((n_phi,), 2 * math.pi / n_phi, dtype=torch.float64)).reshape(-1)
    result = []
    for d, length in zip(directions, distance):
        # Midpoints, denser near the end of the ray
        u = (torch.arange(steps, dtype=torch.float64) + 0.5) / steps
        along = length * (1 - (1 - u) ** 2)
        ds = length * 2 * (1 - u) / steps
        light = 0.0
        for a, step in zip(along, ds):
            x = origin + a * d
            reach = x[2] / m.reshape(-1)
            hits = x + reach[:, None] * down
            radiance = albedo / math.pi * lamp_irradiance(medium, lamp, beta, hits, T(UP))[1][:, 0]
            if specular is not None:
                # All of the ground's light comes straight from the lamp
                layer = SpecularLayer(*specular, T(lamp.position), (T([albedo]), T([albedo])), *(T([]),) * 3)
                to_light = T(lamp.position) - hits
                to_light = to_light / to_light.norm(dim=-1, keepdim=True)
                total, part = specular_factors(layer, T([albedo]), T(UP), to_light, -down)
                radiance = radiance * (total + part)[:, 0]
            arriving = radiance * torch.exp(-optical_depth(medium, x, down, reach) * beta[0])
            source = medium.albedo * (henyey_greenstein(down @ d, medium.anisotropy) * arriving * solid).sum()
            sigma = density(medium, x[None])[0] * beta[0]
            light += float(sigma * torch.exp(-optical_depth(medium, origin, d, a) * beta[0]) * source * step)
        result.append(light)
    return T(result)


HEIGHT_FOG = FOG.model_copy(update={"components": [HeightFog(falloff=2.5)], "sun_attenuation": True})


@pytest.mark.parametrize(
    "medium, lamp",
    [
        (FOG, LAMP),
        (HEIGHT_FOG, LAMP),
        # A street lamp's spot, and a panel, both pointing down
        (HEIGHT_FOG, SpotLight(position=(0.0, 0.0, 3.0), direction=(0.2, 0.0, -1.0), power=(1000.0,), angle=1.4)),
        (FOG, AreaLight(position=(0.0, 0.0, 3.0), direction=(0, 0, -1), axis_u=(1, 0, 0), power=(1e3,), size=(1, 1))),
    ],
)
def test_reflected_light_matches_direct_integration(medium, lamp):
    beta = T([medium.extinction])
    reflection = lamp_reflection(medium, lamp, beta, ground_albedo=(0.4,))
    origin = T([0.5, -9.0, 1.5])
    # Down to the ground, along it, and up, and down to the lamp's foot
    directions = T([[0.0, 1.0, -0.05], [-0.1, 1.0, 0.02], [0.2, 1.0, -0.12], [0.05, 1.0, 0.25], [-0.5, 9.0, -1.5]])
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distance = torch.where(directions[:, 2] < 0, -1.5 / directions[:, 2], torch.full_like(directions[:, 2], 25.0))
    ours = reflection_inscatter(reflection, medium, origin, directions, distance, beta)[:, 0]
    expected = brute_force(medium, lamp, beta, 0.4, origin, directions, distance)
    # Within 3% for these rays, the error of the grids and quadratures
    assert torch.allclose(ours, expected, rtol=0.05)


@pytest.mark.parametrize("roughness, tolerance", [(0.2, 0.08), (0.5, 0.05), (0.8, 0.05)])
def test_specular_reflected_light_matches_direct_integration(roughness, tolerance):
    """The light of a ground with a specular layer, which reflects the lamp's light around the mirror direction, as
    the quadratures over directions and over the ground integrate it, against many directions from many points."""
    beta = T([FOG.extinction])
    reflection = lamp_reflection(FOG, LAMP, beta, ground_albedo=(0.4,), specular=(roughness, 1.5))
    origin = T([0.5, -9.0, 1.5])
    directions = T([[0.0, 1.0, -0.05], [-0.1, 1.0, 0.02], [0.2, 1.0, -0.12], [0.05, 1.0, 0.25], [-0.5, 9.0, -1.5]])
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distance = torch.where(directions[:, 2] < 0, -1.5 / directions[:, 2], torch.full_like(directions[:, 2], 25.0))
    ours = reflection_inscatter(reflection, FOG, origin, directions, distance, beta)[:, 0]
    expected = brute_force(FOG, LAMP, beta, 0.4, origin, directions, distance, specular=(roughness, 1.5))
    # Within 5% from the smallest roughness that the quadratures resolve up, as which smoother layers are taken, within
    # 8% of their own light, which the quadratures of their own lobe would miss more of
    assert torch.allclose(ours, expected, rtol=tolerance)
    # Rays that look towards the ground beyond the lamp's foot, where it reflects the lamp's light towards them, see more
    # light than with a Lambertian ground
    lambertian = reflection_inscatter(
        lamp_reflection(FOG, LAMP, beta, ground_albedo=(0.4,)), FOG, origin, directions, distance, beta
    )[:, 0]
    assert float(ours[4]) > float(lambertian[4])


def test_specular_light_is_kept_towards_the_grid():
    """With a specular layer, the light of the ground and of the wall towards the points of the grid, which doesn't
    depend on the camera, is kept with their weights, in a single channel for grey surfaces, and gives the light
    computed at every lookup, while the light of surfaces lit by nothing is none."""
    lamp, _, maps = wall_scene()
    beta = T([FOG.extinction])
    kwargs = {"ground_albedo": (0.4,), "object_albedo": (0.5,), "maps": maps, "occluders": wall_points()}
    reflection = lamp_reflection(FOG, lamp, beta, specular=(0.5, 1.5), glow=True, **kwargs)
    assert reflection.gathered.toward and not lamp_reflection(FOG, lamp, beta, **kwargs).gathered.toward
    looked_up = reflection._replace(gathered=gather(reflection, FOG, reflection.points, beta))
    origin = T([0.5, -9.0, 1.5])
    directions = T([[0.0, 1.0, -0.05], [-0.1, 1.0, 0.02], [0.2, 1.0, -0.12], [0.05, 1.0, 0.25], [-0.5, 9.0, -1.5]])
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distance = T([30.0] * 5)
    kept = reflection_inscatter(reflection, FOG, origin, directions, distance, beta)
    assert torch.allclose(kept, reflection_inscatter(looked_up, FOG, origin, directions, distance, beta), rtol=1e-5)
    assert float(kept.min()) > 0
    assert torch.equal(reflection_inscatter(dark(reflection), FOG, origin, directions, distance, beta), 0 * kept)
    # The light is kept in every channel for colored objects, and in a single one for grey ones, also when sums round
    # differently in each channel, as on a GPU
    assert reflection.gathered.patch_weights.shape[-1] == 1
    colored = lamp_reflection(
        FOG, lamp, T([FOG.extinction] * 3), specular=(0.5, 1.5), **(kwargs | {"object_albedo": (0.5, 0.4, 0.3)})
    )
    assert colored.gathered.patch_weights.shape[-1] == 3
    light = colored.gathered.patch_weights[..., :1].expand(-1, -1, 3)
    assert _same_channels(light * T([1.0, 1 + 1e-7, 1 - 3e-7])) and not _same_channels(light * T([1.0, 1.0, 1.001]))


def test_specular_layer_reflects_as_much_light_as_its_albedo():
    """The specular layer reflects the Fresnel reflectance when it is perfectly smooth, about 4% at normal incidence, as
    much when it is smooth, more towards grazing angles, and nothing for an index of refraction of one. Over every
    direction, a surface with it reflects what the coating reflects, its directional albedo, and what the base's albedo
    reflects of the rest, both for the light of the lamp and for light arriving evenly from every direction."""
    cosines = torch.linspace(0, 1, 11, dtype=torch.float64)
    for roughness in (0.0, 0.05, 0.2):
        assert float(specular_albedo(T([1.0]), roughness, 1.5)[0]) == pytest.approx(0.04, rel=0.01)
    assert torch.allclose(specular_albedo(cosines[1:], 0.0, 1.5), _fresnel(cosines[1:], 1.5), rtol=0.02)
    assert torch.all(torch.diff(specular_albedo(cosines, 0.5, 1.5)) < 0)
    assert torch.all(specular_albedo(cosines, 0.5, 1.0) == 0)
    # Over the hemisphere of view directions
    t, w = (T(a) for a in np.polynomial.legendre.leggauss(64))
    mu, weights = (t + 1) / 2, w / 2
    phi = (torch.arange(128, dtype=torch.float64) + 0.5) / 128 * 2 * math.pi
    m, p = torch.meshgrid(mu, phi, indexing="ij")
    view = torch.stack([(1 - m * m).sqrt() * torch.cos(p), (1 - m * m).sqrt() * torch.sin(p), m], dim=-1)
    solid = (weights[:, None] * 2 * math.pi / 128).expand_as(m)
    mean = float(2 * (specular_albedo(mu, 0.5, 1.5) * mu * weights).sum())
    albedo = T([0.4])
    layer = SpecularLayer(0.5, 1.5, T([0.0, 0.0, 1.0]), (albedo, albedo), *(T([]),) * 3)
    for elevation in (90.0, 45.0, 10.0):
        e = math.radians(elevation)
        light = T([math.cos(e), 0.0, math.sin(e)]).expand_as(view)
        total, part = specular_factors(layer, albedo, T(UP), light, view)
        # Radiance of a surface lit by a unit irradiance straight from the lamp, and by a glow of unit irradiance
        direct = ((total + part)[..., 0] * 0.4 / math.pi * m * solid).sum()
        glow = (total[..., 0] * 0.4 / math.pi * m * solid).sum()
        coated = float(specular_albedo(T([math.sin(e)]), 0.5, 1.5)[0])
        assert float(direct) == pytest.approx(coated + 0.4 * (1 - mean), rel=0.01)
        assert float(glow) == pytest.approx(mean + 0.4 * (1 - mean), rel=0.01)
    # The BRDF of a perfectly smooth layer stays finite, also in the mirror direction
    smooth = SpecularLayer(0.0, 1.5, T([0.0, 0.0, 1.0]), (albedo, albedo), *(T([]),) * 3)
    mirrored = view * T([-1.0, -1.0, 1.0])
    assert all(torch.isfinite(x).all() for x in specular_factors(smooth, albedo, T(UP), mirrored, view))


def test_smooth_layers_are_taken_as_rough_as_the_quadratures_resolve():
    """Layers smoother than the quadratures resolve are taken as of the smallest roughness they resolve, which their
    light is closer to than the quadratures of their own narrower lobe."""
    beta = T([FOG.extinction])
    for roughness in (0.0, 0.2, MIN_ROUGHNESS, 0.8):
        layer = lamp_reflection(FOG, LAMP, beta, ground_albedo=(0.4,), specular=(roughness, 1.5)).specular
        assert layer is not None and layer.roughness == max(roughness, MIN_ROUGHNESS)


def test_reflected_light_scales_with_albedo_and_vanishes_without_scattering():
    beta = T([FOG.extinction])
    origin, directions, distance = T([0.0, -6.0, 1.5]), T([[0.0, 1.0, -0.1]]), T([15.0])
    light = [
        reflection_inscatter(
            lamp_reflection(FOG, LAMP, beta, ground_albedo=(a,)), FOG, origin, directions, distance, beta
        )
        for a in (0.2, 0.4)
    ]
    assert float(light[1][0, 0]) == pytest.approx(2 * float(light[0][0, 0]), rel=1e-9)
    dark = FOG.model_copy(update={"albedo": 0.0})
    none = reflection_inscatter(
        lamp_reflection(dark, LAMP, beta, ground_albedo=(0.4,)), dark, origin, directions, distance, beta
    )
    assert float(none.abs().max()) == 0.0
    # A spot pointing up lights no ground
    spot = SpotLight(position=(0.0, 0.0, 2.0), direction=UP, power=(1000.0,), angle=1.0)
    up = reflection_inscatter(
        lamp_reflection(FOG, spot, beta, ground_albedo=(0.4,)), FOG, origin, directions, distance, beta
    )
    assert float(up.abs().max()) == 0.0
    with pytest.raises(ValueError):
        reflection_inscatter(
            lamp_reflection(FOG, LAMP, beta, ground_albedo=(0.4,)), FOG, origin[None], directions, distance, beta
        )
    # Lamps further from rays than the grid's extent add nothing to them
    far = LAMP.model_copy(update={"position": (0.0, 60.0, 2.0)})
    away = reflection_inscatter(
        lamp_reflection(FOG, far, beta, ground_albedo=(0.4,)), FOG, origin, -directions, distance, beta
    )
    assert float(away.abs().max()) == 0.0


def wall_scene():
    """A lamp in front of a wall at y = 1, of 4 m by 3 m, standing on the ground."""
    first_hit = box([-2.0, 1.0, 0.0], [2.0, 1.3, 3.0])
    return LAMP.model_copy(update={"position": (0.0, 2.5, 1.5)}), first_hit, equirectangular((0.0, 2.5, 1.5), first_hit)


@pytest.mark.parametrize("level", [0.0, 0.01])
def test_the_ground_casts_no_shadow_on_itself(level):
    """Maps that record the ground, or a road lying on it, give the ground the same light and the wall's shadow."""
    lamp, wall, maps = wall_scene()

    def with_ground(start, directions):
        down = directions[..., 2]
        safe = torch.where(down < 0, down, torch.full_like(down, -1.0))
        hit = torch.where(down < 0, (level - start[..., 2]) / safe, torch.full_like(down, 1e10))
        return torch.minimum(wall(start, directions), hit)

    recorded = equirectangular(lamp.position, with_ground)
    beta, center = T([FOG.extinction]), T([0.0, 2.5, 0.0])
    textures = [ground_texture(FOG, lamp, beta, center, T([0.4]), maps=m, index=0) for m in (maps, recorded)]
    assert float(textures[1].sum()) == pytest.approx(float(textures[0].sum()), rel=0.01)
    # The wall, towards -y from the lamp's foot, still shadows the ground behind it, beyond 3.5 m
    assert float(textures[1][40:, 42:54].sum()) < 0.02 * float(textures[1][40:, 10:22].sum())


def test_lit_surfaces_and_patches_of_a_wall():
    lamp, _, maps = wall_scene()
    positions, normals, solid, _ = lit_surfaces(maps, 0, 0.0)
    # The lamp sees the back of the wall, at y = 1.3, facing it
    assert torch.allclose(positions[:, 1], T(1.3).expand(len(positions)), atol=1e-6)
    assert float((normals @ T([0.0, 1.0, 0.0])).min()) > 0.99
    # Its solid angle as seen from the lamp, that of a rectangle from a point on its axis
    a, b, d = 2.0, 1.5, 1.2

    def rectangle(x, y):
        return math.atan(x * y / (d * math.sqrt(d * d + x * x + y * y)))

    expected = rectangle(a, b) * 2 + rectangle(a, 1.5) * 2
    assert float(solid.sum()) == pytest.approx(expected, rel=0.01)
    # Without a medium, the wall reflects its albedo times the light it receives
    vacuum = Medium(extinction=0.0, components=[Homogeneous()])
    patches = surface_patches(vacuum, lamp, T([0.0]), maps, 0, 0.0, T([0.5]))
    assert float(patches.power.sum()) == pytest.approx(0.5 * 1000 / (4 * math.pi) * expected, rel=0.01)
    assert torch.allclose(patches.normals, T([0.0, 1.0, 0.0]).expand_as(patches.normals), atol=1e-3)
    # Patches are about a fifth of their distance to the lamp wide and deep
    assert 20 < len(patches.areas) < 400


def wall_points(spacing=0.1):
    """Points on the front and back faces of the wall of `wall_scene`, as the shadow maps of a scene record them."""
    x, z = torch.meshgrid(torch.arange(-2, 2.01, spacing), torch.arange(0, 3.01, spacing), indexing="ij")
    front = torch.stack([x, torch.full_like(x, 1.0), z], dim=-1).reshape(-1, 3)
    return torch.cat([front, front + T([0.0, 0.3, 0.0])])


def test_anchor_maps_hide_points_behind_surfaces():
    _, first_hit, _ = wall_scene()
    anchors = T([[0.0, 2.0, 0.0], [0.0, 1.35, 0.0], [3.0, 2.0, 0.0], [-4.0, -1.0, 0.0]])
    seen = anchor_maps(anchors.float(), wall_points().float(), REFLECTION_GRID.maps)
    points = T([[0.0, -3.0, 1.0], [0.0, 0.5, 1.0], [8.0, 0.0, 1.0], [0.0, 4.0, 1.0], [-5.0, 3.0, 2.0], [1.0, -2, 0.5]])
    # Whether the wall is in the way, away from its edges
    offset = points[None] - anchors[:, None]
    expected = first_hit(anchors[:, None].expand_as(offset), offset / offset.norm(dim=-1, keepdim=True)) > offset.norm(
        dim=-1
    )
    assert anchor_visibility(seen, points.float()).T.tolist() == expected.float().tolist()
    assert 0 < int(expected.sum()) < expected.numel()
    # Surfaces close to an anchor leave no holes between their points, down to the ground
    close = T([[x, 0.0, z] for x in np.linspace(-0.5, 0.5, 11) for z in np.linspace(0.0, 1.0, 11)]).float()
    assert float(anchor_visibility(seen, close)[:, 1].max()) == 0.0
    # Points level with anchors only hide what is above the horizon within their disks
    level = anchor_maps(anchors[:1].float(), T([[0.0, 1.0, 0.0]]).float(), REFLECTION_GRID.maps)
    assert anchor_visibility(level, T([[0.0, -1.0, 0.03], [0.0, -1.0, 0.5]]).float())[:, 0].tolist() == [0.0, 1.0]


def test_objects_hide_and_reflect_light():
    lamp, _, maps = wall_scene()
    beta = T([FOG.extinction])
    occluders = wall_points()
    origin = T([0.5, -8.0, 1.5])
    # Towards the ground in front of the wall, towards the wall, and beside the wall
    directions = T([[0.0, 1.0, -0.12], [0.0, 1.0, 0.0], [0.4, 1.0, 0.0]])
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distance = T([7.0, 8.9, 13.0])

    def light(**kwargs):
        reflection = lamp_reflection(FOG, lamp, beta, ground_albedo=(0.4,), maps=maps, **kwargs)
        return reflection_inscatter(reflection, FOG, origin, directions, distance, beta)[:, 0]

    hidden, open_ = light(occluders=occluders), light(occluders=occluders[:0])
    # Rays in front of the wall see little of the ground lit behind it, and the others more of it
    assert float(hidden[0]) < 0.3 * float(open_[0]) and float(hidden[1]) < 0.3 * float(open_[1])
    assert float(hidden[2]) > 0.6 * float(open_[2])
    # Neither the ground, when shadow maps record it, nor a road lying on it, hide its light
    x, y = torch.meshgrid(torch.arange(-12, 12, 0.1), torch.arange(-12, 12, 0.1), indexing="ij")
    ground = torch.stack([x, y, torch.zeros_like(x)], dim=-1).reshape(-1, 3).double()
    road = ground[ground[:, 0].abs() < 2] + T([0.0, 0.0, 0.01])
    assert torch.allclose(light(occluders=torch.cat([occluders, ground, road])), hidden, rtol=1e-9)
    # The wall reflects light towards the lamp, which rays in front of it hardly see, but for the light of the ground
    # beside the wall that the wall lights
    lit = light(occluders=occluders, object_albedo=(0.5,), interreflections=False)
    assert torch.all(lit[:2] < 1.08 * hidden[:2]) and float(lit[2]) > 1.1 * float(hidden[2])
    # Light that keeps bouncing between the wall and the ground adds a little more
    more = light(occluders=occluders, object_albedo=(0.5,))
    assert float(lit[2]) < float(more[2]) < float(lit[2]) + 0.3 * float(lit[2] - hidden[2])


def test_relit_reflection_adds_the_glow():
    """Surfaces lit with the full extinction reflect the glow that reaches them too, the ground's radiance rising by its
    albedo over π times the glow's irradiance, while surfaces lit with the similarity relation are left as they are."""
    beta = T([FOG.extinction])
    direct = lamp_reflection(FOG, LAMP, beta, ground_albedo=(0.4,), glow=True)
    assert direct.albedos is not None
    relit = relit_reflection(
        direct, FOG, beta, lambda points, normals: (points.new_full((len(points), 1), 3.0), points.new_ones(len(points)))
    )
    assert torch.allclose(relit.texture, direct.texture + 0.4 / math.pi * 3.0, rtol=1e-9)
    # Nodes of the quadrature over the ground follow its new radiance, and the light at the grid's points is gathered
    assert len(relit.nodes.areas) == len(direct.nodes.areas) and not torch.equal(relit.nodes.areas, direct.nodes.areas)
    assert not torch.equal(relit.gathered.node_weights, direct.gathered.node_weights)
    # With the full extinction, surfaces are darker than with the similarity relation, until the glow lights them
    similar = lamp_reflection(FOG, LAMP, beta, ground_albedo=(0.4,))
    assert torch.all(direct.texture <= similar.texture) and similar.albedos is None
    assert (
        relit_reflection(similar, FOG, beta, lambda points, normals: (points.new_ones(len(points), 1), points[:, 0]))
        is similar
    )


def test_light_keeps_bouncing_between_surfaces():
    """Each round of the light bouncing between the ground and objects is that which the previous round reflected,
    with the patches of objects receiving it, or other patches, which objects light too, and less each time."""
    vacuum = Medium(extinction=0.0, components=[Homogeneous()])
    zero, center = T([0.0]), T([0.0, 0.0, 0.0])
    albedos = (T([0.4]), T([0.5]))
    texture = torch.full((*REFLECTION_GRID.ground, 1), 2.0, dtype=torch.float64)
    patches = Patches(
        T([[0.0, 0.0, 0.5], [3.0, 0.0, 1.0]]), T([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]), T([0.01, 1.0]), T([[10.0], [5.0]])
    )
    # Facing the second patch, 2 m away along its normal, and facing the ground
    receivers = Patches(
        T([[5.0, 0.0, 1.0], [-2.0, 0.0, 0.5]]), T([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]), T([0.1, 0.2]), T([[0.0], [0.0]])
    )

    def bounced(texture, patches, receivers, rounds, transfer=None):
        return _bounced(
            vacuum, zero, center, REFLECTION_GRID, albedos, texture, patches, 0.0, False, receivers, rounds, transfer
        )

    for targets in (None, receivers):
        once, twice, many = (bounced(texture, patches, targets, rounds) for rounds in (1, 2, 16))
        emitted = (patches if targets is None else targets)._replace(power=once[1])
        again = bounced(once[0], emitted, targets, 1)
        assert torch.allclose(twice[0], once[0] + again[0], rtol=1e-12) and torch.allclose(twice[1], once[1] + again[1])
        assert torch.all(many[0] >= twice[0]) and torch.all(many[1] >= twice[1])
        # The light dies out as it keeps bouncing
        fewer = bounced(texture, patches, targets, 15)
        assert torch.allclose(fewer[1], many[1], rtol=1e-3) and torch.allclose(fewer[0], many[0], rtol=1e-3)
        # The light that surfaces exchange, kept from one round to the next, gives the same light, and kept for the
        # targets, it gives the same light again when they are lit once more
        transfer: dict[str, torch.Tensor] = {}
        kept = bounced(texture, patches, targets, 16, transfer)
        assert all(torch.allclose(x, y, rtol=1e-10, atol=1e-12) for x, y in zip(kept, many)) and transfer
        relit = bounced(once[0], emitted, targets if targets is None else emitted, 16, transfer)
        expected = bounced(once[0], emitted, targets if targets is None else emitted, 16)
        assert all(torch.allclose(x, y, rtol=1e-10, atol=1e-12) for x, y in zip(relit, expected))
    # Receivers are lit by the patches too: the first faces the second patch, whose radiance reaches it on its axis as a
    # disk's would, besides the ground's light
    alone = bounced(texture.new_zeros(0, 0, 1), patches, receivers, 1)[1]
    disk = 5.0 / math.pi / (2.0**2 + 1.0 / (2 * math.pi))
    assert float(alone[0, 0]) == pytest.approx(0.5 * disk * 0.1, rel=1e-9) and float(alone[1, 0]) == 0.0


def test_every_surface_receives_the_light_of_others():
    """Pieces of the surfaces that the maps of the sky record add up to every side of a box standing on the ground but
    its bottom, and patches of them receive the light of the ground and of the faces that the lamp lights, also the
    faces it doesn't light, while the glow of the medium lights them rather than the patches of the lamp's map."""
    from tests.test_occlusion import CELLS, EDGES, occlusion_of, render_maps
    from tests.test_occlusion import box as solid
    from visionsim.medium.occlusion import sky_cell_directions

    low, high = [-1.0, 4.0, 0.0], [1.0, 7.0, 2.0]
    first = solid(low, high)
    sky = render_maps(sky_cell_directions(EDGES, CELLS), [low, high], first, 128)
    occlusion = occlusion_of(render_maps([], [low, high], first), [], sky, [low, high])
    surfels = scene_surfels(occlusion, 0.0)
    for (axis, sign), area in {(0, 1): 6.0, (0, -1): 6.0, (1, 1): 4.0, (1, -1): 4.0, (2, 1): 6.0}.items():
        on = surfels.normals[:, axis].double() * sign > 0.9
        assert float(surfels.areas[on].sum()) == pytest.approx(area, rel=0.05)
        plane = high[axis] if sign > 0 else low[axis]
        off = (surfels.positions[on, axis].double() - plane).abs() > 0.05
        assert float(surfels.areas[on][off].sum()) < 0.01 * area
    # A lamp in front of the box's face at y = 4, which only lights that face
    lamp = PointLight(position=(0.0, 2.0, 1.0), power=(1000.0,), radius=0.05)
    maps = equirectangular(lamp.position, box(low, high))
    beta = T([FOG.extinction])
    kwargs = {"ground_albedo": (0.4,), "object_albedo": (0.5,), "maps": maps, "glow": True, "interreflections": False}
    alone = lamp_reflection(FOG, lamp, beta, **kwargs)
    every = lamp_reflection(FOG, lamp, beta, surfels=surfels, **kwargs)
    receivers = surface_receivers(surfels, every.center, REFLECTION_GRID.extent[1])
    assert every.receivers == len(receivers.areas) > 0 and alone.receivers == 0
    # The patches of the lamp's map keep the lamp's light, and the faces it lights receive the ground's instead of them,
    # and the others some of it
    direct = lamp_reflection(FOG, lamp, beta, **(kwargs | {"bounces": False})).patches.power
    lit = len(every.patches.areas) - every.receivers
    assert torch.equal(every.patches.power[:lit], direct)
    received = every.patches.power[lit:, 0]
    facing = every.patches.normals[lit:, 1] < -0.9
    from_ground = float((alone.patches.power - direct).sum())
    assert float(received[facing].sum()) == pytest.approx(from_ground, rel=0.15)
    assert float(received[every.patches.normals[lit:, 0].abs() > 0.9].sum()) > 0
    # The glow lights the receivers, not the patches of the lamp's map
    relit = relit_reflection(every, FOG, beta, lambda points, normals: (points.new_ones(len(points), 1), points[:, 0]))
    assert torch.equal(relit.patches.power[:lit], every.patches.power[:lit])
    assert torch.all(
        relit.patches.power[lit:] >= every.patches.power[lit:] + 0.5 * every.patches.areas[lit:, None] - 1e-12
    )


def test_surfaces_that_the_maps_of_the_sky_miss_receive_light_too():
    """The underside of a canopy, which the lamp under it lights but no map of the sky records, receives the light of
    the ground and the glow as the patches of the lamp's map do without the pieces of every surface, so that the light
    that surfaces reflect into the medium is the same, but for the glow lighting the canopy's top and sides too."""
    from tests.test_occlusion import CELLS, EDGES, occlusion_of, render_maps
    from tests.test_occlusion import box as solid
    from visionsim.medium.occlusion import sky_cell_directions

    low, high = [-3.0, -3.0, 3.0], [3.0, 3.0, 3.2]
    first = solid(low, high)
    sky = render_maps(sky_cell_directions(EDGES, CELLS), [low, high], first, 128)
    occlusion = occlusion_of(render_maps([], [low, high], first), [], sky, [low, high])
    surfels = scene_surfels(occlusion, 0.0)
    assert not bool((surfels.normals[:, 2] < -0.9).any())
    lamp = LAMP.model_copy(update={"position": (0.0, 0.0, 2.5)})
    maps = equirectangular(lamp.position, box(low, high))
    beta = T([FOG.extinction])
    kwargs = {"ground_albedo": (0.4,), "object_albedo": (0.5,), "maps": maps, "glow": True}
    alone = lamp_reflection(FOG, lamp, beta, **kwargs)
    every = lamp_reflection(FOG, lamp, beta, surfels=surfels, **kwargs)
    direct = lamp_reflection(FOG, lamp, beta, **(kwargs | {"bounces": False})).patches.power
    lit = len(every.patches.areas) - every.receivers
    under = every.patches.normals[lit:, 2] < -0.9
    # The underside joins the receivers with the area that the lamp's map gives it, and receives the ground's light
    assert float(every.patches.areas[lit:][under].sum()) == pytest.approx(36.0, rel=0.02)
    from_ground = float((alone.patches.power - direct).sum())
    assert float(every.patches.power[lit:][under].sum()) == pytest.approx(from_ground, rel=0.01)

    def glow(points, normals):
        return points.new_ones(len(points), 1), points[:, 0]

    relit, relit_alone = (relit_reflection(x, FOG, beta, glow) for x in (every, alone))
    gained = relit.patches.power[lit:] - every.patches.power[lit:]
    assert float(gained[under].sum()) == pytest.approx(
        float((relit_alone.patches.power - alone.patches.power).sum()), rel=0.02
    )
    assert float(gained[~under].sum()) > 0.5 * 0.9 * float(every.patches.areas[lit:][~under].sum())
    origin = T([0.0, -12.0, 1.5])
    directions = T([[0.0, 1.0, e] for e in (-0.1, -0.05, 0.0, 0.05, 0.1, 0.15)])
    directions = directions / directions.norm(dim=-1, keepdim=True)
    distance = T([40.0] * len(directions))
    light = [
        reflection_inscatter(x, FOG, origin, directions, distance, beta) for x in (every, alone, relit, relit_alone)
    ]
    assert torch.allclose(light[0], light[1], rtol=0.01)
    assert torch.all(light[2] >= 0.995 * light[3]) and torch.allclose(light[2], light[3], rtol=0.02)


def test_ground_and_objects_light_each_other():
    """In a vacuum, a small patch facing down 0.5 m above the ground sends all its light to the ground, which reflects
    its albedo of it, and receives all the light of a ground of even radiance, but for the little beyond the texture's
    extent."""
    vacuum = Medium(extinction=0.0, components=[Homogeneous()])
    center = T([0.0, 0.0, 0.0])
    power, area, height = 10.0, 0.01, 0.5
    patches = Patches(T([[0.0, 0.0, height]]), T([[0.0, 0.0, -1.0]]), T([area]), T([[power]]))
    radiance = 2.0
    texture = torch.full((*REFLECTION_GRID.ground, 1), radiance, dtype=torch.float64)
    albedos = (T([0.4]), T([0.5]))
    bounced, received = _bounced(vacuum, T([0.0]), center, REFLECTION_GRID, albedos, texture, patches, 0.0, False)
    _, areas = _texture_cells(center, REFLECTION_GRID)
    # The ground reflects as much light as the texture's radiance over its cells holds, where the cells reach, from
    # half a cell within the smallest distance to half a cell beyond the largest one
    (r_min, r_max), n = REFLECTION_GRID.extent, REFLECTION_GRID.ground[0]
    step = math.log(r_max / r_min) / (n - 1)
    inner, outer = r_min * math.exp(-step / 2), r_max * math.exp(step / 2)
    share = height**2 / (height**2 + inner**2) - height**2 / (height**2 + outer**2)
    reflected = float((math.pi * bounced[..., 0].reshape(-1) * areas).sum())
    assert reflected == pytest.approx(0.4 * power * share, rel=0.005)
    # An infinite ground of even radiance gives an irradiance of π times it, but for the part beyond 40 m
    expected = 0.5 * math.pi * radiance * area * (1 - height**2 / (height**2 + REFLECTION_GRID.extent[1] ** 2))
    assert float(received[0, 0]) == pytest.approx(expected, rel=0.01)


def test_scene_surfaces():
    from tests.test_occlusion import CELLS, EDGES, render_maps, wall
    from visionsim.medium.occlusion import Occlusion, sky_cell_directions

    first = wall(1.0, (-2.0, 2.0), (0.0, 3.0))
    bounds = [[0.5, -2.5, 0.0], [1.5, 2.5, 3.5]]
    sky = render_maps(sky_cell_directions(EDGES, CELLS), bounds, first, resolution=64)
    empty = render_maps([], bounds, first)
    occlusion = Occlusion(empty, T([]).reshape(0, 3), sky, EDGES, CELLS, T(bounds))
    points = map_points(sky)
    assert torch.allclose(points[:, 0], T(1.0).expand(len(points)), atol=1e-6)
    merged = scene_surfaces(occlusion, size=0.25)
    assert len(merged) < len(points) and torch.allclose(merged[:, 0], T(1.0).expand(len(merged)), atol=1e-6)
    assert float(merged[:, 1].min()) > -2.01 and float(merged[:, 2].max()) < 3.01
    # Surfaces on the ground are part of it
    assert float(merged[:, 2].min()) < 0.25 and float(scene_surfaces(occlusion, 0.5, 0.25)[:, 2].min()) > 0.5


def test_apply_medium_adds_light_reflected_by_the_ground():
    camera = {"w": 6, "h": 4, "fl_x": 4.0, "fl_y": 4.0, "cx": 3.0, "cy": 2.0}
    pose = np.eye(4)
    pose[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    pose[:3, 3] = [0.0, -8.0, 1.5]
    depth = np.full((4, 6), 15.0)
    frame = np.zeros((4, 6, 3))
    black = Lighting(points=[LAMP])
    grey = black.model_copy(update={"ground_albedo": (0.4, 0.4, 0.4)})
    dark, lit = (apply_medium(frame, depth, camera, pose, FOG, x).inscatter for x in (black, grey))
    assert torch.all(lit > dark) and 1.05 < float(lit.sum() / dark.sum()) < 3
    # Objects reflect light as the ground does, unless they're given their own albedo
    assert Lighting().object_albedo is None
    # Without lamps, the ground only reflects suns and the sky
    no_lamp = Lighting(sky=(0.1, 0.1, 0.1))
    same = [
        apply_medium(frame, depth, camera, pose, FOG, no_lamp.model_copy(update={"ground_albedo": a})).inscatter
        for a in ((0.0,) * 3, (0.4,) * 3)
    ]
    assert torch.equal(*same)


def test_reflected_light_is_cached_across_frames():
    """Shadow maps loaded in single precision, as by default, are converted for frames rendered in double precision,
    but the light of lamps is cached by the maps given."""
    from tests.test_occlusion import FOG as HEIGHT_FOG_WITH_SUN
    from tests.test_occlusion import demo_scene
    from visionsim.medium import reflections
    from visionsim.medium.occlusion import occlusion_to

    occlusion, lighting, camera, pose, depth, radiance = demo_scene()
    occlusion = occlusion_to(occlusion, dtype=torch.float32)
    lighting = lighting.model_copy(update={"points": [LAMP.model_copy(update={"position": (0.0, 12.0, 2.0)})]})
    reflections._cached_lamp_reflection.cache_clear()
    for _ in range(2):
        apply_medium(radiance, depth, camera, pose, HEIGHT_FOG_WITH_SUN, lighting, occlusion=occlusion)
    info = reflections._cached_lamp_reflection.cache_info()
    assert (info.misses, info.hits) == (1, 1)
