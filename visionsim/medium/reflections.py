"""Light of lamps that surfaces reflect into a medium, which scatters it once towards the camera.

Lamps light the medium directly, see :func:`lamp_inscatter <visionsim.medium.render.lamp_inscatter>`, and the surfaces
around them, which reflect part of their light into the medium: at night, the ground lit by a street lamp glows through
the fog around it. In the night scene of ``examples/medium``, a point light 1.5 m above a ground of albedo 0.39 right
behind a cube, Cycles' light that surfaces reflect and the fog scatters once is about four fifths of the light of the
lamp that the fog scatters once, and about half of the light that went through surfaces, the rest of which the fog
scatters more than once. It is only modeled given the albedo of the ground, or of objects, which default to zero and to
that of the ground, see :class:`Lighting <visionsim.medium.model.Lighting>`, as for the light of suns and of the sky
that the ground reflects.

Surfaces are a Lambertian base under a specular layer, as Blender's Principled BSDF: a dielectric coating of GGX
microfacets, which reflects the lamp's light around the mirror direction, while the base is dimmed by the light that the
coating reflects towards the viewer, see :func:`specular_factors`, given its roughness and index of refraction, see
:class:`Lighting <visionsim.medium.model.Lighting>`, or Lambertian. Layers smoother than a roughness of 0.5 are taken as
that rough, as the quadratures below miss more of the light of narrower lobes than these lobes change it, see
:data:`MIN_ROUGHNESS`. The layer reflects the light that surfaces receive from other surfaces and from the medium's glow
as if it arrived evenly from every direction, by its directional albedo, and the light of area lights and emissive
surfaces around the mirror direction of the position they cast shadows from, as that of a point light. In the night
scene, whose materials have a roughness of 0.8, the layer adds 4.7% to the light of surfaces lit directly that the fog
scatters once, as Cycles' Principled BSDFs add 5.0% to that of Lambertian surfaces of the same albedo. They are lit by
the lamp through the medium as :mod:`visionsim.medium.surfaces` lights them, see :func:`lamp_irradiance
<visionsim.medium.surfaces.lamp_irradiance>`, so that the light they reflect is that which the camera sees them with:
with an extinction reduced by the similarity relation, as the light that the medium scatters forward on its way keeps
lighting them, or, with multiple scattering, with the full extinction, and by the glow of the medium, i.e. by the light
of the lamp that the medium scattered once and more, and the light of surfaces that it scattered, which reaches them
from every direction, as the halos of lamps tabulate it, see :func:`relit_reflection` and :mod:`visionsim.medium.halos`.
Over the outermost cell of halos, 30 to 40 m from the lamp by default, where their glow fades out, surfaces keep the
full extinction, unlike those that the camera sees, which go back to the similarity relation there, as the light they
reflect is negligible: in the night scene, the ground from 30 to 40 m from the lamp receives about 1% of the lamp's
light that reaches the ground without fog. The ground is the plane at the lighting's ``ground_height``, of albedo
``ground_albedo``, in the shadows that objects cast according to the lamp's map, and its radiance is tabulated around
the lamp's foot, see :func:`ground_texture`. Other surfaces are those that the lamp's map records, which are exactly
those that the lamp lights, of albedo ``object_albedo``, grouped into patches that shine as one-sided Lambertian point
emitters, see :func:`surface_patches`. Objects and the ground light each other, see :func:`lamp_reflection`: in the
night scene, the cube 1.5 m from the lamp, which reflects more than a quarter of the light of surfaces, adds about 8% to
the light that reaches the ground within 6 m of the lamp's foot. Light keeps bouncing between them until little is left,
and given the shadow maps of suns or of the sky, which see every side of objects standing on the ground, every surface
receives the light of the ground and of objects, and the medium's glow, also the sides of objects that the lamp doesn't
light, see :func:`scene_surfels` and :func:`surface_receivers`, as do the surfaces that the lamp lights but that these
maps don't record, e.g. under a roof, see :func:`unrecorded_surfaces`. In the night scene, with every surface a
Lambertian of albedo 0.39, the light reflected by two surfaces or more is 7% of Cycles' light of surfaces in the fog,
which these bring within 7%, against 43% too dark when objects and the ground only lit each other once and only the
surfaces that the lamp lights received the ground's light.

The light of the ground reaches a point of the medium from many directions, and is integrated by two quadratures,
combined by the balance heuristic of multiple importance sampling (Veach and Guibas, "Optimally Combining Sampling
Techniques for Monte Carlo Rendering", SIGGRAPH 1995): one over directions, whose nodes follow the phase function
towards the camera as those of the sky do (see :func:`phase_directions`), which integrates the ground around points close
to it, including the ground ahead of camera rays that the medium scatters forward, and one over the ground, whose nodes
follow its radiance, which integrates the bright ground under the lamp from afar, see :func:`ground_nodes`. Objects hide
the light of the ground from points behind them, as seen from anchors on the ground around the lamp, whose maps
reproject the surfaces that the scene's shadow maps record, see :func:`anchor_maps`: those of the sky see every side of
objects standing on the ground, and close to the lamp, where most of the reflected light comes from, the anchors see
objects as the ground does. Surfaces on the ground, such as roads, are part of the ground, which can't hide its own
light from the medium above it, see :data:`GROUND_CLEARANCE`. The light of objects isn't hidden, as it mostly goes back
towards the lamp, through the space that the lamp lights, and their patches' distances to points are softened by their
size, see :func:`gather`.

The light that reaches each point, but for the phase function towards the camera, doesn't depend on the camera, and is
gathered once per lamp and medium on a grid of points around the lamp's foot, spread logarithmically in distance from it
and denser near the ground, see :func:`cached_lamp_reflection`. For every frame, the light that the medium scatters
towards the camera is computed at each point of the grid, and on a grid of points around the camera along the rays of
the frame, where the direction towards the camera changes too quickly across the cells of the first one, and both are
interpolated along each ray, see :func:`reflection_inscatter`. Reflected light is only gathered within the grid's
extent from the lamp's foot, 40 m by default, beyond which lamps whose light reaches no ray of a frame are skipped.
With the default resolution, the light of a lamp takes about 30 MB, and on 4 CPU threads about a second to gather, plus
about 6 s per million points of surfaces within its extent, and about 0.12 s per frame of 320x180 pixels, or 0.06 s at
a quarter of the resolution, for each lamp whose light reaches the frame. For the lamp of the night scene of
``examples/medium``, the patches of every surface (see :func:`surface_receivers`), 588 against the 157 of the surfaces
that the lamp lights, and the specular layer of surfaces, whose light towards the points of the grid, which doesn't
depend on the camera, is kept with the light gathered there rather than computed for every frame, make the light of
the lamp take 44 MB rather than 28 MB, or 98 MB when the light of surfaces differs between channels, e.g. for colored
objects, and about 50 ms more per frame on 4 CPU threads at every resolution: the light of the ground through the
specular layer over the directions that follow the phase function, and that of the patches, is looked up at every
frame at the points of both grids that rays interpolate.

Surfaces don't hide each other's light, and the specular layer only reflects light towards the medium: the light that
surfaces reflect onto each other is that of their Lambertian base, and black surfaces reflect none. Surfaces that
neither the lamp's map nor the maps of suns or of the sky record, such as the bottom of objects, receive no light from
other surfaces. Without multiple scattering, the light that the medium scatters on its way to surfaces beyond what the
similarity relation accounts for isn't modeled. Reflected light is scattered more than once by the halos of lamps, see
:mod:`visionsim.medium.halos`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache
from typing import Any, NamedTuple

import numpy as np
import torch

from visionsim.medium.model import AreaLight, Blob, Lamp, Medium
from visionsim.medium.occlusion import Occlusion, ShadowMaps, lamp_map_index, lamp_visibility, maps_to, visibility
from visionsim.medium.optics import _hg_cdf, _hg_inverse_cdf, _per_channel, density, henyey_greenstein, optical_depth
from visionsim.medium.surfaces import Glow, lamp_irradiance

SURFACE_SPACING = 0.1
"""Size, in meters, of the cubes within which the points of surfaces that hide reflected light are merged, see
:func:`scene_surfaces`"""
GROUND_CLEARANCE = SURFACE_SPACING / 4
"""Height above the ground, in meters, below which surfaces are part of the ground, such as roads lying on it, and don't
hide the light of the ground: segments between the ground and points above it lie above it. It is less than half of
:data:`SURFACE_SPACING`, so that the lowest points of objects standing on the ground still hide its light down to it,
see :func:`anchor_maps`"""
MIN_ROUGHNESS = 0.5
"""Smallest roughness of the specular layer of surfaces, as that of :attr:`Lighting.surface_roughness
<visionsim.medium.model.Lighting.surface_roughness>`, below which layers are taken as this rough. The quadratures that
gather the light of the ground and of objects don't follow the lobe of the layer around the mirror direction, and miss
more and more of its light as it narrows: on the rays of ``tests/test_reflections.py``, the light of the ground that the
medium scatters once is within 5% of a direct integration from a roughness of 0.5 up, but 3% to 9% darker at 0.4, 6% to
15% at 0.3 and up to 17% at 0.2, while layers of roughness 0.1 to 0.45 taken as 0.5 are within 8% of their own light,
see :func:`lamp_reflection`"""
_MIN_ALPHA = 1e-3
"""Smallest width of the distribution of the microfacets of a specular layer, i.e. its roughness squared, which keeps
the BRDF of perfectly smooth layers finite, see :func:`_ggx_reflectance`"""


class ReflectionGrid(NamedTuple):
    """Resolution of the light of a lamp reflected by surfaces, see :func:`lamp_reflection` and
    :func:`reflection_inscatter`."""

    radii: int = 24
    """number of distances to the lamp's foot, spread logarithmically over ``extent``, of the points of the grid"""
    extent: tuple[float, float] = (0.05, 40.0)
    """smallest and largest distance to the lamp's foot, in meters, beyond which no reflected light is gathered"""
    elevations: int = 12
    """number of elevations of the points as seen from the lamp's foot, denser near the ground"""
    azimuths: int = 32
    """number of azimuths of the points around the lamp's foot"""
    view: tuple[int, int, int] = (16, 9, 16)
    """number of directions across and along the bundle of rays of a frame, and of distances to the camera, spread
    logarithmically, of the points of the grid around the camera"""
    ground: tuple[int, int] = (64, 64)
    """number of distances to the lamp's foot, spread logarithmically over ``extent``, and of azimuths, at which the
    radiance of the ground is tabulated"""
    supersampling: int = 4
    """number of samples along each dimension of the cells of the ground's texture, over which its radiance is averaged,
    see :func:`ground_texture`"""
    nodes: tuple[int, int] = (12, 12)
    """number of distances to the lamp's foot and of azimuths of the nodes of the quadrature over the ground"""
    directions: tuple[int, int] = (4, 8)
    """number of angles in each of the three pieces of the quadrature over directions, and of azimuths"""
    anchors: tuple[float, int] = (math.sqrt(2), 16)
    """ratio between the radii of consecutive rings of anchors around the lamp's foot, the first of which is at a
    quarter of the lamp's height above the ground, and number of anchors in each ring"""
    maps: tuple[int, int] = (64, 256)
    """height and width of the maps of anchors, over the directions above their horizon, see :func:`anchor_maps`"""
    patches: float = 0.2
    """size of the patches into which the surfaces of the lamp's map are grouped, in radians as seen from the lamp"""


REFLECTION_GRID = ReflectionGrid()
"""Default resolution of the light of lamps reflected by surfaces"""
_ROUNDS = 16
"""Largest number of times that light bounces between the ground and objects with interreflections, which stop well
before as each bounce reflects a fraction of the light, see :func:`_bounced`"""


class Patches(NamedTuple):
    """Patches of surfaces lit by a lamp, each of which reflects its light as a one-sided Lambertian point emitter."""

    positions: torch.Tensor
    """position of each patch, the mean of its points weighted by the light they reflect, of shape (o, 3)"""
    normals: torch.Tensor
    """unit normal of each patch, on the side of the lamp, of shape (o, 3)"""
    areas: torch.Tensor
    """area of each patch, in m², of shape (o,)"""
    power: torch.Tensor
    """power that each patch reflects per channel, in W, of shape (o, c)"""


class GroundNodes(NamedTuple):
    """Nodes of a quadrature over the ground around a lamp's foot, which follow the radiance of the ground, see
    :func:`ground_nodes`."""

    positions: torch.Tensor
    """position of each node, of shape (n, 3)"""
    areas: torch.Tensor
    """weight of each node, i.e. the area of ground it stands for, in m², of shape (n,)"""
    density: torch.Tensor
    """density of the nodes per unit area at each node, of shape (n,)"""
    anchors: torch.Tensor
    """index of the anchor nearest to each node, of shape (n,), see :func:`nearest_anchor`"""


class Gathered(NamedTuple):
    """Light of a lamp reflected by surfaces that reaches points of a medium, before it is scaled by the radiance of
    the ground or the power of patches, and by the phase function, see :func:`gather`. Channels are those of the
    extinction coefficient, i.e. a single one when it is the same for every channel. When cached, see
    :func:`lamp_reflection`, visibility is kept in half precision and the rest in single precision, which is ample for
    light that is interpolated anyway."""

    visible: torch.Tensor
    """visibility of the points from each anchor on the ground, see :func:`anchor_positions`, of shape (k, m), or
    (k, 1) without anchors"""
    node_weights: torch.Tensor
    """solid angle of ground that each node of the quadrature over the ground stands for as seen from the points, times
    the transmittance and the visibility in between, of shape (k, n, c)"""
    node_density: torch.Tensor
    """density of the nodes of the quadrature over the ground per steradian as seen from the points, of shape (k, n)"""
    patch_weights: torch.Tensor
    """irradiance of the points from each patch of objects, for a unit power, times the transmittance in between, of
    shape (k, o, c)"""
    toward: bool = False
    """whether the weights of nodes and patches are times their radiance and power towards the points too, for the
    channels of their light, or a single one when it is the same for every channel, as they are kept for the points of
    the grid of surfaces with a specular layer, see :func:`lamp_reflection`, whose light towards these points doesn't
    change from one frame to the next, while it depends on the direction towards the points"""


class SpecularLayer(NamedTuple):
    """Specular layer of the surfaces that reflect the light of a lamp, a dielectric coating of GGX microfacets over
    their diffuse base, and the part of their light that reaches them straight from the lamp, which it reflects around
    the mirror direction, see :func:`specular_factors`."""

    roughness: float
    """roughness of the microfacets, whose square is the width of their distribution, as Blender's Principled BSDF's"""
    ior: float
    """index of refraction of the coating, which sets its Fresnel reflectance"""
    light: torch.Tensor
    """position of the lamp, from which its light reaches surfaces directly, of shape (3,)"""
    albedos: tuple[torch.Tensor, torch.Tensor]
    """albedo of the diffuse base of the ground and of objects per channel, each of shape (c,)"""
    texture: torch.Tensor
    """radiance of the ground's diffuse base from the lamp's direct light alone, the part of
    :attr:`LampReflection.texture` that the lamp lights directly, of shape (g, a, c), or (0, 0, c)"""
    node_radiance: torch.Tensor
    """the same at the nodes of the quadrature over the ground, of shape (n, c)"""
    power: torch.Tensor
    """power that the diffuse base of each patch of objects reflects of the lamp's direct light, the part of
    :attr:`Patches.power` that the lamp lights directly, of shape (o, c)"""


class LampReflection(NamedTuple):
    """Light of a lamp reflected by surfaces, gathered on a grid of points around the lamp's foot, but for the phase
    function towards the camera, which changes with every frame, see :func:`lamp_reflection`."""

    center: torch.Tensor
    """foot of the lamp on the ground, or the lamp itself when it isn't above the ground, of shape (3,)"""
    height: float
    """height of the lamp above the ground, which sets the anchors, or zero when the lamp isn't above the ground"""
    grid: ReflectionGrid
    """resolution of the grids and of the quadratures"""
    texture: torch.Tensor
    """radiance of the ground, against distances to the center and azimuths, see :func:`ground_texture`, of shape
    (g, a, c), or (0, 0, c) when the ground reflects no light"""
    nodes: GroundNodes
    """nodes of the quadrature over the ground, of which there are none when the ground reflects no light"""
    node_radiance: torch.Tensor
    """radiance of the ground at each node, of shape (n, c)"""
    cell_density: torch.Tensor
    """density of the nodes of the quadrature over the ground per unit area, in each cell of the texture, see
    :func:`ground_nodes`, of shape (g, a), or (0, 0) when the ground reflects no light"""
    anchors: ShadowMaps | None
    """maps of the anchors, see :func:`anchor_maps`, or None when nothing hides the light of the ground"""
    patches: Patches
    """patches of objects, of which there are none when objects reflect no light"""
    points: torch.Tensor
    """points of the grid around the center, by distance, elevation and azimuth, of shape (k, 3)"""
    gathered: Gathered
    """light that reaches the points, see :func:`gather`"""
    albedos: tuple[torch.Tensor, torch.Tensor] | None = None
    """albedo of the ground and of objects per channel, each of shape (c,), with which the glow of the medium that
    lights them is reflected too, see :func:`relit_reflection`, or None when surfaces are lit with the reduced extinction
    of the similarity relation, which already counts part of their glow, see :func:`lamp_reflection`"""
    bounces: bool = True
    """whether objects and the ground light each other once, see :func:`lamp_reflection`"""
    specular: SpecularLayer | None = None
    """specular layer of surfaces, with the part of their light that the lamp lights directly, or None if they are
    Lambertian, see :func:`specular_factors`"""
    receivers: int = 0
    """number of the last of the ``patches`` that are those of every surface of the scene, see
    :func:`surface_receivers`, which receive the light of the ground and of objects, and the medium's glow, rather than
    the patches of the surfaces that the lamp lights"""
    rounds: int = 1
    """largest number of times that light bounces between the ground and objects, see :func:`_bounced`"""
    transfer: dict[str, torch.Tensor] | None = None
    """light that the ground and objects exchange per unit of their light, see :func:`_bounced`, kept until they are
    relit by the glow of the medium, see :func:`relit_reflection`, or None"""

    @property
    def hemisphere(self) -> bool:
        """Whether points lie above the center, on the ground, rather than all around it."""
        return self.height > 0


def _unit(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0, 1)
    return x * x * (3 - 2 * x)


def _fresnel(cosine: torch.Tensor, ior: float) -> torch.Tensor:
    """Fresnel reflectance of unpolarized light at a dielectric interface of a given index of refraction, for the
    cosine of the angle of incidence, of any shape."""
    g = (ior * ior - 1 + cosine * cosine).clamp_min(0).sqrt()
    a = (g - cosine) / (g + cosine).clamp_min(1e-12)
    b = (cosine * (g + cosine) - 1) / (cosine * (g - cosine) + 1)
    return 0.5 * a * a * (1 + b * b)


def _ggx_reflectance(
    cos_light: torch.Tensor,
    cos_view: torch.Tensor,
    cos_half: torch.Tensor,
    cos_light_half: torch.Tensor,
    alpha: float,
    ior: float,
) -> torch.Tensor:
    """π times the BRDF of a dielectric coating of GGX microfacets (Walter et al., "Microfacet Models for Refraction
    through Rough Surfaces", EGSR 2007), with height-correlated Smith masking (Heitz, "Understanding the
    Masking-Shadowing Function in Microfacet-Based BRDFs", JCGT 2014), given the cosines between the normal and the
    directions towards the light, the viewer and halfway between them, and between the light and the halfway direction,
    of any shape. The width of the distribution of microfacets is at least :data:`_MIN_ALPHA`, which keeps the BRDF of
    perfectly smooth coatings finite."""
    a2 = max(alpha, _MIN_ALPHA) ** 2
    distribution = a2 / (math.pi * (cos_half * cos_half * (a2 - 1) + 1) ** 2)
    value = math.pi * distribution * _masking(cos_light, cos_view, a2) * _fresnel(cos_light_half.clamp(0, 1), ior)
    value = value / (4 * cos_light.clamp_min(1e-6) * cos_view.clamp_min(1e-6))
    return torch.where((cos_light > 0) & (cos_view > 0), value, torch.zeros_like(value))


def _masking(cos_light: torch.Tensor, cos_view: torch.Tensor, a2: float) -> torch.Tensor:
    """Height-correlated Smith masking-shadowing function of GGX microfacets, ``1 / (1 + Λ(l) + Λ(v))``, given the
    cosines between the normal and the directions towards the light and the viewer, of any shape, and the square of the
    width of the distribution of microfacets."""

    def lambda_(cosine: torch.Tensor) -> torch.Tensor:
        c2 = cosine.clamp_min(1e-6) ** 2
        return ((1 + a2 * (1 - c2) / c2).sqrt() - 1) / 2

    return 1 / (1 + lambda_(cos_light) + lambda_(cos_view))


@lru_cache(maxsize=16)
def _specular_albedo_table(roughness: float, ior: float, size: int = 65) -> np.ndarray:
    """Directional albedo of a specular layer, see :func:`specular_albedo`, at ``size`` cosines of the view direction
    spread evenly from zero to one, by a quadrature over the halfway directions that follows the distribution of the
    microfacets, as importance sampling does (Walter et al. 2007), which resolves their lobe however narrow it is:
    Gauss-Legendre over the cumulative distribution ``ξ`` of the polar angle of halfway directions, which
    ``tan² θ = α² ξ / (1 - ξ)`` inverts for a width ``α``, and evenly over their azimuth. Over the density of halfway
    directions ``D(h) cos θ``, the albedo is the mean of ``G F (v · h) / ((n · v) cos θ)``, where the masking ``G`` and
    the Fresnel reflectance ``F`` are those of the direction towards the light that mirrors the viewer's across ``h``,
    see :func:`_ggx_reflectance`."""
    alpha = max(roughness * roughness, _MIN_ALPHA)
    t, w = (torch.as_tensor(x, dtype=torch.float64) for x in np.polynomial.legendre.leggauss(128))
    xi, weights = (t + 1) / 2, w / 2
    cos_half = 1 / (1 + alpha * alpha * xi / (1 - xi)).sqrt()
    # The albedo is symmetric across the plane of the view direction, over which half of the azimuths are enough
    phi = (torch.arange(128, dtype=torch.float64) + 0.5) / 128 * math.pi
    # The albedo at grazing views is its limit, which the masking of microfacets keeps finite
    cos_view = torch.linspace(0, 1, size, dtype=torch.float64).clamp_min(1e-4)
    c, p, v = torch.meshgrid(cos_half, phi, cos_view, indexing="ij")
    sine = (1 - c * c).sqrt()
    half = torch.stack([sine * torch.cos(p), sine * torch.sin(p), c], dim=-1)
    view = torch.stack([(1 - v * v).sqrt(), torch.zeros_like(v), v], dim=-1)
    cos_view_half = (view * half).sum(dim=-1)
    cos_light = 2 * cos_view_half * c - v
    values = _masking(cos_light, v, alpha * alpha) * _fresnel(cos_view_half.clamp(0, 1), ior) * cos_view_half / (v * c)
    values = torch.where((cos_light > 0) & (cos_view_half > 0), values, torch.zeros_like(values))
    return ((values * weights[:, None, None]).sum(dim=0).mean(dim=0)).numpy()


@lru_cache(maxsize=16)
def _specular_albedo_tensor(roughness: float, ior: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """The table of :func:`_specular_albedo_table` on a device, kept there, as copying it to a GPU at every lookup
    would wait for the GPU."""
    return torch.as_tensor(_specular_albedo_table(roughness, ior), dtype=dtype, device=device)


def specular_albedo(cosines: torch.Tensor, roughness: float, ior: float) -> torch.Tensor:
    """Directional albedo of a specular layer, i.e. the share of the light arriving evenly from every direction that it
    reflects towards a viewer, which is also the share of light that it reflects of the light coming from the viewer's
    direction, by reciprocity.

    Args:
        cosines (torch.Tensor): Cosines between the normal and the direction of the viewer, of any shape.
        roughness (float): Roughness of the microfacets, see :class:`SpecularLayer`.
        ior (float): Index of refraction of the coating.

    Returns:
        torch.Tensor: Albedo, of the same shape.
    """
    table = _specular_albedo_tensor(roughness, ior, cosines.dtype, cosines.device)
    x = cosines.clamp(0, 1) * (len(table) - 1)
    low = x.floor().clamp(max=len(table) - 2)
    i = low.long()
    return torch.lerp(table[i], table[i + 1], x - low)


def specular_factors(
    layer: SpecularLayer, albedo: torch.Tensor, normals: torch.Tensor, to_light: torch.Tensor, to_viewer: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factors by which the radiance of Lambertian surfaces becomes that of surfaces with a specular layer, as Blender's
    Principled BSDF: a dielectric coating of GGX microfacets, which reflects the light of the lamp around the mirror
    direction, and the light arriving from every direction, the medium's glow and the light of other surfaces, as much
    as its directional albedo ``E`` towards the viewer, over a diffuse base of albedo ``ρ``, which Blender dims by the
    light that the coating reflects towards the viewer, ``1 - E`` (albedo scaling). Given the radiance of a Lambertian
    surface of albedo ``ρ``, ``L``, and the part of it that the lamp lights directly, ``L_d``, that of the surface with a
    specular layer towards the viewer is ``L · (1 - E + E / ρ) + L_d · (π f - E) / ρ``, where ``f`` is the BRDF of the
    coating, see :func:`specular_albedo`.

    Args:
        layer (SpecularLayer): Specular layer of the surfaces.
        albedo (torch.Tensor): Albedo of their diffuse base per channel, of shape (c,).
        normals (torch.Tensor): Unit normals of the surfaces, of shape (..., 3).
        to_light (torch.Tensor): Unit directions from the surfaces towards the lamp, of shape (..., 3).
        to_viewer (torch.Tensor): Unit directions from the surfaces towards the viewer, of shape (..., 3).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Factors of the radiance ``L`` and of its direct part ``L_d``, each of shape
        (..., c), which are one and zero where the albedo is zero.
    """
    cos_light = (normals * to_light).sum(dim=-1)
    cos_view = (normals * to_viewer).sum(dim=-1)
    half = _unit(to_light + to_viewer)
    reflectance = _ggx_reflectance(
        cos_light, cos_view, (normals * half).sum(dim=-1), (to_light * half).sum(dim=-1), layer.roughness**2, layer.ior
    )
    total, coated, inverse = _coating(layer, albedo, cos_view)
    return total, (reflectance[..., None] - coated) * inverse


def _coating(
    layer: SpecularLayer, albedo: torch.Tensor, cos_view: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Factor of the radiance of Lambertian surfaces lit by light arriving from every direction, ``1 - E + E / ρ``, see
    :func:`specular_factors`, of shape (..., c), the directional albedo ``E`` of the coating, of shape (..., 1), and
    ``1 / ρ``, or zero where the albedo is zero, of shape (c,), given the cosines between the normals of surfaces and the
    directions towards the viewer, of any shape."""
    coated = specular_albedo(cos_view, layer.roughness, layer.ior)[..., None]
    albedo = albedo.to(coated)
    inverse = torch.where(albedo > 0, 1 / albedo.clamp_min(1e-12), torch.zeros_like(albedo))
    total = torch.where(albedo > 0, 1 - coated + coated * inverse, torch.ones_like(coated))
    return total, coated, inverse


def _up(like: torch.Tensor) -> torch.Tensor:
    """Upward unit vector, of shape (3,), made on the device of a tensor without copying it from the CPU, which would
    wait for a GPU."""
    up = like.new_zeros(3)
    up[2] = 1
    return up


def _ground_radiance(
    reflection: LampReflection,
    radiance: torch.Tensor,
    points: torch.Tensor,
    directions: torch.Tensor,
    distance: torch.Tensor,
    radii: torch.Tensor,
    azimuths: torch.Tensor,
) -> torch.Tensor:
    """Radiance of the ground, as looked up in its texture, of shape (k, m, c), where rays from points of shape (k, 3)
    along directions of shape (k, m, 3) reach it after a distance of shape (k, m), at distances to the lamp's foot and
    azimuths around it of shape (k, m), towards the points, with the ground's specular layer, if any."""
    layer = reflection.specular
    if layer is None:
        return radiance
    ground = points[:, None] + distance[..., None] * directions
    to_light = _unit(layer.light.to(ground) - ground)
    half = _unit(to_light - directions)
    # Cosines with the ground's normal, which points up, are the vertical components of directions
    cos_view = -directions[..., 2]
    reflectance = _ggx_reflectance(
        to_light[..., 2], cos_view, half[..., 2], (to_light * half).sum(dim=-1), layer.roughness**2, layer.ior
    )
    total, coated, inverse = _coating(layer, layer.albedos[0], cos_view)
    direct = _texture_lookup(layer.texture, radii, azimuths, reflection.grid.extent)
    return radiance * total + direct * ((reflectance[..., None] - coated) * inverse)


def _weighed(weights: torch.Tensor, light: torch.Tensor) -> torch.Tensor:
    """Sum over sources of weights of shape (k, s, c'), in a single channel or in every channel, times the light of
    each source, of shape (s, c), of shape (k, c)."""
    if weights.shape[-1] == 1:
        return weights[..., 0] @ light.to(weights)
    return torch.einsum("ksc,sc->kc", weights, light.to(weights))


def _layered(
    layer: SpecularLayer,
    albedo: torch.Tensor,
    weights: torch.Tensor,
    cos_view: torch.Tensor,
    light: torch.Tensor,
    direct: torch.Tensor,
    reflectance: torch.Tensor,
) -> torch.Tensor:
    """Weighted sum of the light of sources with a specular layer towards points, ``L (1 - E + E / ρ) + L_d (π f - E) /
    ρ`` for each, see :func:`specular_factors`, as sums of weights times the light of sources, which take the channels
    out of the factors of the layer, i.e. a single channel for weights in a single channel.

    Args:
        layer (SpecularLayer): Specular layer of the sources.
        albedo (torch.Tensor): Albedo of their diffuse base per channel, ``ρ``, of shape (c,).
        weights (torch.Tensor): Weight of each source at each point, of shape (k, s, c').
        cos_view (torch.Tensor): Cosine between the normal of each source and the direction towards each point, of
            shape (k, s).
        light (torch.Tensor): Light of each source as a Lambertian surface, ``L``, of shape (s, c).
        direct (torch.Tensor): Part of it that the lamp lights directly, ``L_d``, of the first sources, of shape (d, c).
        reflectance (torch.Tensor): π times the BRDF of the coating ``f`` towards each point for the lamp's light, of
            the first sources, of shape (k, d).

    Returns:
        torch.Tensor: Weighted light of the sources, of shape (k, c).
    """
    coated = specular_albedo(cos_view, layer.roughness, layer.ior)[..., None]
    albedo = albedo.to(light)
    inverse = torch.where(albedo > 0, 1 / albedo.clamp_min(1e-12), torch.zeros_like(albedo))
    total = _weighed(weights * (1 - coated), light) + _weighed(weights * coated, light * inverse)
    if len(direct):
        lit = len(direct)
        total = total + _weighed(weights[:, :lit] * (reflectance[..., None] - coated[:, :lit]), direct * inverse)
    return total


def _nodes_towards(reflection: LampReflection, weights: torch.Tensor, to_viewer: torch.Tensor) -> torch.Tensor:
    """Weighted sum of the light of the nodes of the quadrature over the ground towards points, with the ground's
    specular layer, given the weight of each node at each point, of shape (k, n, c'), and the unit directions from the
    nodes towards the points, of shape (k, n, 3), of shape (k, c), see :func:`_layered`."""
    layer = reflection.specular
    assert layer is not None
    nodes = reflection.nodes.positions
    to_light = _unit(layer.light.to(nodes) - nodes)
    half = _unit(to_light + to_viewer)
    cos_view = to_viewer[..., 2]
    reflectance = _ggx_reflectance(
        to_light[:, 2], cos_view, half[..., 2], (to_light * half).sum(dim=-1), layer.roughness**2, layer.ior
    )
    return _layered(
        layer, layer.albedos[0], weights, cos_view, reflection.node_radiance, layer.node_radiance, reflectance
    )


def _patches_towards(reflection: LampReflection, weights: torch.Tensor, to_viewer: torch.Tensor) -> torch.Tensor:
    """Weighted sum of the light of the patches of objects towards points, with their specular layer, given the weight
    of each patch at each point, of shape (k, o, c'), and the unit directions from the patches towards the points, of
    shape (k, o, 3), of shape (k, c), see :func:`_layered`."""
    layer, patches = reflection.specular, reflection.patches
    assert layer is not None
    cos_view = (patches.normals * to_viewer).sum(dim=-1)
    # The patches of every surface that follow those of the lamp's map (see `surface_receivers`) aren't lit directly
    # by the lamp, so that only the directional albedo of their coating is needed
    lit = len(patches.areas) - reflection.receivers
    positions, normals = patches.positions[:lit], patches.normals[:lit]
    to_light = _unit(layer.light.to(positions) - positions)
    half = _unit(to_light + to_viewer[:, :lit])
    reflectance = _ggx_reflectance(
        (normals * to_light).sum(dim=-1),
        cos_view[:, :lit],
        (normals * half).sum(dim=-1),
        (to_light * half).sum(dim=-1),
        layer.roughness**2,
        layer.ior,
    )
    return _layered(layer, layer.albedos[1], weights, cos_view, patches.power, layer.power[:lit], reflectance)


def _node_light(reflection: LampReflection, points: torch.Tensor) -> torch.Tensor:
    """Radiance of the ground at the nodes of the quadrature over it, of shape (n, c), or towards points of shape
    (k, 3), of shape (k, n, c), with the ground's specular layer."""
    layer = reflection.specular
    if layer is None:
        return reflection.node_radiance
    nodes = reflection.nodes.positions
    to_light = _unit(layer.light.to(nodes) - nodes)
    total, part = specular_factors(layer, layer.albedos[0], _up(nodes), to_light, _unit(points[:, None] - nodes))
    return reflection.node_radiance * total + layer.node_radiance * part


def _patch_power(reflection: LampReflection, points: torch.Tensor) -> torch.Tensor:
    """Power of the patches of objects, of shape (o, c), or as seen from points of shape (k, 3), i.e. π times their
    radiance towards the points times their area, of shape (k, o, c), with the specular layer of objects."""
    layer, patches = reflection.specular, reflection.patches
    if layer is None:
        return patches.power
    to_viewer = _unit(points[:, None] - patches.positions)
    # The patches of every surface that follow those of the lamp's map (see `surface_receivers`) aren't lit directly
    # by the lamp, so that only the directional albedo of their coating is needed, which saves most of the time
    lit = len(patches.areas) - reflection.receivers
    total = _coating(layer, layer.albedos[1], (patches.normals * to_viewer).sum(dim=-1))[0]
    power = patches.power * total
    if not lit:
        return power
    to_light = _unit(layer.light.to(patches.positions[:lit]) - patches.positions[:lit])
    part = specular_factors(layer, layer.albedos[1], patches.normals[:lit], to_light, to_viewer[:, :lit])[1]
    return torch.cat([power[:, :lit] + layer.power[:lit] * part, power[:, lit:]], dim=1)


def phase_directions(
    view: torch.Tensor, g: float, n_angle: int = 8, n_azimuth: int = 12, below: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quadrature of the phase function over the directions above the horizon, or below it, as unit vectors.

    These are the nodes of :func:`sky_quadrature <visionsim.medium.optics.sky_quadrature>`, i.e. such that
    ``sum(weights * f(nodes))`` approximates ``∫ p(ω·v) f(ω) dω`` over directions ``ω`` above the horizon, for a view
    direction ``v``, as unit vectors ``cos γ · v + sin γ · (cos ψ · e₁ + sin ψ · e₂)``, where ``e₁`` points up within
    the vertical plane of ``v``, and ``e₂`` is horizontal. Directions below the horizon are those above it for the view
    direction mirrored across the horizon, mirrored back.

    Args:
        view (torch.Tensor): Unit view directions ``v``, of shape (..., 3).
        g (float): Asymmetry parameter of the Henyey-Greenstein phase function.
        n_angle (int, optional): Number of nodes of each of the three pieces of the angle's range. Defaults to 8.
        n_azimuth (int, optional): Number of azimuthal nodes. Defaults to 12.
        below (bool, optional): If true, integrate over the directions below the horizon. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Unit directions of the nodes, of shape (..., 3 * n_angle * n_azimuth, 3),
        and their weights, of shape (..., 3 * n_angle * n_azimuth), which sum to the fraction of the phase function
        above (or below) the horizon.
    """
    kwargs: dict[str, Any] = {"dtype": view.dtype, "device": view.device}
    mirror = torch.tensor([1.0, 1.0, -1.0], **kwargs)
    v = view * mirror if below else view
    t, w = (torch.as_tensor(a, **kwargs) for a in np.polynomial.legendre.leggauss(n_angle))
    s, b = (torch.as_tensor(a, **kwargs) for a in np.polynomial.legendre.leggauss(n_azimuth))
    t, w = (t * (3 - t * t) / 2 + 1) / 2, w * 3 * (1 - t * t) / 4
    s, b = s * (3 - s * s) / 2, b * 3 * (1 - s * s) / 2

    v_z = v[..., 2:3].clamp(-1, 1)
    rho = (1 - v_z * v_z).clamp_min(0).sqrt()
    bounds = [torch.zeros_like(v_z), _hg_cdf(-rho, g), _hg_cdf(rho, g), torch.ones_like(v_z)]
    xi = torch.cat([lo + (hi - lo) * t for lo, hi in zip(bounds[:-1], bounds[1:])], dim=-1)
    xi_weights = torch.cat([(hi - lo) * w for lo, hi in zip(bounds[:-1], bounds[1:])], dim=-1)
    cos_gamma = _hg_inverse_cdf(xi, g)
    sin_gamma = (1 - cos_gamma * cos_gamma).clamp_min(0).sqrt()
    psi_max = torch.arccos((-cos_gamma * v_z / (sin_gamma * rho).clamp_min(1e-12)).clamp(-1, 1))
    psi = psi_max[..., None] * s
    weights = xi_weights[..., None] * psi_max[..., None] * b / (2 * math.pi)

    # Frame around v, whose first axis points up within the vertical plane of v, or along x for vertical views
    zenith = torch.tensor([0.0, 0.0, 1.0], **kwargs)
    first = (zenith - v_z * v) / rho.clamp_min(1e-12)
    first = torch.where(rho > 1e-9, first, torch.tensor([1.0, 0.0, 0.0], **kwargs).expand_as(first))
    second = torch.linalg.cross(v, first)
    ring = torch.cos(psi)[..., None] * first[..., None, None, :] + torch.sin(psi)[..., None] * second[..., None, None, :]
    directions = cos_gamma[..., None, None] * v[..., None, None, :] + sin_gamma[..., None, None] * ring
    directions = directions * mirror if below else directions
    return directions.flatten(-3, -2), weights.flatten(-2)


def _shift_rows(values: torch.Tensor, k: int) -> torch.Tensor:
    """Values shifted down by ``k`` rows, repeating the first or last row."""
    rows = (torch.arange(len(values), device=values.device) - k).clamp(0, len(values) - 1)
    return values[rows]


def lit_surfaces(
    maps: ShadowMaps, index: int, ground_height: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Surfaces that a lamp lights, i.e. the first surface in every direction of its map, but for the ground.

    The normal of each texel's surface is estimated from the positions of its neighbors, along each axis of the map from
    the neighbor whose distance to the lamp differs the least, as the other one may lie across an edge of the surface.

    Args:
        maps (ShadowMaps): Maps of lamps, see :func:`lamp_visibility <visionsim.medium.occlusion.lamp_visibility>`.
        index (int): Index of the lamp's map.
        ground_height (float): Height of the ground, on and below which texels are left out, as the ground is a plane of
            its own.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Position of the surface of each texel, of shape
        (t, 3), its unit normal on the side of the lamp, of shape (t, 3), the solid angle of the texel as seen from the
        lamp, of shape (t,), and the index of the texel in the map, row-major from its bottom row, of shape (t,).
    """
    height, width = (int(size) for size in maps.shapes[index])
    first = int(maps.offsets[index])
    kwargs: dict[str, Any] = {"dtype": maps.depths.dtype, "device": maps.depths.device}
    depth = maps.depths[first : first + height * width].reshape(height, width)
    origin = maps.origins[index]
    # Directions of the texels' centers, see `lamp_visibility`
    z = torch.cos((1 - (torch.arange(height, **kwargs) + 0.5) / height) * math.pi)[:, None].expand(height, width)
    phi = (math.pi - 2 * math.pi * (torch.arange(width, **kwargs) + 0.5) / width)[None, :].expand(height, width)
    s = (1 - z * z).clamp_min(0).sqrt()
    positions = origin + depth[..., None] * torch.stack([s * torch.cos(phi), s * torch.sin(phi), z], dim=-1)

    def tangent(axis: int) -> torch.Tensor:
        # Maps wrap around horizontally
        def shift(x: torch.Tensor, k: int) -> torch.Tensor:
            return torch.roll(x, k, dims=1) if axis else _shift_rows(x, k)

        forward = (shift(depth, -1) - depth).abs() <= (shift(depth, 1) - depth).abs()
        return torch.where(forward[..., None], shift(positions, -1) - positions, positions - shift(positions, 1))

    towards = _unit(origin - positions)
    normals = torch.linalg.cross(tangent(0), tangent(1))
    normals = torch.where(((normals * towards).sum(dim=-1) < 0)[..., None], -normals, normals)
    # Where both neighbors along an axis are missing, surfaces are assumed to face the lamp
    length = normals.norm(dim=-1, keepdim=True)
    normals = torch.where(torch.isfinite(length) & (length > 0), normals / length.clamp_min(1e-30), towards)
    solid = (s * (math.pi / height) * (2 * math.pi / width)).expand(height, width)
    # Texels on the ground, within a few times their footprint, are left out
    footprint = depth * (2 * math.pi / width)
    keep = (depth < 1e9) & (positions[..., 2] > ground_height + (2 * footprint).clamp_min(0.01))
    texel = torch.arange(height * width, device=depth.device).reshape(height, width)
    return positions[keep], normals[keep], solid[keep], texel[keep]


def surface_patches(
    medium: Medium,
    lamp: Lamp,
    beta: torch.Tensor,
    maps: ShadowMaps,
    index: int,
    ground_height: float,
    albedo: torch.Tensor,
    size: float = REFLECTION_GRID.patches,
    time: float = 0.0,
    similarity: bool = True,
) -> Patches:
    """Patches of the surfaces of a lamp's map, but for the ground, which reflect the lamp's light, each as a one-sided
    Lambertian point emitter.

    Texels are grouped by blocks of about ``size`` radians of the map, by distance to the lamp, in shells about ``size``
    times the distance thick, and by the main axis of their normal, so that patches are about ``size`` times their
    distance to the lamp wide, and that the faces of a box fall into different patches. Each texel's surface receives
    the light of the lamp within the texel's solid angle, and has the area that the solid angle covers at its distance
    and incidence, which is clamped at grazing incidence, where normals are the least reliable.

    Args:
        medium (Medium): Participating medium.
        lamp (Lamp): Point, spot or area light, or emissive surface.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,).
        maps (ShadowMaps): Maps of lamps.
        index (int): Index of the lamp's map.
        ground_height (float): Height of the ground, which isn't part of the patches.
        albedo (torch.Tensor): Albedo of the surfaces per channel, of shape (c,).
        size (float, optional): Size of the patches, in radians as seen from the lamp. Defaults to 0.2.
        time (float, optional): Time in seconds, used by moving media. Defaults to 0.0.
        similarity (bool, optional): Whether the lamp's light is attenuated with the extinction reduced by the
            similarity relation, rather than the full one, see :func:`lamp_irradiance
            <visionsim.medium.surfaces.lamp_irradiance>`. Defaults to True.

    Returns:
        Patches: Patches of the surfaces, of which there can be none.
    """
    kwargs: dict[str, Any] = {"dtype": beta.dtype, "device": beta.device}
    positions, normals, solid, texel = lit_surfaces(maps, index, ground_height)
    positions, normals, solid = positions.to(**kwargs), normals.to(**kwargs), solid.to(**kwargs)
    if not len(positions):
        return _no_patches(len(beta), **kwargs)
    areas, distance = _texel_areas(maps.origins[index].to(**kwargs), positions, normals, solid)
    power = albedo * lamp_irradiance(medium, lamp, beta, positions, normals, time, similarity)[1] * areas[:, None]

    width = int(maps.shapes[index][1])
    block = max(1, round(size / (2 * math.pi / width)))
    shell = torch.floor(torch.log(distance.clamp_min(1e-6)) / size).long()
    main = normals.abs().argmax(dim=-1)
    side = normals.gather(-1, main[:, None])[:, 0] > 0
    keys = torch.stack([texel // width // block, texel % width // block, shell, 2 * main + side], dim=-1)
    patch = torch.unique(keys, dim=0, return_inverse=True)[1]
    count = int(patch.max()) + 1

    def total(values: torch.Tensor) -> torch.Tensor:
        return values.new_zeros(count, *values.shape[1:]).index_add_(0, patch, values)

    weight = power.mean(dim=-1)
    weights = total(weight)
    lit = weights > 0
    centers = total(weight[:, None] * positions)[lit] / weights[lit, None]
    return Patches(centers, _unit(total(areas[:, None] * normals))[lit], total(areas)[lit], total(power)[lit])


def _texel_areas(
    origin: torch.Tensor, positions: torch.Tensor, normals: torch.Tensor, solid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Area of the surface of the texels of a lamp's map, see :func:`lit_surfaces`, i.e. the area that the solid angle
    of each texel covers at its distance and incidence, which is clamped at grazing incidence, where normals are the
    least reliable, of shape (t,), and the distance of each to the lamp, of shape (t,), given the lamp's position, of
    shape (3,), and the positions, normals and solid angles of the texels."""
    to_lamp = origin - positions
    distance = to_lamp.norm(dim=-1)
    incidence = (normals * to_lamp).sum(dim=-1) / distance.clamp_min(1e-12)
    return distance**2 * solid / incidence.clamp_min(0.1), distance


def _no_patches(channels: int, **kwargs) -> Patches:
    empty = torch.zeros(0, 3, **kwargs)
    return Patches(empty, empty, torch.zeros(0, **kwargs), torch.zeros(0, channels, **kwargs))


def map_points(maps: ShadowMaps, equirectangular: bool = False) -> torch.Tensor:
    """Points of the surfaces that shadow maps record, i.e. the first surface of each of their texels.

    Args:
        maps (ShadowMaps): Shadow maps, orthographic as those of suns and of the sky (see :class:`ShadowMaps
            <visionsim.medium.occlusion.ShadowMaps>`), or equirectangular as those of lamps (see :func:`lamp_visibility
            <visionsim.medium.occlusion.lamp_visibility>`).
        equirectangular (bool, optional): Whether the maps are equirectangular. Defaults to False.

    Returns:
        torch.Tensor: World-space points, of shape (t, 3).
    """
    points = [maps.depths.new_zeros(0, 3)]
    for i in range(len(maps.offsets)):
        height, width = (int(size) for size in maps.shapes[i])
        first = int(maps.offsets[i])
        depth = maps.depths[first : first + height * width].reshape(height, width)
        rows, columns = (depth < 1e9).nonzero(as_tuple=True)
        distance = depth[rows, columns][:, None]
        rows, columns = rows.to(depth.dtype), columns.to(depth.dtype)
        if equirectangular:
            z = torch.cos((1 - (rows + 0.5) / height) * math.pi)
            phi = math.pi - 2 * math.pi * (columns + 0.5) / width
            s = (1 - z * z).clamp_min(0).sqrt()
            points.append(maps.origins[i] + distance * torch.stack([s * torch.cos(phi), s * torch.sin(phi), z], -1))
        else:
            # Maps look along the opposite of their third axis, from a plane spanned by the other two
            right, up, towards = maps.axes[i]
            across = ((columns - width / 2 + 0.5) * maps.texels[i])[:, None] * right
            along = ((rows - height / 2 + 0.5) * maps.texels[i])[:, None] * up
            points.append(maps.origins[i] + across + along - distance * towards)
    return torch.cat(points)


@lru_cache(maxsize=8)
def scene_surfaces(
    occlusion: Occlusion, ground_height: float | None = None, size: float = SURFACE_SPACING
) -> torch.Tensor:
    """Points of the surfaces of a scene that its shadow maps record, those of suns, of the sky and of lamps, which
    together see every side of objects standing on the ground, merged into one point per cube of ``size`` meters.

    Maps record the ground too, unless it was left out of them, and surfaces lying on it, such as roads, which are left
    out before merging, so that they don't drag down the points of objects that stand on the ground.

    Args:
        occlusion (Occlusion): Shadow maps of the scene, see :func:`load_occlusion
            <visionsim.medium.occlusion.load_occlusion>`.
        ground_height (float | None, optional): Height of the ground, on and below which, up to
            :data:`GROUND_CLEARANCE` above it, surfaces are left out. Defaults to None, i.e. all surfaces are kept.
        size (float, optional): Size of the cubes, in meters. Defaults to :data:`SURFACE_SPACING`.

    Returns:
        torch.Tensor: Mean point of the surfaces within each cube, of shape (t, 3).
    """
    parts = [map_points(occlusion.sun_maps), map_points(occlusion.sky_maps)]
    if occlusion.lamp_maps is not None:
        parts.append(map_points(occlusion.lamp_maps, equirectangular=True))
    points = torch.cat(parts).double()
    if ground_height is not None:
        points = points[points[:, 2] > ground_height + GROUND_CLEARANCE]
    if not len(points):
        return points.to(occlusion.sky_maps.depths.dtype)
    cells = torch.floor(points / size).long()
    # Cubes are keyed by a single integer, which sorts faster than rows of three
    low, span = cells.min(dim=0).values, cells.max(dim=0).values - cells.min(dim=0).values + 1
    keys = ((cells - low) * torch.stack([span[1] * span[2], span[2], torch.ones_like(span[0])])).sum(dim=-1)
    cube = torch.unique(keys, return_inverse=True)[1]
    counts = points.new_zeros(int(cube.max()) + 1 if len(cube) else 0).index_add_(0, cube, torch.ones_like(points[:, 0]))
    merged = points.new_zeros(len(counts), 3).index_add_(0, cube, points) / counts[:, None]
    return merged.to(occlusion.sky_maps.depths.dtype)


class Surfels(NamedTuple):
    """Small pieces of the surfaces of a scene, each with its normal and area, see :func:`scene_surfels`."""

    positions: torch.Tensor
    """position of each piece, of shape (t, 3)"""
    normals: torch.Tensor
    """unit normal of each piece, on the side from which the maps see it, of shape (t, 3)"""
    areas: torch.Tensor
    """area of each piece, in m², of shape (t,)"""
    maps: tuple[ShadowMaps, ...] = ()
    """orthographic shadow maps that the pieces were found in, those of suns and of the sky, in which the pieces of
    other surfaces are looked up, see :func:`unrecorded_surfaces`"""


@lru_cache(maxsize=8)
def scene_surfels(occlusion: Occlusion, ground_height: float | None = None, sharpness: float = 4.0) -> Surfels:
    """Pieces of every surface that the orthographic shadow maps of a scene record, those of suns and of the sky, which
    together see every side of objects standing on the ground but their bottom, with their normals and areas.

    The normal of each texel's surface is estimated from the positions of its neighbors, along each axis of the map from
    the neighbor whose depth differs the least, as for the maps of lamps, see :func:`lit_surfaces`. A surface is seen by
    many maps, among which each texel takes a share of it, ``max(n · d, 0)^k`` over its sum over the directions ``d``
    of the maps that see it, for the normal ``n`` of its surface, so that the shares of these maps add up to one, and
    that those that face it most take most of it, where its texels are largest: a texel stands for an area of its size
    squared over ``n · d``. Maps that don't see a piece of surface, hidden by another object or beyond their edges, are
    found by looking its point up in them, as shadows are, see :func:`visibility
    <visionsim.medium.occlusion.visibility>`. In the night scene of ``examples/medium``, the areas of the sides and tops
    of the boxes are within 4% of their true areas, and the 700,000 pieces take 20 MB, cached by the identity of the
    shadow maps for the last 8 of them, and 4 to 6 s on 4 CPU threads to find.

    Args:
        occlusion (Occlusion): Shadow maps of the scene, see :func:`load_occlusion
            <visionsim.medium.occlusion.load_occlusion>`.
        ground_height (float | None, optional): Height of the ground, on and below which, up to
            :data:`GROUND_CLEARANCE` above it, surfaces are left out, as the ground is a plane of its own. Defaults to
            None, i.e. all surfaces are kept.
        sharpness (float, optional): Exponent ``k`` of the shares of the maps. Defaults to 4.

    Returns:
        Surfels: Pieces of the surfaces, one per texel of the maps but those of other surfaces, in the precision of the
        maps, with the maps.
    """
    sets = [m for m in (occlusion.sun_maps, occlusion.sky_maps) if len(m.offsets)]
    dtype, device = occlusion.sky_maps.depths.dtype, occlusion.sky_maps.depths.device
    maps = [(m, i) for m in sets for i in range(len(m.offsets))]
    if not maps:
        empty = torch.zeros(0, 3, dtype=dtype, device=device)
        return Surfels(empty, empty, empty[:, 0])
    directions = torch.cat([m.axes[:, 2] for m in sets]).double()
    pieces = []
    for k, (m, i) in enumerate(maps):
        positions, normals, texel = _map_surfaces(m, i)
        keep = torch.ones_like(positions[:, 0], dtype=torch.bool)
        if ground_height is not None:
            keep = positions[:, 2] > ground_height + GROUND_CLEARANCE
        positions, normals = positions[keep], normals[keep]
        facing = (normals @ directions.T).clamp_min(0) ** sharpness
        # The map records its own texels, whose points the others may see or not
        seen = torch.cat([_seen_in(other, positions) for other in sets], dim=-1)
        seen[:, k] = 1
        share = facing[:, k] / (facing * seen).sum(dim=-1).clamp_min(1e-30)
        areas = share * texel * texel / (normals @ directions[k]).clamp_min(0.2)
        kept = share > 1e-4
        pieces.append((positions[kept], normals[kept], areas[kept]))
    positions, normals, areas = (torch.cat([piece[j] for piece in pieces]).to(dtype) for j in range(3))
    return Surfels(positions, normals, areas, tuple(sets))


def unrecorded_surfaces(
    surfels: Surfels, positions: torch.Tensor, normals: torch.Tensor, areas: torch.Tensor
) -> Surfels:
    """Pieces of other surfaces, such as those that a lamp lights, that none of the maps of the pieces of every surface
    records, e.g. the underside of a roof, which the maps of suns and of the sky don't see, so that these pieces lack
    them, see :func:`scene_surfels`.

    A map records a piece of surface if it records its point, as found by looking it up in the map as the pieces of
    every surface are, and if the piece faces it, as maps only record the pieces that face them. Pieces that maps record
    in part, where the lookup is filtered at the edges of what they record, keep the share of their area that no map
    records.

    Args:
        surfels (Surfels): Pieces of every surface, with the maps they were found in, see :func:`scene_surfels`.
        positions (torch.Tensor): Positions of the other pieces, of shape (t, 3).
        normals (torch.Tensor): Their unit normals, on the side from which they are seen, of shape (t, 3).
        areas (torch.Tensor): Their areas, in m², of shape (t,).

    Returns:
        Surfels: The pieces that no map records, without maps, in the precision of the pieces given.
    """
    hidden = torch.ones_like(areas, dtype=torch.float64)
    for maps in surfels.maps:
        if not len(positions):
            break
        # Looked up where the maps are, which can differ from where the pieces are
        seen = _seen_in(maps, positions.to(maps.origins))
        facing = (normals.to(maps.axes) @ maps.axes[:, 2].T > 0).double()
        hidden = hidden * (1 - seen * facing).prod(dim=-1).to(hidden)
    kept = hidden > 1e-3
    return Surfels(positions[kept], normals[kept], (areas * hidden.to(areas))[kept])


def _seen_in(maps: ShadowMaps, points: torch.Tensor, bias: float = 2.0) -> torch.Tensor:
    """Whether each of a set of orthographic maps records points of surfaces, of shape (t, 3), rather than surfaces in
    front of them or nothing, of shape (t, k), filtered as shadows are, see :func:`visibility
    <visionsim.medium.occlusion.visibility>`. Points may lie behind the surface that a map records by ``bias`` texels,
    as maps record points of the surface beside them, at depths that differ by up to a texel over ``n · d``."""
    count = len(maps.offsets)
    if not len(points) or not count:
        return points.new_zeros(len(points), count)
    dtype = maps.origins.dtype
    points = points.to(dtype)
    # Points are looked up along rays from a common origin, away from all of them
    origin = points.min(dim=0).values - 1.0
    offset = points - origin
    distance = offset.norm(dim=-1, keepdim=True)
    seen = points.new_zeros(len(points), count)
    for i in range(0, len(points), max(1, int(2e6 // count))):
        chunk = slice(i, i + max(1, int(2e6 // count)))
        seen[chunk] = visibility(maps, origin, offset[chunk] / distance[chunk], distance[chunk], bias)[:, 0]
    # Points beyond the edges of a map, which it doesn't record, count as lit there
    return seen.double() * _within(maps, points).double()


def _within(maps: ShadowMaps, points: torch.Tensor) -> torch.Tensor:
    """Whether points of shape (t, 3) lie within the extent of each of a set of orthographic maps, of shape (t, k)."""
    half = maps.shapes.to(maps.texels) / 2 * maps.texels[:, None]
    coordinates = torch.einsum("tkd,kad->tka", points[:, None] - maps.origins, maps.axes)
    return (coordinates[..., 0].abs() <= half[:, 1]) & (coordinates[..., 1].abs() <= half[:, 0])


def _map_surfaces(maps: ShadowMaps, index: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Position and unit normal of the first surface of each texel of an orthographic shadow map that records one, in
    double precision, each of shape (t, 3), facing the map, and the size of its texels. Normals are estimated from the
    positions of the texel's neighbors, along each axis from the neighbor whose depth differs the least."""
    height, width = (int(size) for size in maps.shapes[index])
    first = int(maps.offsets[index])
    depth = maps.depths[first : first + height * width].reshape(height, width).double()
    right, up, towards = maps.axes[index].double()
    texel = float(maps.texels[index])
    rows = (torch.arange(height, dtype=depth.dtype, device=depth.device) - height / 2 + 0.5) * texel
    columns = (torch.arange(width, dtype=depth.dtype, device=depth.device) - width / 2 + 0.5) * texel
    # Maps look along the opposite of their third axis, from a plane spanned by the other two, see `map_points`
    origin = maps.origins[index].double()
    positions = origin + columns[None, :, None] * right + rows[:, None, None] * up - depth[..., None] * towards

    def tangent(axis: int) -> torch.Tensor:
        forward = (torch.roll(depth, -1, axis) - depth).abs() <= (torch.roll(depth, 1, axis) - depth).abs()
        ahead, behind = torch.roll(positions, -1, axis), torch.roll(positions, 1, axis)
        return torch.where(forward[..., None], ahead - positions, positions - behind)

    normals = _unit(torch.linalg.cross(tangent(1), tangent(0)))
    normals = torch.where(((normals * towards).sum(dim=-1) < 0)[..., None], -normals, normals)
    seen = depth < 1e9
    return positions[seen], normals[seen], texel


def surface_receivers(
    surfels: Surfels, center: torch.Tensor, extent: float, size: float = REFLECTION_GRID.patches
) -> Patches:
    """Patches of every surface within a distance of a lamp's foot, which receive the light of the ground and of the
    other objects that the lamp lights, see :func:`lamp_reflection`, grouped as :func:`surface_patches` groups the
    surfaces of the lamp's map: by blocks of about ``size`` radians of azimuth and elevation around the lamp's foot, by
    distance to it, in shells about ``size`` times the distance thick, and by the main axis of their normal.

    Args:
        surfels (Surfels): Pieces of the surfaces of the scene, see :func:`scene_surfels`.
        center (torch.Tensor): Foot of the lamp on the ground, of shape (3,).
        extent (float): Distance to the lamp's foot beyond which surfaces are left out, in meters.
        size (float, optional): Size of the patches, in radians as seen from the lamp's foot. Defaults to 0.2.

    Returns:
        Patches: Patches of the surfaces, which reflect no power yet, of which there can be none.
    """
    kwargs: dict[str, Any] = {"dtype": center.dtype, "device": center.device}
    positions, normals, areas = (x.to(**kwargs) for x in (surfels.positions, surfels.normals, surfels.areas))
    offset = positions - center
    distance = offset.norm(dim=-1)
    within = distance < extent
    positions, normals, areas, offset, distance = (x[within] for x in (positions, normals, areas, offset, distance))
    if not len(positions):
        return _no_patches(1, **kwargs)
    azimuth = torch.atan2(offset[:, 1], offset[:, 0])
    elevation = torch.asin((offset[:, 2] / distance.clamp_min(1e-12)).clamp(-1, 1))
    shell = torch.floor(torch.log(distance.clamp_min(1e-6)) / size)
    main = normals.abs().argmax(dim=-1)
    side = normals.gather(-1, main[:, None])[:, 0] > 0
    keys = torch.stack([torch.floor(azimuth / size), torch.floor(elevation / size), shell, 2 * main + side], dim=-1)
    patch = torch.unique(keys.long(), dim=0, return_inverse=True)[1]
    count = int(patch.max()) + 1

    def total(values: torch.Tensor) -> torch.Tensor:
        return values.new_zeros(count, *values.shape[1:]).index_add_(0, patch, values)

    weights = total(areas)
    return Patches(
        total(areas[:, None] * positions) / weights[:, None],
        _unit(total(areas[:, None] * normals)),
        weights,
        positions.new_zeros(count, 1),
    )


def _anchor_rings(height: float, grid: ReflectionGrid) -> tuple[float, int]:
    """Radius of the first ring of anchors and number of rings, up to the grid's extent."""
    first, ratio = height / 4, grid.anchors[0]
    return first, max(1, math.floor(math.log(max(grid.extent[1] / first, 1.0)) / math.log(ratio)) + 1)


def anchor_positions(center: torch.Tensor, height: float, grid: ReflectionGrid = REFLECTION_GRID) -> torch.Tensor:
    """Anchors on the ground around a lamp's foot, from which objects hide the light of the ground: the foot itself,
    and rings of anchors, whose radii grow geometrically, see :attr:`ReflectionGrid.anchors`.

    Args:
        center (torch.Tensor): Foot of the lamp on the ground, of shape (3,).
        height (float): Height of the lamp above the ground.
        grid (ReflectionGrid, optional): Resolution of the grid. Defaults to :data:`REFLECTION_GRID`.

    Returns:
        torch.Tensor: Position of each anchor, the foot first, then ring by ring, of shape (m, 3).
    """
    first, rings = _anchor_rings(height, grid)
    ratio, count = grid.anchors
    radii = first * ratio ** torch.arange(rings, dtype=center.dtype, device=center.device)
    azimuths = torch.arange(count, dtype=center.dtype, device=center.device) / count * 2 * math.pi
    r, a = torch.meshgrid(radii, azimuths, indexing="ij")
    ring = torch.stack([r * torch.cos(a), r * torch.sin(a), torch.zeros_like(r)], dim=-1).reshape(-1, 3)
    return torch.cat([center[None], center + ring])


def nearest_anchor(
    radii: torch.Tensor, azimuths: torch.Tensor, height: float, grid: ReflectionGrid = REFLECTION_GRID
) -> torch.Tensor:
    """Index of the anchor nearest to points of the ground, in log distance to the lamp's foot and in azimuth around it,
    see :func:`anchor_positions`.

    Args:
        radii (torch.Tensor): Distance of the points to the lamp's foot, of any shape.
        azimuths (torch.Tensor): Azimuth of the points around the lamp's foot, of the same shape.
        height (float): Height of the lamp above the ground.
        grid (ReflectionGrid, optional): Resolution of the grid. Defaults to :data:`REFLECTION_GRID`.

    Returns:
        torch.Tensor: Index of the anchors, of the same shape.
    """
    first, rings = _anchor_rings(height, grid)
    ratio, count = grid.anchors
    ring = torch.round(torch.log(radii.clamp_min(1e-12) / first) / math.log(ratio)).clamp(0, rings - 1).long()
    sector = torch.remainder(torch.round(azimuths / (2 * math.pi) * count).long(), count)
    # Points closer to the foot than the first ring, by half the ratio between rings, belong to the foot
    return torch.where(radii < first / math.sqrt(ratio), 0, 1 + ring * count + sector)


def anchor_maps(
    anchors: torch.Tensor,
    surfaces: torch.Tensor,
    size: tuple[int, int],
    spacing: float = SURFACE_SPACING,
    reach: int = 24,
) -> ShadowMaps:
    """Maps of the distance from anchors on the ground to the nearest of surface points in every direction above their
    horizon, where the medium lies, as the upper half of the maps of lamps, see :func:`lamp_visibility
    <visionsim.medium.occlusion.lamp_visibility>`: rows spread elevations evenly from the horizon up to the zenith, and
    columns spread azimuths as in the maps of lamps.

    Surfaces are sampled by points about ``spacing`` apart, each of which stands for a disk of surface (a surfel) of
    radius three quarters of the spacing, which covers the texels around its own within the angle that the disk spans
    as seen from an anchor, up to ``reach`` texels away, so that surfaces close to an anchor leave no holes between
    their points. This widens surfaces by up to a quarter of their spacing. Disks face anchors, so points level with an
    anchor should belong to objects, not to the ground around it, which they would raise to the disks' height, see
    :data:`GROUND_CLEARANCE`. Distances are stored in half precision, which is ample against the bias with which they
    are compared, see :func:`anchor_visibility`.

    Args:
        anchors (torch.Tensor): Position of each anchor, of shape (m, 3).
        surfaces (torch.Tensor): Points of the surfaces, of shape (t, 3), e.g. from :func:`scene_surfaces`.
        size (tuple[int, int]): Height and width of the maps.
        spacing (float, optional): Distance between neighboring points of the surfaces, in meters. Defaults to
            :data:`SURFACE_SPACING`.
        reach (int, optional): Number of texels, around that of a point, up to which a point covers the map. Defaults
            to 24, which covers the whole disks of points further from an anchor than about their spacing, in maps of
            64 rows.

    Returns:
        ShadowMaps: One map per anchor, whose texels are in radians.
    """
    height, width = size
    kwargs: dict[str, Any] = {"dtype": anchors.dtype, "device": anchors.device}
    texels = height * width
    depths = torch.full((len(anchors) * texels,), 1e10, **kwargs)
    step = max(1, int(2e7 // max(1, len(surfaces))))
    for a in range(0, len(anchors), step):
        offset = surfaces - anchors[a : a + step, None]
        distance = offset.norm(dim=-1).clamp_min(1e-9)
        sine = (offset[..., 2] / distance).clamp(-1, 1)
        u = (math.pi - torch.atan2(offset[..., 1], offset[..., 0])) / (2 * math.pi)
        # Rows of points below the horizon are negative, and only the parts of their disks above it are kept
        column = torch.remainder(torch.floor(u * width).long(), width).reshape(-1)
        row = torch.floor(torch.asin(sine) / (math.pi / 2) * height).long().clamp(max=height - 1).reshape(-1)
        first = (
            (torch.arange(a, a + len(offset), device=anchors.device) * texels)[:, None].expand_as(distance).reshape(-1)
        )
        distance = distance.reshape(-1)
        above = row >= 0
        depths.scatter_reduce_(0, (first + row * width + column)[above], distance[above], reduce="amin")
        # Half-widths of the disks in texels, along rows, and along columns, which are narrower towards the zenith
        angle = torch.atan(0.75 * spacing / distance)
        rows = torch.floor(angle / (math.pi / 2 / height) + 0.5).clamp(max=reach).long()
        cosine = (1 - sine * sine).clamp_min(0).sqrt().reshape(-1)
        columns = torch.floor(angle / (2 * math.pi / width * cosine.clamp_min(1 / width)) + 0.5).clamp(max=reach).long()
        wide = ((rows > 0) | (columns > 0)).nonzero().squeeze(-1)
        if not len(wide):
            continue
        rows, columns = rows[wide], columns[wide]
        for shift in range(-int(rows.max()), int(rows.max()) + 1):
            within = rows >= abs(shift)
            shifted = row[wide[within]] + shift
            kept = shifted >= 0
            chosen, shifted, spans = (
                wide[within][kept],
                shifted[kept].clamp(max=height - 1),
                2 * columns[within][kept] + 1,
            )
            pair = torch.repeat_interleave(chosen, spans)
            starts = torch.repeat_interleave(torch.cumsum(spans, dim=0) - spans, spans)
            across = torch.arange(len(pair), device=anchors.device) - starts - torch.repeat_interleave(spans // 2, spans)
            index = torch.repeat_interleave(shifted, spans) * width + torch.remainder(column[pair] + across, width)
            depths.scatter_reduce_(0, first[pair] + index, distance[pair], reduce="amin")
    return ShadowMaps(
        depths=depths.half(),
        offsets=torch.arange(len(anchors), device=anchors.device) * texels,
        shapes=torch.tensor([[height, width]] * len(anchors), device=anchors.device),
        origins=anchors,
        axes=torch.eye(3, **kwargs).expand(len(anchors), 3, 3),
        texels=torch.full((len(anchors),), 2 * math.pi / width, **kwargs),
    )


def anchor_visibility(maps: ShadowMaps, points: torch.Tensor, bias: float = 1.0) -> torch.Tensor:
    """Visibility of points from every anchor, as :func:`lamp_visibility <visionsim.medium.occlusion.lamp_visibility>`
    sees points from a lamp, but for all anchors at once, in their maps of the directions above their horizon.

    Args:
        maps (ShadowMaps): Maps of anchors, see :func:`anchor_maps`.
        points (torch.Tensor): Points, of shape (k, 3), which are taken as level with anchors when below their horizon.
        bias (float, optional): Distance, in texels at the distance of each point, by which points can lie behind the
            recorded surface and still be lit. Defaults to 1.0.

    Returns:
        torch.Tensor: Visibility in [0, 1], of shape (k, m), in the precision of the points.
    """
    height, width = (int(size) for size in maps.shapes[0])
    texel = float(maps.texels[0])
    result = []
    step = max(1, int(4e6 // max(1, len(maps.origins))))
    for i in range(0, len(points), step):
        offset = points[i : i + step, None].to(maps.origins.dtype) - maps.origins
        distance = offset.norm(dim=-1)
        u = (math.pi - torch.atan2(offset[..., 1], offset[..., 0])) / (2 * math.pi)
        v = torch.asin((offset[..., 2] / distance.clamp_min(1e-12)).clamp(-1, 1)) / (math.pi / 2)
        x, y = u * width - 0.5, v * height - 0.5
        x0, y0 = x.floor(), y.floor()
        fx, fy = x - x0, y - y0
        columns = [torch.remainder(x0, width).long(), torch.remainder(x0 + 1, width).long()]
        rows = [y0.clamp(0, height - 1).long(), (y0 + 1).clamp(0, height - 1).long()]
        behind = distance * (1 - bias * texel)
        lit = [[(behind <= maps.depths[maps.offsets + r * width + c]).to(offset.dtype) for c in columns] for r in rows]
        bottom, upper = (torch.lerp(row[0], row[1], fx) for row in lit)
        result.append(torch.lerp(bottom, upper, fy).to(points.dtype))
    return torch.cat(result) if result else points.new_zeros(0, len(maps.origins))


def ground_nodes(
    texture: torch.Tensor, center: torch.Tensor, height: float, grid: ReflectionGrid = REFLECTION_GRID
) -> tuple[GroundNodes, torch.Tensor]:
    """Nodes of a quadrature over the ground around a lamp's foot, whose density follows the radiance of the ground.

    The mean radiance over channels of a :func:`ground_texture` is taken as constant over each of its cells, between
    consecutive tabulated distances and azimuths, and within the smallest distance, which makes a density per unit area
    proportional to it. Rings of nodes split its marginal over distances evenly, at the middle of each share, and the
    nodes of each ring split its marginal over azimuths within the ring likewise, shifted by half a share every other
    ring. Each node then stands for a share ``1 / n`` of the radiance, i.e. for an area of ``1 / (n · density)``. The
    density in every cell weighs this quadrature against others, see :func:`scatter`.

    Args:
        texture (torch.Tensor): Radiance of the ground, see :func:`ground_texture`, of shape (g, a, c).
        center (torch.Tensor): Foot of the lamp on the ground, of shape (3,).
        height (float): Height of the lamp above the ground, which sets its anchors, see :func:`anchor_positions`.
        grid (ReflectionGrid, optional): Resolution of the grid, see :attr:`ReflectionGrid.nodes`. Defaults to
            :data:`REFLECTION_GRID`.

    Returns:
        tuple[GroundNodes, torch.Tensor]: Nodes of the quadrature, of which there are none without light, and their
        density per unit area in each cell, by distance and azimuth, of shape (g, a), see :func:`_cell_density`.
    """
    kwargs: dict[str, Any] = {"dtype": center.dtype, "device": center.device}
    (r_min, r_max), (n_rings, n_nodes) = grid.extent, grid.nodes
    n_radii, n_azimuths = texture.shape[:2]
    radii = r_min * (r_max / r_min) ** (torch.arange(n_radii, **kwargs) / (n_radii - 1))
    edges = torch.cat([radii.new_zeros(1), radii])
    mean = texture.mean(dim=-1).to(**kwargs)
    # Cells within the smallest distance take its radiance, as the texture does
    mean = torch.cat([mean[:1], (mean[:-1] + mean[1:]) / 2])
    values = (mean + torch.roll(mean, -1, dims=1)) / 2
    mass = values * ((edges[1:] ** 2 - edges[:-1] ** 2) / 2 * 2 * math.pi / n_azimuths)[:, None]
    total = float(mass.sum())
    if total <= 0:
        empty = torch.zeros(0, **kwargs)
        return GroundNodes(torch.zeros(0, 3, **kwargs), empty, empty, empty.long()), torch.zeros_like(values)

    def split(masses: torch.Tensor, fractions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cell of each fraction of the masses' total, by rows, and the fraction within it."""
        cumulative = torch.cat([masses.new_zeros(*masses.shape[:-1], 1), masses.cumsum(dim=-1)], dim=-1)
        targets = fractions * cumulative[..., -1:]
        cell = torch.searchsorted(cumulative[..., 1:].contiguous(), targets.contiguous(), right=True)
        cell = cell.clamp(max=masses.shape[-1] - 1)
        within = (targets - cumulative.gather(-1, cell)) / masses.gather(-1, cell).clamp_min(1e-300)
        return cell, within.clamp(0, 1)

    ring, within = split(mass.sum(dim=-1), (torch.arange(n_rings, **kwargs) + 0.5) / n_rings)
    r = (edges[ring] ** 2 + within * (edges[ring + 1] ** 2 - edges[ring] ** 2)).sqrt()
    shift = 0.5 * (torch.arange(n_rings, **kwargs) % 2)[:, None]
    fractions = torch.remainder((torch.arange(n_nodes, **kwargs)[None, :] + 0.5 + shift) / n_nodes, 1.0)
    cell, within = split(mass[ring], fractions)
    azimuths = (cell + within) / n_azimuths * 2 * math.pi
    r = r[:, None].expand_as(azimuths)
    positions = center + torch.stack([r * torch.cos(azimuths), r * torch.sin(azimuths), torch.zeros_like(r)], dim=-1)
    table = n_rings * n_nodes * values / total
    density = table[ring[:, None], cell]
    return GroundNodes(
        positions.reshape(-1, 3),
        (1 / density.clamp_min(1e-300)).reshape(-1),
        density.reshape(-1),
        nearest_anchor(r.reshape(-1), azimuths.reshape(-1), height, grid),
    ), table


def ground_texture(
    medium: Medium,
    lamp: Lamp,
    beta: torch.Tensor,
    center: torch.Tensor,
    albedo: torch.Tensor,
    grid: ReflectionGrid = REFLECTION_GRID,
    maps: ShadowMaps | None = None,
    index: int | None = None,
    time: float = 0.0,
    similarity: bool = True,
) -> torch.Tensor:
    """Radiance of the ground lit by a lamp, a Lambertian plane, tabulated against the distance to the lamp's foot,
    spread logarithmically over the grid's extent, and the azimuth around it, see :attr:`ReflectionGrid.ground`.

    The radiance of each texel is its mean over the cell around it, between the middles of its neighbors, sampled
    evenly in log distance and azimuth and weighted by area, see :attr:`ReflectionGrid.supersampling`: sharp edges of
    light, such as those of shadows and of the cones of spot lights, which are only a few texels apart around a narrow
    beam, then keep their area, which samples at the texels alone would round to whole texels.

    Args:
        medium (Medium): Participating medium.
        lamp (Lamp): Point, spot or area light, or emissive surface.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,).
        center (torch.Tensor): Foot of the lamp on the ground, of shape (3,).
        albedo (torch.Tensor): Albedo of the ground per channel, of shape (c,).
        grid (ReflectionGrid, optional): Resolution of the grid. Defaults to :data:`REFLECTION_GRID`.
        maps (ShadowMaps | None, optional): Maps of lamps, through which objects cast the lamp's shadows onto the
            ground. Defaults to None.
        index (int | None, optional): Index of the lamp's map. Defaults to None, i.e. no shadows.
        time (float, optional): Time in seconds, used by moving media. Defaults to 0.0.
        similarity (bool, optional): Whether the lamp's light is attenuated with the extinction reduced by the
            similarity relation, rather than the full one, see :func:`lamp_irradiance
            <visionsim.medium.surfaces.lamp_irradiance>`. Defaults to True.

    Returns:
        torch.Tensor: Radiance per channel, of shape (g, a, c).
    """
    kwargs: dict[str, Any] = {"dtype": beta.dtype, "device": beta.device}
    points = _texture_samples(center, grid)
    up = torch.tensor([0.0, 0.0, 1.0], **kwargs)
    radiance = albedo / math.pi * lamp_irradiance(medium, lamp, beta, points, up, time, similarity)[1]
    if maps is not None and index is not None:
        # Maps can record the ground, or roads lying on it, whose depth varies across a texel by up to the texel's
        # footprint over the tangent of the lamp's elevation, beyond the bias of the comparison at grazing angles:
        # points are looked up a texel's footprint above the ground (a normal offset, as in shadow mapping), so that
        # it doesn't shadow itself, which only shortens the shadows of objects by about a texel
        lift = (points - maps.origins[index].to(points)).norm(dim=-1) * float(maps.texels[index])
        raised = points + lift[:, None] * up
        radiance = radiance * lamp_visibility(maps, index, raised.to(maps.depths.dtype)).to(radiance)[:, None]
    return _cell_means(radiance, center, grid)


def _texture_samples(center: torch.Tensor, grid: ReflectionGrid) -> torch.Tensor:
    """Points of the ground at which the radiance of a :func:`ground_texture` is sampled, spread evenly in log distance
    to the lamp's foot and in azimuth around it within each of its cells, by distance, then sample of distance within
    the cell, azimuth, and sample of azimuth, of shape (g * s * a * s, 3)."""
    kwargs: dict[str, Any] = {"dtype": center.dtype, "device": center.device}
    (r_min, r_max), (n_radii, n_azimuths), k = grid.extent, grid.ground, grid.supersampling
    step = math.log(r_max / r_min) / (n_radii - 1)
    within = (torch.arange(k, **kwargs) + 0.5) / k - 0.5
    radii = r_min * torch.exp((torch.arange(n_radii, **kwargs)[:, None] + within) * step).reshape(-1)
    azimuths = ((torch.arange(n_azimuths, **kwargs)[:, None] + within) / n_azimuths * 2 * math.pi).reshape(-1)
    r, a = torch.meshgrid(radii, azimuths, indexing="ij")
    return center + torch.stack([r * torch.cos(a), r * torch.sin(a), torch.zeros_like(r)], dim=-1).reshape(-1, 3)


def _cell_means(radiance: torch.Tensor, center: torch.Tensor, grid: ReflectionGrid) -> torch.Tensor:
    """Means of the radiance at the samples of :func:`_texture_samples`, of shape (g * s * a * s, c), over each cell
    of the texture, weighted by area, of shape (g, a, c)."""
    (n_radii, n_azimuths), k = grid.ground, grid.supersampling
    # Areas grow as r² in log distance
    r = (_texture_samples(center, grid)[:, :2] - center[:2]).norm(dim=-1).to(radiance)
    area = (r * r).reshape(n_radii, k, n_azimuths, k, 1)
    radiance = radiance.reshape(n_radii, k, n_azimuths, k, radiance.shape[-1])
    return (radiance * area).sum(dim=(1, 3)) / area.sum(dim=(1, 3))


def _texture_lookup(
    texture: torch.Tensor, radii: torch.Tensor, azimuths: torch.Tensor, extent: tuple[float, float]
) -> torch.Tensor:
    """Bilinear interpolation of a :func:`ground_texture` at given distances to the lamp's foot and azimuths, of shape
    (...), of shape (..., c), which is zero beyond the texture's extent."""
    n_radii, n_azimuths = texture.shape[:2]
    r_min, r_max = extent
    x = (torch.log(radii.clamp_min(r_min) / r_min) / math.log(r_max / r_min) * (n_radii - 1)).clamp(0, n_radii - 1)
    y = torch.remainder(azimuths / (2 * math.pi) * n_azimuths, n_azimuths)
    x0, y0 = x.floor().clamp(max=n_radii - 2), y.floor()
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    i, j = x0.long(), y0.long() % n_azimuths
    j1 = (j + 1) % n_azimuths
    near = torch.lerp(texture[i, j], texture[i, j1], fy)
    far = torch.lerp(texture[i + 1, j], texture[i + 1, j1], fy)
    return torch.where((radii <= r_max)[..., None], torch.lerp(near, far, fx), torch.zeros_like(near))


def _cell_density(
    table: torch.Tensor, radii: torch.Tensor, azimuths: torch.Tensor, extent: tuple[float, float]
) -> torch.Tensor:
    """Density of the nodes of :func:`ground_nodes` per unit area in the cells of its table that contain points of the
    ground at given distances to the lamp's foot and azimuths, of shape (...), of the same shape, which is zero beyond
    the table's extent. Cells lie between consecutive tabulated distances, or within the smallest one, and azimuths."""
    n_radii, n_azimuths = table.shape
    r_min, r_max = extent
    x = torch.log(radii.clamp_min(r_min) / r_min) / math.log(r_max / r_min) * (n_radii - 1)
    ring = torch.where(radii < r_min, 0, x.floor().long() + 1).clamp(max=n_radii - 1)
    cell = torch.remainder(torch.floor(azimuths / (2 * math.pi) * n_azimuths).long(), n_azimuths)
    return torch.where(radii <= r_max, table[ring, cell], torch.zeros_like(radii))


def _elevations(n: int, hemisphere: bool, **kwargs) -> torch.Tensor:
    """Elevations of the points of the grid around the lamp's foot, ``π/2 · t |t|`` for ``t`` spread evenly over
    [0, 1], or over [-1, 1] when points lie all around it, which makes them denser near the ground."""
    t = torch.linspace(0 if hemisphere else -1, 1, n, **kwargs)
    return math.pi / 2 * t * t.abs()


def _grid_points(center: torch.Tensor, grid: ReflectionGrid, hemisphere: bool) -> torch.Tensor:
    """Points of the grid around the lamp's foot, by distance, elevation and azimuth, of shape (r * e * a, 3)."""
    kwargs: dict[str, Any] = {"dtype": center.dtype, "device": center.device}
    r_min, r_max = grid.extent
    radii = r_min * (r_max / r_min) ** (torch.arange(grid.radii, **kwargs) / (grid.radii - 1))
    elevations = _elevations(grid.elevations, hemisphere, **kwargs)
    azimuths = torch.arange(grid.azimuths, **kwargs) / grid.azimuths * 2 * math.pi
    r, e, a = torch.meshgrid(radii, elevations, azimuths, indexing="ij")
    offsets = torch.stack([torch.cos(e) * torch.cos(a), torch.cos(e) * torch.sin(a), torch.sin(e)], dim=-1)
    return (center + r[..., None] * offsets).reshape(-1, 3)


def gather(
    reflection: LampReflection, medium: Medium, points: torch.Tensor, beta: torch.Tensor, time: float = 0.0
) -> Gathered:
    """Gather the light of a lamp reflected by surfaces that reaches points, but for the phase function.

    Seen from a point, a node of the quadrature over the ground of area ``dA``, at a distance ``r`` and an angle ``θ``
    from the vertical, stands for a solid angle ``dA cos θ / r²``, and its light is hidden as seen from its nearest
    anchor. A patch of objects of power ``P`` and area ``A`` reaches a point at a distance ``r`` from it and an angle
    ``θ`` from its normal with an irradiance ``P / π · cos θ / (r² + A / 2π)``: that of a point emitter far from it, and
    which tends to that of an infinite plane of the same radiance close to it, as for a disk of area ``A`` on its axis.

    Args:
        reflection (LampReflection): Reflected light, whose ground, anchors and patches are used, see
            :func:`lamp_reflection`.
        medium (Medium): Participating medium.
        points (torch.Tensor): Points of the medium, of shape (k, 3).
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,), or (1,) when it is the same for every
            channel.
        time (float, optional): Time in seconds, used by moving media. Defaults to 0.0.

    Returns:
        Gathered: Light that reaches the points.
    """
    k = len(points)

    def transmittance(ends: torch.Tensor) -> torch.Tensor:
        """Transmittance between the points and other points, of shape (n, 3), of shape (k, n, c)."""
        offset = ends[None] - points[:, None]
        distance = offset.norm(dim=-1)
        depth = optical_depth(
            medium, points[:, None], offset / distance.clamp_min(1e-12)[..., None], distance, time=time
        )
        return torch.exp(-depth[..., None] * beta)

    visible = points.new_ones(k, 1) if reflection.anchors is None else anchor_visibility(reflection.anchors, points)

    nodes = reflection.nodes
    to_nodes = nodes.positions[None] - points[:, None]
    distance = to_nodes.norm(dim=-1).clamp_min(1e-12)
    above = (points[:, 2:3] - float(reflection.center[2])).clamp_min(0)
    seen = visible[:, nodes.anchors] if visible.shape[1] > 1 else visible
    solid = seen * above * nodes.areas / distance**3
    node_weights = solid[..., None] * transmittance(nodes.positions)
    node_density = (nodes.density * distance**3 / above.clamp_min(1e-12)).clamp_max(1e30)

    patches = reflection.patches
    offset = points[:, None] - patches.positions[None]
    distance = offset.norm(dim=-1).clamp_min(1e-12)
    cosine = ((offset * patches.normals[None]).sum(dim=-1) / distance).clamp_min(0)
    patch_weights = (cosine / (math.pi * (distance**2 + patches.areas / (2 * math.pi))))[..., None]
    return Gathered(visible, node_weights, node_density, patch_weights * transmittance(patches.positions))


def scatter(
    reflection: LampReflection,
    gathered: Gathered,
    points: torch.Tensor,
    view: torch.Tensor,
    medium: Medium,
    beta: torch.Tensor,
    time: float = 0.0,
) -> torch.Tensor:
    """Reflected light that the medium scatters along view directions at points, per unit of extinction, i.e. the
    albedo times the radiance arriving from every direction weighted by the phase function.

    The light of the ground is the sum of two quadratures over the directions below the horizon, each weighted by the
    balance heuristic, i.e. by its density of nodes per steradian over the sum of both densities: one over directions,
    whose nodes follow the phase function, see :func:`phase_directions`, and whose density is taken as the number of
    nodes times the phase function over its integral below the horizon, and one over the ground, see
    :func:`ground_nodes`, whose density is that of the cell of the ground where a direction reaches it, so that both
    weights add up to one in every direction. Each direction of the first sees the radiance of the ground interpolated
    in the :func:`ground_texture`, attenuated by the medium, and hidden as seen from the anchor nearest to where it
    reaches the ground. With a specular layer, the radiance of the ground and of patches is that towards each point,
    which the layer reflects around the mirror direction of the lamp, see :func:`specular_factors`.

    Args:
        reflection (LampReflection): Reflected light, see :func:`lamp_reflection`.
        gathered (Gathered): Light that reaches the points, see :func:`gather`.
        points (torch.Tensor): Points of the medium, of shape (k, 3).
        view (torch.Tensor): Unit view directions at the points, i.e. the directions of the camera's rays, opposite to
            that of the light scattered towards the camera, of shape (k, 3).
        medium (Medium): Participating medium.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,), or (1,) when it is the same for every
            channel.
        time (float, optional): Time in seconds, used by moving media. Defaults to 0.0.

    Returns:
        torch.Tensor: Scattered radiance per unit of extinction, of shape (k, c'), where c' is the number of channels
        of the light, which can differ from that of ``beta``.
    """
    g, grid = medium.anisotropy, reflection.grid
    channels = max(reflection.node_radiance.shape[-1], reflection.patches.power.shape[-1])
    total = points.new_zeros(len(points), channels)
    nodes = reflection.nodes
    if len(nodes.areas):
        count = 3 * grid.directions[0] * grid.directions[1]
        ground_height = float(reflection.center[2])
        size = max(1, int(4e6 // (count * (8 + 2 * channels))))
        for i in range(0, len(points), size):
            chunk = slice(i, i + size)
            x, v = points[chunk], view[chunk]
            directions, weights = phase_directions(v, g, *grid.directions, below=True)
            mass = weights.sum(dim=-1, keepdim=True).clamp_min(1e-30)
            above = (x[:, 2:3] - ground_height).clamp_min(0)
            down = (-directions[..., 2]).clamp_min(1e-9)
            distance = above / down
            offset = x[:, None, :2] + distance[..., None] * directions[..., :2] - reflection.center[:2]
            radii, azimuths = offset.norm(dim=-1), torch.atan2(offset[..., 1], offset[..., 0])
            radiance = _texture_lookup(reflection.texture, radii, azimuths, grid.extent)
            # Balance heuristic, with densities per steradian, r² / cos θ = above² / cos³ θ for the ground's
            phase = count * henyey_greenstein((v[:, None] * directions).sum(dim=-1), g) / mass
            ground = _cell_density(reflection.cell_density, radii, azimuths, grid.extent) * above**2 / down**3
            share = phase / (phase + ground).clamp_min(1e-30)
            visible = gathered.visible[chunk]
            if visible.shape[1] > 1:
                visible = visible.gather(1, nearest_anchor(radii, azimuths, reflection.height, grid))
            depth = optical_depth(medium, x[:, None], directions, distance, time=time)
            radiance = _ground_radiance(reflection, radiance, x, directions, distance, radii, azimuths)
            radiance = radiance * (visible[..., None] * torch.exp(-depth[..., None] * beta))
            total[chunk] += ((weights * share)[..., None] * radiance).sum(dim=-2)

            towards = _unit(nodes.positions[None] - x[:, None])
            phase = henyey_greenstein((v[:, None] * towards).sum(dim=-1), g)
            density = gathered.node_density[chunk]
            share = phase * density / (count * phase / mass + density).clamp_min(1e-30)
            light = share[..., None] * gathered.node_weights[chunk]
            if gathered.toward or reflection.specular is None:
                light = light if gathered.toward else light * reflection.node_radiance
                total[chunk] += light.sum(dim=-2)
            else:
                total[chunk] += _nodes_towards(reflection, light, -towards)
    patches = reflection.patches
    if len(patches.areas):
        # In chunks of points, as patches of every surface can be many, but large ones, as each takes a few operations
        size = max(1, int(32e6 // (len(patches.areas) * (8 + 2 * channels))))
        for i in range(0, len(points), size):
            x, chunk = points[i : i + size], slice(i, i + size)
            towards = _unit(patches.positions[None] - x[:, None])
            phase = henyey_greenstein((view[chunk, None] * towards).sum(dim=-1), g)
            light = phase[..., None] * gathered.patch_weights[chunk]
            if gathered.toward or reflection.specular is None:
                light = light if gathered.toward else light * patches.power
                total[chunk] = total[chunk] + light.sum(dim=-2)
            else:
                total[chunk] = total[chunk] + _patches_towards(reflection, light, -towards)
    return medium.albedo * total


def arriving_light(
    reflection: LampReflection,
    medium: Medium,
    points: torch.Tensor,
    directions: torch.Tensor,
    density: Callable[[torch.Tensor], torch.Tensor],
    beta: torch.Tensor,
    time: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Light of a lamp reflected by surfaces that reaches points directly, before the medium scatters it, through a
    quadrature over given directions around each point, combined with those of the ground and of objects.

    As in :func:`scatter`, the light of the ground is the sum of two quadratures, each weighted by the balance
    heuristic: one over the given directions, along which the radiance of the ground is looked up in its
    :func:`ground_texture`, attenuated by the medium and hidden as seen from the anchor nearest to where they reach the
    ground, and whose density per steradian is given, and the quadrature over the ground, see :func:`ground_nodes`.
    Unlike :func:`scatter`'s, neither depends on the direction towards which the light is then scattered, so that the
    light arriving at points can be scattered towards many directions at once, as by the halos of lamps, see
    :mod:`visionsim.medium.halos`. Patches of objects shine as the point emitters they are. With a specular layer, the
    radiance of the ground and of patches is that towards each point, see :func:`specular_factors`.

    Args:
        reflection (LampReflection): Reflected light, whose ground, anchors and patches are used, see
            :func:`lamp_reflection`.
        medium (Medium): Participating medium.
        points (torch.Tensor): Points of the medium, of shape (k, 3).
        directions (torch.Tensor): Unit directions from each point, towards where the light comes from, of shape
            (k, m, 3).
        density (Callable[[torch.Tensor], torch.Tensor]): Density per steradian of the quadrature over the directions
            around each point, given unit directions of shape (k, q, 3), of shape (k, q), which should be the
            reciprocal of the solid angle that each of ``directions`` stands for at those directions.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,).
        time (float, optional): Time in seconds, used by moving media. Defaults to 0.0.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Radiance arriving from each direction, weighted by the
        balance heuristic, of shape (k, m, c'), where c' is the number of channels of the reflected light, unit
        directions from the points towards the nodes of the ground and the patches of objects, of shape (k, q, 3), and
        the light arriving from each, i.e. its radiance times the solid angle it stands for, or the irradiance of
        patches, weighted by the balance heuristic, of shape (k, q, c').
    """
    k, m = directions.shape[:2]
    inner = _channels(beta)
    channels = max(reflection.node_radiance.shape[-1], reflection.patches.power.shape[-1])
    gathered = gather(reflection, medium, points, inner, time)
    radiance = points.new_zeros(k, m, channels)
    sources, light = [points.new_zeros(k, 0, 3)], [points.new_zeros(k, 0, channels)]
    nodes, grid = reflection.nodes, reflection.grid
    if len(nodes.areas):
        above = (points[:, 2:3] - float(reflection.center[2])).clamp_min(0)
        down = -directions[..., 2]
        reaching = down > 0
        distance = torch.where(reaching, above / down.clamp_min(1e-9), torch.zeros_like(down))
        offset = points[:, None, :2] + distance[..., None] * directions[..., :2] - reflection.center[:2]
        radii, azimuths = offset.norm(dim=-1), torch.atan2(offset[..., 1], offset[..., 0])
        seen = _texture_lookup(reflection.texture, radii, azimuths, grid.extent)
        visible = gathered.visible
        if visible.shape[1] > 1:
            visible = visible.gather(1, nearest_anchor(radii, azimuths, reflection.height, grid))
        depth = optical_depth(medium, points[:, None], directions, distance, time=time)
        # Balance heuristic, with densities per steradian, r² / cos θ = above² / cos³ θ for the ground's
        own = density(directions)
        ground = (
            _cell_density(reflection.cell_density, radii, azimuths, grid.extent) * above**2 / down.clamp_min(1e-9) ** 3
        )
        share = own / (own + ground).clamp_min(1e-30)
        seen = _ground_radiance(reflection, seen, points, directions, distance, radii, azimuths)
        arriving = (share * visible * reaching)[..., None] * seen * torch.exp(-depth[..., None] * inner)
        radiance = torch.nan_to_num(arriving, nan=0.0, posinf=0.0)

        towards = _unit(nodes.positions[None] - points[:, None])
        own = density(towards)
        share = gathered.node_density / (gathered.node_density + own).clamp_min(1e-30)
        sources.append(towards)
        light.append(share[..., None] * gathered.node_weights * _node_light(reflection, points))
    patches = reflection.patches
    if len(patches.areas):
        sources.append(_unit(patches.positions[None] - points[:, None]))
        light.append(gathered.patch_weights * _patch_power(reflection, points))
    return (
        radiance.expand(k, m, channels),
        torch.cat(sources, dim=1),
        torch.cat([x.expand(k, -1, channels) for x in light], dim=1),
    )


def lamp_reflection(
    medium: Medium,
    lamp: Lamp,
    beta: torch.Tensor,
    ground_height: float = 0.0,
    ground_albedo: tuple[float, ...] = (0.0,),
    object_albedo: tuple[float, ...] = (0.0,),
    maps: ShadowMaps | None = None,
    occluders: torch.Tensor | None = None,
    grid: ReflectionGrid = REFLECTION_GRID,
    time: float = 0.0,
    map_tolerance: float = 0.01,
    glow: bool = False,
    bounces: bool = True,
    shadowed: bool = True,
    occluded: bool = True,
    specular: tuple[float, float] | None = None,
    surfels: Surfels | None = None,
    interreflections: bool = True,
) -> LampReflection:
    """Gather the light of a lamp reflected by surfaces on a grid of points around the lamp's foot, but for the phase
    function towards the camera.

    The grid's points are spread logarithmically in distance to the lamp's foot, over the grid's extent, by elevation
    as seen from it, denser near the ground, and evenly in azimuth around it, see :class:`ReflectionGrid`. Surfaces are
    lit by the lamp through the medium with the extinction reduced by the similarity relation, which also counts the
    light that the medium scatters forward on its way to them, or, with ``glow``, with the full extinction, as the light
    that reaches them directly, to which the light of the medium's glow is added later, see :func:`relit_reflection`.
    Objects and the ground light each other, see :func:`_bounced`: the patches of objects that the lamp lights receive
    the light of the ground, or, given the pieces of every surface of the scene, ``surfels``, the patches of every
    surface within the grid's extent receive the light of the ground and of the objects that the lamp lights instead,
    see :func:`surface_receivers`, which then reflect it, while the patches of the lamp's map keep its direct light.
    Given the roughness and index of refraction of a specular layer, the radiance of surfaces towards each point of the
    medium is that of a Principled BSDF over their Lambertian radiance, see :func:`specular_factors`.

    Args:
        medium (Medium): Participating medium.
        lamp (Lamp): Point, spot or area light, or emissive surface.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,).
        ground_height (float, optional): Height of the ground. Defaults to 0.0.
        ground_albedo (tuple[float, ...], optional): Albedo of the ground per channel, or a single one. Defaults to
            (0.0,), i.e. the ground reflects no light.
        object_albedo (tuple[float, ...], optional): Albedo of objects per channel, or a single one. Defaults to (0.0,),
            i.e. objects reflect no light.
        maps (ShadowMaps | None, optional): Maps of lamps, see :attr:`Occlusion.lamp_maps
            <visionsim.medium.occlusion.Occlusion.lamp_maps>`, among which the lamp's map is found by its position, which
            casts its shadows onto the ground, and whose surfaces reflect its light. Defaults to None, i.e. only the
            ground reflects light.
        occluders (torch.Tensor | None, optional): Points of the surfaces that hide the light of the ground, of shape
            (t, 3), see :func:`scene_surfaces`. Defaults to None, i.e. the surfaces of the lamp's map, if any.
        grid (ReflectionGrid, optional): Resolution of the grids and of the quadratures. Defaults to
            :data:`REFLECTION_GRID`.
        time (float, optional): Time in seconds, used by moving media. Defaults to 0.0.
        map_tolerance (float, optional): Distance, in meters, within which the lamp uses a map rendered at another
            position, see :attr:`Occlusion.lamp_tolerance <visionsim.medium.occlusion.Occlusion.lamp_tolerance>`.
            Defaults to 0.01.
        glow (bool, optional): Whether surfaces are lit by the light of the lamp that reaches them directly, with the
            full extinction, rather than with the reduced extinction of the similarity relation. Defaults to False.
        bounces (bool, optional): Whether objects and the ground light each other, here and once relit by the glow,
            see :func:`relit_reflection`, once, or until little is left with ``interreflections``, see
            :func:`_bounced`. Defaults to True.
        shadowed (bool, optional): Whether the ground lies in the shadows that objects cast from the lamp, as its map
            tells, see :func:`ground_texture`. Otherwise, the lamp lights the whole ground. Defaults to True.
        occluded (bool, optional): Whether objects hide the light of the ground from points behind them, as seen from
            anchors on the ground, see :func:`anchor_maps`. Otherwise, the light of the ground reaches every point.
            Defaults to True.
        specular (tuple[float, float] | None, optional): Roughness and index of refraction of the specular layer of
            surfaces, see :func:`specular_factors`, which is taken as :data:`MIN_ROUGHNESS` if smoother. Defaults to
            None, i.e. surfaces are Lambertian.
        surfels (Surfels | None, optional): Pieces of every surface of the scene, see :func:`scene_surfels`, whose
            patches within the grid's extent receive the light of the ground and of objects, and the medium's glow,
            with ``bounces``, see :func:`surface_receivers`, along with the surfaces of the lamp's map that their maps
            don't record, see :func:`unrecorded_surfaces`. Defaults to None, i.e. only the surfaces that the lamp
            lights receive them.
        interreflections (bool, optional): Whether light keeps bouncing between the ground and objects, with
            ``bounces``, until little is left, rather than once. Defaults to True.

    Returns:
        LampReflection: Reflected light on the grid.
    """
    kwargs: dict[str, Any] = {"dtype": beta.dtype, "device": beta.device}
    n = len(beta)
    position = torch.as_tensor(lamp.position, **kwargs)
    # Maps of area lights are rendered just in front of them
    if isinstance(lamp, AreaLight):
        position = position + 1e-3 * _unit(torch.as_tensor(lamp.direction, **kwargs))
    maps = maps_to(maps, device=beta.device, dtype=torch.float32) if maps is not None else None
    index = lamp_map_index(maps, position, map_tolerance)
    height = max(float(position[2]) - ground_height, 0.0)
    center = torch.cat([position[:2], position.new_tensor([ground_height])]) if height > 0 else position
    ground = _per_channel(ground_albedo, n, "ground albedo", **kwargs)
    objects = _per_channel(object_albedo, n, "object albedo", **kwargs)
    surfaces = lit_surfaces(maps, index, ground_height) if maps is not None and index is not None else None

    texture, node_radiance, anchors = torch.zeros(0, 0, n, **kwargs), torch.zeros(0, n, **kwargs), None
    empty = torch.zeros(0, **kwargs)
    nodes, cells = GroundNodes(torch.zeros(0, 3, **kwargs), empty, empty, empty.long()), torch.zeros(0, 0, **kwargs)
    patches = _no_patches(n, **kwargs)
    if surfaces is not None and len(surfaces[0]) and bool((objects > 0).any()):
        assert maps is not None and index is not None
        patches = surface_patches(
            medium, lamp, beta, maps, index, ground_height, objects, grid.patches, time, similarity=not glow
        )
    lit, lit_nodes, lit_cells = texture, nodes, cells
    # The light that reaches surfaces directly from the lamp, which their specular layer reflects around the mirror
    # direction, unlike the light of other surfaces and of the medium
    direct, direct_power = lit, patches.power
    reflecting = height > 0 and bool((ground > 0).any())
    receivers = None
    transfer: dict[str, torch.Tensor] | None = None
    rounds = _ROUNDS if interreflections else 1
    if reflecting:
        shadowing = index if shadowed else None
        lit = ground_texture(medium, lamp, beta, center, ground, grid, maps, shadowing, time, similarity=not glow)
        direct = lit
        if bounces:
            # Objects and the ground light each other, or every surface of the scene, which objects light too, if the
            # maps record any within the grid's extent, with the surfaces that the lamp lights but that no map of suns
            # or of the sky records, e.g. under a roof
            if surfels is not None:
                origin = maps.origins[index] if maps is not None and index is not None else position
                every = _every_surface(surfels, surfaces, origin, center, grid.extent[1])
                receivers = surface_receivers(every, center, grid.extent[1], grid.patches)
                receivers = receivers._replace(power=receivers.power.expand(-1, n)) if len(receivers.areas) else None
            # The light that surfaces exchange is computed once for every round, and kept for the glow to relight them
            transfer = {} if rounds > 1 or receivers is not None else None
            bounced, power = _bounced(
                medium, beta, center, grid, (ground, objects), lit, patches, time, not glow, receivers, rounds, transfer
            )
            lit = lit + bounced
            patches = (
                patches._replace(power=patches.power + power)
                if receivers is None
                else _joined(patches, receivers, power)
            )
        lit_nodes, lit_cells = ground_nodes(lit, center, height, grid)
    # The ground only reflects light where the lamp lights it, or, with the medium's glow, wherever it reaches it
    if len(lit_nodes.areas):
        texture, nodes, cells = lit, lit_nodes, lit_cells
        node_radiance = _node_radiance(texture, nodes, center, grid)
    layer = None
    if specular is not None:
        direct = direct if texture.numel() else texture
        direct_nodes = _node_radiance(direct, nodes, center, grid) if len(nodes.areas) else node_radiance
        # Receivers only reflect the light of other surfaces and of the medium
        direct_power = torch.cat([direct_power, patches.power.new_zeros(len(patches.areas) - len(direct_power), n)])
        roughness, ior = max(specular[0], MIN_ROUGHNESS), specular[1]
        layer = SpecularLayer(roughness, ior, position, (ground, objects), direct, direct_nodes, direct_power)
    if occluded and (len(lit_nodes.areas) or (glow and reflecting)):
        hiding = occluders if occluders is not None else surfaces[0] if surfaces is not None else None
        if hiding is not None:
            hiding = hiding.to(device=beta.device, dtype=torch.float32)
            # Surfaces on the ground don't hide its light from the medium above it, and those beyond the grid's extent
            # from the lamp's foot don't cross segments between its points, the anchors and the ground within it
            reach = (hiding - center.float()).norm(dim=-1) < grid.extent[1] + SURFACE_SPACING
            hiding = hiding[(hiding[:, 2] > ground_height + GROUND_CLEARANCE) & reach]
        if hiding is not None and len(hiding):
            anchors = anchor_maps(anchor_positions(center, height, grid).float(), hiding, grid.maps)

    points = _grid_points(center, grid, hemisphere=height > 0)
    nothing = Gathered(*(torch.zeros(0, **kwargs) for _ in range(4)))
    albedos = (ground, objects) if glow else None
    reflection = LampReflection(
        center,
        height,
        grid,
        texture,
        nodes,
        node_radiance,
        cells,
        anchors,
        patches,
        points,
        nothing,
        albedos,
        bounces,
        layer,
        len(receivers.areas) if receivers is not None else 0,
        rounds,
        transfer if glow else None,
    )
    return _gathered(reflection, medium, beta, time)


def _every_surface(
    surfels: Surfels,
    surfaces: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    origin: torch.Tensor,
    center: torch.Tensor,
    extent: float,
) -> Surfels:
    """Pieces of every surface, see :func:`scene_surfels`, with the surfaces of a lamp's map, see :func:`lit_surfaces`,
    within a distance of the lamp's foot that none of their maps records, see :func:`unrecorded_surfaces`, each texel a
    piece of the area it covers, given the position of the lamp's map, of shape (3,)."""
    if surfaces is None or not len(surfaces[0]):
        return surfels
    positions, normals, solid, _ = surfaces
    near = (positions - center.to(positions)).norm(dim=-1) < extent
    positions, normals, solid = positions[near], normals[near], solid[near]
    areas = _texel_areas(origin.to(positions), positions, normals, solid)[0]
    unrecorded = unrecorded_surfaces(surfels, positions, normals, areas)
    positions, normals, areas = (torch.cat([mine, theirs.to(mine)]) for mine, theirs in zip(surfels[:3], unrecorded[:3]))
    return Surfels(positions, normals, areas, surfels.maps)


def _joined(patches: Patches, receivers: Patches, power: torch.Tensor) -> Patches:
    """Patches followed by receivers, which reflect a given power, of shape (r, c)."""
    return Patches(
        torch.cat([patches.positions, receivers.positions]),
        torch.cat([patches.normals, receivers.normals]),
        torch.cat([patches.areas, receivers.areas]),
        torch.cat([patches.power, power]),
    )


def _texture_cells(center: torch.Tensor, grid: ReflectionGrid) -> tuple[torch.Tensor, torch.Tensor]:
    """Center and area of each cell of a :func:`ground_texture`, between the middles of its neighbors in log distance
    to the lamp's foot and in azimuth, by distance then azimuth, of shape (g * a, 3) and (g * a,)."""
    kwargs: dict[str, Any] = {"dtype": center.dtype, "device": center.device}
    (r_min, r_max), (n_radii, n_azimuths) = grid.extent, grid.ground
    step = math.log(r_max / r_min) / (n_radii - 1)
    index = torch.arange(n_radii, **kwargs)
    inner, outer = r_min * torch.exp((index - 0.5) * step), r_min * torch.exp((index + 0.5) * step)
    azimuths = torch.arange(n_azimuths, **kwargs) / n_azimuths * 2 * math.pi
    r, a = torch.meshgrid(r_min * torch.exp(index * step), azimuths, indexing="ij")
    positions = center + torch.stack([r * torch.cos(a), r * torch.sin(a), torch.zeros_like(r)], dim=-1).reshape(-1, 3)
    areas = ((outer**2 - inner**2) / 2 * 2 * math.pi / n_azimuths)[:, None].expand(n_radii, n_azimuths)
    return positions, areas.reshape(-1)


def _bounced(
    medium: Medium,
    beta: torch.Tensor,
    center: torch.Tensor,
    grid: ReflectionGrid,
    albedos: tuple[torch.Tensor, torch.Tensor],
    texture: torch.Tensor,
    patches: Patches,
    time: float,
    similarity: bool,
    receivers: Patches | None = None,
    rounds: int = 1,
    transfer: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Light that the ground and objects reflect onto each other.

    The patches of objects light the ground as the Lambertian point emitters they are, softened by their size, see
    :func:`gather`, at the samples of each cell of the ground's texture, and the cells of the texture light the patches
    likewise, from their center, both attenuated by the medium as the lamp's light is on its way to surfaces. Objects
    don't hide each other's light, nor the ground's, which only cosines hide from the back of surfaces. Given other
    patches that receive light, ``receivers``, such as those of every surface of the scene (see
    :func:`surface_receivers`), these receive the light of the ground and of the patches instead, so that objects light
    each other too. Light keeps bouncing between the ground and the patches, or the receivers, for up to a number of
    ``rounds``, the light of each round being the light that the previous round reflected, until it is less than a
    thousandth of the first round's.

    The light that one unit of the light of each cell of the ground or patch brings to the others through the medium
    doesn't change from one round to the next, nor when the same surfaces are lit again, by the glow of the medium, see
    :func:`relit_reflection`. Given a ``transfer``, it is computed once and kept there, about 15 MB for the patches of
    every surface of the night scene of ``examples/medium``, and rounds only weigh it by the light of the ground and of
    the patches, rather than computing it again for every round.

    Args:
        medium (Medium): Participating medium.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,).
        center (torch.Tensor): Foot of the lamp on the ground, of shape (3,).
        grid (ReflectionGrid): Resolution of the ground's texture.
        albedos (tuple[torch.Tensor, torch.Tensor]): Albedo of the ground and of objects per channel, of shape (c,).
        texture (torch.Tensor): Radiance of the ground, see :func:`ground_texture`, of shape (g, a, c).
        patches (Patches): Patches of objects, with the power they reflect.
        time (float): Time in seconds, used by moving media.
        similarity (bool): Whether light is attenuated with the extinction reduced by the similarity relation, rather
            than the full one, see :func:`lamp_irradiance <visionsim.medium.surfaces.lamp_irradiance>`.
        receivers (Patches | None, optional): Patches that receive the light of the ground and of the patches, rather
            than the patches themselves. Defaults to None.
        rounds (int, optional): Largest number of times that light bounces off surfaces. Defaults to 1.
        transfer (dict[str, torch.Tensor] | None, optional): Light exchanged between the ground and the receivers, or
            the patches without receivers, per unit of light, kept from earlier calls with the same surfaces, medium and
            ``similarity``, if any, and where it is kept for later calls. Defaults to None, i.e. it is computed for
            every round.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Radiance that the ground reflects of the light of objects, of shape
        (g, a, c), and power that the patches, or the receivers, reflect of the light of the ground and of objects, of
        shape (o, c).
    """
    ground, objects = albedos
    reduced = 1 - medium.albedo * medium.anisotropy if similarity else 1.0
    targets = patches if receivers is None else receivers
    channels = patches.power.shape[-1]
    up = center.new_tensor([0.0, 0.0, 1.0])

    def exchange(
        points: torch.Tensor,
        normals: torch.Tensor,
        emitters: torch.Tensor,
        facing: torch.Tensor,
        areas: torch.Tensor,
        extinction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Irradiance of points of shape (n, 3) facing unit normals of shape (n, 3) or (3,), from one-sided Lambertian
        emitters of shape (m, 3) facing unit normals ``facing`` of shape (m, 3), of area ``areas`` of shape (m,), per
        unit of their power, of shape (n, m), and the transmittance of the medium in between, for an extinction
        coefficient per channel of shape (c,), of shape (n, m, c)."""
        offset = points[:, None] - emitters[None]
        distance = offset.norm(dim=-1).clamp_min(1e-12)
        leaving = ((offset * facing[None]).sum(dim=-1) / distance).clamp_min(0)
        arriving = (-(offset * normals[..., None, :]).sum(dim=-1) / distance).clamp_min(0)
        weights = leaving * arriving / (math.pi * (distance**2 + areas / (2 * math.pi)))
        depth = optical_depth(medium, points[:, None], -offset / distance[..., None], distance, time=time)
        return weights, torch.exp(-reduced * depth[..., None] * extinction)

    def irradiance(
        points: torch.Tensor,
        normals: torch.Tensor,
        emitters: torch.Tensor,
        facing: torch.Tensor,
        light: torch.Tensor,
        areas: torch.Tensor,
    ) -> torch.Tensor:
        """Irradiance of points from emitters, see `exchange`, of power ``light`` of shape (m, c), of shape (n, c)."""
        weights, through = exchange(points, normals, emitters, facing, areas, beta)
        return torch.einsum("nm,nmc->nc", weights, light[None] * through)

    # The transfer is kept in a single channel when the extinction is the same in every channel, as for fog
    inner = _channels(beta)

    def kept(name: str, compute: Callable[[], torch.Tensor], keep: bool) -> torch.Tensor:
        """Transfer of a given name, computed if it isn't kept yet, and then kept if asked."""
        assert transfer is not None
        key = f"{name} {reduced}"
        if key in transfer:
            return transfer[key]
        value = compute()
        if keep:
            transfer[key] = value
        return value

    def onto_cells(emitters: Patches) -> torch.Tensor:
        """Mean irradiance over the samples of each cell of the ground's texture from each patch per unit of its power,
        through the medium, of shape (g, a, m, c')."""
        samples = _texture_samples(center, grid)
        k, (n_radii, n_azimuths) = grid.supersampling, grid.ground
        width = len(samples) // n_radii
        step = max(1, int(4e6 // (width * len(emitters.areas))))
        # Areas grow as r² in log distance, see `_cell_means`
        squared = ((samples[:, :2] - center[:2]).norm(dim=-1) ** 2).reshape(n_radii, k, n_azimuths, k, 1, 1)
        parts = []
        for i in range(0, n_radii, step):
            weights, through = exchange(
                samples[i * width : (i + step) * width],
                up,
                emitters.positions,
                emitters.normals,
                emitters.areas,
                inner,
            )
            values = (weights[..., None] * through).reshape(-1, k, n_azimuths, k, *through.shape[1:])
            area = squared[i : i + step]
            parts.append((values * area).sum(dim=(1, 3)) / area.sum(dim=(1, 3)))
        return torch.cat(parts)

    def onto_targets(emitters: torch.Tensor, facing: torch.Tensor, areas: torch.Tensor) -> torch.Tensor:
        """Irradiance of the targets from emitters per unit of their power, through the medium, of shape (t, m, c')."""
        size = max(1, int(4e6 // len(areas)))
        parts = []
        for i in range(0, len(targets.areas), size):
            chunk = slice(i, i + size)
            weights, through = exchange(targets.positions[chunk], targets.normals[chunk], emitters, facing, areas, inner)
            parts.append(weights[..., None] * through)
        return torch.cat(parts) if parts else targets.positions.new_zeros(0, len(areas), len(inner))

    def weigh(values: torch.Tensor, light: torch.Tensor) -> torch.Tensor:
        """Light that a transfer of shape (..., m, c') brings from emitters of light (m, c), of shape (..., c)."""
        if values.shape[-1] == 1:
            return torch.einsum("...m,mc->...c", values[..., 0], light)
        return torch.einsum("...mc,mc->...c", values, light)

    def onto_ground(emitters: Patches, name: str) -> torch.Tensor:
        """Radiance that the ground reflects of the light of patches, of shape (g, a, c), which are the targets, or the
        given patches, by name."""
        if not texture.numel() or not len(emitters.areas) or not bool((ground > 0).any()):
            return torch.zeros_like(texture)
        if transfer is not None:
            light = weigh(kept(f"ground<{name}", lambda: onto_cells(emitters), name == "targets"), emitters.power)
            return ground / math.pi * light
        samples = _texture_samples(center, grid)
        size = max(1, int(1e6 // len(emitters.areas)))
        light = torch.cat(
            [
                irradiance(
                    samples[i : i + size], up, emitters.positions, emitters.normals, emitters.power, emitters.areas
                )
                for i in range(0, len(samples), size)
            ]
        )
        return _cell_means(ground / math.pi * light, center, grid)

    def onto_objects(radiance: torch.Tensor, emitters: Patches | None, name: str) -> torch.Tensor:
        """Power that the targets reflect of the light of the ground of a given radiance, of shape (g, a, c), and of
        given patches, if any, which are the targets, or the given patches, by name, of shape (t, c)."""
        arriving = targets.positions.new_zeros(len(targets.areas), channels)
        if not len(targets.areas) or not bool((objects > 0).any()):
            return arriving
        if texture.numel():
            cells, areas = _texture_cells(center, grid)
            # Cells shine with the power of a Lambertian emitter of their radiance
            light = math.pi * radiance.reshape(-1, radiance.shape[-1]) * areas[:, None]
            facing = up.expand_as(cells)
            if transfer is not None:
                arriving = arriving + weigh(
                    kept("targets<cells", lambda: onto_targets(cells, facing, areas), True), light
                )
            else:
                size = max(1, int(1e6 // len(cells)))
                arriving = arriving + torch.cat(
                    [
                        irradiance(
                            targets.positions[i : i + size], targets.normals[i : i + size], cells, facing, light, areas
                        )
                        for i in range(0, len(targets.areas), size)
                    ]
                )
        if emitters is not None and len(emitters.areas):
            if transfer is not None:
                values = kept(
                    f"targets<{name}",
                    lambda: onto_targets(emitters.positions, emitters.normals, emitters.areas),
                    name == "targets",
                )
                arriving = arriving + weigh(values, emitters.power)
            else:
                size = max(1, int(1e6 // len(emitters.areas)))
                arriving = arriving + torch.cat(
                    [
                        irradiance(
                            targets.positions[i : i + size],
                            targets.normals[i : i + size],
                            emitters.positions,
                            emitters.normals,
                            emitters.power,
                            emitters.areas,
                        )
                        for i in range(0, len(targets.areas), size)
                    ]
                )
        return objects * arriving * targets.areas[:, None]

    if receivers is None and (not len(patches.areas) or not texture.numel()):
        return torch.zeros_like(texture), torch.zeros_like(patches.power)
    # The patches are the targets but for the first round of the light of objects lit by the lamp onto every surface
    first = "targets" if receivers is None or receivers.positions is patches.positions else "patches"
    bounced = onto_ground(patches, first)
    power = onto_objects(texture, patches if receivers is not None else None, first)
    if rounds > 1:
        energy, latest = _energy(bounced, power, center, grid), (bounced, power)
        for _ in range(rounds - 1):
            emitted = targets._replace(power=latest[1])
            latest = (
                onto_ground(emitted, "targets"),
                onto_objects(latest[0], emitted if receivers is not None else None, "targets"),
            )
            bounced, power = bounced + latest[0], power + latest[1]
            if _energy(*latest, center, grid) <= 1e-3 * energy:
                break
    return bounced, power


def _energy(radiance: torch.Tensor, power: torch.Tensor, center: torch.Tensor, grid: ReflectionGrid) -> float:
    """Power that the ground reflects, given its radiance in the cells of its texture, of shape (g, a, c), and that
    objects reflect, of shape (o, c), summed over channels."""
    total = float(power.sum())
    if radiance.numel():
        areas = _texture_cells(center, grid)[1].reshape(radiance.shape[:2])
        total += float((math.pi * radiance * areas[..., None]).sum())
    return total


def _node_radiance(
    texture: torch.Tensor, nodes: GroundNodes, center: torch.Tensor, grid: ReflectionGrid
) -> torch.Tensor:
    """Radiance of the ground at the nodes of the quadrature over it, of shape (n, c)."""
    offset = nodes.positions[:, :2] - center[:2]
    return _texture_lookup(texture, offset.norm(dim=-1), torch.atan2(offset[:, 1], offset[:, 0]), grid.extent)


def _gathered(reflection: LampReflection, medium: Medium, beta: torch.Tensor, time: float) -> LampReflection:
    """Reflected light with the light that reaches the grid's points gathered from the rest, see :func:`gather`, kept
    in a lower precision."""
    gathered = gather(reflection, medium, reflection.points, _channels(beta), time)
    gathered = Gathered(gathered.visible.half(), *(x.float() for x in gathered[1:4]))
    if reflection.specular is None:
        return reflection._replace(gathered=gathered)
    # With a specular layer, the light of surfaces towards the points is kept with the weights, as it doesn't depend on
    # the camera either, rather than computed at every frame, which would take longer than the rest of the light, in
    # chunks of points that take up about as much memory as the weights of a few thousand points
    points, nodes, patches = reflection.points, [], []
    size = max(1, int(4e6 // max(1, len(reflection.patches.areas) + len(reflection.nodes.areas))))
    for i in range(0, len(points), size):
        x, chunk = points[i : i + size], slice(i, i + size)
        nodes.append((gathered.node_weights[chunk] * _node_light(reflection, x)).float())
        patches.append((gathered.patch_weights[chunk] * _patch_power(reflection, x)).float())
    lights = [
        torch.cat(light) if light else weights for light, weights in ((nodes, gathered[1]), (patches, gathered[3]))
    ]
    # A single channel is kept when the light is the same in every channel, as for grey surfaces under a white lamp,
    # within the rounding of single precision, as sums over many terms on a GPU don't round alike in every channel
    lights = [light[..., :1] if _same_channels(light) else light for light in lights]
    return reflection._replace(gathered=gathered._replace(node_weights=lights[0], patch_weights=lights[1], toward=True))


def relit_reflection(
    reflection: LampReflection,
    medium: Medium,
    beta: torch.Tensor,
    glow: Glow,
    time: float = 0.0,
) -> LampReflection:
    """Light of a lamp reflected by surfaces lit by the medium's glow too, i.e. by the light of the lamp that the
    medium scattered on its way to them, from every direction.

    The irradiance of the glow is added to the light that reaches surfaces directly, with the full extinction (see
    :func:`lamp_reflection`'s ``glow``): over the samples of each cell of the ground's texture, which then reaches the
    ground in the lamp's shadows too, and at the center of each patch of objects, for its normal: the patches of every
    surface, if the reflection has them (see :func:`lamp_reflection`'s ``surfels``), or else those of the surfaces
    that the lamp lights, so that objects that the lamp doesn't light then reflect none of their glow. The ground and
    objects lit by the glow light each other too, as they do with the lamp's light. The nodes of the quadrature over the
    ground then follow the radiance of the ground lit by both, and the light that reaches the grid's points is gathered
    again.

    Args:
        reflection (LampReflection): Light reflected by surfaces lit directly, see :func:`lamp_reflection`.
        medium (Medium): Participating medium.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,).
        glow (Glow): Irradiance of surfaces from the medium's glow, given points of shape (n, 3) and unit normals of
            shape (n, 3), of shape (n, c), along with the share of the lamp's light whose glow it is, of shape (n,),
            such as :func:`halo_glow <visionsim.medium.halos.halo_glow>`.
        time (float, optional): Time in seconds, used by moving media. Defaults to 0.0.

    Returns:
        LampReflection: Light reflected by surfaces lit directly and by the medium's glow.
    """
    if reflection.albedos is None:
        return reflection
    ground, objects = reflection.albedos
    grid, center, texture = reflection.grid, reflection.center, reflection.texture
    nodes, node_radiance, cells = reflection.nodes, reflection.node_radiance, reflection.cell_density
    patches, lit = reflection.patches, len(reflection.patches.areas) - reflection.receivers
    targets = Patches(*(x[lit:] for x in patches)) if reflection.receivers else patches
    power = objects * glow(targets.positions, targets.normals)[0].to(patches.power) * targets.areas[:, None]
    if reflection.hemisphere and bool((ground > 0).any()):
        samples = _texture_samples(center, grid)
        up = samples.new_tensor([0.0, 0.0, 1.0]).expand_as(samples)
        glowing = _cell_means(ground / math.pi * glow(samples, up)[0].to(samples), center, grid)
        if reflection.bounces:
            # The light of the glow that objects and the ground reflect lights each other too
            emitted = targets._replace(power=power)
            receivers = emitted if reflection.receivers else None
            bounced, more = _bounced(
                medium,
                beta,
                center,
                grid,
                (ground, objects),
                glowing,
                emitted,
                time,
                False,
                receivers,
                reflection.rounds,
                reflection.transfer,
            )
            glowing, power = glowing + bounced, power + more
        texture = glowing if not texture.numel() else texture + glowing
        nodes, cells = ground_nodes(texture, center, reflection.height, grid)
        node_radiance = _node_radiance(texture, nodes, center, grid) if len(nodes.areas) else node_radiance[:0]
    if reflection.receivers:
        power = torch.cat([power.new_zeros(lit, power.shape[-1]), power])
    patches = patches._replace(power=patches.power + power)
    layer = reflection.specular
    if layer is not None:
        # The ground lit by the glow alone, where the lamp doesn't light it, has none of the lamp's direct light
        direct = layer.texture if layer.texture.numel() else torch.zeros_like(texture)
        direct_nodes = _node_radiance(direct, nodes, center, grid) if len(nodes.areas) else node_radiance
        layer = layer._replace(texture=direct, node_radiance=direct_nodes)
    # Surfaces relit by the glow aren't relit again
    relit = reflection._replace(
        texture=texture,
        nodes=nodes,
        node_radiance=node_radiance,
        cell_density=cells,
        patches=patches,
        specular=layer,
        transfer=None,
    )
    return _gathered(relit, medium, beta, time)


def dark(reflection: LampReflection) -> LampReflection:
    """The surfaces of reflected light lit by nothing, which :func:`relit_reflection` then lights by a glow alone,
    whose light it then holds, e.g. that of a single order of the light of a lamp scattered more than once.

    Args:
        reflection (LampReflection): Light reflected by surfaces lit with the full extinction, see
            :func:`lamp_reflection`.

    Returns:
        LampReflection: The same surfaces, which reflect no light.
    """
    layer = reflection.specular
    if layer is not None:
        layer = layer._replace(
            texture=torch.zeros_like(layer.texture),
            node_radiance=torch.zeros_like(layer.node_radiance),
            power=torch.zeros_like(layer.power),
        )
    # The weights of the points of the grid may hold the light of the surfaces towards them, which is then none
    gathered = reflection.gathered
    if gathered.toward:
        gathered = gathered._replace(node_weights=gathered.node_weights * 0, patch_weights=gathered.patch_weights * 0)
    return reflection._replace(
        texture=torch.zeros_like(reflection.texture),
        node_radiance=torch.zeros_like(reflection.node_radiance),
        patches=reflection.patches._replace(power=torch.zeros_like(reflection.patches.power)),
        specular=layer,
        gathered=gathered,
    )


def _channels(beta: torch.Tensor) -> torch.Tensor:
    """Extinction coefficient per channel, or a single one when it is the same for every channel, as for fog."""
    return beta[:1] if bool((beta == beta[0]).all()) else beta


def _same_channels(light: torch.Tensor, tolerance: float = 1e-5) -> bool:
    """Whether light of shape (..., c) is the same in every channel, but for differences of up to ``tolerance`` times
    the light of the first channel, plus a tenth of that times the largest light, which the rounding of single precision
    stays well within: up to 5e-7 times the light of grey surfaces on a GPU in the night scene of ``examples/medium``."""
    if not light.numel():
        return True
    spread = (light - light[..., :1]).abs()
    return bool((spread <= tolerance * (light[..., :1].abs() + 0.1 * light.abs().amax())).all())


def _corners(
    sizes: tuple[int, int, int], coordinates: list[torch.Tensor], periodic: tuple[bool, bool, bool]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Flat index and weight of each of the eight neighbors of fractional indices along the first three dimensions of a
    grid of the given sizes, each of shape (n,), wrapping around the periodic ones, as trilinear interpolation weighs
    them."""
    strides = (sizes[1] * sizes[2], sizes[2], 1)
    # Offset and weight of the lower and upper neighbors along each dimension
    axes = []
    for x, size, wrap, stride in zip(coordinates, sizes, periodic, strides):
        if wrap:
            low = x.floor()
            lower = torch.remainder(low.long(), size)
            upper = torch.remainder(lower + 1, size)
        else:
            x = x.clamp(0, size - 1)
            low = x.floor().clamp(max=max(size - 2, 0))
            lower = low.long()
            upper = (lower + 1).clamp(max=size - 1)
        fraction = x - low
        axes.append(((lower * stride, 1 - fraction), (upper * stride, fraction)))
    corners = []
    for corner in range(8):
        (i, a), (j, b), (k, c) = (axis[(corner >> d) & 1] for d, axis in enumerate(axes))
        corners.append((i + j + k, a * b * c))
    return corners


def _trilinear(values: torch.Tensor, coordinates: list[torch.Tensor], periodic: tuple[bool, bool, bool]) -> torch.Tensor:
    """Trilinear interpolation of values of shape (a, b, c, channels) at fractional indices along their first three
    dimensions, each of shape (n,), wrapping around the periodic ones, of shape (n, channels)."""
    flat = values.reshape(-1, values.shape[-1])
    result = flat.new_zeros(len(coordinates[0]), values.shape[-1])
    sizes = (values.shape[0], values.shape[1], values.shape[2])
    for index, weight in _corners(sizes, coordinates, periodic):
        result = result.addcmul_(weight[:, None], flat.index_select(0, index))
    return result


def _foot_coordinates(reflection: LampReflection, points: torch.Tensor) -> list[torch.Tensor]:
    """Fractional indices of points in the grid around the lamp's foot, by log distance, elevation and azimuth."""
    grid = reflection.grid
    r_min, r_max = grid.extent
    offset = points - reflection.center
    r = offset.norm(dim=-1).clamp_min(1e-12)
    elevation = torch.asin((offset[:, 2] / r).clamp(-1, 1))
    t = (elevation.abs() / (math.pi / 2)).sqrt() * elevation.sign()
    return [
        torch.log(r.clamp_min(r_min) / r_min) / math.log(r_max / r_min) * (grid.radii - 1),
        t * (grid.elevations - 1) if reflection.hemisphere else (t + 1) / 2 * (grid.elevations - 1),
        torch.remainder(torch.atan2(offset[:, 1], offset[:, 0]) / (2 * math.pi) * grid.azimuths, grid.azimuths),
    ]


class _ViewGrid(NamedTuple):
    """Grid of points along the rays of a frame, by direction, in the azimuthal equidistant projection around the mean
    direction of the rays, and by log distance to their origin."""

    axes: torch.Tensor
    """mean direction of the rays, and two directions completing it into an orthonormal frame, of shape (3, 3)"""
    bounds: torch.Tensor
    """smallest and largest coordinates of the rays' directions in the projection, of shape (2, 2)"""
    radii: tuple[float, float]
    """smallest and largest distance to the origin"""
    points: torch.Tensor
    """points of the grid, by first and second coordinate of directions, then distance, of shape (u * v * r, 3)"""


def _project(directions: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
    """Azimuthal equidistant projection of unit directions around the first axis, of shape (n, 2)."""
    along = directions @ axes[0]
    across = directions @ axes[1:].T
    sine = across.norm(dim=-1, keepdim=True)
    return torch.atan2(sine, along[:, None]) * across / sine.clamp_min(1e-12)


def _view_grid(
    origin: torch.Tensor,
    directions: torch.Tensor,
    distance: torch.Tensor,
    near: float,
    far: float,
    size: tuple[int, int, int],
) -> _ViewGrid:
    """Grid of points along the rays of a frame, from a distance ``near`` up to ``far``, see
    :attr:`ReflectionGrid.view`."""
    axis = _unit(directions.mean(dim=0))
    helper = directions.new_tensor([0.0, 0.0, 1.0] if abs(float(axis[2])) < 0.9 else [1.0, 0.0, 0.0])
    first = _unit(torch.linalg.cross(helper, axis))
    axes = torch.stack([axis, first, torch.linalg.cross(axis, first)])
    projected = _project(directions, axes)
    bounds = torch.stack([projected.amin(dim=0), projected.amax(dim=0)])
    n_u, n_v, n_r = size
    u, v = (
        torch.linspace(float(bounds[0, i]), float(bounds[1, i]), n, **_like(directions)) for i, n in ((0, n_u), (1, n_v))
    )
    u, v = torch.meshgrid(u, v, indexing="ij")
    angle = torch.hypot(u, v)
    sine = torch.where(angle > 1e-12, torch.sin(angle) / angle.clamp_min(1e-12), torch.ones_like(angle))
    rays = torch.cos(angle)[..., None] * axes[0] + sine[..., None] * (u[..., None] * axes[1] + v[..., None] * axes[2])
    finite = distance[torch.isfinite(distance)]
    longest = min(float(finite.max()) if len(finite) else far, far) if len(distance) else far
    radii = near * (max(longest, 2 * near) / near) ** (torch.arange(n_r, **_like(directions)) / (n_r - 1))
    points = origin + rays.reshape(-1, 1, 3) * radii[:, None]
    return _ViewGrid(axes, bounds, (near, max(longest, 2 * near)), points.reshape(-1, 3))


def _like(tensor: torch.Tensor) -> dict[str, Any]:
    return {"dtype": tensor.dtype, "device": tensor.device}


def _above_ground(reflection: LampReflection, points: torch.Tensor) -> torch.Tensor:
    """Points moved up to the ground, if below it, as camera rays end there."""
    if not reflection.hemisphere:
        return points
    return torch.cat([points[:, :2], points[:, 2:].clamp_min(float(reflection.center[2]))], dim=-1)


def _ray_nodes(
    origin: torch.Tensor,
    directions: torch.Tensor,
    distance: torch.Tensor,
    center: torch.Tensor,
    spread: float,
    t: torch.Tensor,
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nodes along rays of a Gauss-Legendre quadrature, of nodes ``t`` and weights ``w`` over [-1, 1], over the angle at
    which the lamp's foot sees them, from rays no closer to it than ``spread``: distance of each node to the origin and
    its weight, of shape (r, n), and its position, of shape (r * n, 3)."""
    offset = center - origin
    along_ray = directions @ offset
    closest = ((offset * offset).sum() - along_ray**2).clamp_min(0).sqrt().clamp_min(spread)
    start, end = torch.atan2(-along_ray, closest), torch.atan2(distance - along_ray, closest)
    theta = (start + end)[:, None] / 2 + (end - start)[:, None] / 2 * t
    weights = (end - start)[:, None] / 2 * w * closest[:, None] / torch.cos(theta) ** 2
    s = (along_ray[:, None] + closest[:, None] * torch.tan(theta)).clamp_min(0)
    return s, weights, (origin + s[..., None] * directions[:, None, :]).reshape(-1, 3)


def _foot_share(
    reflection: LampReflection, s: torch.Tensor, points: torch.Tensor, center: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Share of the light of the grid around the lamp's foot at nodes along rays, at a distance ``s`` from the origin of
    shape (r, n) and at given points of shape (r * n, 3), see :func:`reflection_inscatter`, and whether they lie within
    the grid's extent, both of shape (r * n,)."""
    to_foot = (points - center).norm(dim=-1)
    share = _smoothstep(torch.log2(s.reshape(-1).clamp_min(1e-12) / to_foot.clamp_min(1e-12)) + 0.5)
    inside = to_foot <= reflection.grid.extent[1]
    if reflection.hemisphere:
        inside = inside & (points[:, 2] >= float(center[2]) - 1e-6)
    return share, inside


def reflection_inscatter(
    reflection: LampReflection,
    medium: Medium,
    origin: torch.Tensor,
    directions: torch.Tensor,
    distance: torch.Tensor,
    beta: torch.Tensor,
    nodes: int = 12,
    time: float = 0.0,
) -> torch.Tensor:
    """Light of a lamp reflected by surfaces that the medium scatters once towards the common origin of rays.

    The light scattered towards the origin per unit of extinction, which varies smoothly, is computed for every frame on
    two grids, see :func:`scatter`, at the points that nodes along rays interpolate: the grid around the lamp's foot,
    whose light is gathered once, and a grid along the rays, by direction and log distance to the origin, whose light is
    gathered for the frame, see :attr:`ReflectionGrid.view`. As the direction towards the origin changes across the
    cells of the first grid, by an angle that grows with their size over their distance to the origin, points closer to
    the origin than to the lamp's foot by a factor √2 take the light of the second grid, those further by a factor √2
    the light of the first, and those in between a smooth mix of both, in log of their ratio of distances. It is
    interpolated at the nodes of a Gauss-Legendre quadrature over the angle at which the lamp's foot sees them, as for
    the light of lamps (see :func:`point_light_inscatter <visionsim.medium.optics.point_light_inscatter>`), and scaled
    by the extinction of the medium there. Unlike the light of a lamp, reflected light doesn't diverge at the lamp's
    foot, around which the ground is brightest over about the lamp's height, so rays closer to the foot than half that
    height take the nodes of rays that far, which keeps nodes spread over the bright part of rays that reach the ground
    under the lamp.

    Args:
        reflection (LampReflection): Reflected light, see :func:`lamp_reflection`.
        medium (Medium): Participating medium.
        origin (torch.Tensor): Common origin of the rays, e.g. the camera, of shape (3,).
        directions (torch.Tensor): Unit ray directions, of shape (r, 3).
        distance (torch.Tensor): Ray lengths in meters, of shape (r,), can be infinite.
        beta (torch.Tensor): Extinction coefficient per channel, of shape (c,).
        nodes (int, optional): Number of quadrature nodes along each ray. Defaults to 12.
        time (float, optional): Time in seconds, used by moving media. Defaults to 0.0.

    Raises:
        ValueError: raised if rays don't share their origin.

    Returns:
        torch.Tensor: Radiance that reaches the origin of each ray, of shape (r, c).
    """
    if origin.ndim > 1:
        raise ValueError("Reflected light is only scattered towards the common origin of rays.")
    kwargs: dict[str, Any] = _like(directions)
    grid, center = reflection.grid, reflection.center.to(**kwargs)
    if not len(directions) or not (len(reflection.nodes.areas) or len(reflection.patches.areas)):
        return directions.new_zeros(len(directions), len(beta))
    # Rays that pass further than the grid's extent from the lamp's foot receive none of its reflected light
    offset = center - origin
    along = torch.minimum((directions @ offset).clamp_min(0), distance)
    if not bool(((origin + along[:, None] * directions - center).norm(dim=-1) <= grid.extent[1]).any()):
        return directions.new_zeros(len(directions), len(beta))
    inner = _channels(beta.to(reflection.points))
    camera = origin.to(reflection.points)
    t, w = (torch.as_tensor(a, **kwargs) for a in np.polynomial.legendre.leggauss(nodes))
    spread = max(grid.extent[0], reflection.height / 2)
    size = max(1, int(4e7 // (nodes * 48)))
    chunks = [slice(i, i + size) for i in range(0, len(directions), size)]
    sizes, periodic = (grid.radii, grid.elevations, grid.azimuths), (False, False, True)
    channels = max(reflection.node_radiance.shape[-1], reflection.patches.power.shape[-1])

    # The grid along the rays, up to where the grid around the foot takes over
    reach = 2 * float((origin - center).norm()) + grid.extent[1]
    rays = _view_grid(origin, directions, distance, grid.extent[0], reach, grid.view)
    projected = _project(directions, rays.axes)
    scale = (rays.bounds[1] - rays.bounds[0]).clamp_min(1e-12)
    across = ((projected - rays.bounds[0]) / scale * (torch.tensor(grid.view[:2], **kwargs) - 1)).T
    log_near, log_far = math.log(rays.radii[0]), math.log(rays.radii[1])

    def along_coordinates(chunk: slice, s: torch.Tensor, chosen: torch.Tensor) -> list[torch.Tensor]:
        """Fractional indices in the grid along the rays of the nodes of a chunk of rays, of which some are chosen."""
        index = [x[:, None].expand(-1, nodes).reshape(-1)[chosen] for x in across[:, chunk]]
        far = (torch.log(s.reshape(-1)[chosen].clamp_min(rays.radii[0])) - log_near) / (log_far - log_near)
        return [*index, far * (grid.view[2] - 1)]

    # Light towards the origin is only computed at the points of both grids that nodes along rays interpolate, and
    # around the lamp's foot, where the origin is far enough to be used
    points = reflection.points
    needed = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    needed_along = torch.zeros(math.prod(grid.view), dtype=torch.bool, device=directions.device)
    for chunk in chunks:
        s, _, positions = _ray_nodes(origin, directions[chunk], distance[chunk], center, spread, t, w)
        share, inside = _foot_share(reflection, s, positions, center)
        for index, _ in _corners(sizes, _foot_coordinates(reflection, positions[(share > 0) & inside]), periodic):
            needed[index] = True
        chosen = ((share < 1) & inside).nonzero().squeeze(-1)
        for index, _ in _corners(grid.view, along_coordinates(chunk, s, chosen), (False, False, False)):
            needed_along[index] = True
    useful = needed & ((points - camera).norm(dim=-1) > 0.25 * (points - reflection.center).norm(dim=-1))
    foot = points.new_zeros(len(points), channels)
    if bool(useful.any()):
        kept = reflection.gathered
        gathered = kept._replace(
            visible=kept.visible[useful],
            node_weights=kept.node_weights[useful],
            node_density=kept.node_density[useful],
            patch_weights=kept.patch_weights[useful],
        )
        view = _unit(points[useful] - camera)
        foot[useful] = scatter(reflection, gathered, points[useful], view, medium, inner, time)
    foot = foot.reshape(*sizes, -1).to(**kwargs)
    along = points.new_zeros(len(rays.points), channels)
    if bool(needed_along.any()):
        chosen_points = rays.points[needed_along].to(reflection.points)
        near = _above_ground(reflection, chosen_points)
        gathered = gather(reflection, medium, near, inner, time)
        along[needed_along] = scatter(reflection, gathered, near, _unit(chosen_points - camera), medium, inner, time)
    along = along.reshape(*grid.view, -1).to(**kwargs)

    parts = []
    for chunk in chunks:
        d = directions[chunk]
        s, weights, positions = _ray_nodes(origin, d, distance[chunk], center, spread, t, w)
        share, inside = _foot_share(reflection, s, positions, center)
        # Each grid is only interpolated at the nodes within the grid's extent that take some of its light
        light = positions.new_zeros(len(positions), foot.shape[-1])
        if len(chosen := ((share > 0) & inside).nonzero().squeeze(-1)):
            coordinates = _foot_coordinates(reflection, positions[chosen])
            light[chosen] += share[chosen, None] * _trilinear(foot, coordinates, periodic)
        if len(chosen := ((share < 1) & inside).nonzero().squeeze(-1)):
            index = along_coordinates(chunk, s, chosen)
            light[chosen] += (1 - share)[chosen, None] * _trilinear(along, index, (False, False, False))
        light = light.reshape(*s.shape, -1)
        tau = optical_depth(medium, origin, d[:, None, :], s, time=time)[..., None] * beta
        extinction = density(medium, positions.reshape(*s.shape, 3), time)[..., None] * beta
        integrand = extinction * torch.exp(-tau) * light * weights[..., None]
        # Far along rays going down forever, the density overflows where no light is left anyway
        parts.append(torch.nan_to_num(integrand, nan=0.0, posinf=0.0).sum(dim=-2))
    return torch.cat(parts)


@lru_cache(maxsize=16)
def _cached_lamp_reflection(
    medium: str,
    lamp: Lamp,
    beta: tuple,
    ground_height: float,
    ground_albedo: tuple[float, ...],
    object_albedo: tuple[float, ...],
    maps: ShadowMaps | None,
    occluders: torch.Tensor | None,
    grid: ReflectionGrid,
    time: float,
    dtype: torch.dtype,
    device: str,
    map_tolerance: float,
    glow: bool,
    bounces: bool,
    shadowed: bool,
    occluded: bool,
    specular: tuple[float, float] | None,
    surfels: Surfels | None,
    interreflections: bool,
) -> LampReflection:
    return lamp_reflection(
        Medium.model_validate_json(medium),
        lamp,
        torch.tensor(beta, dtype=dtype, device=device),
        ground_height,
        ground_albedo,
        object_albedo,
        maps,
        occluders,
        grid,
        time,
        map_tolerance,
        glow,
        bounces,
        shadowed,
        occluded,
        specular,
        surfels,
        interreflections,
    )


def cached_lamp_reflection(
    medium: Medium,
    lamp: Lamp,
    beta: torch.Tensor,
    ground_height: float = 0.0,
    ground_albedo: tuple[float, ...] = (0.0,),
    object_albedo: tuple[float, ...] = (0.0,),
    maps: ShadowMaps | None = None,
    occluders: torch.Tensor | None = None,
    grid: ReflectionGrid = REFLECTION_GRID,
    time: float = 0.0,
    *,
    map_tolerance: float = 0.01,
    glow: bool = False,
    bounces: bool = True,
    shadowed: bool = True,
    occluded: bool = True,
    specular: tuple[float, float] | None = None,
    surfels: Surfels | None = None,
    interreflections: bool = True,
) -> LampReflection:
    """Same as :func:`lamp_reflection`, cached for the last 16 lamps and media, as it doesn't depend on the camera and
    can be reused for every frame, unless the medium moves. Maps, occluders and surfels are keyed by identity, so the
    same ones should be given for every frame, rather than copies of them. Each lamp takes about 30 MB with the default
    grid, mostly the maps of its anchors and the light gathered on its grid, and more with the patches of every
    surface, see :func:`surface_receivers`."""
    moving = any(isinstance(c, Blob) and any(c.velocity) for c in medium.components)
    return _cached_lamp_reflection(
        medium.model_dump_json(),
        lamp,
        tuple(beta.tolist()),
        ground_height,
        tuple(ground_albedo),
        tuple(object_albedo),
        maps,
        occluders,
        grid,
        time if moving else 0.0,
        beta.dtype,
        str(beta.device),
        map_tolerance,
        glow,
        bounces,
        shadowed,
        occluded,
        (float(specular[0]), float(specular[1])) if specular is not None else None,
        surfels,
        interreflections,
    )
