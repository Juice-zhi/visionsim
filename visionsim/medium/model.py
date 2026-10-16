from __future__ import annotations

import math
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveFloat, model_validator
from typing_extensions import Self

Vector3 = tuple[float, float, float]
_PI = math.pi + 1e-6
"""Upper bound of angles of up to π, which tolerates their rounding to single precision, e.g. by Blender"""


class Homogeneous(BaseModel):
    """Constant density that fills all of space."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["homogeneous"] = "homogeneous"
    """component discriminator"""
    density: NonNegativeFloat = 1.0
    """relative density, which scales the medium's extinction coefficient"""


class HeightFog(BaseModel):
    """Density that decays exponentially with height, as is the case for ground or valley fog.

    The relative density at a point of height ``z`` (world-space, z is up) is ``density * exp(-(z - base_height) / falloff)``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["height"] = "height"
    """component discriminator"""
    density: NonNegativeFloat = 1.0
    """relative density at ``base_height``"""
    base_height: float = 0.0
    """height, in meters, at which the relative density equals ``density``"""
    falloff: PositiveFloat = 10.0
    """height difference, in meters, over which the density decreases by a factor of e"""


class Blob(BaseModel):
    """Isotropic Gaussian puff of density, such as a patch of fog or a plume of smoke.

    The relative density at a point ``x`` and time ``t`` is ``density * exp(-|x - center - velocity * t|^2 / (2 * radius^2))``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["blob"] = "blob"
    """component discriminator"""
    density: NonNegativeFloat = 1.0
    """relative density at the center of the blob"""
    center: Vector3
    """world-space position, in meters, of the center of the blob at time zero"""
    radius: PositiveFloat
    """standard deviation, in meters, of the blob's density profile"""
    velocity: Vector3 = (0.0, 0.0, 0.0)
    """velocity of the blob, in meters per second, for instance due to wind"""


Component = Annotated[Union[Homogeneous, HeightFog, Blob], Field(discriminator="type")]


class Medium(BaseModel):
    """A participating medium such as fog, haze or smoke.

    The medium's extinction coefficient, at a point ``x`` and wavelength ``λ``, is
    ``extinction * (λ / reference_wavelength) ** -angstrom * density(x)`` where the relative density is the sum
    of the densities of all components. All components share the same scattering properties (albedo and phase
    function), which is what enables closed-form solutions for the light transport, see :mod:`visionsim.medium.optics`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    extinction: NonNegativeFloat
    """extinction coefficient, in 1/m, at the reference wavelength and for a relative density of one"""
    reference_wavelength: PositiveFloat = 550.0
    """wavelength, in nm, at which ``extinction`` is specified"""
    angstrom: float = 0.0
    """Ångström exponent, which models how extinction depends on wavelength. It is close to zero for fog,
    and around 1.3 for haze, see :func:`kim_exponent` for an empirical model based on visibility"""
    albedo: float = Field(1.0, ge=0.0, le=1.0)
    """single scattering albedo, i.e. the ratio of scattering to extinction, which is close to one for water droplets"""
    anisotropy: float = Field(0.85, gt=-1.0, lt=1.0)
    """asymmetry parameter of the Henyey-Greenstein phase function, fog droplets mostly scatter forward"""
    components: list[Component] = Field(default_factory=lambda: [Homogeneous()])
    """density components, whose relative densities add up"""
    sun_attenuation: bool = False
    """if true, light from suns and from the sky is attenuated as it travels through the medium before being
    scattered. This is only supported, in closed form, for media made of a single :class:`HeightFog` component.
    Otherwise, light is assumed to reach every point of the medium unattenuated"""
    multiple_scattering: bool = False
    """if true, also approximate light scattered more than once by the medium, which is significant in dense fog as
    its droplets absorb little light, see :mod:`visionsim.medium.scattering`. This requires ``sun_attenuation``"""

    @model_validator(mode="after")
    def _validate_sun_attenuation(self) -> Self:
        if self.sun_attenuation and (len(self.components) != 1 or not isinstance(self.components[0], HeightFog)):
            raise ValueError("Sun attenuation is only supported for media made of a single `HeightFog` component.")
        if self.multiple_scattering and not self.sun_attenuation:
            raise ValueError("Multiple scattering is only supported along with sun attenuation.")
        return self

    @classmethod
    def from_visibility(cls, visibility: float, contrast_threshold: float = 0.02, **kwargs) -> Self:
        """Create a medium with a given meteorological visibility, at the reference wavelength and unit density.

        Following Koschmieder's law, the visibility is the distance at which the contrast of a black object against
        the horizon drops to ``contrast_threshold``, meaning that the extinction coefficient is
        ``-ln(contrast_threshold) / visibility``.

        Args:
            visibility (float): Visibility in meters, e.g. under 1000m for fog, and a few kilometers for haze.
            contrast_threshold (float, optional): Contrast threshold of the observer. Defaults to 0.02, as used by
                Koschmieder, whereas the World Meteorological Organization uses 0.05.
            **kwargs: Additional fields of the medium, such as its ``components`` or ``albedo``.

        Returns:
            Self: Medium with the requested visibility.
        """
        return cls(extinction=-math.log(contrast_threshold) / visibility, **kwargs)

    def extinction_at(self, wavelengths: tuple[float, ...]) -> tuple[float, ...]:
        """Extinction coefficient, for a relative density of one, at the given wavelengths.

        Args:
            wavelengths (tuple[float, ...]): Wavelengths in nm.

        Returns:
            tuple[float, ...]: Extinction coefficient in 1/m, one per wavelength.
        """
        return tuple(self.extinction * (w / self.reference_wavelength) ** -self.angstrom for w in wavelengths)


def kim_exponent(visibility: float) -> float:
    """Empirical Ångström exponent of fog and haze, as a function of their visibility.

    This follows the model of Kim et al. (2001), commonly used for free-space optics and lidar in adverse weather,
    in which dense fog scatters all wavelengths alike whereas light haze scatters shorter wavelengths more.

    Args:
        visibility (float): Visibility in meters.

    Returns:
        float: Ångström exponent, to be used as :attr:`Medium.angstrom`.
    """
    km = visibility / 1000
    if km > 50:
        return 1.6
    if km > 6:
        return 1.3
    if km > 1:
        return 0.16 * km + 0.34
    if km > 0.5:
        return km - 0.5
    return 0.0


class Sun(BaseModel):
    """Distant light source, such as the sun."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    direction: Vector3
    """world-space direction pointing towards the light, which is normalized when used"""
    irradiance: tuple[float, ...]
    """irradiance per color channel, in W/m², received by a surface facing the light"""


Falloff = Literal["quadratic", "linear", "constant"]
"""How the light of a lamp falls off with the distance ``r`` to it: physically, as ``1 / r²`` (quadratic), or as
``1 / r`` (linear) or not at all (constant), as with the outputs of Blender's Light Falloff node, which scale the
lamp's intensity by ``r`` or ``r²``"""


class PointLight(BaseModel):
    """Point light source, such as a street lamp, whose radiant intensity is its power divided by 4π, as in Cycles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    position: Vector3
    """world-space position of the light, in meters"""
    power: tuple[float, ...]
    """radiant power per color channel, in W"""
    radius: NonNegativeFloat = 0.0
    """radius of the light, in meters, below which distances to the light are clamped"""
    falloff: Falloff = "quadratic"
    """how the light falls off with the distance to it, see :data:`Falloff`"""
    smooth: NonNegativeFloat = 0.0
    """smoothing of the light near it, whose intensity is scaled by ``r² / (smooth + r²)`` at a distance ``r``, as with
    Blender's Light Falloff node"""


class SpotLight(BaseModel):
    """Point light source that only shines within a cone, such as a ceiling spot or a car's headlight.

    As in Cycles, its radiant intensity within the cone is its power divided by 4π, i.e. as for a point light of the
    same power, and fades towards the edge of the cone as ``smoothstep((cos φ - cos(angle / 2)) / ((1 - cos(angle / 2))
    * blend))``, where ``φ`` is the angle between the spot's direction and the direction from the light.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    position: Vector3
    """world-space position of the light, in meters"""
    direction: Vector3
    """world-space direction in which the spot points, which is normalized when used"""
    power: tuple[float, ...]
    """radiant power per color channel, in W, of a point light with the same intensity"""
    angle: float = Field(math.pi / 4, gt=0.0, le=_PI)
    """angle of the cone, in radians, between its opposite edges"""
    blend: float = Field(0.15, ge=0.0, le=1.0)
    """softness of the edge of the cone, as a fraction of ``1 - cos(angle / 2)``"""
    radius: NonNegativeFloat = 0.0
    """radius of the light, in meters, below which distances to the light are clamped"""
    falloff: Falloff = "quadratic"
    """how the light falls off with the distance to it, see :data:`Falloff`"""
    smooth: NonNegativeFloat = 0.0
    """smoothing of the light near it, whose intensity is scaled by ``r² / (smooth + r²)`` at a distance ``r``, as with
    Blender's Light Falloff node"""


class AreaLight(BaseModel):
    """Planar light source, such as a ceiling panel or a window lit from outside.

    As in Cycles, it is a one-sided Lambertian emitter, whose radiance is its power divided by π times its area, which
    can be restricted to a narrower cone of directions by its spread (like the grid of a softbox).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    position: Vector3
    """world-space position of the center of the light, in meters"""
    direction: Vector3
    """world-space normal of the light, towards which it shines, which is normalized when used"""
    axis_u: Vector3
    """world-space direction, within the light's plane, along which its first size is measured"""
    size: tuple[PositiveFloat, PositiveFloat]
    """size of the light, in meters, along ``axis_u`` and along the other axis of its plane"""
    shape: Literal["rectangle", "ellipse"] = "rectangle"
    """shape of the light, whose sizes are the lengths of either the sides of a rectangle or the axes of an ellipse"""
    power: tuple[float, ...]
    """radiant power per color channel, in W"""
    spread: float = Field(math.pi, gt=0.0, le=_PI)
    """angle, in radians, of the cone of directions in which each point of the light shines"""
    falloff: Falloff = "quadratic"
    """how the light falls off with the distance to it, see :data:`Falloff`"""
    smooth: NonNegativeFloat = 0.0
    """smoothing of the light near it, whose intensity is scaled by ``r² / (smooth + r²)`` at a distance ``r`` to each
    of its points, as with Blender's Light Falloff node"""

    @property
    def area(self) -> float:
        """Area of the light, in m²."""
        return self.size[0] * self.size[1] * (math.pi / 4 if self.shape == "ellipse" else 1.0)


class EmissiveSurface(BaseModel):
    """Surfaces of meshes that emit light, such as screens, bulbs or neon tubes, which shine and cast shadows as a lamp.

    As in Cycles, both sides of the surfaces emit a constant radiance in every direction (Lambertian emission), except
    for the light that the surfaces block themselves, such as the light emitted inside a closed surface. The surfaces
    are described by patches, each a part of them of a given position, area and radiance, e.g. clusters of their faces,
    see :meth:`BlenderService.lighting_info <visionsim.simulate.blender.BlenderService.exposed_lighting_info>`. A patch
    shines in each direction in proportion to the area of its faces seen from there whose light leaves towards it,
    which is approximated from the moments of their normals, exactly for flat patches, spheres, cylinders and
    hemispheres, see :func:`projected_area <visionsim.medium.lights.projected_area>`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    position: Vector3
    """world-space position from which the surfaces cast shadows, in meters, typically their center if they surround
    it, or a point on them near it"""
    positions: tuple[Vector3, ...]
    """world-space center of each patch, in meters"""
    areas: tuple[PositiveFloat, ...]
    """area of each patch, in m²"""
    radiance: tuple[tuple[float, ...], ...]
    """radiance of each patch per color channel, in W/(m²·sr), i.e. the emission strength times its color in Blender"""
    orientation: tuple[tuple[float, float, float, float, float, float], ...]
    """mean of ``n nᵀ`` over the unit world-space normals ``n`` of the faces of each patch, weighted by their area and
    by the mean of the fractions of their light that leave their two sides, as ``(xx, yy, zz, xy, xz, yz)``. It is
    ``n nᵀ`` for a flat patch whose two sides emit, ``n nᵀ / 2`` if only its front does, and a sixth of the identity
    for a sphere"""
    facing: tuple[Vector3, ...] = ()
    """mean of the unit normals of the faces of each patch, weighted by their area and by half the difference between
    the fractions of their light that leave their front and their back. It is ``n / 2`` for a flat patch of normal ``n``
    only the front of which emits. Defaults to zero, i.e. faces whose light leaves both sides alike"""
    spread: tuple[tuple[float, float, float, float, float, float], ...] = ()
    """covariance of the world-space points of each patch around its center, in m², as ``(xx, yy, zz, xy, xz, yz)``.
    The square root of its trace, the root mean square distance between the points and the center, is the radius of the
    patch, below which distances to it are clamped, e.g. the radius of a sphere, and flat patches are seen as the
    rectangle of the same spread by rays that pass close to them, see :func:`patch_emitters
    <visionsim.medium.lights.patch_emitters>`. Defaults to patches seen as a point, whose radius is half the square
    root of their area"""

    @model_validator(mode="after")
    def _validate_patches(self) -> Self:
        sizes = {len(self.positions), len(self.areas), len(self.radiance), len(self.orientation)}
        if not self.areas or len(sizes) != 1 or {len(self.facing), len(self.spread)} - {0, len(self.areas)}:
            raise ValueError(
                "Expected as many positions, areas, radiances, orientations, and facings and spreads if any, one per "
                "patch, and at least one patch."
            )
        return self

    @property
    def area(self) -> float:
        """Area of the surfaces, in m²."""
        return sum(self.areas)


Lamp = Union[PointLight, SpotLight, AreaLight, EmissiveSurface]
"""Light source at a finite distance, as opposed to suns"""


class Lighting(BaseModel):
    """Lighting that illuminates a participating medium.

    This is typically exported from the Blender scene, using :meth:`BlenderService.lighting_info
    <visionsim.simulate.blender.BlenderService.exposed_lighting_info>`, so that the medium and the surfaces
    are lit consistently.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sky: tuple[float, ...] = (0.0, 0.0, 0.0)
    """average radiance per color channel of the environment above the horizon, e.g. the sky. Light from below the
    horizon is assumed to be blocked by the ground, as is the case for scenes captured from near the ground"""
    ambient: tuple[float, ...] = (0.0, 0.0, 0.0)
    """radiance per color channel of an environment which lights the medium uniformly from all directions, including
    from below the horizon, and which is never attenuated. This is useful for scenes without a ground"""
    suns: list[Sun] = Field(default_factory=list)
    """distant light sources"""
    points: list[PointLight] = Field(default_factory=list)
    """point light sources, whose light is attenuated by the medium on its way, and scattered once"""
    spots: list[SpotLight] = Field(default_factory=list)
    """spot light sources, whose light is attenuated by the medium on its way, and scattered once"""
    areas: list[AreaLight] = Field(default_factory=list)
    """area light sources, whose light is attenuated by the medium on its way, and scattered once"""
    emissive: list[EmissiveSurface] = Field(default_factory=list)
    """surfaces that emit light, whose light is attenuated by the medium on its way, and scattered once"""
    ground_albedo: tuple[float, ...] = (0.0, 0.0, 0.0)
    """albedo per color channel of the ground, modeled as a Lambertian plane at ``ground_height``, which reflects the
    light of suns, the sky and lamps into the medium. Light of suns and the sky is only reflected when sunlight is
    attenuated by the medium, see :attr:`Medium.sun_attenuation`, and light of lamps is reflected by any medium, see
    :mod:`visionsim.medium.reflections`, by the ground and by a specular layer over it, see ``surface_ior``"""
    object_albedo: tuple[float, ...] | None = None
    """albedo per color channel of objects, i.e. the surfaces other than the ground, which reflect the light of lamps
    into the medium, as Lambertian surfaces under a specular layer, see ``surface_ior`` and
    :mod:`visionsim.medium.reflections`. Defaults to None, i.e. that of the ground, as an estimate of the mean albedo of
    the scene's surfaces"""
    ground_height: float = 0.0
    """height of the ground, in meters"""
    surface_roughness: float = Field(default=0.5, ge=0.0, le=1.0)
    """roughness of the specular layer of the ground and of objects (see ``surface_ior``), as that of Blender's
    Principled BSDF, whose default it is: the width of the distribution of the layer's GGX microfacets is its square.
    Smoother layers are taken as of roughness 0.5, see :data:`MIN_ROUGHNESS
    <visionsim.medium.reflections.MIN_ROUGHNESS>`: the light that surfaces reflect into the medium is within 5% of a
    direct integration from 0.5 up, and that of smoother layers within 8% from 0.1 up, while the closed form, which
    doesn't follow their narrower lobes, would miss up to 17% of it. Below 0.1, surfaces such as wet roads mirror
    lamps, which isn't modeled"""
    surface_ior: float = Field(default=1.5, ge=1.0)
    """index of refraction of a specular layer over the ground and objects, a dielectric coating of GGX microfacets
    over their Lambertian base, of albedo ``ground_albedo`` and ``object_albedo``, as Blender's Principled BSDF, which
    reflects the light of lamps around the mirror direction into the medium, see :mod:`visionsim.medium.reflections`.
    It sets the share of light that the layer reflects, 4% at normal incidence for the default of the Principled BSDF,
    1.5, and more at grazing angles, while one leaves the layer out, i.e. surfaces are Lambertian. The light of suns and
    the sky that the ground and objects reflect is that of Lambertian surfaces"""

    @property
    def lamps(self) -> list[Lamp]:
        """Light sources at a finite distance, i.e. point, spot and area lights, and emissive surfaces."""
        return [*self.points, *self.spots, *self.areas, *self.emissive]


class AnimatedLighting(BaseModel):
    """Lighting of the frames of an animation, which changes over them, e.g. as lamps move or turn on and off.

    This is typically exported from the Blender scene, along with the lighting of its first frame, when lights are
    animated, see :meth:`BlenderService.save_lighting <visionsim.simulate.blender.BlenderService.exposed_save_lighting>`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    frames: dict[int, Lighting]
    """lighting by frame number, at the frames at which it changes, which holds until the next of them"""

    @model_validator(mode="after")
    def _validate_frames(self) -> Self:
        if not self.frames:
            raise ValueError("Expected the lighting of at least one frame.")
        return self

    def at(self, frame: int) -> Lighting:
        """Lighting of a frame, i.e. that of the last frame at or before it at which the lighting changed, or that of the
        first frame for frames before it.

        Args:
            frame (int): Frame number.

        Returns:
            Lighting: Lighting of the frame.
        """
        before = [number for number in self.frames if number <= frame]
        return self.frames[max(before) if before else min(self.frames)]


EFFECT_GROUPS: dict[str, str] = {
    "daylight": "light of suns and the sky in the medium, and the shadows that objects cast in it",
    "lamps": "light of lamps in the medium, scattered once and more, and reflected by surfaces",
    "surfaces": "light of the surfaces seen through the medium, which it dims while its glow lights them",
}
"""Groups of the switches of :class:`Effects`, by name, with what their terms model"""

EFFECT_INPUTS: dict[str, str] = {
    "shadows": "objects cast shadows, as shadow maps of the scene or shadows traced from them beforehand are given",
    "occlusion": "shadow maps of the scene are given, which some terms need themselves, beyond the shadows traced from "
    "them along camera rays",
    "normals": "the normals of the surfaces are given, through which surfaces are lit through the medium",
    "albedo": "the albedo of the surfaces seen by each pixel is given, e.g. Blender's diffuse color pass",
    "irradiance": "the irradiance that the surfaces seen by each pixel received without the medium is given, e.g. "
    "from Blender's diffuse light passes",
    "suns": "a sun above the horizon shines",
    "daylight": "a sun above the horizon or the sky shines",
    "lamps": "lamps or emissive surfaces shine",
    "direct_light": "a sun above the horizon, lamps or emissive surfaces shine",
    "scattering": "the medium scatters light, i.e. Medium.albedo isn't zero",
    "sun_attenuation": "the medium attenuates the light of suns and the sky, see Medium.sun_attenuation",
    "multiple_scattering": "the medium scatters light more than once, see Medium.multiple_scattering",
    "ground_albedo": "the ground reflects light, i.e. Lighting.ground_albedo isn't zero",
    "object_albedo": "objects reflect light, i.e. Lighting.object_albedo, or the ground's when they have none, isn't "
    "zero",
    "measured": "the light that the surface seen by each pixel received without the medium is measured: its irradiance "
    "is given, or its albedo, from which its radiance tells that light",
    "received": "the light that surfaces received without the medium can be told from their radiance: it is measured "
    "for each pixel, or the ground or objects reflect light, whose mean albedo tells it on average",
    "reflectors": "surfaces reflect the light of lamps: the ground, or objects, which the shadow maps of lamps record, "
    "given shadow maps",
    "specular": "surfaces have a specular layer, i.e. Lighting.surface_ior isn't one",
}
"""Conditions on the inputs of the model that the switches of :class:`Effects` can require, by name, see
:meth:`Effects.conditions`"""

EFFECT_GIVEN: tuple[str, ...] = ("shadows", "occlusion", "normals", "albedo", "irradiance")
"""Conditions of :data:`EFFECT_INPUTS` on what is given with each frame, which :meth:`Effects.conditions` takes by name,
while the others are found from the medium and the lighting"""


class Effects(BaseModel):
    """Switches of the terms of the closed-form model, which can each be turned off, to save time or to see what they do.

    Each field turns one term of the model on (the default) or off, and says in its description what the term models.
    All on, the model is complete, and a term that is off costs nothing, as it isn't computed at all. Unlike the
    medium, the lighting and the inputs of a frame (such as albedos, :attr:`Medium.multiple_scattering`, normals or
    shadow maps), which describe the scene, switches only choose which terms of the model of that scene are computed.

    The ``json_schema_extra`` of each field declares, as JSON, the ``group`` of its term (one of
    :data:`EFFECT_GROUPS`), and its prerequisites: ``requires``, the other switches that the term needs, e.g. a term
    that refines shadows needs the shadows, and ``requires_inputs``, the conditions on the inputs it needs, among the
    fixed vocabulary of :data:`EFFECT_INPUTS`, e.g. shadow maps, without which objects cast no shadows. A switch is
    effective if it is on and all of its prerequisites are met, recursively for the switches it requires, see
    :meth:`effective`, which :meth:`missing` explains, and a switch that isn't effective is treated exactly as a switch
    that is off, see :meth:`resolve`. For instance::

        effects = Effects(halos=False, reflected_daylight=False)
        apply_medium(radiance, depth, camera, pose, medium, lighting, occlusion=occlusion, effects=effects)

    Prerequisites are checked against the medium, the lighting and what is given with frames, not against the data of
    each frame, so that an effective switch can still change nothing where its term vanishes: ``halos`` where the
    medium around lamps is too thin to be worth them (see :func:`halo_threshold
    <visionsim.medium.halos.halo_threshold>`), the terms of shadow maps when the maps given lack those of a sun or a
    lamp, ``received_irradiance`` and ``measured_indirect`` where the medium dims the light of every surface alike, e.g.
    without any light, or where the light measured for each pixel is zero, ``received_irradiance`` also, given the mean
    albedo of the ground and objects only, where surfaces received no more light than the share of the frame's direct
    light that it replaces, and ``measured_indirect`` where every surface received at least that share beyond its
    direct light, ``sky_occlusion`` when the medium only receives the direct light of suns, under a black sky, above a
    black ground and black objects, without multiple scattering, ``ground_sky_occlusion`` when the ground only receives
    the direct light of suns, under a black sky, without multiple scattering, and ``surface_sky_occlusion`` when
    surfaces only receive the direct light of suns, under a black sky, above a black ground, through a medium that
    scatters nothing.

    To add a term of the model:

    - add a field, on by default, with its description, group and prerequisites, declared like the others. A new
      condition on the inputs is added to :data:`EFFECT_INPUTS`, and found by :meth:`conditions` from the medium and
      the lighting, or, if it is on what is given with frames, to :data:`EFFECT_GIVEN` too, which :func:`apply_medium
      <visionsim.medium.render.apply_medium>` passes;
    - compute the term only if its switch is on among those that :func:`apply_medium
      <visionsim.medium.render.apply_medium>` resolves, and add the switch to the key of every cache that it changes;
    - add the functions that compute it to ``WORK`` in ``tests/test_effects.py``, which checks that they are called
      with every switch on and never with this one off;
    - measure what it costs and the error it removes with ``ablation.py`` of ``examples/medium/experiment``, and add it
      to the table of ``docs/source/sections/medium.rst``.

    The switches, with their group and prerequisites, are:
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sun_shadows: bool = Field(
        default=True,
        description="Objects cast the shadows of suns onto the medium, which shows light shafts: along the stretches of "
        "camera rays in a sun's shadow, the medium scatters none of the sun's light.",
        json_schema_extra={
            "group": "daylight",
            "requires": [],
            "requires_inputs": ["shadows", "sun_attenuation", "suns", "scattering"],
        },
    )
    sky_occlusion: bool = Field(
        default=True,
        description="Objects hide part of the sky from the medium around them, e.g. in front of a wall: from the "
        "directions they hide, the medium receives none of the light of the sky, nor of the ground below the horizon.",
        json_schema_extra={
            "group": "daylight",
            "requires": [],
            "requires_inputs": ["shadows", "sun_attenuation", "daylight", "scattering"],
        },
    )
    scattering_occlusion: bool = Field(
        default=True,
        description="Objects that hide part of the sky from the medium also hide the light that the medium scatters "
        "more than once from those directions, as the medium in their shadows glows less.",
        json_schema_extra={
            "group": "daylight",
            "requires": ["sky_occlusion"],
            "requires_inputs": ["multiple_scattering"],
        },
    )
    azimuthal_occlusion: bool = Field(
        default=True,
        description="Objects that hide directions towards a sun, e.g. from the medium in their shadow, hide more of "
        "the light that the medium scatters more than once than objects that hide directions away from it, as this "
        "light still goes mostly the way sunlight went, rather than as much as the mean over the directions of the "
        "same elevation.",
        json_schema_extra={
            "group": "daylight",
            "requires": ["scattering_occlusion"],
            "requires_inputs": ["suns"],
        },
    )
    ground_shadows: bool = Field(
        default=True,
        description="The ground in the shadows of suns lacks their light, so that the light it reflects into the medium "
        "is darker where it starts on the ground in a shadow, and, with scattering_occlusion, so is the light of the "
        "ground that the medium scatters on its way up.",
        json_schema_extra={
            "group": "daylight",
            "requires": ["sky_occlusion"],
            "requires_inputs": ["shadows", "suns", "ground_albedo"],
        },
    )
    ground_sky_occlusion: bool = Field(
        default=True,
        description="The ground next to objects, which hide part of the sky from it, lacks that part of the light of "
        "the sky and of the glow of the medium, so that the light it reflects into the medium is darker where it starts "
        "on the ground next to objects, and, with scattering_occlusion, so is the light of the ground that the medium "
        "scatters on its way up.",
        json_schema_extra={
            "group": "daylight",
            "requires": ["sky_occlusion"],
            "requires_inputs": ["shadows", "ground_albedo"],
        },
    )
    reflected_daylight: bool = Field(
        default=True,
        description="Objects reflect the light of suns and the sky into the medium, which scatters it towards the "
        "camera, e.g. the fog in front of a sunlit wall is brighter. It takes a quarter of the time of a frame by day.",
        json_schema_extra={
            "group": "daylight",
            "requires": ["sky_occlusion"],
            "requires_inputs": ["occlusion", "object_albedo"],
        },
    )
    scattered_daylight_reflections: bool = Field(
        default=True,
        description="The light of suns and the sky that objects reflect is scattered more than once too: the part that "
        "the medium scatters forward on its way keeps going towards each point, which the reduced extinction of the "
        "similarity relation accounts for, rather than all of it being taken out.",
        json_schema_extra={
            "group": "daylight",
            "requires": ["reflected_daylight"],
            "requires_inputs": ["multiple_scattering"],
        },
    )
    lamp_shadows: bool = Field(
        default=True,
        description="Objects cast the shadows of lamps onto the medium, so that the medium in their shadow, e.g. in "
        "front of a lamp hidden by a wall, receives none of the lamp's light, scattered once or in its halo. Surfaces "
        "keep them, see reflection_shadows and surface_shadows.",
        json_schema_extra={"group": "lamps", "requires": [], "requires_inputs": ["occlusion", "lamps", "scattering"]},
    )
    halos: bool = Field(
        default=True,
        description="The light of lamps scattered more than once spreads a halo around them, much wider than the glow "
        "of their light scattered once. It is computed once per lamp and medium, in about a minute on the CPU.",
        json_schema_extra={
            "group": "lamps",
            "requires": [],
            "requires_inputs": ["lamps", "multiple_scattering", "scattering"],
        },
    )
    halo_occlusion: bool = Field(
        default=True,
        description="Objects stop the light that spreads the halos of lamps, so that the medium behind an object "
        "doesn't receive the halo's light from its other side, through the voxels that objects occupy.",
        json_schema_extra={"group": "lamps", "requires": ["halos"], "requires_inputs": ["occlusion"]},
    )
    halo_shadow_edges: bool = Field(
        default=True,
        description="The halos of lamps fade across the edges of the lamps' shadows and right behind objects, rather "
        "than as steps blurred over their coarse grid, which is refined around the objects that cast them, by whole "
        "rows and columns of directions, and objects don't stop the light that passes right beside them. Halos take "
        "2.5 to 4 times as long to compute, and twice as much memory to keep.",
        json_schema_extra={"group": "lamps", "requires": ["halos", "lamp_shadows"], "requires_inputs": []},
    )
    lamp_reflections: bool = Field(
        default=True,
        description="Surfaces lit by lamps reflect part of their light into the medium, which scatters it towards the "
        "camera, e.g. the fog above the ground lit by a street lamp glows.",
        json_schema_extra={"group": "lamps", "requires": [], "requires_inputs": ["lamps", "reflectors", "scattering"]},
    )
    reflection_shadows: bool = Field(
        default=True,
        description="The ground in the shadows that objects cast from lamps, as the maps of lamps tell, receives none "
        "of their light, so that it reflects none of it into the medium, e.g. behind a wall lit by a street lamp.",
        json_schema_extra={
            "group": "lamps",
            "requires": ["lamp_reflections"],
            "requires_inputs": ["occlusion", "ground_albedo"],
        },
    )
    reflection_occlusion: bool = Field(
        default=True,
        description="Objects hide the light that the ground lit by lamps reflects from the medium behind them, e.g. the "
        "fog behind a wall doesn't receive the light of the ground on the lamp's side, as seen from points on the "
        "ground around the lamp.",
        json_schema_extra={
            "group": "lamps",
            "requires": ["lamp_reflections"],
            "requires_inputs": ["occlusion", "ground_albedo"],
        },
    )
    reflection_bounces: bool = Field(
        default=True,
        description="Objects and the ground lit by lamps light each other once, which adds to the light they reflect "
        "into the medium.",
        json_schema_extra={
            "group": "lamps",
            "requires": ["lamp_reflections"],
            "requires_inputs": ["occlusion", "ground_albedo", "object_albedo"],
        },
    )
    bounce_surfaces: bool = Field(
        default=True,
        description="Every surface around lamps, that the shadow maps of suns or of the sky record or that lamps light, "
        "receives the light of the ground and of the objects that lamps light, and their glow, also the sides of "
        "objects that lamps don't light, e.g. those that face the camera, so that objects light each other too, rather "
        "than only the surfaces that lamps light receiving the light of the ground.",
        json_schema_extra={"group": "lamps", "requires": ["reflection_bounces"], "requires_inputs": []},
    )
    interreflections: bool = Field(
        default=True,
        description="Light keeps bouncing between the ground and objects lit by lamps until little is left, rather than "
        "once, which adds to the light they reflect into the medium.",
        json_schema_extra={"group": "lamps", "requires": ["reflection_bounces"], "requires_inputs": []},
    )
    scattered_reflections: bool = Field(
        default=True,
        description="The light of lamps that surfaces reflect into the medium is scattered more than once in the "
        "halos of lamps, rather than once only.",
        json_schema_extra={"group": "lamps", "requires": ["halos", "lamp_reflections"], "requires_inputs": []},
    )
    relit_reflections: bool = Field(
        default=True,
        description="Surfaces that reflect the light of lamps into the medium are also lit by the glow of the lamps' "
        "halos, from every direction, rather than with the reduced extinction of the similarity relation only.",
        json_schema_extra={"group": "lamps", "requires": ["halos", "lamp_reflections"], "requires_inputs": []},
    )
    specular_reflections: bool = Field(
        default=True,
        description="Surfaces that reflect the light of lamps into the medium have a specular layer, as Blender's "
        "Principled BSDF, which reflects the lamps' light around the mirror direction, e.g. far over the ground away "
        "from a street lamp, and dims their diffuse light, rather than being Lambertian.",
        json_schema_extra={"group": "lamps", "requires": ["lamp_reflections"], "requires_inputs": ["specular"]},
    )
    relit_scattering: bool = Field(
        default=True,
        description="The light that surfaces reflect of the glow of halos beyond the lamps' light scattered once, i.e. "
        "of their light scattered more than once and of the light of surfaces that the medium scattered, is scattered "
        "more than once too, order by order, rather than once only.",
        json_schema_extra={
            "group": "lamps",
            "requires": ["scattered_reflections", "relit_reflections"],
            "requires_inputs": [],
        },
    )
    surface_shadows: bool = Field(
        default=True,
        description="Surfaces in the shadows of suns and lamps, as their shadow maps tell, receive none of their "
        "light, so that the medium dims the rest of their light instead.",
        json_schema_extra={
            "group": "surfaces",
            "requires": [],
            "requires_inputs": ["normals", "occlusion", "direct_light"],
        },
    )
    surface_sky_occlusion: bool = Field(
        default=True,
        description="Surfaces only receive the light of the parts of the sky they see, and the glow of the medium in "
        "proportion, e.g. under a roof, rather than as if they stood in the open.",
        json_schema_extra={
            "group": "surfaces",
            "requires": [],
            "requires_inputs": ["normals", "occlusion", "daylight"],
        },
    )
    daylight_glow: bool = Field(
        default=True,
        description="The glow of the medium, i.e. the light of suns and the sky it scatters, lights surfaces from every "
        "direction, which makes up for part of the light it takes out, more so for surfaces facing the sun.",
        json_schema_extra={
            "group": "surfaces",
            "requires": [],
            "requires_inputs": ["normals", "sun_attenuation", "daylight", "scattering"],
        },
    )
    glow_shadows: bool = Field(
        default=True,
        description="Surfaces in the shadow of a sun lack the glow of its light scattered once by the medium, which "
        "lies in the shadow too.",
        json_schema_extra={"group": "surfaces", "requires": ["daylight_glow"], "requires_inputs": ["occlusion", "suns"]},
    )
    glow_shadow_edges: bool = Field(
        default=True,
        description="Near the edges of a sun's shadows, and on the sides of objects turned away from the sun, surfaces "
        "keep the part of that glow that comes from the medium the sun lights beside them.",
        json_schema_extra={"group": "surfaces", "requires": ["glow_shadows"], "requires_inputs": []},
    )
    lamp_glow: bool = Field(
        default=True,
        description="The glow of the halos of lamps lights surfaces from every direction, also in the shadows of lamps, "
        "rather than the reduced extinction of the similarity relation making up for part of it.",
        json_schema_extra={"group": "surfaces", "requires": ["halos"], "requires_inputs": ["normals"]},
    )
    received_irradiance: bool = Field(
        default=True,
        description="The light that surfaces receive from other surfaces, by which the glow of the medium is weighed, "
        "is at least the rest of what they received, as measured for each pixel, or as their radiance and the mean "
        "albedo of the ground and objects imply, rather than a share of the frame's direct light only.",
        json_schema_extra={"group": "surfaces", "requires": [], "requires_inputs": ["normals", "received"]},
    )
    measured_indirect: bool = Field(
        default=True,
        description="Given the light that the surface seen by each pixel received, as its irradiance or its albedo, all "
        "of it beyond its direct light reaches it from other surfaces, even where it is less than the share of the "
        "frame's direct light that only guesses it otherwise, e.g. on the ground in the shadow of a lamp.",
        json_schema_extra={
            "group": "surfaces",
            "requires": ["received_irradiance"],
            "requires_inputs": ["normals", "measured"],
        },
    )

    @classmethod
    def prerequisites(cls) -> dict[str, dict[str, Any]]:
        """Group and prerequisites of each switch, as declared by the ``json_schema_extra`` of its field.

        Returns:
            dict[str, dict[str, Any]]: For each switch, by name, its ``group``, the names of the switches it
            ``requires``, and the conditions on the inputs it requires, ``requires_inputs``.
        """
        declared = {}
        for name, field in cls.model_fields.items():
            extra = field.json_schema_extra
            assert isinstance(extra, dict), f"Effect {name} doesn't declare its group and prerequisites."
            declared[name] = {
                "group": extra["group"],
                "requires": list(extra.get("requires") or []),
                "requires_inputs": list(extra.get("requires_inputs") or []),
            }
        return declared

    @staticmethod
    def conditions(medium: Medium, lighting: Lighting, **given: bool) -> dict[str, bool]:
        """Which conditions on the inputs of the model hold, see :data:`EFFECT_INPUTS`.

        Args:
            medium (Medium): Participating medium.
            lighting (Lighting): Lighting of the scene.
            **given (bool): Whether each input of :data:`EFFECT_GIVEN` is given with frames, by name, those left out
                being missing: ``shadows``, shadows traced beforehand, see :func:`trace_shadows
                <visionsim.medium.render.trace_shadows>`, which shadow maps give too; ``occlusion``, shadow maps;
                ``normals``, the normals of surfaces; ``albedo``, the albedo of the surfaces seen by each pixel; and
                ``irradiance``, the irradiance that they received without the medium.

        Raises:
            ValueError: If an input isn't among :data:`EFFECT_GIVEN`.

        Returns:
            dict[str, bool]: Whether each condition holds, by name, in the order of :data:`EFFECT_INPUTS`.
        """
        if unknown := sorted(set(given) - set(EFFECT_GIVEN)):
            raise ValueError(f"Unknown inputs {unknown}, which should be among {list(EFFECT_GIVEN)}.")
        holds = {name: bool(given.get(name, False)) for name in EFFECT_GIVEN}
        # Shadows are traced from shadow maps
        holds["shadows"] = holds["shadows"] or holds["occlusion"]
        suns = any(sun.direction[2] > 0 and any(v != 0 for v in sun.irradiance) for sun in lighting.suns)
        ground = max(lighting.ground_albedo, default=0.0) > 0
        objects = max(lighting.ground_albedo if lighting.object_albedo is None else lighting.object_albedo, default=0.0)
        holds |= {
            "suns": suns,
            "daylight": suns or any(v != 0 for v in lighting.sky),
            "lamps": bool(lighting.lamps),
            "direct_light": suns or bool(lighting.lamps),
            "scattering": medium.albedo > 0,
            "sun_attenuation": medium.sun_attenuation,
            "multiple_scattering": medium.multiple_scattering,
            "ground_albedo": ground,
            "object_albedo": objects > 0,
            # The light received by the surface of each pixel, which its irradiance, or its albedo and radiance, give,
            # or else that of surfaces of the mean albedo of the ground and of objects
            "measured": holds["albedo"] or holds["irradiance"],
            "received": holds["albedo"] or holds["irradiance"] or ground or objects > 0,
            # Objects only reflect the light of lamps whose maps record them
            "reflectors": ground or (objects > 0 and holds["occlusion"]),
            "specular": lighting.surface_ior != 1.0,
        }
        return {name: holds[name] for name in EFFECT_INPUTS}

    def missing(self, medium: Medium, lighting: Lighting, **given: bool) -> dict[str, list[str]]:
        """Prerequisites of each switch that aren't met, which make it ineffective, whether it is on or off.

        The switches that a switch requires are followed recursively: a switch that is off is missing, as are the
        prerequisites of the switches it requires that aren't met, whether these are on or off.

        Args:
            medium (Medium): Participating medium.
            lighting (Lighting): Lighting of the scene.
            **given (bool): Whether each input of :data:`EFFECT_GIVEN` is given with frames, by name, see
                :meth:`conditions`.

        Returns:
            dict[str, list[str]]: For each switch, by name, the names of the switches it needs that are off, and of the
            conditions on the inputs it needs that don't hold, see :data:`EFFECT_INPUTS`, each once, or an empty list if
            all of its prerequisites are met.
        """
        declared = self.prerequisites()
        holds = self.conditions(medium, lighting, **given)

        def unmet(name: str, seen: set[str]) -> list[str]:
            result = [condition for condition in declared[name]["requires_inputs"] if not holds[condition]]
            for required in declared[name]["requires"]:
                if required in seen:
                    continue
                seen.add(required)
                result += ([] if getattr(self, required) else [required]) + unmet(required, seen)
            return list(dict.fromkeys(result))

        return {name: unmet(name, {name}) for name in declared}

    def effective(self, medium: Medium, lighting: Lighting, **given: bool) -> dict[str, bool]:
        """Which switches are effective for given inputs, i.e. on, with all their prerequisites met, see
        :meth:`missing`.

        Args:
            medium (Medium): Participating medium.
            lighting (Lighting): Lighting of the scene.
            **given (bool): Whether each input of :data:`EFFECT_GIVEN` is given with frames, by name, see
                :meth:`conditions`.

        Returns:
            dict[str, bool]: Whether each switch is effective, by name.
        """
        missing = self.missing(medium, lighting, **given)
        return {name: bool(getattr(self, name)) and not missing[name] for name in missing}

    def resolve(self, medium: Medium, lighting: Lighting, **given: bool) -> Effects:
        """Switches whose terms are computed for given inputs, i.e. those that are :meth:`effective`, the others off.

        Args:
            medium (Medium): Participating medium.
            lighting (Lighting): Lighting of the scene.
            **given (bool): Whether each input of :data:`EFFECT_GIVEN` is given with frames, by name, see
                :meth:`conditions`.

        Returns:
            Effects: Effective switches.
        """
        return self.model_copy(update=self.effective(medium, lighting, **given))


def _list_effects() -> str:
    """List of the switches of :class:`Effects`, with their group, prerequisites and description, which ends its
    docstring, so that the documentation follows the fields."""
    lines = [""]
    for name, declared in Effects.prerequisites().items():
        needs = [f"``{required}``" for required in declared["requires"] + declared["requires_inputs"]]
        prerequisites = f", requires {', '.join(needs)}" if needs else ""
        lines.append(f"    - ``{name}`` ({declared['group']}{prerequisites}): {Effects.model_fields[name].description}")
    return "\n".join(lines) + "\n"


Effects.__doc__ = (Effects.__doc__ or "").rstrip() + "\n" + _list_effects()

EFFECT_PRESETS: dict[str, tuple[Effects, str]] = {
    "complete": (
        Effects(),
        (
            "Every term of the model. In the scenes of examples/medium, each changes whole frames, the medium's light "
            "or surfaces by at least 0.1 points of relative L1 error, and removes more error per millisecond than "
            "computing the medium at every pixel rather than at a quarter of them does: computing it at a quarter of "
            "the resolution, with every term, is the balanced choice."
        ),
    ),
    "fast": (
        Effects(
            azimuthal_occlusion=False,
            ground_shadows=False,
            ground_sky_occlusion=False,
            scattered_daylight_reflections=False,
            glow_shadows=False,
            glow_shadow_edges=False,
            halo_shadow_edges=False,
            bounce_surfaces=False,
            interreflections=False,
            relit_scattering=False,
        ),
        (
            "Only the terms that change whole frames by at least 0.5 points of relative L1 error, or the medium's light "
            "or surfaces by at least 1 point, in the scenes of examples/medium, and those they require."
        ),
    ),
}
"""Presets of the switches, by name, with their description, chosen by stated rules from the error that each term
removes in the day and night scenes of ``examples/medium`` (``ablation.py`` of ``examples/medium/experiment``, see the
documentation): ``complete`` keeps every term, as each changes whole frames, the medium's light or surfaces by at least
0.1 points of relative L1 error in a scene where it is effective, and ``fast`` only keeps those that change whole
frames by at least 0.5 points, or the medium's light or surfaces by at least 1 point, and the switches they require.
Changes are taken in absolute value, as a term whose removal brings frames closer to a reference only by cancelling
another error of the model still matters."""
