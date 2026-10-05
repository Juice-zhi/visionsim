from __future__ import annotations

import math
from typing import Annotated, Literal, Union

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
    """albedo per color channel of the ground, modeled as a Lambertian plane at ``ground_height`` lit by suns and
    the sky, which reflects light into the medium. This is only used when sunlight is attenuated by the medium"""
    ground_height: float = 0.0
    """height of the ground, in meters"""

    @property
    def lamps(self) -> list[Lamp]:
        """Light sources at a finite distance, i.e. point, spot and area lights, and emissive surfaces."""
        return [*self.points, *self.spots, *self.areas, *self.emissive]
