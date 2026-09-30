from __future__ import annotations

import math
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveFloat, model_validator
from typing_extensions import Self

Vector3 = tuple[float, float, float]


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


class PointLight(BaseModel):
    """Point light source, such as a street lamp, whose radiant intensity is its power divided by 4π, as in Cycles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    position: Vector3
    """world-space position of the light, in meters"""
    power: tuple[float, ...]
    """radiant power per color channel, in W"""
    radius: NonNegativeFloat = 0.0
    """radius of the light, in meters, below which distances to the light are clamped"""


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
    ground_albedo: tuple[float, ...] = (0.0, 0.0, 0.0)
    """albedo per color channel of the ground, modeled as a Lambertian plane at ``ground_height`` lit by suns and
    the sky, which reflects light into the medium. This is only used when sunlight is attenuated by the medium"""
    ground_height: float = 0.0
    """height of the ground, in meters"""
