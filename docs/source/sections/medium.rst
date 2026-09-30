Participating Media
===================

Fog, haze or smoke can be added to rendered scenes without volumetric path tracing. Besides being slow, path tracing
leaves Monte Carlo noise in its renders, which photon-level emulators, such as those of single photon cameras or
event cameras, would pick up as if it were photon noise or brightness changes. Instead, a medium is described by a
handful of physical parameters and added after rendering, using the linear radiance and depth of each frame. Its
effect on every pixel is computed in closed form, which yields a unique and noise-free result that is then shared by
all sensor emulators, so that they all see the same medium.

|

Workflow
--------

First, render linear frames and depth maps, as well as the lighting of the scene::

    $ visionsim blender.render-animation scene.blend renders/ --config.include-depths --config.include-lighting \
        --config.frames.file-format OPEN_EXR --config.frames.bit-depth 32

Then, describe the medium in a JSON file, here a ground fog lit by the sun:

.. code-block:: json

    {
      "extinction": 0.04,
      "anisotropy": 0.6,
      "sun_attenuation": true,
      "components": [{"type": "height", "density": 1.0, "base_height": 0.0, "falloff": 6.0}]
    }

And apply it to the renders::

    $ visionsim medium.apply --input-dir renders/ --output-dir renders-fog/ --medium fog.json

The frames in ``renders-fog/frames`` are linear EXRs that can be used as the input of any :doc:`sensor emulator
<emulation>`. The transmittance, optical depth and in-scattered radiance of each pixel are saved alongside them
as ground truth, in ``renders-fog/transmittance``, ``renders-fog/optical-depth`` and ``renders-fog/inscatter``,
as well as the medium and lighting that were used. The same can be done using the API, with
:func:`apply_medium <visionsim.medium.render.apply_medium>`.

|

Describing a medium
-------------------

A :class:`Medium <visionsim.medium.model.Medium>` is made of density components, which all share the same optical
properties:

- ``extinction``: extinction coefficient, in 1/m, at a reference wavelength of 550nm. It can also be set from a
  visibility using :meth:`Medium.from_visibility <visionsim.medium.model.Medium.from_visibility>`, following
  Koschmieder's law, e.g. a visibility of 100m corresponds to an extinction of 0.039/m.
- ``angstrom``: exponent of the wavelength dependence of extinction, close to zero for fog and around 1.3 for haze.
  :func:`kim_exponent <visionsim.medium.model.kim_exponent>` provides an empirical value based on visibility.
  Each color channel is attenuated according to its effective wavelength.
- ``albedo``: ratio of scattering to extinction, close to one for water droplets.
- ``anisotropy``: asymmetry of the Henyey-Greenstein phase function, fog droplets mostly scatter light forward.
- ``components``: :class:`Homogeneous <visionsim.medium.model.Homogeneous>` density, exponential
  :class:`HeightFog <visionsim.medium.model.HeightFog>`, and Gaussian :class:`Blob <visionsim.medium.model.Blob>`\s,
  which can move with a given velocity, for instance to model patches of fog drifting in the wind. The medium is
  defined in world space, so it is consistent across frames and cameras.

The medium is lit by the same lights as the scene, which are exported from Blender to ``lighting.json`` (see
:meth:`lighting_info <visionsim.simulate.blender.BlenderService.exposed_lighting_info>`): sun and point lights, and the
average radiance of the world background above the horizon, i.e. the sky. Light from below the horizon is assumed to
be blocked by the ground, and, with ``sun_attenuation``, both sunlight and skylight are attenuated by the fog they travel
through before being scattered. The ground can also reflect light into the fog, given its ``ground_albedo`` and
``ground_height`` in the lighting, which are not exported from Blender.

Dense fog absorbs little light, so a large share of the light it scatters towards the camera was already scattered
before: in the ground fog of ``examples/medium``, whose albedo is one, light scattered more than once makes up 63% of
the fog's light. Setting ``multiple_scattering`` on the medium approximates it, see below. Note that Blender's default
maximum number of volume bounces, zero, also renders single scattering only.

|

Light transport
---------------

Along the ray of each pixel, which travels a distance :math:`d` before hitting a surface of radiance
:math:`L_{surf}`, the radiance reaching the camera is:

.. math::
    L = T(d) \, L_{surf} + \int_0^d \sigma(s) \, T(s) \, J(s) \, ds \,, \qquad T(s) = e^{-\tau(s)} \,, \qquad
    \tau(s) = \int_0^s \sigma(r) \, dr

where :math:`\sigma` is the extinction coefficient and :math:`J` the light scattered towards the camera, which is
the albedo times the skylight weighted by the phase function over the sky, plus the phase function times the irradiance
of each sun. Every component
has a closed-form optical depth :math:`\tau`, which relies on error functions for blobs. Moreover, when :math:`J` is
constant along the ray, the integral is exactly :math:`J \, (1 - T(d))` regardless of the density, which is a
generalization of Koschmieder's model. When sunlight is attenuated by an exponential height fog before being
scattered, this integral also has a closed form:

.. math::
    \int_0^d \sigma(s) \, T(s) \, T_{sun}(s) \, ds = \frac{\ell_z}{\ell_z - v_z} \left( T_{sun}(0) - T(d) \, T_{sun}(d) \right)

where :math:`v_z` and :math:`\ell_z` are the vertical components of the ray direction and of the direction towards
the sun, and :math:`T_{sun}` the transmittance between a point and the sun. See :mod:`visionsim.medium.optics` for
details.

Skylight, light reflected by the ground and light scattered more than once reach the fog from many directions, so that
the light they scatter towards a ray only depends on the ray's elevation and on the height of each point. Their
integral along rays is tabulated once per frame against the rays' elevation and optical depth, as all camera rays start
from the same point, and interpolated for every pixel, which is within 0.1% of integrating every direction of the sky
for each pixel, while the cost of the table does not depend on the resolution. Directions of the sky and the ground are integrated with a fixed
quadrature (:func:`sky_quadrature <visionsim.medium.optics.sky_quadrature>`), so the result stays deterministic and
noise-free. Light scattered more than once is approximated following Hillaire (EGSR 2020): light scattered once, by the
fog or the ground, is gathered at a set of heights and scattered again with the phase function, and higher orders are
summed as a geometric series. See :mod:`visionsim.medium.scattering` for details.

Point lights are integrated along each ray over the angle at which the light sees it (equi-angular sampling), with
Gauss-Legendre quadrature, which cancels the singularity near the light, see :func:`point_light_inscatter
<visionsim.medium.optics.point_light_inscatter>`. Their radiant intensity is their power divided by :math:`4\pi`, as in
Cycles.

|

Validation
----------

The ``examples/medium`` directory renders a scene through visionsim, and the same scene with an equivalent fog
volume using Cycles' volumetric path tracing, then compares them. When restricted to what is modeled here
(single scattering, no shadows cast onto the fog), the in-scattered radiance agrees with Cycles' volume direct
pass with a median relative error of 0.26%, and the radiance of objects and of the sky seen through the fog is
within 0.3%. The remaining differences are Monte Carlo noise, and object edges.

In the same scene without objects, the light scattered by the fog towards the camera, including multiple scattering
and light reflected by the ground, is within 2% of Cycles with 32 volume bounces overall, and within 8% at any
elevation. Light scattered by a point light is within 1% of Cycles' single scattering.
``examples/medium/experiment`` compares every method as seen by the sensor emulators.

|

Limitations
-----------

The following are not yet modeled:

- Shadows cast onto the medium, i.e. light shafts, and the occlusion of the sky by nearby objects.
- Spot and area lights, as well as emissive surfaces, lighting the medium. Light from point lights is only
  scattered once, which misses about 30% of their glow in the dense fog of ``examples/medium``.
- Multiple scattering in media other than a single height fog, and skies whose radiance varies with direction.
- The dimming of surfaces lit through the medium, and their lighting by the fog's glow.
- Anti-aliasing: depth maps are not anti-aliased, so edges between near and far objects can show halos in dense
  media. Rendering at a higher resolution and downsampling the results reduces these.
