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

For objects to cast shadows onto the medium (light shafts) and hide part of the sky from it, also render shadow maps
of the scene by adding ``--config.include-occlusion`` when rendering, and exclude any large ground plane, which would
otherwise make the maps span all of it, with ``--config.occlusion.exclude Plane``. ``medium.apply`` then uses the
``occlusion.npz`` it finds in the renders.

Tracing where objects hide the lights along the rays of a frame takes most of the time of shadows, but doesn't depend
on the medium. To add several media to the same frame, such as fogs of different visibilities, trace shadows once with
:func:`trace_shadows <visionsim.medium.render.trace_shadows>` and pass them to :func:`apply_medium
<visionsim.medium.render.apply_medium>` as ``shadows``, which then only integrates each medium. As skylight is sampled
where the medium the shadows were traced with scatters light, this is approximate for media of other densities: in
``examples/medium``, the light scattered by fogs from a quarter to four times as dense differs by less than 0.4% from
tracing shadows for each of them.

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
:meth:`lighting_info <visionsim.simulate.blender.BlenderService.exposed_lighting_info>`): sun, point, spot and area
lights, and the average radiance of the world background above the horizon, i.e. the sky. The export accounts for each
light's power, color, exposure, temperature, volume factor and object scale, and for the node trees that commonly set
their emission: an Emission node whose color can come from a Blackbody node, and whose strength can come from a Light
Falloff node (including its linear and constant outputs, and its smoothing near the light) or an IES Texture node, whose
angular profile is not supported. Light from below the horizon is assumed to be blocked by the ground, and, with
``sun_attenuation``, both sunlight and skylight are attenuated by the fog they travel through before being scattered.
The ground can also reflect light into the fog, given its ``ground_albedo`` and ``ground_height`` in the lighting, which
are not exported from Blender.

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

Lamps emit as in Cycles (see :mod:`visionsim.medium.lights`): point lights with a radiant intensity of their power
divided by :math:`4\pi`, spot lights likewise within a cone towards the edge of which their light fades smoothly, and
area lights as one-sided Lambertian emitters of radiance :math:`P / (\pi A)`, possibly restricted by their spread.
Light from each point emitter is integrated along each ray over the angle at which the emitter sees it (equi-angular
sampling), with Gauss-Legendre quadrature, which cancels the singularity near the emitter, see
:func:`point_light_inscatter <visionsim.medium.optics.point_light_inscatter>`. Seen from the emitter, a ray sweeps
a great circle, which crosses a cone of light over a single range of angles: only that range is integrated, so that
the edge of a spot's cone doesn't spoil the quadrature. Area lights are split into patches, each a point emitter
whose intensity is proportional to the cosine of the angle to the light's normal: rays that pass within a radius of the
light see a grid of 16 patches along its longest side, those that pass within 3 radii a grid of 4, those that pass
within 8 radii a grid of 2, and those further away a single emitter, with smooth transitions in between. This is within
about 1% of a fine grid over a frame.

Objects also cast the shadows of lamps onto the medium, through an equirectangular map of the distance to the first
surface around each lamp, rendered by Cycles from the lamp's position. Their visibility is sampled along each ray at
angles evenly spread as seen from the lamp, which matches the map's angular resolution whatever the distance, and the
weight of each node of the quadrature is scaled by the visibility averaged between its neighbors, rather than at the
node, so that the edges of shadows don't spoil the quadrature, see :func:`lamp_shadow
<visionsim.medium.occlusion.lamp_shadow>`. Area lights cast the shadows of their center.

In dense media, light from lamps scattered more than once spreads a halo around them, much wider than the glow of light
scattered once. With ``multiple_scattering``, it is computed on a grid of points around each lamp, spread
logarithmically in distance from it: at each point, the light scattered once is gathered from directions concentrated
towards the lamp, where most of it comes from, and scattered again with the phase function. As this light mostly keeps
going away from the lamp, it is tabulated against the angle between the direction it is scattered towards and the
direction away from the lamp. Higher orders are iterated by integrating each order along the gathering rays, and later
ones are extrapolated geometrically at each point, see :mod:`visionsim.medium.halos`. The tables don't depend on the
camera, so they are computed once per lamp and medium, and integrated along camera rays as light scattered once is.

Surfaces are lit through the medium too: their light is attenuated on its way to them, while the medium's glow lights
them. Given the normals of surfaces (``normals`` of :func:`apply_medium <visionsim.medium.render.apply_medium>`, which
``medium.apply`` reads from renders that include them), the radiance of each surface is scaled by the ratio of the
irradiance it receives with and without the medium, see :mod:`visionsim.medium.surfaces`. Sunlight, skylight and light
reflected by the ground are attenuated in closed form, and the glow arrives from every direction: the light of suns
scattered once, with the phase function towards each direction, so that surfaces facing the sun receive much more of
it than those facing away, the light of the sky and the ground scattered once, and, with ``multiple_scattering``, light
scattered more times. As the glow only depends on the height of surfaces and on the orientation of their normal, it is
tabulated once per frame. Light from lamps is attenuated with the share of the extinction that isn't scattered within
the forward peak of the phase function, ``1 - albedo · g²`` (delta-Eddington), as light scattered forward keeps going
towards surfaces. Light that bounces off other surfaces first is assumed to be dimmed as the frame's direct light is on
average, and the light that surfaces emit, given by an emission pass (``--include-emission``), isn't dimmed at all.

Objects block sunlight from the medium behind them, which casts light shafts, and hide part of the sky from the medium
around them, such as the fog in front of a wall. Both are modeled with shadow maps, i.e. orthographic depth maps of the
scene rendered by Cycles along the direction of each sun, and along the central direction of each of the cells into
which the sky is split. Visibility is sampled along each ray where objects can cast shadows, and the closed-form
integral of each source between consecutive samples is weighted by the visibility in between, so the result stays
deterministic, and exact wherever nothing is occluded. The shadows of suns are sharp: rays are split into intervals of
a few texels of the sun's map, intervals that lie entirely above or below the surfaces recorded around them are lit or
hidden as a whole, and the others, which the edge of a shadow may cross, are sampled every texel. Skylight and light
scattered more than once are sampled more coarsely, as each point is lit by many cells, each weighted by how much
light it scatters towards the camera. As rays close to each other sample places close to each other, these places are
snapped to a grid finer than the texels of the sky's maps, whose cells are each looked up once. Light that reaches
the medium from below the horizon, reflected by the ground or scattered by the medium beneath, is hidden by objects
standing on the ground as the mirrored cells above the horizon are, unless they stand further than where this light
starts on the ground. See :mod:`visionsim.medium.occlusion` for details.

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
elevation but looking down at the ground, where it is 12% brighter. Light scattered by lamps is within about 1% of
Cycles' single scattering overall: 0.6% for a point light, 0.1% for a spot light, 1.2% for an area light, and 0.9% for
a point light whose light is smoothed by a Light Falloff node, with per-pixel differences of 1% to 2% once Cycles' noise
is blurred out, the largest right next to the lamps. With a point light right behind a cube, which hides it from the
camera, objects casting its shadows bring the light scattered by the fog within 0.3% of Cycles', with per-pixel
differences of 0.9% once blurred, against eleven times too much light without shadows. With multiple scattering, all the
light of a point lamp scattered by the fog, in the scene without objects and with a black ground, is within 1.4% of
Cycles' with 32 volume bounces, and within 5% at any distance from the lamp, with per-pixel differences of 2.5% once
blurred, against 15% too dark with single scattering only. Note that Cycles clamps indirect light by default (Clamp
Indirect of 10), which darkens light scattered more than once near bright lamps, by 30% for this lamp: references are
rendered without clamping (``render_passes.py --no-clamp``).

Surfaces lit through the fog (``validate_surfaces.py``) are within 1% of Cycles' overall, both with single and multiple
scattering, against 13% too bright in the render without fog for single scattering, with per-pixel differences of 3.5%
and 4.6%, against 14% and 5%. Walls, which only see the glow of the fog around them, are within about 3%, against 26% too
bright. Surfaces lit by a point, spot or area light are within 1% to 2.4%, against 8% to 11%. Note that Cycles' renders
with no volume bounces include light scattered once on its way to surfaces too, as the light is sampled from where it
scatters.

With objects casting shadows, the sun's light scattered by the fog is within 1.3% of Cycles' over surfaces and within
0.8% over the sky, against 17% and 3% without shadows, and the sky's is within 1% over both, against 25% and 3%.
``examples/medium/experiment`` compares every method as seen by the sensor emulators.

|

Limitations
-----------

The following are not yet modeled:

- Shadows of moving objects, as shadow maps are rendered once per scene, and the soft shadows of large lamps, which
  cast the hard shadows of their center.
- Emissive surfaces lighting the medium, IES profiles and other node trees of lamps, square spot lights, and the
  elliptical cones of spot lights scaled unevenly. Light that surfaces lit by lamps reflect into the medium, which, in
  ``examples/medium``, adds as much again to the light of a lamp scattered twice near the lit ground.
- Multiple scattering in media other than a single height fog, and skies whose radiance varies with direction.
- Multiple scattering near objects, which is overestimated, by about 30% to 60% within 5 m of the objects of
  ``examples/medium``: the medium around objects is itself in their shadow, and so darker than the open medium that
  the approximation assumes.
- The shadows of objects on the glow that lights surfaces, which assumes that surfaces stand in the open, unless shadow
  maps show that they don't see the sky, and the glow of lamps on surfaces beyond what the reduced extinction of
  delta-Eddington accounts for. Surfaces are assumed to be diffuse, and light that bounces off other surfaces to be
  dimmed as the frame's direct light is on average.
- Anti-aliasing: depth maps are not anti-aliased, so edges between near and far objects can show halos in dense
  media. Rendering at a higher resolution and downsampling the results reduces these.
