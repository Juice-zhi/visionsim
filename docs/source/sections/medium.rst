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

Lights can be animated: lighting is exported at every rendered frame, after keyframes are moved by
``--config.keyframe-multiplier``, and when it changes, e.g. as lamps move or their power changes, the lighting of each
frame at which it changes is saved to ``animated-lighting.json``, which ``medium.apply`` uses by the number of each
frame (see :class:`AnimatedLighting <visionsim.medium.model.AnimatedLighting>`). Moving lamps get shadow maps along
their way, rendered at the frames from which they move further than ``--config.occlusion.lamp-spacing`` (10 cm by
default) from where their previous maps were, and cast the shadows of the nearest one. Emissive surfaces are exported
again at every frame if they, their parents or their materials are animated. ``--config.lighting.no-animated`` only
exports the lighting of the first frame.

Tracing where objects hide the lights along the rays of a frame takes most of the time of shadows, but doesn't depend
on the medium. To add several media to the same frame, such as fogs of different visibilities, trace shadows once with
:func:`trace_shadows <visionsim.medium.render.trace_shadows>` and pass them to :func:`apply_medium
<visionsim.medium.render.apply_medium>` as ``shadows``, which then only integrates each medium. As skylight is sampled
where the medium the shadows were traced with scatters light, this is approximate for media of other densities: in
``examples/medium``, the light scattered by fogs from a quarter to four times as dense differs by about 0.8% per pixel
(mean absolute difference, relative to the light) from tracing shadows for each of them.

Most of the time per frame grows with the number of pixels: tracing shadows along each ray, integrating lamps, their
halos and the light that surfaces reflect along it, and lighting each surface. The light that the medium scatters is
smooth across the image, though, so that it can be computed along the rays of a fraction of the pixels and interpolated
in between: :func:`apply_medium_upsampled <visionsim.medium.upsampling.apply_medium_upsampled>` adds the medium along
the rays of every ``factor``-th pixel along each axis (``--factor`` of ``medium.apply``), and upsamples the light it
scatters per unit of optical depth, and the ratio by which it scales the light of surfaces, to every pixel, weighted by
how close the depths and normals of the samples are to each pixel's (joint bilateral upsampling), see
:mod:`visionsim.medium.upsampling`. The transmittance and the optical depth of every pixel are exact, so that the edges
of objects stay sharp, while the edges of the shadows that objects cast onto the medium are blurred over ``factor``
pixels. In ``examples/medium``, at a quarter of the resolution of 320x180 frames, the light scattered by the fog differs
by 0.6% (relative L1) by day and by 1.1% at night, where the cube's shadow cuts the light of a lamp, from adding it at
every pixel, and by 0.4% and 0.1% at a quarter of 1920x1080, an order of magnitude below the error of the model
against Cycles, which it barely changes. It is worth it at high resolutions, where it saves most of the time, but
should be avoided when the medium's light is used as ground truth at each pixel, e.g. for the sharp edges of light
shafts.

Everything that doesn't depend on the camera is computed once and kept for the next frames: light scattered more than
once, the tables through which surfaces are lit, the halos of lamps and the light of lamps that surfaces reflect, the
share of the sky that the ground sees around objects, and so are the tables of light from the sky, the ground and light
scattered more than once integrated along camera rays, which only depend on the height of the camera, and the copies of
the shadow maps on the GPU. These copies are kept for the maps' tensors, which should be the same objects from one frame
to the next rather than copies of them, and should only be changed in place through PyTorch's operations, as changes
made otherwise, e.g. through a NumPy array that shares their memory, aren't seen. Only the tables along camera rays are
computed again when the camera changes height, which takes about 10 ms on a GPU and 0.2 s on the CPU, while the shadows
along camera rays and the light of suns and the sky that objects reflect towards them are traced and gathered for every
frame. With these, on 12 CPU threads, a frame of the day scene of ``examples/medium`` takes about 0.2 s at a quarter of
320x180 (1.3 s at every pixel) and 2.6 s at a quarter of 1920x1080, and a frame of the night scene 0.15 s (1.2 s at
every pixel) and 3.8 s. On a GPU, at 320x180, most of the time is spent launching thousands of small kernels, which a
lower resolution doesn't save.

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
lights, meshes that emit light, and the average radiance of the world background above the horizon, i.e. the sky. The export accounts for each
light's power, color, exposure, temperature, volume factor and object scale, and for the node trees that commonly set
their emission: an Emission node whose color can come from a Blackbody node, and whose strength can come from a Light
Falloff node (including its linear and constant outputs, and its smoothing near the light) or an IES Texture node, whose
angular profile is not supported. Light from below the horizon is assumed to be blocked by the ground, and, with
``sun_attenuation``, both sunlight and skylight are attenuated by the fog they travel through before being scattered.
The ground can also reflect light into the fog, given its ``ground_albedo`` and ``ground_height`` in the lighting, which
are not exported from Blender, and so can other surfaces lit by lamps, given their ``object_albedo``, the ground's by
default. Renders made with ``--config.include-diffuse-pass`` save Blender's diffuse color pass, the albedo of the
surfaces seen by each pixel, to ``diffuse/color``, along with its diffuse direct and indirect light passes. Asked to
(``--estimate-albedos``), ``medium.apply`` estimates from them the ground's albedo, when the lighting's is zero, and the
objects', when it gives none, from up to 16 frames spread over the sequence, as the mean albedo of the pixels that see
them, weighted by the light that their surfaces received, see :func:`mean_albedos
<visionsim.medium.render.mean_albedos>`, and saves them with the lighting. They are estimated once for all frames rather
than for each of them, as the light that surfaces reflect into the fog doesn't depend on what the camera sees:
estimated for each frame, they would make the fog flicker as objects come into view, and compute the halos of lamps and
the light that surfaces reflect again at every frame. The diffuse color pass leaves out the specular reflection of
glossy surfaces, which reflects light into the fog too: the ground of ``examples/medium``, a Principled BSDF whose base
colors are those of a checkerboard of mean 0.39, has a mean diffuse albedo of 0.366, weighted by the light it received,
while 13% of its light in the frames, at the camera's angles, is its specular reflection, and the fog's light is further
from Cycles' with the albedos estimated than with 0.39 (see the validation).

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
the light they scatter towards a ray only depends on the ray's elevation and on the height of each point, and, for the
light of suns scattered more than once, on the ray's azimuth from the sun's. Their integral along rays is tabulated once
per frame against the rays' elevation (and azimuth) and optical depth, as all camera rays start from the same point,
and interpolated for every pixel, which is within 0.1% of integrating every direction of the sky for each pixel, while
the cost of the table does not depend on the resolution, and the table is kept for the next frames as long as the
camera stays at the same height. Directions of the sky and the ground are integrated with a fixed quadrature
(:func:`sky_quadrature <visionsim.medium.optics.sky_quadrature>`), so the result stays deterministic and noise-free.
Light scattered more than once is computed by successive orders of scattering, once per medium and
lighting: at a set of heights, the light of each order is gathered from all directions and scattered again with the
phase function, while the ground reflects the light of each order that reaches it, until further orders are negligible.
In fog, which scatters light strongly forward, light keeps going in about the same direction, so that near the horizon
each order is nearly as bright as the previous one, and the light that the fog takes out of sunlight mostly still
reaches the ground. As the fog doesn't change horizontally, each harmonic of the azimuth from suns is scattered
independently, which keeps light scattered more than once brighter towards suns. See
:mod:`visionsim.medium.scattering` for details.

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

Meshes that emit light, through an Emission shader or the emission of a Principled BSDF, possibly mixed or added with
other shaders, light the medium too: their radiance is the emission's strength times its color, which can come from a
Blackbody node or from an image, whose mean color is used, while emission that only the camera sees (mixed in by a
Light Path node, as for "shadeless" materials) is left out, as it lights nothing. Emissive surfaces further than half a
meter apart are grouped into different lamps, up to 8 of them, beyond which lamps are gathered by position, and lamps
are split into 32 patches in total, those of the most light times size first (both can be set with
``--config.lighting``), see :meth:`lighting_info <visionsim.simulate.blender.BlenderService.exposed_lighting_info>`. As in
Cycles, both sides of a surface emit, but surfaces block each other's light, such as the light emitted inside a closed
surface: rays are traced from each side of a sample of their faces to find how much of their light leaves them. Each
patch then shines as a point emitter in proportion to the area of its faces seen from each direction, counting those
whose light leaves towards it, which is approximated from the first two moments of their normals: exactly for flat
patches, spheres, cylinders and hemispheres, see :func:`projected_area <visionsim.medium.lights.projected_area>`.
Patches whose faces it approximates poorly, such as those of a box, are split by the orientation of their faces.
Distances to a patch are clamped at the root mean square distance between its points and its center, e.g. the radius
of a ball, and rays that pass close to flat patches see them as grids over the rectangle of the same spread of points,
as they do area lights, see :func:`patch_emitters <visionsim.medium.lights.patch_emitters>`.

Objects also cast the shadows of lamps onto the medium, through an equirectangular map of the distance to the first
surface around each lamp, rendered by Cycles from the lamp's position. Their visibility is sampled along each ray at
angles evenly spread as seen from the lamp, which matches the map's angular resolution whatever the distance, and the
weight of each node of the quadrature is scaled by the visibility averaged between its neighbors, rather than at the
node, so that the edges of shadows don't spoil the quadrature, see :func:`lamp_shadow
<visionsim.medium.occlusion.lamp_shadow>`. Area lights cast the shadows of their center, and so do emissive lamps
that surround their center, such as a ball, otherwise they cast those of the center of their face nearest to it, as
it could lie within another object, such as the middle of a ring of light. Their maps see through emissive surfaces,
as these don't hide their own light.

In dense media, light from lamps scattered more than once spreads a halo around them, much wider than the glow of light
scattered once. With ``multiple_scattering``, it is computed on a grid of points around each lamp, spread
logarithmically in distance from it and evenly in solid angle around it: at each point, the light scattered once is
gathered from directions concentrated towards the lamp, where most of it comes from, and scattered again with the phase
function. As this light mostly keeps going away from the lamp, it is tabulated against the angle between the direction
it is scattered towards and the direction away from the lamp, and against the azimuth around the latter, as in the
shadows of objects it comes from the medium lit beside them instead. Higher orders are iterated by integrating each
order along the gathering rays, and later ones are extrapolated geometrically at each point, see
:mod:`visionsim.medium.halos`. Given shadow maps, objects stop this light: the gathering rays stop at the first object
they hit, found in the visual hull of the objects seen by the maps, which is baked into voxels, so that points in the
shadow of an object don't gather the light on its other side, and points lit by the lamp, or in its shadow, are only
interpolated from points of the grid alike. Light scattered twice fades across the edges of the lamp's shadows, and into
them, over angles much smaller than the spacing of the grid's directions near the objects that cast them, as it only
comes from the medium lit beside them, at angles from the lamp's direction that grow into the shadow, so the directions
are refined threefold around the edges of these objects, as the lamp's map records them, by whole rows and columns of
directions: light scattered twice is gathered at every point, and so is light scattered more than twice, which the
shadows shape alike, in the lamp's shadows and next to them, while elsewhere it is interpolated from the points of the
even grid through its ratio to the light it is gathered from, the light scattered twice and the light of surfaces
scattered once, which varies slowly there. Most of the light that reaches the medium right behind an object passes right
beside it, across the voxels occupied up to about a voxel around it, so on the refined grid, gathering rays only stop at
a run of occupied voxels that they go through over about a voxel, where they entered it (``halo_shadow_edges``). The
light gathered at each point is also the glow of the medium there, which lights surfaces: its irradiance is tabulated
against the orientation of surfaces, exactly for the light that arrives evenly around the direction of the lamp, as the
light scattered once mostly does, and through spherical harmonics of degree up to 2 for the rest. The tables don't
depend on the camera, so they are computed once per lamp and medium, which takes about 50 s on 4 CPU threads for a lamp
of ``examples/medium``, including 3 s for the voxels, and keeps about 10 MB, or 54 MB with the light that surfaces
reflect (see below), which is computed and kept along with it, in about 15 s more, or 108 MB when that light differs
between channels, e.g. for colored objects, while refining the grid around objects, as around the cube and the walls of the night scene, makes it take 3 to 4 times as long and keep about 5 times as much for the lamp's own light, and take up to 2.9 GB of memory to compute rather than 1.1 GB, and they are integrated along camera rays as light scattered once is. As
halos spread far beyond lamps, each lamp is a single emitter there: area lights aren't split into grids, and the patches
of emissive surfaces are merged into one.

Surfaces are lit through the medium too: their light is attenuated on its way to them, while the medium's glow lights
them. Given the normals of surfaces (``normals`` of :func:`apply_medium <visionsim.medium.render.apply_medium>`, which
``medium.apply`` reads from renders that include them), the radiance of each surface is scaled by the ratio of the
irradiance it receives with and without the medium, see :mod:`visionsim.medium.surfaces`. Sunlight, skylight and light
reflected by the ground are attenuated in closed form, and the glow arrives from every direction: the light of suns
scattered once, with the phase function towards each direction, so that surfaces facing the sun receive much more of it
than those facing away, the light of the sky and the ground scattered once, and, with ``multiple_scattering``, light
scattered more times, which is still brighter towards suns. As the glow only depends on the height of surfaces and on
the orientation of their normal, it is tabulated once per medium and lighting. Given shadow maps, surfaces in the shadow
of a sun lack the glow of its light scattered once by the medium in the shadow, which comes from directions close to the
sun's, looked up in its map: all of it deep in a shadow, and part of it near its edges, or on the sides of objects
turned away from the sun, whose medium the sun mostly lights. Light from lamps is attenuated with a reduced extinction,
``1 - albedo · g`` times it (the similarity relation of radiative transfer), as light scattered forward keeps going
towards surfaces, or, with ``multiple_scattering``, with the full extinction, while the glow of each lamp's halo lights
surfaces from every direction, in the lamp's shadows too, which includes the light that the lamp sends away from
surfaces and that the medium scatters back towards them. The glow fades out over the outermost cell of the halo's grid,
where the similarity relation takes over, which holds beyond the halo's extent. Light that bounces off other surfaces
first makes up an ambient share of the frame's direct light, or, given the albedo of the ground and of objects, at least
the rest of the light that the radiance of surfaces implies they received, and is assumed to be dimmed as the frame's
direct light is on average. Next to brightly lit surfaces, such as in the shadow of a lamp next to the ground it lights,
this light can be most of the light of surfaces, by which the glow is weighed, so that they reflect as much of the glow
as their albedo does. The light that each surface received can also be measured for each pixel, rather than
estimated from the mean albedo of the ground and of objects, which takes the dark squares of a checkerboard for less lit
than its light ones: by its irradiance (``irradiance`` of :func:`apply_medium <visionsim.medium.render.apply_medium>`,
π times the sum of Blender's diffuse direct and indirect light passes, which ``medium.apply`` reads from
``diffuse/direct`` and ``diffuse/indirect`` in renders made with ``--config.include-diffuse-pass``), or by its albedo
(``albedo``, Blender's diffuse color pass, which ``medium.apply`` reads from ``diffuse/color`` when the light passes
are missing) and its radiance. The radiance of glossy surfaces also holds their specular reflection, which their diffuse
albedo leaves out, so that the albedo makes them seem more lit than they were: in ``examples/medium``, whose surfaces
are Principled BSDFs, the ground seems 20% more lit by day (30% on its dark squares, 10% on its light ones) and objects
12%, and at night the ground in the lamp's shadow, a third of whose light is specular, 65%. The rest of the light
measured beyond the direct light of a surface is all the light that reaches it indirectly (``measured_indirect``),
which replaces the ambient share, a guess of this light, even where it is less, such as on the ground in the shadow of a
lamp, far from the surfaces it lights. Each surface then reflects the glow with its own albedo, which follows textures
and colors. Where the light measured is zero, as for metals, glass and surfaces that only emit light, the ambient share
is kept. As the ratio by which the medium then scales the light of a surface grows without bound where it received
little light, while the light of the glow it reflects doesn't, this ratio isn't upsampled from a lower resolution, but
formed at every pixel from the irradiances of surfaces with and without the medium, which are upsampled, guided by the
light received, and from the light that the pixel's own surface received, see :mod:`visionsim.medium.upsampling`. The
light that surfaces emit, given by an emission pass (``--include-emission``), isn't dimmed at all.

Objects block sunlight from the medium behind them, which casts light shafts, and hide part of the sky from the medium
around them, such as the fog in front of a wall. Both are modeled with shadow maps, i.e. orthographic depth maps of the
scene rendered by Cycles along the direction of each sun, and along the central direction of each of the cells into
which the sky is split. Visibility is sampled along each ray where objects can cast shadows, and the closed-form
integral of each source between consecutive samples is weighted by the visibility in between, so the result stays
deterministic, and exact wherever nothing is occluded. The shadows of suns are sharp: rays are split into intervals of a
few texels of the sun's map, intervals that lie entirely above or below the surfaces recorded around them are lit or
hidden as a whole, and the others, which the edge of a shadow may cross, are sampled every texel. Skylight and light
scattered more than once are sampled more coarsely, as each point is lit by many cells, each weighted by how much light
it scatters towards the camera. As rays close to each other sample places close to each other, these places are snapped
to a grid finer than the texels of the sky's maps, whose cells are each looked up once. A hidden cell brings none of the
light scattered more than once, as the medium in front of the object that hides it is in the object's shadows too. Light
scattered at least once arrives brighter from the cells towards suns, as the fog scatters sunlight forward and keeps it
going about the way it went: in the day scene of ``examples/medium``, the cells of the lowest band towards the sun bring
2.5 times the mean of their band, and 0.6 times away from it, and those close to the sun's elevation up to 5.5 and 0.4
times. Cells are weighted by the radiance arriving from them, the mean of their band times their profile, from the
harmonics of this light's variation with the azimuth from each sun averaged over each cell, see :func:`cell_profile
<visionsim.medium.occlusion.cell_profile>`, so that objects that hide cells towards a sun, e.g. from the medium in their
shadow, hide more of it than those that hide cells away from it. Light that reaches the medium from below the horizon,
reflected by the ground or scattered by the medium beneath, is looked up in the maps of the cells half a turn around,
which look down along the mirrored directions: it is hidden by the objects these maps record under each point, and, for
points under the surfaces these maps record, by objects standing on the ground as the mirrored cells above the horizon
are, unless they stand further than where this light starts on the ground. Where it starts on the ground in the shadow
of a sun, the ground lacks the sun's light, directly and through the glow of its light scattered once, and where it
starts on the ground next to objects, which hide part of the sky from the ground, the ground lacks that part of the rest
of its light, from the sky and the glow of the medium: the share of the sky that the ground sees, each cell weighted by
its solid angle times the mean cosine of its elevations, as for surfaces facing up, is found once per scene from the
maps of the sky, on a grid over the ground up to three times the height of objects around them (see :func:`ground_sky
<visionsim.medium.occlusion.ground_sky>`). The light of the ground that the medium scatters on its way up lacks them
too. Objects reflect the light of suns and the sky into the medium too, which scatters it towards the camera: each cell
that an object hides from a point brings the light of the object's surface where the cell's direction enters it, found
in the map of a cell that sees the point from the other side, lit as surfaces are (see above), attenuated on its way,
with ``multiple_scattering`` by the extinction reduced by the similarity relation, ``1 - albedo · g`` times it, as much
of the light that the fog takes out of it on its way is scattered forward and still reaches the point, as for the light
of lamps on surfaces. This light depends on the medium, and is gathered for each medium on a grid four times coarser
than that of the sky's visibility, which takes a quarter of the time of a frame by day on the CPU, and a third at a
quarter of the resolution; ``apply_medium(..., effects=Effects(reflected_daylight=False))`` leaves it out (see below).
See :mod:`visionsim.medium.occlusion` for details.

Surfaces lit by lamps reflect part of their light into the medium too: at night, the ground under a street lamp glows
through the fog around it. Given the albedo of the ground or of objects, surfaces are taken as a Lambertian base of that
albedo under a specular layer, as Blender's Principled BSDF (``specular_reflections``): a dielectric coating of GGX
microfacets, of the lighting's ``surface_roughness`` and ``surface_ior`` (0.5 and 1.5 by default, the Principled
BSDF's), which reflects the light of lamps around the mirror direction, and over which the base is dimmed by the light
that the coating reflects towards the viewer, while ``surface_ior`` 1 leaves the coating out. Coatings smoother than a
roughness of 0.5 are taken as that rough, as the quadratures below miss more of the light of their narrower lobes than
these lobes change it (see the validation below). The coating reflects the light that surfaces receive from other
surfaces and from the medium's glow as if it arrived evenly from every direction, and the light of area lights and
emissive surfaces as that of a point at the position they cast shadows from. They are lit by each lamp as surfaces are
lit through the medium (see above): the ground is the plane at ``ground_height``, in the lamp's shadows, and objects are
the surfaces that the lamp's map records, i.e. those it lights, grouped into patches. Their light reaches each point of
the medium from many directions, and is integrated by two quadratures combined by multiple importance sampling: one over
directions, whose nodes follow the phase function towards the camera, which integrates the ground around points close to
it, including the ground ahead of camera rays, whose light fog scatters forward, and one over the ground, whose nodes
follow its radiance, which integrates the bright ground under a lamp from afar. Objects hide the light of the ground
from the points behind them, as seen from anchors on the ground around the lamp, whose maps reproject the surfaces that
the scene's shadow maps record, those of the sky seeing every side of objects standing on the ground, while the ground
and surfaces lying on it, such as roads, don't. All of it but the phase function towards the camera is gathered once per
lamp and medium, on a grid of points within 40 m of the lamp's foot. For every frame, the light scattered towards the
camera is computed on this grid, and on a grid along the rays of the frame, which takes over where the direction towards
the camera changes too quickly across the cells of the first one, i.e. closer to the camera than to the lamp, and both
are interpolated along each ray, see :mod:`visionsim.medium.reflections`. Objects and the ground also light each other
(``reflection_bounces``), as Lambertian surfaces that don't hide each other's light, until little of it is left
(``interreflections``), and given the shadow maps of suns or of the sky, which see every side of objects standing on the
ground, every surface within 40 m of the lamp's foot receives the light of the ground and of the objects that the lamp
lights, also the sides of objects that the lamp doesn't light, such as those that face the camera behind them, and
reflects it (``bounce_surfaces``), rather than only the surfaces that the lamp lights, along with the surfaces that the
lamp lights but that these maps don't record, such as the underside of a roof. With ``multiple_scattering``, surfaces
are lit with the full extinction and by the glow of the lamp's halo too, and the light they reflect is scattered more
than once along with the lamp's light in its halo: the light of surfaces lit directly, by the light of the lamp
scattered once, which halos gather first, and by the glow of each later order but the last, which is gathered along with
the next order (``relit_scattering``).

|

Switching terms off
-------------------

Each optional term of the model can be turned off, to save time or to see what it does. :class:`Effects
<visionsim.medium.model.Effects>` holds one switch per term, all on by default, which :func:`apply_medium
<visionsim.medium.render.apply_medium>`, :func:`apply_medium_upsampled
<visionsim.medium.upsampling.apply_medium_upsampled>` and :func:`trace_shadows <visionsim.medium.render.trace_shadows>`
take as ``effects``, and ``medium.apply`` as a JSON file (``--effects``), e.g. ``{"halos": false}``. A term that is off
isn't computed at all. Switches only choose which terms of the model are computed, while the scene is described by the
medium, the lighting and the inputs of each frame, such as albedos, ``multiple_scattering``, normals and shadow maps.

Each switch declares, in its JSON schema, its group (daylight, lamps or surfaces) and its prerequisites: the switches
whose terms it needs, e.g. the edges of the sun's shadows on the glow that lights surfaces need these shadows, and
conditions on the inputs, among a fixed vocabulary (:data:`EFFECT_INPUTS <visionsim.medium.model.EFFECT_INPUTS>`). Some
are on what is given with frames (:data:`EFFECT_GIVEN <visionsim.medium.model.EFFECT_GIVEN>`): shadows (``shadows``),
given as shadow maps or as shadows traced from them beforehand, which serve the shadows of suns and the sky along camera
rays; the shadow maps themselves (``occlusion``), which the other terms of objects need, such as the light that objects
reflect, the shadows of lamps and those on surfaces; normals (``normals``); and the albedo (``albedo``) or the
irradiance (``irradiance``) of each pixel, either of which measures the light that its surface received (``measured``).
The others are on the lighting: a sun above the horizon (``suns``), a sun or the sky (``daylight``), lamps or emissive
surfaces (``lamps``), either suns or lamps (``direct_light``), and a ground or objects that reflect light into the
medium (``ground_albedo``, ``object_albedo``, ``reflectors`` for the surfaces that reflect the light of lamps: the
ground, or objects, which only the shadow maps of lamps record, and ``received`` for the light that surfaces received,
measured, or estimated from the mean albedo of the ground and of objects), and surfaces whose specular layer isn't left
out (``specular``: their index of refraction isn't 1); or on the medium, which scatters light (``scattering``),
attenuates sunlight (``sun_attenuation``) or scatters light more than once (``multiple_scattering``). A switch is
effective if it is on and its prerequisites are met, recursively for the switches it requires, which
:meth:`Effects.effective <visionsim.medium.model.Effects.effective>` tells and :meth:`Effects.missing
<visionsim.medium.model.Effects.missing>` explains, e.g. to grey out the switches that would do nothing in an interface,
and a switch that isn't effective is treated exactly as a switch that is off.

Prerequisites are checked on the medium, the lighting and what is given with frames, not on the data of each frame, so
that an effective switch can still change nothing where its term vanishes: ``halos`` where the medium around lamps is
too thin to be worth them, the terms of shadow maps when the maps given lack those of a sun or a lamp,
``received_irradiance`` and ``measured_indirect`` where the medium dims the light of every surface alike, e.g. without
any light, or where the light measured for each pixel is zero, ``received_irradiance`` also, given the mean albedo of
the ground and objects only, where surfaces received no more light than the share of the frame's direct light that it
replaces, and ``measured_indirect`` where every surface received at least that share beyond its direct light. Three
combinations of the lighting aren't told apart either: ``sky_occlusion`` changes nothing when the medium only receives
the direct light of suns, under a black sky, above a black ground and black objects, without multiple scattering,
``ground_sky_occlusion`` when the ground only receives the direct light of suns, under a black sky, without multiple
scattering, and ``surface_sky_occlusion`` when surfaces only receive the direct light of suns, under a black sky, above a
black ground, through a medium that scatters nothing. On the small scene of ``tests/test_effects.py``, over 1920
combinations of media (one of which scatters nothing), suns, sky, lamps and albedos, and of what is given with frames
(nothing, shadows traced beforehand or shadow maps, with or without normals, and nothing, the albedo or the irradiance
of each pixel), turning off a switch that isn't effective changes nothing, bitwise, and every effective switch changes
the frame but in these cases.

The table below gives each switch, its prerequisites, and what turning it off alone does in the scenes of
``examples/medium`` (frames 255 and 345 at 320x180, with the inputs and references of the validation below: by day,
Cycles with 16 volume bounces; at night, the lamp behind the cube, Cycles with 65536 samples, both given the roughness
of their materials, 0.8): the time it saves per frame, once what is kept from one frame to the next is computed, on 4
CPU threads at every pixel and at a quarter of the resolution (``factor`` 4), then likewise on an NVIDIA RTX 5080 GPU,
both in single precision, and the error it adds at every pixel, in percentage points of the relative L1 error of whole
frames, of the fog's light and of surfaces (``ablation.py`` of ``examples/medium/experiment``). With every switch on, a
frame takes 81 ms by day and 126 ms at night on the GPU (69 ms and 57 ms at a quarter of the resolution), with errors of
3.71%, 5.79% and 2.77% by day (3.75%, 5.80% and 2.86%), and 4.15%, 7.79% and 4.61% at night (4.38%, 8.02% and 4.85%); on
4 CPU threads, about 2 s per frame at every pixel and 0.2 s to 0.4 s at a quarter of the resolution. The GPU's times and
the errors are those of every switch of this table, measured together, timing variants back to back in both orders, so
that differences below about 2 ms are noise. The CPU's times were measured as each switch was added, on a CPU shared
with other jobs, where differences below about 50 ms at every pixel and 15 ms at a quarter of the resolution are noise:
those of ``azimuthal_occlusion``, ``ground_sky_occlusion`` and ``scattered_daylight_reflections`` (marked (*)) are those
they add to tracing shadows, timed alone in the same process, alternating with and without them, which vary by about a
factor of two from run to run (50 to 130 ms and 110 to 200 ms at every pixel), while the reduced extinction of
``scattered_daylight_reflections`` costs nothing. Switches whose prerequisites a scene doesn't meet, such as those of
lamps by day, are ineffective there.

.. list-table::
    :header-rows: 1
    :widths: 20 7 24 13 15 13 15

    * - Switch
      - Group
      - Requires
      - Day: ms saved
      - Day: error added
      - Night: ms saved
      - Night: error added
    * - ``sun_shadows``
      - daylight
      - ``shadows``, ``sun_attenuation``, ``suns``, ``scattering``
      - 350 / 26; 7.1 / 5.9
      - +0.66 / +1.56 / +0.00
      - ineffective
      -
    * - ``sky_occlusion``
      - daylight
      - ``shadows``, ``sun_attenuation``, ``daylight``, ``scattering``
      - 1654 / 322; 52.2 / 39.8
      - +4.57 / +10.07 / -0.00
      - ineffective
      -
    * - ``scattering_occlusion``
      - daylight
      - ``sky_occlusion``, ``multiple_scattering``
      - 150 / 15; 3.7 / 5.2
      - +3.27 / +7.58 / +0.00
      - ineffective
      -
    * - ``azimuthal_occlusion``
      - daylight
      - ``scattering_occlusion``, ``suns``
      - 90 / 25; 1.1 / -1.1 (*)
      - +0.06 / +0.18 / +0.00
      - ineffective
      -
    * - ``ground_shadows``
      - daylight
      - ``sky_occlusion``, ``shadows``, ``suns``, ``ground_albedo``
      - 201 / 30; 4.7 / 0.1
      - -0.10 / +0.04 / +0.00
      - ineffective
      -
    * - ``ground_sky_occlusion``
      - daylight
      - ``sky_occlusion``, ``shadows``, ``ground_albedo``
      - 150 / 40; 3.0 / 2.7 (*)
      - -0.23 / -0.40 / +0.00
      - ineffective
      -
    * - ``reflected_daylight``
      - daylight
      - ``sky_occlusion``, ``occlusion``, ``object_albedo``
      - 538 / 137; 16.7 / 15.9
      - +2.38 / +4.19 / -0.00
      - ineffective
      -
    * - ``scattered_daylight_reflections``
      - daylight
      - ``reflected_daylight``, ``multiple_scattering``
      - 0 / 0; -0.7 / 0.7 (*)
      - +0.42 / +0.71 / +0.00
      - ineffective
      -
    * - ``lamp_shadows``
      - lamps
      - ``occlusion``, ``lamps``, ``scattering``
      - ineffective
      -
      - 573 / 32; 35.0 / 5.0
      - +183.28 / +357.35 / +0.52
    * - ``halos``
      - lamps
      - ``lamps``, ``multiple_scattering``, ``scattering``
      - ineffective
      -
      - 1138 / 68; 68.3 / 13.8
      - +15.62 / +24.57 / +2.42
    * - ``halo_occlusion``
      - lamps
      - ``halos``, ``occlusion``
      - ineffective
      -
      - -0 / 3; 1.2 / -0.5
      - +14.46 / +27.23 / +0.54
    * - ``halo_shadow_edges``
      - lamps
      - ``halos``, ``lamp_shadows``
      - ineffective
      -
      - 30 / 8; 0.7 / 0.5
      - +0.15 / +0.29 / +0.01
    * - ``lamp_reflections``
      - lamps
      - ``lamps``, ``reflectors``, ``scattering``
      - ineffective
      -
      - 226 / 125; 29.4 / 28.2
      - +23.37 / +43.73 / -0.63
    * - ``reflection_shadows``
      - lamps
      - ``lamp_reflections``, ``occlusion``, ``ground_albedo``
      - ineffective
      -
      - -5 / 7; -2.1 / 0.8
      - +2.03 / +4.39 / +0.18
    * - ``reflection_occlusion``
      - lamps
      - ``lamp_reflections``, ``occlusion``, ``ground_albedo``
      - ineffective
      -
      - 28 / 25; 5.6 / -0.3
      - +6.90 / +13.09 / +0.06
    * - ``reflection_bounces``
      - lamps
      - ``lamp_reflections``, ``occlusion``, ``ground_albedo``, ``object_albedo``
      - ineffective
      -
      - -16 / -1; -2.6 / -1.2
      - +0.83 / +2.18 / -0.17
    * - ``bounce_surfaces``
      - lamps
      - ``reflection_bounces``
      - ineffective
      -
      - 46 / 18; 0.1 / 0.0
      - +0.23 / +0.77 / -0.03
    * - ``interreflections``
      - lamps
      - ``reflection_bounces``
      - ineffective
      -
      - 40 / 2; -2.0 / 0.1
      - +0.06 / +0.18 / -0.02
    * - ``scattered_reflections``
      - lamps
      - ``halos``, ``lamp_reflections``
      - ineffective
      -
      - -5 / 2; 0.9 / -1.9
      - +5.52 / +8.68 / -0.63
    * - ``relit_reflections``
      - lamps
      - ``halos``, ``lamp_reflections``
      - ineffective
      -
      - -14 / 3; -1.3 / -0.5
      - +0.39 / +1.07 / -0.09
    * - ``specular_reflections``
      - lamps
      - ``lamp_reflections``, ``specular``
      - ineffective
      -
      - 45 / 20; 8.1 / 6.0
      - +0.51 / +1.26 / -0.11
    * - ``relit_scattering``
      - lamps
      - ``scattered_reflections``, ``relit_reflections``
      - ineffective
      -
      - 15 / 1; -2.7 / 0.6
      - +0.11 / +0.26 / -0.10
    * - ``surface_shadows``
      - surfaces
      - ``normals``, ``occlusion``, ``direct_light``
      - 13 / -3; -1.6 / 1.6
      - +0.79 / -0.00 / +1.61
      - -14 / -0; 1.2 / -2.3
      - +0.53 / +0.00 / -0.29
    * - ``surface_sky_occlusion``
      - surfaces
      - ``normals``, ``occlusion``, ``daylight``
      - 136 / 8; 3.9 / 3.5
      - -0.02 / +0.00 / +1.97
      - ineffective
      -
    * - ``daylight_glow``
      - surfaces
      - ``normals``, ``sun_attenuation``, ``daylight``, ``scattering``
      - 92 / 5; 6.9 / 4.7
      - +12.63 / +0.00 / +28.34
      - ineffective
      -
    * - ``glow_shadows``
      - surfaces
      - ``daylight_glow``, ``occlusion``, ``suns``
      - 77 / 4; 3.6 / 6.7
      - -0.12 / +0.00 / +0.07
      - ineffective
      -
    * - ``glow_shadow_edges``
      - surfaces
      - ``glow_shadows``
      - 81 / -0; 2.6 / 7.6
      - +0.30 / +0.00 / +0.65
      - ineffective
      -
    * - ``lamp_glow``
      - surfaces
      - ``halos``, ``normals``
      - ineffective
      -
      - 27 / 6; 2.5 / 3.0
      - +2.60 / +0.00 / +2.42
    * - ``received_irradiance``
      - surfaces
      - ``normals``, ``received``
      - 21 / -12; -1.5 / 2.2
      - +0.02 / +0.00 / -0.01
      - -23 / 0; 1.8 / -1.5
      - +0.64 / +0.00 / +1.74
    * - ``measured_indirect``
      - surfaces
      - ``received_irradiance``, ``normals``, ``measured``
      - ineffective
      -
      - ineffective
      -
    * - ``multiple_scattering`` of the medium
      -
      -
      - 142 / 11; 5.2 / 6.5
      - +26.35 / +39.69 / +8.84
      - 1139 / 70; 67.3 / 13.3
      - +15.62 / +24.57 / +2.42

Times are given on the CPU at every pixel / at a quarter of the resolution, then on the GPU likewise, and errors as
whole frames / fog / surfaces. Turning off ``sky_occlusion`` also turns off the terms that require it,
``scattering_occlusion``, ``azimuthal_occlusion``, ``ground_shadows``, ``ground_sky_occlusion``, ``reflected_daylight``
and ``scattered_daylight_reflections``, whose time the 1.7 s it saves include. The terms of the light that surfaces
reflect and of halos other than the halos themselves (``halo_occlusion``, ``reflection_shadows``,
``reflection_bounces``, ``interreflections``, ``scattered_reflections``, ``relit_reflections`` and ``relit_scattering``)
cost nothing per frame: they are computed with each halo, once per lamp and medium (about 52 s for the night's lamp on 4
CPU threads and 7 s on the GPU, or 66 s and 8 s with the four switches of the light that surfaces reflect added since,
against 3 s and 0.2 s without halos), and turning one of them off computes the halo again. ``halo_shadow_edges`` makes
the night's halo take about 4 times as long to compute on 4 CPU threads shared with other jobs (231 s and 279 s against
61 s and 67 s with the light that surfaces reflect, back to back in both orders), and 2.5 times on the GPU (19.5 s and
20.1 s against 8.2 s and 7.2 s), as its grid is refined by whole rows and columns of directions, 34 by 58 of them rather
than 12 by 24, around the cube and walls 10 to 40 m from the lamp, out of the 36 by 72 that refining every direction
would give, while a single small object would refine about twice as many as the even grid. The halo keeps 52 MB rather
than 9 MB of tables (78 MB rather than 36 MB with the light that surfaces reflect), and computing it takes up to 2.9 GB
of memory on the GPU rather than 1.1 GB. Looking its finer tables up costs 3 to 4 ms more per frame on the GPU (up to 1
ms at a quarter of the resolution, within the noise), and 30 ms (8 ms) on the CPU, within its noise.
``reflection_occlusion``, whose maps of anchors are built with each halo too, in about 1.5 s on the CPU, costs about 25
ms per frame on the CPU and 1 ms on the GPU, to look the points of the medium up in these maps. ``specular_reflections``
and ``bounce_surfaces`` each cost about 20 ms per frame on 4 CPU threads at every resolution, the larger savings at
every pixel in the table being mostly noise, for each lamp whose light surfaces reflect, and about 8 ms together on the
GPU, most of it ``specular_reflections``': the light of the ground through the specular layer over the directions that
follow the phase function towards the camera, and the light of more patches, 588 rather than the 157 of the surfaces
that the night's lamp lights, are looked up at every frame at the points of the grids of reflected light that rays
interpolate, while the light of surfaces towards the points of the grid around the lamp, which doesn't depend on the
camera, is kept with the light gathered there. With these four switches, the light that surfaces reflect, kept with the
halo, takes 44 MB rather than 28 MB for the night's lamp, or 98 MB when it differs between channels, e.g. for colored
objects, and the pieces of every surface, found once per set of shadow maps in the maps of suns and the sky, in about 2
s on 4 CPU threads, take 20 MB. At night, the halo is all the light scattered more than once, so that turning off
``halos`` or ``multiple_scattering`` gives the same frames. Lamps' terms change surfaces too, through the glow of halos,
which holds the light that surfaces reflect: without it, surfaces are 0.43 points closer to Cycles' at night (0.66 with
the four switches added since), as they are 1% too bright with it (1.4%), while whole frames are much further. These
four switches make surfaces 0.23 points further from Cycles' in all, as the light that they add to the fog lights them
too, while whole frames get 1.18 points closer.

Some switches remove error from a part of the frame while adding to another, or by making up for another error. By day,
the fog's light is darker than Cycles' (0.96 times it), mostly as Cycles' Principled ground reflects more light into it
than the Lambertian ground modeled here (see the validation below), so that the terms that darken it bring it further
from Cycles' in total, while they bring it closer to Cycles' with a Lambertian ground. Without ``ground_sky_occlusion``,
whole frames are 0.23 points closer to Cycles' and the fog's light 0.40 points, while against Cycles with a Lambertian
ground and objects of the ground's grey albedo, the fog's light is 0.29 points further, and its per-pixel differences
once blurred are 5.9% where rays see the ground in the shadows of objects, against 3.2%. Without ``ground_shadows``,
whole frames are 0.10 points closer to Cycles', while the fog's light where rays see the ground in the shadows of
objects is much further from Cycles'. Without ``sun_shadows``, the fog's light is closer to Cycles' in total (1.003
times it), while its error per pixel grows by 1.6 points; without ``glow_shadows``, surfaces are 0.07 points further
from Cycles', but whole frames 0.12 points closer, as surfaces in the shadows of objects, brighter, make up for the
fog's light, too dark. ``received_irradiance`` removes no error by day, where objects are given the ground's grey
albedo, and 1.5 points of the error of surfaces at night. Given the irradiance of each pixel, from Blender's diffuse
light passes, it adds 0.02 points to the error of whole frames and 0.04 to that of surfaces by day, and removes 0.57 and
1.61 at night, of which ``measured_indirect``, which these inputs make effective, adds 0.02 and 0.05 by day, and removes
0.19 and 0.20 at night (see the validation below). Given the albedo of each pixel, from Blender's diffuse color pass,
which takes glossy surfaces for more lit than they were, it removes 0.03 and 0.06 by day, and 0.63 and 1.82 at night, of
which ``measured_indirect`` 0.01 and 0.01 by day, and 0.20 and 0.33 at night. Neither the light measured nor
``measured_indirect`` costs measurable time at every pixel: on the GPU, frames take 74 to 76 ms by day and 110 to 114 ms
at night with either input, with or without ``measured_indirect``, against 76 ms and 111 ms without them, and on the CPU
the differences are within the noise of the timings. At a quarter of the resolution, forming the ratio of surfaces at
every pixel from the irradiances upsampled costs about 5 ms per frame at night on the GPU (61 ms, against 56 ms), and
nothing measurable by day on the GPU (70 ms), nor on the CPU. The shadows of lamps on surfaces (``surface_shadows``)
remove none of the error of surfaces at night, which is 0.18 points lower without them, while whole frames are 0.72
points closer to Cycles' with them. Without the shadows of the lamp on the ground that reflects its light
(``reflection_shadows``), whole frames are 0.16 points closer to Cycles' at night, as the fog, 0.90 times as bright as
Cycles' in total, gets brighter (0.96 times), while its error per pixel grows by 0.32 points, and that of surfaces, lit
by the glow of the halo, by 0.14 points. ``halo_shadow_edges`` removes 0.16 points of the error of whole frames at
night, and 0.30 of the fog's, within 0.03 points of gathering light scattered more than twice at every point of its
grid, while surfaces are as close to Cycles' in all: the cube's faces in the lamp's shadow, lit by the glow of the halo,
which follows the lamp's shadows more closely, and the ground in its shadow get closer, while lit walls 10 to 30 m from
the lamp, already too bright, get 1.8% brighter. At a quarter of the resolution, surfaces get 0.13 points further from
Cycles', as the glow on the ground, which fades across the edge of the lamp's shadow over a few pixels, is upsampled
across it, which adds 0.15 points of error to surfaces within 4 pixels of that edge. In the pixels where the cube
shadows the lamp, the fog goes from 0.82 to 0.74 times Cycles', as the lamp's halo, 1.26 times Cycles' there before,
made up for part of the light that surfaces reflect, scattered more than once, which is too dark there (see the
validation below), while it goes from 0.90 to 0.91 times Cycles' elsewhere.

Presets of the switches are given by name in :data:`EFFECT_PRESETS <visionsim.medium.model.EFFECT_PRESETS>`, chosen from
the table above by stated rules, taking the error each term changes in absolute value, as a term whose removal brings
frames closer to Cycles' only by cancelling another error, such as the light of Cycles' Principled ground by day, still
matters. Every term changes whole frames, the fog's light or surfaces by at least 0.1 points in a scene where it is
effective, so ``complete`` keeps them all, and each removes more error per millisecond than computing the medium at
every pixel rather than at a quarter of them does, at night 0.23 points of whole frames for 69 ms per frame on the GPU:
the balanced choice is every term at a quarter of the resolution (:func:`apply_medium_upsampled
<visionsim.medium.upsampling.apply_medium_upsampled>` with ``factor`` 4). ``fast`` only keeps the terms that change
whole frames by at least 0.5 points, or the fog's light or surfaces by at least 1 point, and those they require, which
leaves out ``azimuthal_occlusion``, ``ground_shadows``, ``ground_sky_occlusion``, ``scattered_daylight_reflections``,
``glow_shadows``, ``glow_shadow_edges``, ``halo_shadow_edges``, ``bounce_surfaces``, ``interreflections`` and
``relit_scattering``. On the GPU, in single precision, at every pixel and at a quarter of the resolution, ``complete``
takes 86 ms and 69 ms per frame by day, with whole frames 3.71% and 3.75% from Cycles', and 127 ms and 59 ms at night,
4.15% and 4.38%, and ``fast`` 71 ms and 63 ms by day, 4.07% and 4.11%, and 124 ms and 59 ms at night, 4.72% and 4.84%:
by day, ``fast`` saves a sixth of the time for a tenth more error, while at night it saves little per frame, but its
halo takes 8 s to compute on the GPU rather than 31 s, and keeps 36 MB rather than 94 MB, which matters for lamps that
move, whose halos are computed again at every frame (``presets.py`` of ``examples/medium/experiment``).

|

Validation
----------

The ``examples/medium`` directory renders a scene through visionsim, and the same scene with an equivalent fog
volume using Cycles' volumetric path tracing, then compares them. When restricted to what is modeled here
(single scattering, no shadows cast onto the fog), the in-scattered radiance agrees with Cycles' volume direct
pass with a median relative error of 0.26%, and the radiance of objects and of the sky seen through the fog is
within 0.3%. The remaining differences are Monte Carlo noise, and object edges.

In the same scene without objects, the light scattered by the fog towards the camera, including multiple scattering and
light reflected by the ground, is within 3% of Cycles with 32 volume bounces overall, with per-pixel differences of 2.8%
once blurred, and within 5% at any elevation and azimuth from the sun. The light other than single scattering is 4% too
dark, as Cycles' ground, a Principled BSDF whose specular layer reflects more at grazing angles, is brighter than the
Lambertian ground of the same albedo modeled here: with the ground of the same render replaced by a Lambertian one (a
Diffuse BSDF of the same color), Cycles' fog is 3.3% darker, and up to 8% looking down on the side of the sun, and the
closed form is within 0.7% of it overall, with per-pixel differences of 1.1% once blurred, and within 2.4% at any
elevation and azimuth from the sun, while the light other than single scattering is 1.2% too bright. With a black
ground, the fog's own light is within 0.6% of Cycles', the light it scatters twice within 0.7%, three times within 1.5%,
and more within 4%, though 5% to 8% too bright 1.5° to 25° above the horizon. With the camera turned towards the sun,
whose rays come within 13° of it, the fog's light is within 0.6% of Cycles' with a Lambertian ground, and within 1.6% at
any elevation and azimuth from the sun, so that light scattered more than once still goes forward as it should. Light
scattered by lamps is within about 1% of Cycles' single scattering overall: 0.6% for a point light, 0.1% for a spot
light, 1.2% for an area light, and 0.9% for a point light whose light is smoothed by a Light Falloff node, with
per-pixel differences of 1% to 2% once Cycles' noise is blurred out, the largest right next to the lamps. With a point
light right behind a cube, which hides it from the camera, objects casting its shadows bring the light scattered by the
fog within 0.3% of Cycles', with per-pixel differences of 0.9% once blurred, against eleven times too much light without
shadows. The light of meshes that emit light is within 1% of Cycles' too: 0.2% for a panel of 2 by 1 m whose two sides
emit, where the area light is, 0.9% for a ball of 15 cm radius and 0.2% for a tilted box of 40 cm, with per-pixel
differences of 1.3%, 2.7% and 1.6% once blurred, leaving out the few pixels whose rays go through the closed meshes,
which the camera of these renders doesn't see. With the ball right behind the cube, objects casting its shadows bring
its light within 0.5% of Cycles', with per-pixel differences of 1.4% once blurred, against eleven times too much light
without shadows. With multiple scattering, all the light of a point lamp scattered by the fog, in the scene without
objects and with a black ground, is within 0.9% of Cycles' with 32 volume bounces, and within 5% at any distance from
the lamp, with per-pixel differences of 2.3% once blurred, against 15% too dark with single scattering only. With the
point light behind the cube, whose shadow holds the camera, the light of the lamp scattered more than once is 0.96 times
the light that Cycles' fog scatters more than once (in a render whose surfaces are black, with 16 volume bounces and
65536 samples), with per-pixel differences of 4.7% once blurred, and as close where the cube shadows the whole ray,
which holds 9% of this light (0.96 times, with per-pixel differences of 6.3% once blurred), and on either side of the
edge of the shadow in the image (0.97 and 0.94 times), against 2.7 times too much light, and 15 times as much in the
shadow, when objects don't stop the light gathered by the halo. The light scattered twice alone is 0.97 times Cycles'
(with one volume bounce) in the shadow and elsewhere. Gathering light scattered more than twice at every point of the
refined grid changes none of these by more than 0.001. On the even grid of halos, without refining it around the edges
of the cube (``halo_shadow_edges``), the light scattered more than once is 1.26 times Cycles' in the shadow and 0.92
times elsewhere, with per-pixel differences of 10% once blurred, and the light scattered twice 1.34 times and 0.93
times. On the refined grid, gathering rays that stop at the first voxel occupied around the cube make the light
scattered more than once 0.90 times Cycles' in the shadow and 0.94 times elsewhere, while on the even grid, rays that go
through these voxels over less than a voxel make it 1.35 times Cycles' in the shadow: there, the voxels partly make up
for the coarse grid. Note that Cycles clamps indirect light by default (Clamp Indirect of 10), which darkens light
scattered more than once near bright lamps, by 30% for this lamp: references are rendered without clamping
(``render_passes.py --no-clamp``).

Surfaces lit through the fog (``validate_surfaces.py``) are within 1% of Cycles' overall, both with single and multiple
scattering, against 13% too bright in the render without fog for single scattering, with per-pixel differences of 3.5%
and 4.6%, against 14% and 5%. Walls, which only see the glow of the fog around them, are within about 3%, against 26% too
bright. Surfaces lit by a point, spot or area light are within 1.3%, against 4% to 6% too bright, with per-pixel
differences of 4% to 6%, against 7% to 9%, and surfaces lit by the emissive panel or ball within 0.6%, against 5% and 4%
too bright, with per-pixel differences of 3.2% and 5.8%, against 5.8% and 7.4%. Note that Cycles' renders with no volume bounces include light scattered once
on its way to surfaces too, as the light is sampled from where it scatters, and that references lit by lamps are rendered
without clamping indirect light, see below.

With objects casting shadows, the sun's light scattered by the fog is within 1.3% of Cycles' over surfaces and within
0.8% over the sky, against 17% and 3% without shadows, and the sky's is within 1% over both, against 25% and 3%.

By day, in the scene with objects, with multiple scattering and a ground of albedo 0.39, at frames 255 and 345, objects
dim the fog's light, by hiding the sky and the ground, and by casting the sun's shadows onto the fog and the ground, and
brighten it by reflecting light into it, about 8% of the fog's light in Cycles. Before the terms of the next paragraph,
which change the following figures as described there, against Cycles with black objects (16 volume bounces, without
clamping), the closed form's fog without the light that objects reflect is within 0.2% overall, with per-pixel
differences of 3.6% once blurred, against 4.7% too bright and 6.1% when the sun's shadows didn't reach the ground and
objects under points didn't hide it, and 7% too bright where the ground is in the shadows of objects, against 35% for
light other than single scattering. In this scene, light scattered twice, integrated by brute force with every shadow,
is dimmed by objects as Cycles dims it to within 2% where rays see the sky or the lit ground, though 10% less where they
see the shadowed ground, and as hidden cells dim it, while counting the glow of the medium in front of objects as in the
open would make it 10% to 18% brighter. The light that objects reflect and the fog scatters once is 0.90 times Cycles'
(with one volume bounce), and about half of all the light that Cycles' objects reflect into the fog, and 0.90 times the
light that the same surfaces reflect through the exact geometry of the scene. As the scene's objects are colored, while
the closed form gives them the ground's grey albedo, it is 2% brighter than Cycles' in red, and 14% darker in green and
blue. Against Cycles with its objects, whole frames differ by 3.6% per pixel once blurred, against 4.8%, the fog's light
by 5.6%, against 7.7%, and surfaces by 1.9%, against 2.0%: the ground in the shadows of objects is within 1% of Cycles',
against 17% too bright, while it would be 12% too bright with all of the glow of the sun's light scattered once, and 9%
too dark without any of it. Estimating the light that surfaces received from their radiance brings the ground closer to
Cycles' (per-pixel differences of 2.7%, against 2.8% without it), but objects further from it (3.9%, against 3.4%), as
they are given the ground's grey albedo. Measured for each pixel by its irradiance, from Blender's diffuse light passes,
the light that surfaces received brings objects within 3.2% of Cycles' per pixel (0.991 times its light, against 0.976),
but the ground in the shadows of objects is then 6.7% too bright, against 0.5%, and the lit ground 1.4% too dark,
against 0.6%: surfaces differ by 2.79% per pixel (1.93% once blurred), against 2.77% (1.95%), and whole frames by 3.96%,
against 3.92%. With the grey albedo, the light squares of the shadowed ground seemed more lit than they were, which hid
the excess of their glow, now 7.2%, and 5.7% on the dark squares. Without ``measured_indirect``, i.e. with the share of
the frame's direct light kept as a lower bound of the light that reaches surfaces indirectly, the shadowed ground is
3.1% too bright and surfaces differ by 2.74%. Measured by the albedo of each pixel instead, from Blender's diffuse color
pass, surfaces differ by 2.69% and whole frames by 3.91%, but only as the specular reflection that the ground's radiance
holds, 13% of it, makes it seem 20% more lit than it was, which weighs its glow down: on the lit ground, about as much
as the share of the frame's direct light that the light measured replaces, and on the shadowed ground, by part of the
excess of its glow (then 4.1% too bright). These are errors that cancel, while objects, 8% of whose light is specular,
are within 3.2% of Cycles' either way. Objects given the mean albedo of the pixels that see them, weighted by the light
they received, (0.28, 0.55, 0.46), rather than the ground's grey, reflect greener and bluer light into the fog, which is
then 0.961, 0.977 and 0.969 times Cycles' in red, green and blue, against 0.974, 0.959 and 0.961, with per-pixel
differences of 6.11%, against 6.22%, and whole frames differ by 3.78% (3.86% with the irradiance of each pixel). The
ground's own mean diffuse albedo, 0.366, rather than the 0.39 of its checker's base colors, leaves out its specular
reflection, which reflects light into the fog too, which is then further from Cycles' (7.16% per pixel, with the
objects' albedo estimated as well). Without the light that objects reflect, whole frames would differ by 5.4%, and
without the sun's shadows on the ground's light, by 3.5%, as the fog is too dark overall, but by 12% where rays see the
shadowed ground, against 5%.

The Principled ground of that scene makes up for some of the closed form's errors, as it reflects more light into the
fog than the Lambertian ground modeled here (see the scene without objects above). Renders of the scene whose ground's
Principled BSDF is replaced by a Diffuse BSDF of the same color, with black objects or objects of the ground's grey
albedo, also lit by the sun or the sky alone (16 volume bounces, without clamping), tell these errors apart. Against
them, three terms of the light that the fog scatters more than once near objects remove error. The ground next to
objects, which sees less of the sky, lacks part of the light of the sky and of the glow of the fog
(``ground_sky_occlusion``): with black objects, the light of the sky other than single scattering is then within 0.7% of
Cycles' overall, against 5.4% too bright, and within 2.3% where rays pass next to objects, against 12%. Cells of the sky
towards the sun hide more of the light scattered more than once (``azimuthal_occlusion``), and the light that objects
reflect keeps the light that the fog scatters forward on its way (``scattered_daylight_reflections``), which brings it
from 0.70 to 0.87 times Cycles', with per-pixel differences of 18% of this light once blurred, against 31%. With them,
the fog with grey objects is within 0.5% of Cycles' overall, with per-pixel differences of 2.6% once blurred, against
0.8% too bright and 3.4% without them, and 1.8% too bright where rays see the ground in the shadows of objects and 2.8%
where they pass next to objects, against 6.2% and 7.3%; with black objects, it is 1.5% too bright, with per-pixel
differences of 3.1%, against 3.1% and 4.1%. Over Cycles' Principled ground, with black objects, it is 1.4% too dark,
with per-pixel differences of 3.2% once blurred, against 0.2% too bright and 3.6%, and 0.7% too dark where rays see the
ground in the shadows of objects (3.4%), against 5.4% too bright (6.8%). Against Cycles with its Principled ground and
colored objects, the same terms make the fog darker, as that ground's light made up for what they remove: whole frames
differ by 3.7% per pixel (3.4% once blurred) and the fog's light by 5.8% (5.2%), against 3.9% (3.6%) and 6.2% (5.6%)
without them, and 3.5% and 5.4% without the ground's share of the sky alone. Of the azimuths from the sun, those up to
20° only hold a pixel of these frames, and the fog of rays 20° to 40° from it is within 1.1% of Cycles' (3.2% too bright
against the Lambertian ground).

At night, with the point light right behind the cube and a ground of albedo 0.39, Cycles' light that surfaces reflect
and the fog scatters once (the difference between renders with one volume bounce and no surface bounces, with surfaces
of their own albedo, and with black surfaces) is 0.82 times the light of the lamp that the fog scatters once, and 53% of
all the light that went through surfaces in renders with 16 volume bounces, most of the rest of which the fog scatters
more than once. With surfaces lit with the full extinction, as in these renders, and Lambertian, as with the specular
layer of their materials off, the closed form's is 6% darker than Cycles' overall, 7% darker in pixels whose rays pass
within 3 m of the lamp's foot and 12% darker in the cube's shadow, with per-pixel differences of 17% once blurred,
mostly Cycles' noise, but for the light that objects and the ground reflect onto each other, which these renders leave
out, and which adds 6%. As surfaces are lit with the similarity relation without multiple scattering, which also counts
the light that the fog scatters on its way to them, it is then 63% of all the light that went through surfaces. Against
a direct integration of the same model, the light of the ground is within about 3% along rays of every direction,
including rays that end under the lamp, for point, spot and area lights 2 to 4 m above the ground, within 6% for a
headlight 60 cm above it, and within 12% for a point light 15 cm above it, whose light varies within a few centimeters
of the lamp's foot. With a specular layer, for a point light 2 m above the ground, it is within 5% of a direct
integration from a roughness of 0.5 up, which the quadratures resolve, but 3% to 9% darker at 0.4, 6% to 15% at 0.3 and
up to 17% at 0.2, as they miss more of the narrower lobes of smoother layers, which are therefore taken as of roughness
0.5: within 8% of their own light from a roughness of 0.1 up.

With multiple scattering, against Cycles with 65536 samples per pixel and 16 volume bounces, and given the roughness of
the scene's materials (0.8), the night's whole frames are 2% darker than Cycles', with per-pixel differences of 4.2%,
against 5% darker with per-pixel differences of 5.5% with Lambertian surfaces, of which only those that the lamp lights
receive the light of the ground, once, and whose light lit by the glow beyond the lamp's light scattered once is
scattered once only, and against 15% darker with per-pixel differences of 15% when surfaces are also lit with the
similarity relation and no glow, and reflected light is scattered once only. The fog's light is 5% darker, with
per-pixel differences of 7.8%, against 10% and 11%, and 22% and 22%: 12% darker in the cube's shadow and 5% darker
elsewhere, against 18% and 10%, and 41% and 21% (4% and 6% darker on the even grid of halos, without
``halo_shadow_edges``, where the lamp's own light was too bright in the shadow, see below). Of it, the light that
surfaces reflect is 10% darker than Cycles' (12% in the cube's shadow), against 19% (30%), on the even grid of halos.
With the halo's grid refined around the edges of the lamp's shadows (``halo_shadow_edges``), the lamp's own light
scattered more than once is as close to Cycles' in the cube's shadow as elsewhere, while on the even grid it was 1.26
times too bright there, which made up for part of the light that surfaces reflect, scattered more than once, that is
missing there. Renders whose surfaces are all Lambertian, of the albedo of 0.39 that the closed form is given, with as
many bounces off surfaces as the scene's (4), split this light by the surfaces it went through: the closed form's is 6%
darker than Cycles' overall, against 11%. The light of surfaces lit directly, which the fog scatters once, 55% of it, is
4% darker (10% darker in the cube's shadow), which is the resolution of the grid of reflected light: twice as many
directions and azimuths of the quadrature over directions, distances and azimuths of its nodes over the ground and of
the ground's texture, rings of anchors and anchors per ring, and rows and columns of their maps, bring the light of the
ground lit directly and scattered once within 1% of Cycles' (1.1% in the shadow), while maps of anchors alone, four
times as fine, only bring it from 11% to 8.5% darker in the shadow; the light of these surfaces scattered more than
once, 16% of it, is 12% darker; the light reflected by two surfaces or more, 7% of it, is 6% darker, against 43%; and
the light of surfaces lit by the glow, 22% of it, 7% darker, against 16%. With light bouncing between surfaces once only
(``interreflections`` off), so that it went through two surfaces exactly, as in renders that stop there, the light of
two surfaces that the fog scatters once, 4.5% of the light that surfaces reflect, is within 1% of Cycles' (0.992), but
that which it scatters more than once, 1.8%, is 36% darker (0.637), while the light of three surfaces or more, with
``interreflections``, 0.7%, is 31% brighter (1.31), neither of which is explained. The scene's Principled BSDFs reflect
5.8% more of this light than Lambertian surfaces of the same albedo, and 5.0% more of the light of surfaces lit directly
scattered once, which the specular layer brings to 5.6% and 4.7%. The scene's colored objects, which the closed form
gives the ground's grey albedo, make most of the rest: the fog's light is 4.7% brighter with the scene's materials than
in the Lambertian renders, 2.8 points of which come from the specular layer of its materials.

The glow lights the ground the camera sees within 1% of gathering the light of the lamp scattered once and more, and of
the light of surfaces scattered once and more, along rays from each point of the ground, and surfaces are 1.4% brighter
than Cycles', with per-pixel differences of 4.6%, against 1% and 4.4% without the four switches of the light that
surfaces reflect above, whose light lights them too, and against 7% darker with per-pixel differences of 7% with the
similarity relation alone: the lit ground, which holds 84% of their light, is within 1%, the cube's faces in the lamp's
shadow, lit by the ground, 9% too bright, against 17% too dark without the glow and up to three times too bright when
the glow is weighed by the ambient share of light alone, and lit walls 10 to 30 m from the lamp, a third of whose light
comes from the ground in front of them, 11% too bright. Measured for each pixel by its irradiance, from Blender's
diffuse light passes (rendered with 4096 samples per pixel for this comparison), before the terms of the light that
surfaces reflect and of the halo's shadow edges above were added, the light that surfaces received brought them within
4.28% of Cycles' per pixel (3.01% once blurred), against 4.39% (3.09%), and whole frames within 5.37%, against 5.49%:
the ground in the lamp's shadow within 30 m of it, for which the ambient share was more light than it received, is 8%
too dark, against 61%, while the cube's faces in the lamp's shadow are 13% too bright, against 7%, an excess of the glow
that the ground lights them with, and lit walls 11%, against 10%. Without ``measured_indirect``, surfaces differ by
4.48%: the gain comes from the light that reaches surfaces indirectly, measured rather than guessed. Measured by the
albedo of each pixel instead, the light of the ground in the lamp's shadow, a third of which is its specular reflection,
seems 65% more than it was, and that of the cube's faces 23%, which weighs their glow down: surfaces then differ by
4.08%, with the ground in the lamp's shadow 33% too dark and the cube's faces 10% too bright, as the excess of the
faces' glow is hidden along with part of the ground's. Objects given the mean albedo of the pixels that see them,
weighted by the light they received, (0.29, 0.46, 0.57), rather than the ground's grey, reflect more light into the fog,
which is then 0.907 times Cycles', against 0.897, and 0.922, 0.880 and 0.920 times in red, green and blue, against
0.939, 0.868 and 0.888, with per-pixel differences of 10.0%, against 10.9%, while whole frames differ by 4.92% (with the
irradiance of each pixel), and surfaces, lit by the glow of the halo that holds this light, by 4.45%. Part of this comes
from adding light to a fog that lacks the light of surfaces reflected more than once, as the red, which these objects
reflect less, moves further from Cycles'. The ground's albedo of 0.39, here and by day, is the mean of its checker's
base colors: its mean diffuse albedo, 0.37, which leaves out the specular layer of its material, makes the fog's light
differ from Cycles' by 11.5% per pixel at night (with the objects' albedo estimated as well), as the light that the
ground reflects is short of Cycles' then, see above. ``examples/medium/experiment`` compares every method as seen by the
sensor emulators.

|

Limitations
-----------

The following are not yet modeled:

- Shadows of moving objects, as shadow maps are rendered at the first frame, except those of moving lamps, which are
  rendered at the frames at which lamps reach new positions, and the soft shadows of large lamps, which cast the hard
  shadows of a single point. Suns that move cast the shadows of the first frame. Halos of moving lamps are computed
  again at every frame, along with the voxels that objects occupy, and are cached up to
  :data:`HALO_CACHE_BYTES <visionsim.medium.halos.HALO_CACHE_BYTES>` (1 GB, about 10 MB per halo, or 36 MB with the
  light that surfaces reflect without the switches that need more of it, and 94 MB with every switch for the night's
  lamp, whose grid is refined around the cube and the walls, so that it holds about 10 of these: the halos of a scene
  with more lamps are all computed again at every frame).
- IES profiles and other node trees of lamps, square spot lights, and the elliptical cones of spot lights scaled
  unevenly. Emission whose strength is set by other nodes, or whose color varies over a surface, as it does with an
  image, which is averaged. Close to curved emissive surfaces, within about the size of their patches, their light is
  that of point emitters.
- Multiple scattering in media other than a single height fog, and skies whose radiance varies with direction.
- Part of the dimming of light scattered more than twice by objects: in the day scene of ``examples/medium``, with a
  Lambertian ground and objects of the same albedo, Cycles dims it to 0.50 times the light without objects, against
  0.52 times here, 0.75 times against 0.78 where rays see the sky, and 0.59 against 0.66 where they see the ground in
  the shadows of objects, see :mod:`visionsim.medium.occlusion`. The sun's shadows on light scattered more than once,
  but for that of the ground and through the cells towards the sun that objects hide: with black objects, the sun's
  light other than single scattering is 7% too bright where rays see the ground in the shadows of objects and 6% where
  they pass next to objects, against 1% for the sky's. The light that objects reflect, scattered sideways on its way,
  which then reaches points from other directions, or that lights other surfaces first, such as the ground next to
  sunlit walls, which the ground's share of the sky counts as dark: with the similarity relation, it is 0.87 times
  Cycles' with objects of the ground's grey albedo. Cells of the sky are hidden or seen as a whole, as their central
  direction is. A hidden cell brings none of the glow of the medium in front of the object that hides it: in the day
  scene, whose sun is 30° high, this medium lies in the shadows of the object and glows little, but in front of
  objects that the sun lights from behind points, it glows, and the fog there is too dark. All objects have the same
  albedo.
- Grounds other than Lambertian, such as a Principled BSDF, whose glossy reflection brightens the ground at grazing
  angles, from which the fog near the horizon receives most of the ground's light: in the day scene of
  ``examples/medium``, without objects, Cycles' fog is 3.3% brighter over a Principled ground (roughness 0.8, specular
  0.5) than over a Lambertian one of the same color, and 8% brighter looking down on the side of the sun, which makes
  up most of the difference between the closed form and Cycles there. With objects, scaling the closed form's fog by
  this ratio, pixel by pixel, would bring whole frames from 3.7% to 3.1% of Cycles' per pixel, and the fog's light
  from 5.8% to 4.9%, and the ground's share of the sky would then bring the fog's light closer to Cycles' (4.9%
  against 5.1% without it) rather than further, with whole frames as close, see the validation above. Only the light
  of lamps is reflected by a specular layer (``specular_reflections``).
- The halos of lamps are stopped by objects, as seen by the visual hull of the shadow maps, which takes places hidden
  from the whole sky and from every lamp and emissive surface, which are dark, for the inside of objects, and leaves out
  emissive meshes, which the maps of emissive surfaces see through. Objects don't stop the halos of lamps without maps,
  as they cast none of their shadows either. The grid of halos is only refined around the edges of the objects that cast
  the shadows of lamps, by whole rows and columns of directions, so that a single small object about doubles them, and
  objects that span many directions, or many objects, refine most of them, which takes up to 9 times as long for the
  light scattered twice, and for the light scattered more than twice where the lamp's shadows fill the grid, which is
  gathered at every point in and next to them, at every frame for lamps that move, while the edges of shadows are still
  steps between points of the grid. Rays that graze the faces of objects, within a voxel of them, are stopped, as the
  voxels around objects are occupied, and so are rays that pass right beside objects on the even grid, which makes up
  for part of the light of the medium lit beside the shadows of lamps that its coarse grid spreads into them.
- The shadows of objects on the glow of suns and the sky that lights surfaces but for the sun's shadows on the glow of
  its light scattered once, which otherwise assumes that surfaces stand in the open, unless shadow maps show that they
  don't see the sky, and the glow of lamps on surfaces beyond what the reduced extinction accounts for, without
  ``multiple_scattering``, or beyond the extent of halos, 40 m: in the night scene, 30 m from the lamp, the ground
  receives 1.2 to 1.5 times as much light as without fog with the glow, against 0.7 times with the similarity
  relation. Surfaces are assumed to be diffuse, and their specular reflection of the glow isn't modeled. Light that
  bounces off other surfaces is assumed to be dimmed as the frame's direct light is on average, while at night it
  crosses meters of fog: lit walls 10 to 30 m from the lamp, a third of whose light comes from the ground in front of
  them, are 10% too bright. Unless the light that each surface received is measured, it is at least an ambient share
  of the frame's direct light, by which the glow is weighed, which can be more than surfaces receive, such as the ground
  in the cube's shadow at night, 1% of the light of surfaces, which is 61% too dark (8% given the irradiance of each
  pixel), and the albedo of the ground and of objects is that of their surfaces on average, which flattens the glow on
  textured surfaces where light from other surfaces dominates, and misjudges the light that surfaces of another color
  received: by day, the colored objects, given the ground's grey albedo, are 1% darker than without this estimate. With
  the light measured, the glow is too bright on surfaces that other surfaces light: the cube's faces in the lamp's
  shadow at night, 13% too bright, and the ground in the shadows of objects by day, 6.7%. Measured by the albedo of each
  pixel, the specular reflection of glossy surfaces, which their radiance holds, makes them seem more lit than they
  were, most so on dark surfaces and in shadows: by 20% for the ground by day, 30% on its dark squares, and 65% for the
  ground in the lamp's shadow at night, 80% on its dark squares, which weighs their glow down and hides part of these
  errors rather than removing them.
- Anti-aliasing: depth maps are not anti-aliased, so edges between near and far objects can show halos in dense
  media. Rendering at a higher resolution and downsampling the results reduces these.
- Surfaces hiding the light of lamps that other surfaces reflect onto them, which is that of their Lambertian base, as
  their specular layer only reflects the light of lamps towards the medium, and the light of surfaces that neither the
  lamp's map nor the shadow maps of suns or of the sky record, such as the bottom of objects, which receive no light
  from other surfaces. Surfaces that no shadow map records hide no reflected light, the light of objects isn't hidden,
  and close to objects, within about their size, their light is that of point emitters. Objects all have the same
  albedo, and the same specular layer as the ground, whose single roughness stands for every material of the scene, and
  which is at least 0.5, as the quadratures of reflected light don't follow narrower lobes: below a roughness of 0.1,
  which the validation doesn't reach, surfaces such as wet roads mirror lamps. The specular layer reflects the light
  that surfaces receive from other surfaces and from the medium's glow as if it arrived evenly from every direction, and
  the light of area lights and emissive surfaces as that of a point, and its microfacets reflect light once, while those
  of the Principled BSDF's default, multiscatter GGX, also reflect the light that they reflect onto each other.
  Reflected light is only gathered within 40 m of the lamp's foot. In the night scene, the light of surfaces lit
  directly that the fog scatters more than once is 12% darker than Cycles', that of surfaces lit by the glow 7% darker,
  the light of two surfaces exactly that the fog scatters more than once 36% darker, and that of three surfaces or more
  31% brighter, which isn't explained, while the light of the ground that the fog scatters once is 4% darker (11% in the
  cube's shadow) at the default resolution of the grid of reflected light, which a grid twice as fine in every
  dimension, with four times as many directions and nodes over the ground to look up at every frame, brings within 1%.
