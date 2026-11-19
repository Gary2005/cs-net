"""
demo_extract — per-frame CS2 demo extraction with real fire and smoke.

    from demo_extract import extract, codec

    data = extract("match.dem")            # every tick (~64 Hz)
    codec.save(data, "match.csz")          # compressed, single file
    data2 = codec.load("match.csz")        # bit-for-bit identical

    codec.equal(data, data2)               # -> (True, '$')

What you get per round
----------------------
* ``ticks`` / ``round_seconds`` — the sampled timeline
* ``players`` — 10 columnar records: position, view angles, velocity, health,
  armor, equipment, weapon, place, flash, shots, footsteps, (v6) ground
  contact plus the discrete jump/landing events, and (v7) the crouch amount
  plus the FL_DUCKING collision-hull state
* ``events`` — kills, damage, bomb, grenades, sound
* ``fire`` — every inferno as change-snapshots of its 16 burning cells
  (3D position + surface normal + spread-tree parent + active flag)
* ``smoke`` — every smoke as an append-only voxel journal plus the per-tick
  ``(size, update)`` curve that slices it

Fire and smoke are the two things demoparser2 cannot reach on any access path;
they come from a small demoinfocs-golang helper.  See :mod:`demo_extract.sidechannel`.

`recipes.py` has the accessors you want on top of the raw dict: stack players
into a ``(T, P, F)`` tensor, rebuild fire geometry at a tick, voxelise smoke.
``recipes.PLAYER_FEATURES_V6`` adds ``on_ground`` / ``jump`` / ``land`` so a
model can work out movement noise from the raw physics instead of a hand-tuned
speed threshold.  ``recipes.PLAYER_FEATURES_V7`` adds ``crouch``; the crouch
*amount* is a continuous ramp while the collision hull is a 72 -> 54 step, so
``crouch_amount`` and ``crouching_mask`` are deliberately two functions -- use
the first for the camera, the second for hitboxes.

See ``README.md`` for the on-disk format and the size numbers.
"""

from . import codec, recipes, sidechannel, voxel
from .extract import FORMAT_VERSION, PLAYER_PROPS, extract

__all__ = [
    "extract",
    "codec",
    "voxel",
    "recipes",
    "sidechannel",
    "FORMAT_VERSION",
    "PLAYER_PROPS",
]
__version__ = "0.1.0"