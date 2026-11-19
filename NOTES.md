# Technical notes

How the AnimGraph2 migration was worked out, what was measured, and the traps
that cost the most time.

The community pack I used as a test base was nozb1's, whose models are rigged on
CS2's own skeleton — every canonical bone sits within 0.73 units of Valve's
agents. That is a large part of why this was solvable with text edits rather
than 3D work, and a pack rigged from scratch may not port as cleanly.

Reference set: Valve's shipped agents (`ctm_fbi`, `ctm_sas`), one custom model
that had been hand-ported to AG2, and 26 unported models from a 2024-era
community pack. Every rule in the scripts comes from a difference between those
groups, not from guesswork.

**Confirmed working in game.** All 26 models from the test pack animate.

Four bugs turned up while auditing, two of which would have silently shipped
broken models — those are section 2, and they're the most useful part of this
document if you're doing something similar.

---

## 1. Bone hierarchy — the thing that actually matters

CS2's canonical AG2 skeleton (`worldmodel.vnmskel`) has 74 bones rooted at
`root_motion`, with `pelvis` as its child.

| | canonical bones present | parent mismatches |
|---|---|---|
| reference model (**animates**) | 64 / 74 | **0** |
| a ported model | 64 / 74 | **0** |

Every shared bone has an identical parent in both. The 10 absent bones
(`wpnHand_L/R`, `wpnTip`, `wpnEnd`, `attachHand_L/R`, `attachFoot_L/R`,
`wpnAimIntent`, `attachWorld`) are absent from the reference model too, so they are optional.

**Across all 26 models after porting:** `root_motion` is the sole skeleton root
in every one, and `wpnPivot`/`wpn` are present in every one. 22 of 26 have zero
parent mismatches. The other four differ only in finger metacarpal parenting — e.g.
`finger_ring_0_L` attached to `hand_L` instead of `finger_ring_meta_L`. That is
a rigging quirk in the source pack's originals, affects finger posing only, and does not
prevent body animation.

`romasha` has an extra custom root called `lean_root`. Harmless — `pelvis` is
correctly re-parented regardless.

## 2. Traps that cost the most time

**(a) one model in the test pack would have shipped broken.** It is the only model in the
pack with lowercase `_l`/`_r` bones. Without a case flip it cannot bind to the
AG2 graph. Now auto-detected and flipped — 52 bones.

**(b) The case flip was renaming physics shapes.** Physics bodies, hitboxes and
attachments carry their own `name =` fields with their own conventions — a test model
has `leg_upper_l` as a *physics shape* while its skeleton bone is
`leg_upper_L`. Renaming those breaks physics binding. Now scoped to `Bone`
nodes only; verified byte-identical before/after on that model's physics names.

**(c) The case flip desynchronised the mesh — this one was serious.** The mesh
binds to bones **by name**, through a `jointList` in the DMX. Renaming only the
`.vmdl` left the DMX pointing at names that no longer existed, which produces
`Invalid skinning bone index :: -1` for every joint and a mesh that does not
deform. The flip is length-preserving, so the DMX string table is now patched
in place — 5 DMX files for the affected model, byte-identical file sizes, **0 unresolved
joints** afterwards.

**(d) Three PowerShell case-insensitivity traps.** `-ne`, `-eq` and
`Hashtable.ContainsKey` all ignore case by default, which is precisely the
distinction this work depends on. Each silently made a fix into a no-op that
still reported success. Now uses `-cne` / `-ccontains`.

## 3. Mesh binding

| check | result |
|---|---|
| a test model DMX `jointList` casing | uppercase — matches its `.vmdl` skeleton |
| a test model DMX joints unresolved against skeleton | **0 of 70** |
| that model after flip + DMX patch | **0 unresolved** across all 3 body meshes |
| `.vmdl`↔DMX casing consistency, all 26 | consistent in every model |
| `m_remappingTable` in our compile | present and regenerated correctly |

## 4. The skinning warnings are cosmetic

Investigated properly rather than assumed:

- Every DMX joint resolves to a skeleton bone by name (0 unresolved)
- The compiler still emits `m_remappingTable` correctly
- The untouched original decompile produces the identical count, so it is a
  VRF round-trip artefact rather than anything the port introduces
- ModelDoc loads the compiled result with *"No Errors or Warnings"* and renders
  the mesh with all materials bound

Not a blocker. Worth watching if the model renders visibly deformed.

## 5. Differences left alone deliberately

Community models often carry three AnimGraph1-era nodes the reference does not: `MovementSettings`,
`Feet`/`Foot`, and `character_arm_config`. The compiler accepts all three and
AG2 appears not to read them — the reference model has none. They are left in
place rather than stripped, because removing `character_arm_config` would take
away first-person arms and the others are inert.

The `Feet` node did reference lowercase bones (`ankle_l`) that do not exist on
uppercase skeletons — a latent bug in the source pack's originals. Now corrected to match
the actual skeleton, 4 references per model.

Valve's agents and the reference both carry `AnimConstraint*` nodes (aim constraints, tilt-twist,
jigglebone drivers) and 526 bones to a test model's 72. Those are authoring richness,
not AG2 requirements.

## 6. Compiled output vs the original

| block | original | ours | note |
|---|---|---|---|
| mesh | `MBUF` 255,826 | `MVTX` 224,243 + `MIDX` 18,764 | newer compiler splits the buffer |
| `MDAT` | 10,184 | 10,273 | comparable |
| `PHYS` | 6,324 | 2,879 | **ours is smaller — see below** |
| `m_remappingTable` | 88 entries | 91 entries | ours has 2 extra bones, so expected |

The `PHYS` shrinkage is unexplained. It affects hitboxes and collision, not
animation. Flagged rather than solved.

## 7. What static analysis couldn't answer

Everything below was unknown until the models were loaded in game. Recorded
because the same limits apply to anyone repeating this work:

1. **Whether AG2 binds.** Static structure said it should. Only the game could
   confirm it. It does.
2. **Weapon placement.** `wpnPivot`/`wpn` are at identity, not the model's real
   weapon position. In the reference the two are exact negations (net zero), which is
   consistent with the graph driving them — but if the body animates and the
   gun sits wrong, that is the knob. the reference model's real values are in
   `canonical_skeleton.txt`.
3. **Whether `PHYS` shrinkage matters.** Hit registration, not animation. Still
   unmeasured.
4. **First-person arms.** A separate port entirely — `*_pm_arm.vmdl_c` carries
   no animgraph reference at all. Even the reference model's arms were never conformed by her
   author.

## 8. Still open

**Twist constraints.** Valve's agents carry 16 `CTiltTwistConstraint` lists
driving forearm and leg twist bones. Ported models have none, so limbs deform
poorly under rotation. The ModelDoc source format takes bone *names*, so the
bone hash isn't the obstacle — for the record it's MurmurHash2 with seed
`0x31415926`, Valve's standard. The obstacle is `relative_origin` and
`relative_angles`, which encode each model's own geometry. Get them wrong and
limbs look worse than leaving the constraints out.

**Sample size.** Every rule here is inferred from three models of one lineage:
two Valve agents, plus one custom model that had been conformed to match them.
A working AG2 port by an unrelated author would be worth more than any further
static analysis.

**Viewmodel arms.** `*_pm_arm.vmdl` files carry no animgraph reference at all
and are a separate port. AG2 decouples arms from the player model — Valve ships
them under `agents/models/shared/arms/glove_*` — so ported models fall back to
CS2's standard arms, which is what Valve's own agents do.
