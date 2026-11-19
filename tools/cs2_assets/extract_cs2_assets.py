#!/usr/bin/env python3
"""Extract web-ready CS2 assets from the local game install.

One entry point that wraps Source2Viewer-CLI with the flags that actually matter
for a browser replay viewer, then prunes the agent GLBs down to a usable size.

Everything is derived from measurements in `notes/research-gameassets.md`; the
notable ones are that a CS2 agent exports 2091 clips / 1062 MB by default, that
`--gltf_animation_list` matches exact names only (no globs), and that textures
land next to the GLB as loose PNGs rather than embedded.

Examples
--------
    # what is available
    python extract_cs2_assets.py --list agents
    python extract_cs2_assets.py --list weapon-icons

    # one agent, locomotion clips only, into ./assets
    python extract_cs2_assets.py --agent ctm_sas --out ./assets

    # every competitive agent
    python extract_cs2_assets.py --agent-all --out ./assets

    # weapon models + the 2D HUD icons
    python extract_cs2_assets.py --weapons ak47,awp,deagle --out ./assets
    python extract_cs2_assets.py --weapon-icons --out ./assets

    # smoke/fire particle definitions + their sprite textures
    python extract_cs2_assets.py --particles --out ./assets

    # radar images + the overview transform that places them in world space
    python extract_cs2_assets.py --radars --out ./assets
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]           # notes/scratch -> notes -> repo root
DEFAULT_CLI = ROOT / "tools" / "Source2Viewer-CLI" / "Source2Viewer-CLI.exe"
DEFAULT_GAME = Path(
    r"F:\Program Files\steam\steamapps\common\Counter-Strike Global Offensive"
)
VPK_REL = Path("game/csgo/pak01_dir.vpk")

# Clips a third-person replay actually needs: idle/walk/run/jump/crouch in the
# eight compass directions, for the two "held item" poses a demo distinguishes.
LOCOMOTION_KEEP = [r"world/(rifle|pistol)/_default"]
LOCOMOTION_DROP = [r"ladder|planted|turn_|_deploy|reload|shoot|silencer|_draw|_holster"]

# The competitive agent roster (folder name -> representative .vmdl_c).
AGENTS = {
    "ctm_sas": "agents/models/ctm_sas/ctm_sas.vmdl_c",
    "ctm_fbi": "agents/models/ctm_fbi/ctm_fbi.vmdl_c",
    "ctm_swat": "agents/models/ctm_swat/ctm_swat_variantk.vmdl_c",
    "ctm_st6": "agents/models/ctm_st6/ctm_st6_variantn.vmdl_c",
    "ctm_gendarmerie": "agents/models/ctm_gendarmerie/ctm_gendarmerie_variante.vmdl_c",
    "ctm_diver": "agents/models/ctm_diver/ctm_diver_variantc.vmdl_c",
    "tm_phoenix": "agents/models/tm_phoenix/tm_phoenix.vmdl_c",
    "tm_leet": "agents/models/tm_leet/tm_leet_variantk.vmdl_c",
    "tm_balkan": "agents/models/tm_balkan/tm_balkan_variantl.vmdl_c",
    "tm_jungle_raider": "agents/models/tm_jungle_raider/tm_jungle_raider_variantf.vmdl_c",
    "tm_professional": "agents/models/tm_professional/tm_professional_varj.vmdl_c",
}

# Full CS2 arsenal. Key = the asset name used everywhere downstream (extraction
# output dir, --weapons argument, manifest entry). The *value* is the VPK path.
#
# Do not try to derive the key from the VPK path: Valve's model names disagree
# with the HUD-icon names in several places (`weapon_rif_m4a4` is the "M4A4"
# icon but `weapon_rif_m4a1` is the M4A4 *model*, `weapon_knife_m9` is the
# "M9 Bayonet" icon `weapon_knife_m9_bayonet`, `weapon_incendiarygrenade` is the
# `incgrenade` icon). The mapping from display name to key lives in
# `visualizer/static/js/weapon-index.js`; this table is the VPK side of it.
WEAPON_MODELS = {
    # rifles
    "ak47": "weapons/models/ak47/weapon_rif_ak47.vmdl_c",
    "m4a4": "weapons/models/m4a4/weapon_rif_m4a4.vmdl_c",
    "m4a1_silencer": "weapons/models/m4a1_silencer/weapon_rif_m4a1_silencer.vmdl_c",
    "galilar": "weapons/models/galilar/weapon_rif_galilar.vmdl_c",
    "famas": "weapons/models/famas/weapon_rif_famas.vmdl_c",
    "aug": "weapons/models/aug/weapon_rif_aug.vmdl_c",
    "sg556": "weapons/models/sg556/weapon_rif_sg556.vmdl_c",
    # snipers
    "awp": "weapons/models/awp/weapon_snip_awp.vmdl_c",
    "ssg08": "weapons/models/ssg08/weapon_snip_ssg08.vmdl_c",
    "scar20": "weapons/models/scar20/weapon_snip_scar20.vmdl_c",
    "g3sg1": "weapons/models/g3sg1/weapon_snip_g3sg1.vmdl_c",
    # smgs
    "mp9": "weapons/models/mp9/weapon_smg_mp9.vmdl_c",
    "mac10": "weapons/models/mac10/weapon_smg_mac10.vmdl_c",
    "mp7": "weapons/models/mp7/weapon_smg_mp7.vmdl_c",
    "mp5sd": "weapons/models/mp5sd/weapon_smg_mp5sd.vmdl_c",
    "ump45": "weapons/models/ump45/weapon_smg_ump45.vmdl_c",
    "p90": "weapons/models/p90/weapon_smg_p90.vmdl_c",
    "bizon": "weapons/models/bizon/weapon_smg_bizon.vmdl_c",
    # pistols
    "deagle": "weapons/models/deagle/weapon_pist_deagle.vmdl_c",
    "glock18": "weapons/models/glock18/weapon_pist_glock18.vmdl_c",
    "usp_silencer": "weapons/models/usp_silencer/weapon_pist_usp_silencer.vmdl_c",
    "hkp2000": "weapons/models/hkp2000/weapon_pist_hkp2000.vmdl_c",
    "p250": "weapons/models/p250/weapon_pist_p250.vmdl_c",
    "fiveseven": "weapons/models/fiveseven/weapon_pist_fiveseven.vmdl_c",
    "tec9": "weapons/models/tec9/weapon_pist_tec9.vmdl_c",
    "elite": "weapons/models/elite/weapon_pist_elite.vmdl_c",
    "cz75a": "weapons/models/cz75a/weapon_pist_cz75a.vmdl_c",
    "revolver": "weapons/models/revolver/weapon_pist_revolver.vmdl_c",
    "taser": "weapons/models/taser/weapon_pist_taser.vmdl_c",
    # shotguns
    "mag7": "weapons/models/mag7/weapon_shot_mag7.vmdl_c",
    "nova": "weapons/models/nova/weapon_shot_nova.vmdl_c",
    "sawedoff": "weapons/models/sawedoff/weapon_shot_sawedoff.vmdl_c",
    "xm1014": "weapons/models/xm1014/weapon_shot_xm1014.vmdl_c",
    # machine guns
    "m249": "weapons/models/m249/weapon_mach_m249.vmdl_c",
    "negev": "weapons/models/negev/weapon_mach_negev.vmdl_c",
    # knives
    "knife_karambit": "weapons/models/knife/knife_karambit/weapon_knife_karambit.vmdl_c",
    "knife_bayonet": "weapons/models/knife/knife_bayonet/weapon_knife_bayonet.vmdl_c",
    "knife_m9": "weapons/models/knife/knife_m9/weapon_knife_m9.vmdl_c",
    "knife_butterfly": "weapons/models/knife/knife_butterfly/weapon_knife_butterfly.vmdl_c",
    "knife_skeleton": "weapons/models/knife/knife_skeleton/weapon_knife_skeleton.vmdl_c",
    "knife_default_ct": "weapons/models/knife/knife_default_ct/weapon_knife_default_ct.vmdl_c",
    "knife_default_t": "weapons/models/knife/knife_default_t/weapon_knife_default_t.vmdl_c",
    "knife_bowie": "weapons/models/knife/knife_bowie/weapon_knife_bowie.vmdl_c",
    "knife_canis": "weapons/models/knife/knife_canis/weapon_knife_canis.vmdl_c",
    "knife_cord": "weapons/models/knife/knife_cord/weapon_knife_cord.vmdl_c",
    "knife_css": "weapons/models/knife/knife_css/weapon_knife_css.vmdl_c",
    "knife_falchion": "weapons/models/knife/knife_falchion/weapon_knife_falchion.vmdl_c",
    "knife_flip": "weapons/models/knife/knife_flip/weapon_knife_flip.vmdl_c",
    "knife_gut": "weapons/models/knife/knife_gut/weapon_knife_gut.vmdl_c",
    "knife_kukri": "weapons/models/knife/knife_kukri/weapon_knife_kukri.vmdl_c",
    "knife_navaja": "weapons/models/knife/knife_navaja/weapon_knife_navaja.vmdl_c",
    "knife_outdoor": "weapons/models/knife/knife_outdoor/weapon_knife_outdoor.vmdl_c",
    "knife_push": "weapons/models/knife/knife_push/weapon_knife_push.vmdl_c",
    "knife_stiletto": "weapons/models/knife/knife_stiletto/weapon_knife_stiletto.vmdl_c",
    "knife_tactical": "weapons/models/knife/knife_tactical/weapon_knife_tactical.vmdl_c",
    "knife_talon": "weapons/models/knife/knife_talon/weapon_knife_talon.vmdl_c",
    "knife_ursus": "weapons/models/knife/knife_ursus/weapon_knife_ursus.vmdl_c",
    # grenades
    "hegrenade": "weapons/models/grenade/hegrenade/weapon_hegrenade.vmdl_c",
    "flashbang": "weapons/models/grenade/flashbang/weapon_flashbang.vmdl_c",
    "smokegrenade": "weapons/models/grenade/smokegrenade/weapon_smokegrenade.vmdl_c",
    "molotov": "weapons/models/grenade/molotov/weapon_molotov.vmdl_c",
    "incendiary": "weapons/models/grenade/incendiary/weapon_incendiarygrenade.vmdl_c",
    "decoy": "weapons/models/grenade/decoy/weapon_decoy.vmdl_c",
    # equipment
    "c4": "weapons/models/c4/weapon_c4.vmdl_c",
    "healthshot": "weapons/models/healthshot/weapon_healthshot.vmdl_c",
}

#: Maps the replay viewer can load (``maps/optimized_obj_files/*.obj``), plus the
#: two maps that ship a separate lower-level radar.  The image itself is
#: ``<map>_radar_psd.vtex_c``; the world->pixel transform lives in
#: ``resource/overviews/<map>.txt`` and is extracted alongside it, because a
#: radar image without that transform is just a picture.
RADAR_MAPS = [
    "de_ancient",
    "de_anubis",
    "de_cache",
    "de_dust2",
    "de_inferno",
    "de_mirage",
    "de_nuke",
    "de_nuke_lower",
    "de_overpass",
    "de_train",
    "de_train_lower",
    "de_vertigo",
    "de_vertigo_lower",
]

PARTICLE_PATHS = [
    "particles/explosions_fx/explosion_smokegrenade_voxel.vpcf_c",
    "particles/explosions_fx/explosion_smokegrenade_adaptive.vpcf_c",
    "particles/explosions_fx/explosion_smokegrenade.vpcf_c",
    "particles/inferno_fx/fire_core.vsnap_c",
    "particles/inferno_fx/fire_filler.vsnap_c",
    "particles/inferno_fx/fire_edge.vsnap_c",
    "particles/inferno_fx/molotov_groundfire.vpcf_c",
    "particles/inferno_fx/incendiary_groundfire.vpcf_c",
    "particles/inferno_fx/extinguish_fire.vpcf_c",
]

PARTICLE_TEXTURE_DIRS = [
    "materials/particle/smoke/smokeburst/",
    "materials/particle/flames/",
    "materials/particle/fire_onsurface/",
]

ICON_DIR = "panorama/images/econ/weapons/base_weapons/"


def run_cli(cli: Path, vpk: Path, args: list[str], quiet: bool = False,
            capture: bool = False) -> int:
    cmd = [str(cli), "-i", str(vpk), *args]
    if not quiet:
        print("  $ " + " ".join(args))
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if capture and proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-6:])
        print(f"  ! exit {proc.returncode}\n{tail}", file=sys.stderr)
    return proc.returncode


def do_list(cli: Path, vpk: Path, what: str) -> int:
    if what == "agents":
        for k, v in sorted(AGENTS.items()):
            print(f"{k:<20} {v}")
    elif what == "weapons":
        for k, v in sorted(WEAPON_MODELS.items()):
            print(f"{k:<20} {v}")
    elif what == "weapon-icons":
        run_cli(cli, vpk, ["--vpk_list", "-f", ICON_DIR], quiet=True, capture=True)
    elif what == "particles":
        for p in PARTICLE_PATHS:
            print(p)
    elif what == "radars":
        for m in RADAR_MAPS:
            print(f"{m:<20} panorama/images/overheadmaps/{m}_radar_psd.vtex_c")
    else:
        print(f"unknown list target: {what}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    ap.add_argument("--game", type=Path, default=DEFAULT_GAME)
    ap.add_argument("--out", type=Path, default=ROOT / "assets")
    ap.add_argument("--agent", action="append", default=[],
                    help="agent key (see --list agents)")
    ap.add_argument("--agent-all", action="store_true")
    ap.add_argument("--weapons", default="", help="comma-separated weapon keys")
    ap.add_argument("--weapon-icons", action="store_true")
    ap.add_argument("--particles", action="store_true")
    ap.add_argument("--radars", action="store_true",
                    help="radar overview images + their world transforms")
    ap.add_argument("--all-clips", action="store_true",
                    help="do NOT prune animations (produces ~1 GB per agent)")
    ap.add_argument("--list", metavar="WHAT",
                    help="agents | weapons | weapon-icons | particles | radars")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv[1:])

    vpk = args.game / VPK_REL
    if not args.cli.is_file():
        print(f"Source2Viewer-CLI not found at {args.cli}", file=sys.stderr)
        return 2
    if not vpk.is_file():
        print(f"pak01_dir.vpk not found at {vpk}", file=sys.stderr)
        return 2

    if args.list:
        return do_list(args.cli, vpk, args.list)

    args.out.mkdir(parents=True, exist_ok=True)
    gltf_flags = ["-d", "--gltf_export_format", "glb",
                  "--gltf_export_materials", "--gltf_textures_adapt"]

    # --- agents -------------------------------------------------------------
    keys = list(AGENTS) if args.agent_all else args.agent
    for key in keys:
        if key not in AGENTS:
            print(f"unknown agent {key!r}; see --list agents", file=sys.stderr)
            continue
        print(f"[agent] {key}")

        # VRF writes the GLB *and* its loose texture PNGs into the output
        # directory, and the GLB references them by bare filename. The GLB must
        # therefore end up in that same directory -- relocating it on its own
        # leaves the model untextured, because nothing will resolve the PNGs.
        dest = args.out / "agents" / key
        dest.mkdir(parents=True, exist_ok=True)
        raw = dest / f"{key}.glb"

        run_cli(args.cli, vpk, [
            "-f", AGENTS[key], "-o", str(raw), *gltf_flags,
            "--gltf_export_animations",
        ])
        if not raw.is_file():
            print(f"  !! export produced no GLB for {key}", file=sys.stderr)
            continue
        if args.all_clips:
            continue

        import prune_glb  # local module

        pruned = dest / f"{key}.pruned.glb"
        rc = prune_glb.main([
            "prune", str(raw), str(pruned),
            *sum((["--keep", k] for k in LOCOMOTION_KEEP), []),
            *sum((["--drop", d] for d in LOCOMOTION_DROP), []),
        ])
        if rc == 0:
            # Replace in place so the textures sitting beside it stay valid.
            pruned.replace(raw)
        else:
            print(f"  !! prune failed for {key}; keeping the full-clip GLB",
                  file=sys.stderr)
            pruned.unlink(missing_ok=True)

        n_tex = len(list(dest.glob("*.png")))
        print(f"  -> {raw.relative_to(args.out)}  ({n_tex} texture PNG(s) beside it)")

    # --- weapons ------------------------------------------------------------
    if args.weapons:
        wanted = [w.strip() for w in args.weapons.split(",") if w.strip()]
        paths = [WEAPON_MODELS[w] for w in wanted if w in WEAPON_MODELS]
        for w in wanted:
            if w not in WEAPON_MODELS:
                print(f"unknown weapon {w!r}; see --list weapons", file=sys.stderr)
        if paths:
            print(f"[weapons] {len(paths)} model(s)")
            run_cli(args.cli, vpk, [
                "-f", ",".join(paths), "-o", str(args.out / "weapons") + "\\",
                *gltf_flags, "--threads", str(args.threads),
            ])

        # Write an exact key -> GLB index. VRF preserves the VPK directory
        # structure under the -o prefix, so the output path is derivable; but
        # deriving it downstream by *searching* is how you silently pick
        # `weapon_rif_m4a1_silencer.glb` when you asked for `m4a1`. Record it.
        #
        # Merged with any existing index: weapons are extracted in batches and a
        # later batch must not forget the earlier ones.
        idx_path = args.out / "weapons_index.json"
        index: dict[str, str] = {}
        if idx_path.is_file():
            try:
                index = json.loads(idx_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                index = {}
        added = 0
        for w in wanted:
            vmdl = WEAPON_MODELS.get(w)
            if not vmdl:
                continue
            rel = Path("weapons") / Path(vmdl).with_suffix("").with_suffix(".glb")
            if (args.out / rel).is_file():
                index[w] = rel.as_posix()
                added += 1
            else:
                print(f"  !! {w}: expected {rel} but it is missing", file=sys.stderr)
        if added:
            idx_path.write_text(json.dumps(index, indent=2, sort_keys=True),
                                encoding="utf-8")
            print(f"[weapons] index -> {idx_path.relative_to(args.out)} "
                  f"({added} from this batch, {len(index)} total)")

    # --- weapon icons -------------------------------------------------------
    if args.weapon_icons:
        print("[weapon-icons] HUD icons")
        run_cli(args.cli, vpk, [
            "-f", ICON_DIR, "-o", str(args.out / "weapon_icons") + "\\",
            "-d", "--threads", str(args.threads),
        ])

    # --- particles ----------------------------------------------------------
    if args.particles:
        print("[particles] definitions")
        run_cli(args.cli, vpk, [
            "-f", ",".join(PARTICLE_PATHS),
            "-o", str(args.out / "particles") + "\\", "-d",
        ])
        for d in PARTICLE_TEXTURE_DIRS:
            print(f"[particles] textures {d}")
            run_cli(args.cli, vpk, [
                "-f", d, "-o", str(args.out / "particles") + "\\",
                "-d", "--threads", str(args.threads),
            ])

    # --- radar --------------------------------------------------------------
    if args.radars:
        print(f"[radar] {len(RADAR_MAPS)} image(s) + overview transforms")
        images = [f"panorama/images/overheadmaps/{m}_radar_psd.vtex_c"
                  for m in RADAR_MAPS]
        run_cli(args.cli, vpk, [
            "-f", ",".join(images), "-o", str(args.out / "radar") + "\\",
            "-d", "--threads", str(args.threads),
        ])
        # de_nuke_lower has no overview of its own: it reuses de_nuke's
        # pos_x/pos_y/scale, which is exactly what the game does via the
        # `verticalsections` block.  Extract the base names, not the suffixes.
        overviews = sorted({m.removesuffix("_lower") for m in RADAR_MAPS})
        run_cli(args.cli, vpk, [
            "-f", ",".join(f"resource/overviews/{m}.txt" for m in overviews),
            "-o", str(args.out / "radar") + "\\", "-d",
        ])

    print(f"\ndone -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main(sys.argv))
