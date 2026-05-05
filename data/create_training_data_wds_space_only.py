import argparse
import io
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Iterable, Set
import random
import zstandard as zstd

cctx = zstd.ZstdCompressor(level=3)

import numpy as np
import yaml

import webdataset as wds

T_Window_Size = 1 # The history window size for each sample, 32/4 = 8s
Space_Size = 31 # The number of tokens in one tick
N_Test_Map = 4 # Number of each map in test set
Kepp_rate = (1/(4 * 2)) # Rate to process a tick

print(f"Sample every {(1/Kepp_rate)/4} seconds")

map_config = {
	"maps": {
		"de_mirage": {
			"center": [-605.8900146484375, -866.8900146484375, -171.6199951171875],
		},
		"de_dust2": {
			"center": [-199.0, 977.0, 32.220001220703125],
		},
		"de_inferno": {
			"center": [481.07000732421875, 1396.47998046875, 137.91000366210938],
		},
		"de_nuke": {
			"center": [265.9599914550781, -772.5, -381.8999938964844],
		},
		"de_overpass": {
			"center": [-2027.3900146484375, -812.9000244140625, 324.95001220703125],
		},
		"de_ancient": {
			"center": [-435.5, -348.0, 43.650001525878906],
		},
		"de_anubis": {
			"center": [-77.38999938964844, 618.9000244140625, -6.800000190734863],
		},
		"de_train": {
			"center": [-118.25, -2.0, -128.52000427246094],
		},
	},
	"ranges": {
		"x": [-5000, 5000],
		"y": [-5000, 5000],
		"z": [-2000, 2000],
	},
}

map_name_to_idx = {map_name: idx for idx, map_name in enumerate(map_config["maps"].keys())}

print(f"Number of maps: {len(map_name_to_idx)}")

def clip_and_scale(value, range):
	min_val, max_val = range
	if value < min_val:
		value = min_val
	elif value > max_val:
		value = max_val
	return value / max(abs(min_val), abs(max_val))


with open("demoparser_utils/tokenizer.yaml", "r", encoding="utf-8") as f:
	config = yaml.safe_load(f)

def weapon_name_to_idx(weapon_name: str) -> int:

	weapons = config.get("weapons", [])
	weapon2idx = {name: i for i, name in enumerate(weapons)}
	if weapon_name not in weapon2idx:
		weapon_name = "knife"
	return weapon2idx[weapon_name]

print(f"Number of weapons: {len(config.get('weapons', []))}")

def projectile_name_to_idx(projectile_name: str) -> int:

	projectiles = config.get("projectiles", [])
	projectile2idx = {name: i for i, name in enumerate(projectiles)}
	if projectile_name not in projectile2idx:
		raise ValueError(f"Unknown projectile name: {projectile_name}")
	return projectile2idx[projectile_name]

print(f"Number of projectiles: {len(config.get('projectiles', []))}")

def parse_args():
	parser = argparse.ArgumentParser(
		description="Create WebDataset shards from processed jsons in zip archives"
	)
	parser.add_argument(
		"--zip-root",
		required=True,
		help="Root directory that contains zip archives",
	)
	parser.add_argument(
		"--output-dir",
		required=True,
		help="Output directory for WebDataset shards",
	)
	parser.add_argument(
		"--processed-list",
		default="processed_jsons.txt",
		help="Text file that stores processed json names",
	)
	parser.add_argument(
		"--pattern",
		default="*.zip",
		help="Glob pattern for zip files (default: *.zip)",
	)

	parser.add_argument(
		"--only-append-to-train",
		action="store_true",
		help="If set, all samples will only be written to train, not to test."
	)
	return parser.parse_args()


def load_processed(path: Path) -> Set[str]:
	if not path.exists():
		return set()
	return set(
		line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line
	)


def append_processed(path: Path, name: str):
	with path.open("a", encoding="utf-8") as f:
		f.write(name + "\n")


def iter_zip_files(root: Path, pattern: str) -> Iterable[Path]:
	return sorted(root.rglob(pattern))


def find_start_shard(output_dir: Path) -> int:
	output_dir.mkdir(parents=True, exist_ok=True)
	shard_re = re.compile(r"^shards-(\d{5})\.tar$")
	max_idx = -1
	for path in output_dir.iterdir():
		if not path.is_file():
			continue
		match = shard_re.match(path.name)
		if not match:
			continue
		idx = int(match.group(1))
		if idx > max_idx:
			max_idx = idx
	return max_idx + 1


def parse_json_name_metadata(json_name: str) -> dict:
	base_name = json_name
	window_time_s = None
	if "_**_" in json_name:
		base_name, time_part = json_name.split("_**_", 1)
		time_part = time_part.strip()
		if time_part.endswith("s"):
			try:
				window_time_s = float(time_part[:-1])
			except ValueError:
				window_time_s = None

	stem = Path(base_name).stem
	team1 = None
	team2 = None
	map_name = None
	match_index = None

	match = re.match(r"^(?P<teams>.+)-m(?P<match>\d+)-(?P<map>.+)$", stem)
	if match:
		teams_part = match.group("teams")
		map_name = match.group("map")
		match_index = int(match.group("match"))
		if "-vs-" in teams_part:
			team1, team2 = teams_part.split("-vs-", 1)

	metadata = {
		"source_json": base_name,
		"teams": [team1, team2],
		"match_index": match_index,
		"time": window_time_s,
	}
	return metadata


def is_valid_round(round_data):
    times = []

    for tick in round_data:
        t = tick.get("round_seconds", None)
        if t is None:
            return False
        times.append(t)

    if len(times) == 0:
        return False

    min_time = min(times)
    max_time = max(times)

    if max_time > 170 or min_time < 0:
        return False

    return True

def process_one_data(json_datas, json_name):
	player_id_permutation = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	random.shuffle(player_id_permutation) # Randomly permute player IDs to prevent model from overfitting to specific player positions

	player_name_to_idx = {}

	result = {}
	result["__key__"] = f"{hash(json_name)}_{random.randint(0,1e9)}"

	metadata = parse_json_name_metadata(json_name)
	metadata["window_size"] = T_Window_Size
	metadata["map_config"] = map_config
	metadata["players"] = [None for _ in range(10)]
	for idx, player_info in enumerate(json_datas[-1]["players_info"]):
		metadata["players"][player_id_permutation[idx]] = {
			"steamid": player_info["steamid"],
			"name": player_info["name"],
		}
		player_name_to_idx[player_info["name"]] = player_id_permutation[idx]

	metadata["n_maps"] = len(map_name_to_idx)
	metadata["n_weapons"] = len(config.get("weapons", []))
	metadata["n_projectiles"] = len(config.get("projectiles", []))
	metadata["T_Window_Size"] = T_Window_Size
	metadata["Space_Size"] = Space_Size

	metadata["map_name"] = json_datas[-1]["map_name"]
	metadata["winner_info"] = json_datas[-1]["round_label"]["round_info"]
	metadata["future_kills"] = []
	for f_kill in json_datas[-1]["future_kills"]:
		metadata["future_kills"].append([player_name_to_idx.get(f_kill["attacker_name"], 10), player_name_to_idx.get(f_kill["victim_name"], 10)])
	metadata["alive_in_the_end"] = [1 for _ in range(10)]
	metadata["name_to_idx"] = player_name_to_idx
	for idx, player_info in enumerate(json_datas[-1]["players_info"]):
		if player_info["is_alive"] == 0:
			metadata["alive_in_the_end"][player_name_to_idx[player_info["name"]]] = 0
	for f_kill in metadata["future_kills"]:
		victim_idx = f_kill[1]  # Get the index of the victim
		if 0 <= victim_idx < 10:
			metadata["alive_in_the_end"][victim_idx] = 0

	result["meta"] = metadata

	def init_token_slots(default=None):
		return [[default for _ in range(Space_Size)] for __ in range(T_Window_Size)]

	# MLP1(x,y,z,Embedding2(map_id)) for token j in the sequence i
	result["MLP1_f"] = init_token_slots([0,0,0]) # list of [x,y,z], float
	result["MLP1_i"] = init_token_slots(0) # list of [map_id], int
	result["MLP1_mask"] = init_token_slots(False) # list of bool

	# MLP2(x_0, x_1, ..., x_n) for token j in the sequence i, where x_0, x_1, ..., x_n are the features
	result["MLP2_f"] = init_token_slots([0 for _ in range(14)]) # list of [x_0, x_1, ..., x_n], float
	result["MLP2_mask"] = init_token_slots(False) # list of bool

	# MLP3(duration, Embedding3(projectile_name)) for token j in the sequence i
	result["MLP3_f"] = init_token_slots([0]) # list of [duration], float
	result["MLP3_i"] = init_token_slots(0) # list of [projectile_name], int
	result["MLP3_mask"] = init_token_slots(False) # list of bool

	# MLP4(x_0, x_1, ..., x_n) for token j in the sequence i, where x_0, x_1, ..., x_n are the features
	result["MLP4_f"] = init_token_slots([0 for _ in range(4)]) # list of [x_0, x_1, ..., x_n], float
	result["MLP4_mask"] = init_token_slots(False) # list of bool

	# MLP5(x_0, x_1, ..., x_n, Embedding4(id)) for token j in the sequence i, where x_0, x_1, ..., x_n are the features
	result["MLP5_f"] = init_token_slots([[0 for _ in range(13)] for __ in range(9)]) # list of [x_0, x_1, ..., x_n], float
	result["MLP5_i"] = init_token_slots([0 for _ in range(9)]) # list of [id], int
	result["MLP5_mask"] = init_token_slots([False for _ in range(9)]) # list of bool

	# Embedding1(inventory) for token j in the sequence i
	result["EMB1_i"] = init_token_slots([0 for _ in range(9)]) # list of [inventory], int
	result["EMB1_mask"] = init_token_slots([0 for _ in range(9)]) # list of bool

	# Embedding2(map_id) for token j in the sequence i
	result["EMB2_i"] = init_token_slots(0) # list of [map_id], int
	result["EMB2_mask"] = init_token_slots(False) # list of bool

	# Embedding4(j) for token j in the sequence i
	# no need to store the index for this one since it's just the player index

	# Embedding5(i) for token j in the sequence i
	# no need to store the index for this one since it's just the tick index

	# Deadmask: whether (i,j) is a dead player
	result["DEAD_mask"] = init_token_slots(False) # list of bool

	# Padmask: whether (i,j) is a padding token
	result["PAD_mask"] = init_token_slots(False) # list of bool


	def process_xyz(x, y, z, map_name):
		center = map_config["maps"][map_name]["center"]
		x -= center[0]
		y -= center[1]
		z -= center[2]
		return [clip_and_scale(x, map_config["ranges"]["x"]), clip_and_scale(y, map_config["ranges"]["y"]), clip_and_scale(z, map_config["ranges"]["z"])]
	
	pad_front = T_Window_Size - len(json_datas)
	if pad_front < 0:
		raise ValueError(f"Number of ticks in json data ({len(json_datas)}) exceeds T_Window_Size ({T_Window_Size})")
	if pad_front > 0:
		for _ in range(pad_front):
			for __ in range(Space_Size):
				result["PAD_mask"][_][__] = True
	

	for sequence_id_, data in enumerate(json_datas):
		sequence_id = pad_front + sequence_id_
		# tick_info = []

		if data["is_bomb_planted"] == False:
			data["bomb_planted_duration"] = 0

		num_tokens = 11 + len(data["projectiles"])
		assert num_tokens <= Space_Size, f"Number of tokens ({num_tokens}) exceeds Space_Size ({Space_Size})"

		for i in range(num_tokens, Space_Size):
			result["PAD_mask"][sequence_id][i] = True

		for player_info in data["players_info"]:
			idx = player_name_to_idx[player_info["name"]]
			# tick_info.append({
			# 	"player_idx": idx,
			# 	"armor": player_info["armor"] > 0,
			# 	"helmet": player_info["has_helmet"],
			# 	"defuser": player_info["has_defuser"],
			# 	"is_alive": player_info["is_alive"],
			# 	"is_blind": player_info["flash_duration"] > 0,
			# 	"X": player_info["X"],
			# 	"Y": player_info["Y"],
			# 	"Z": player_info["Z"],
			# 	"pitch": player_info["pitch"],
			# 	"yaw": player_info["yaw"],
			# 	"health": player_info["health"],
			# 	"inventory": player_info["inventory"],
			# 	"team": player_info["team_num"],
			# 	"velocity": player_info["velocity"],
			# 	"velocity_X": player_info["velocity_X"],
			# 	"velocity_Y": player_info["velocity_Y"],
			# 	"velocity_Z": player_info["velocity_Z"],
			# })
			if player_info["is_alive"] == 0:
				result["DEAD_mask"][sequence_id][idx] = True
			else:
				result["MLP1_f"][sequence_id][idx] = process_xyz(player_info["X"], player_info["Y"], player_info["Z"], data["map_name"])
				result["MLP1_i"][sequence_id][idx] = map_name_to_idx[data["map_name"]]
				result["MLP1_mask"][sequence_id][idx] = True

				result["MLP2_f"][sequence_id][idx] = [player_info["armor"] > 0, player_info["has_helmet"], player_info["has_defuser"], player_info["flash_duration"] > 0, math.cos(math.radians(player_info["pitch"])), math.sin(math.radians(player_info["pitch"])), math.cos(math.radians(player_info["yaw"])), math.sin(math.radians(player_info["yaw"])), player_info["health"] / 100.0, player_info["team_num"] == "CT", clip_and_scale(player_info["velocity"], [0, 8000]), clip_and_scale(player_info["velocity_X"], [-8000, 8000]), clip_and_scale(player_info["velocity_Y"], [-8000, 8000]), clip_and_scale(player_info["velocity_Z"], [-1000, 1000])] 
				result["MLP2_mask"][sequence_id][idx] = True
				
				inventory_ids = [weapon_name_to_idx(w) for w in player_info["inventory"]]
				assert len(inventory_ids) <= 9, f"Inventory size exceeds 9: {player_info['inventory']}"
				result["EMB1_i"][sequence_id][idx] = inventory_ids + [0] * (9 - len(inventory_ids))
				result["EMB1_mask"][sequence_id][idx] = [True] * len(inventory_ids) + [False] * (9 - len(inventory_ids))


		# tick_info.append({
		# 	"map_name": data["map_name"],
		# 	"time": data["round_seconds"],
		# 	"c4_planted": data["is_bomb_planted"],
		# 	"c4_dropped": data["is_bomb_dropped"],
		# 	"c4_planted_duration": data["bomb_planted_duration"],
		# 	"c4_X": data["bomb_position"][0],
		# 	"c4_Y": data["bomb_position"][1],
		# 	"c4_Z": data["bomb_position"][2],
		# })
		result["MLP1_f"][sequence_id][10] = process_xyz(data["bomb_position"][0], data["bomb_position"][1], data["bomb_position"][2], data["map_name"])
		result["MLP1_i"][sequence_id][10] = map_name_to_idx[data["map_name"]]
		result["MLP1_mask"][sequence_id][10] = True

		result["EMB2_i"][sequence_id][10] = map_name_to_idx[data["map_name"]]
		result["EMB2_mask"][sequence_id][10] = True

		result["MLP4_f"][sequence_id][10] = [data["round_seconds"]/160, data["is_bomb_planted"], data["is_bomb_dropped"], data["bomb_planted_duration"]/40]
		result["MLP4_mask"][sequence_id][10] = True

		# shuffle projectiles to prevent model from overfitting to specific projectile order
		random.shuffle(data["projectiles"])

		for idx_, projectile in enumerate(data["projectiles"]):
			idx = idx_ + 11
			# tick_info.append({
			# 	"projectile_name": projectile["type"],
			# 	"duration": projectile["duration"],
			# 	"X": projectile["position"][0],
			# 	"Y": projectile["position"][1],
			# 	"Z": projectile["position"][2],
			# })
			result["MLP1_f"][sequence_id][idx] = process_xyz(projectile["position"][0], projectile["position"][1], projectile["position"][2], data["map_name"])
			result["MLP1_i"][sequence_id][idx] = map_name_to_idx[data["map_name"]]
			result["MLP1_mask"][sequence_id][idx] = True

			result["MLP3_f"][sequence_id][idx] = [projectile["duration"] / 25]
			result["MLP3_i"][sequence_id][idx] = projectile_name_to_idx(projectile["type"])
			result["MLP3_mask"][sequence_id][idx] = True



		rel_matrix = [[None for _ in range(Space_Size)] for __ in range(Space_Size)]

		for player_info in data["players_info"]:
			idx = player_name_to_idx[player_info["name"]]

			mlp5_features = []
			mlp5_index = []

			for player_info_2 in data["players_info"]:
				idx2 = player_name_to_idx[player_info_2["name"]]
				if idx == idx2 or player_info["is_alive"] == 0 or player_info_2["is_alive"] == 0:
					continue
				rel_matrix[idx][idx2] = {}
				rel_matrix[idx][idx2]["dx"] = player_info_2["X"] - player_info["X"]
				rel_matrix[idx][idx2]["dy"] = player_info_2["Y"] - player_info["Y"]
				rel_matrix[idx][idx2]["dz"] = player_info_2["Z"] - player_info["Z"]
				rel_matrix[idx][idx2]["distance"] = (rel_matrix[idx][idx2]["dx"] ** 2 + rel_matrix[idx][idx2]["dy"] ** 2 + rel_matrix[idx][idx2]["dz"] ** 2) ** 0.5
				rel_matrix[idx][idx2]["j_is_teammate"] = player_info["team_num"] == player_info_2["team_num"]
				rel_matrix[idx][idx2]["j_is_enemy"] = player_info["team_num"] != player_info_2["team_num"]
				rel_matrix[idx][idx2]["j_is_projectile"] = False
				rel_matrix[idx][idx2]["j_is_spotted_by_i"] = rel_matrix[idx][idx2]["j_is_enemy"] and (player_info["steamid"] in player_info_2["spotted_by"])
				rel_matrix[idx][idx2]["i_is_spotted_by_j"] = rel_matrix[idx][idx2]["j_is_enemy"] and (player_info_2["steamid"] in player_info["spotted_by"])

				dx = rel_matrix[idx][idx2]["dx"]
				dy = rel_matrix[idx][idx2]["dy"]
				dz = rel_matrix[idx][idx2]["dz"]

				yaw_deg = float(player_info.get("yaw", 0.0) or 0.0)
				pitch_deg = float(player_info.get("pitch", 0.0) or 0.0)
				yaw_rad = math.radians(yaw_deg)
				pitch_rad = math.radians(pitch_deg)

				# Angle between facing direction and target vector on XY plane.
				xy_norm = math.hypot(dx, dy)
				if xy_norm > 0:
					fwd_x = math.cos(yaw_rad)
					fwd_y = math.sin(yaw_rad)
					dot_xy = (dx * fwd_x + dy * fwd_y) / xy_norm
					dot_xy = max(-1.0, min(1.0, dot_xy))
					d_theta_xy = math.degrees(math.acos(dot_xy))
				else:
					d_theta_xy = 0.0

				# Angle between pitch and target elevation angle.
				xy_plane_dist = math.hypot(dx, dy)
				if xy_plane_dist > 0:
					target_pitch = math.atan2(dz, xy_plane_dist)
				else:
					target_pitch = math.pi / 2 if dz > 0 else (-math.pi / 2 if dz < 0 else 0.0)
				d_theta_z = abs(math.degrees(pitch_rad - target_pitch))

				rel_matrix[idx][idx2]["d_theta_xy_ij"] = d_theta_xy
				rel_matrix[idx][idx2]["d_theta_z_ij"] = d_theta_z

				rel_data = rel_matrix[idx][idx2]

				# result["MLP5_f"][sequence_id][idx] = [clip_and_scale(rel_data["dx"], map_config["ranges"]["x"]), clip_and_scale(rel_data["dy"], map_config["ranges"]["y"]), clip_and_scale(rel_data["dz"], map_config["ranges"]["z"]), math.log(clip_and_scale(rel_data["distance"], [0, 5000]) + 1), rel_data["j_is_teammate"], rel_data["j_is_enemy"], rel_data["j_is_projectile"], rel_data["j_is_spotted_by_i"], rel_data["i_is_spotted_by_j"], math.cos(math.radians(rel_data["d_theta_xy_ij"])), math.sin(math.radians(rel_data["d_theta_xy_ij"])), math.cos(math.radians(rel_data["d_theta_z_ij"])), math.sin(math.radians(rel_data["d_theta_z_ij"]))]
				# result["MLP5_i"][sequence_id][idx] = idx2
				# result["MLP5_mask"][sequence_id][idx] = True
				mlp5_features.append([clip_and_scale(rel_data["dx"], map_config["ranges"]["x"]), clip_and_scale(rel_data["dy"], map_config["ranges"]["y"]), clip_and_scale(rel_data["dz"], map_config["ranges"]["z"]), math.log(clip_and_scale(rel_data["distance"], [0, 5000]) + 1), rel_data["j_is_teammate"], rel_data["j_is_enemy"], rel_data["j_is_projectile"], rel_data["j_is_spotted_by_i"], rel_data["i_is_spotted_by_j"], math.cos(math.radians(rel_data["d_theta_xy_ij"])), math.sin(math.radians(rel_data["d_theta_xy_ij"])), math.cos(math.radians(rel_data["d_theta_z_ij"])), math.sin(math.radians(rel_data["d_theta_z_ij"]))])
				mlp5_index.append(idx2)

			# for idx_, projectile in enumerate(data["projectiles"]):
			# 	idx2 = 11 + idx_
			# 	if player_info["is_alive"] == 0:
			# 		continue
			# 	rel_matrix[idx][idx2] = {}
			# 	rel_matrix[idx][idx2]["dx"] = projectile["position"][0] - player_info["X"]
			# 	rel_matrix[idx][idx2]["dy"] = projectile["position"][1] - player_info["Y"]
			# 	rel_matrix[idx][idx2]["dz"] = projectile["position"][2] - player_info["Z"]
			# 	rel_matrix[idx][idx2]["distance"] = (rel_matrix[idx][idx2]["dx"] ** 2 + rel_matrix[idx][idx2]["dy"] ** 2 + rel_matrix[idx][idx2]["dz"] ** 2) ** 0.5
			# 	rel_matrix[idx][idx2]["j_is_teammate"] = False
			# 	rel_matrix[idx][idx2]["j_is_enemy"] = False
			# 	rel_matrix[idx][idx2]["j_is_projectile"] = True
			# 	rel_matrix[idx][idx2]["j_is_spotted_by_i"] = False
			# 	rel_matrix[idx][idx2]["i_is_spotted_by_j"] = False

			# 	dx = rel_matrix[idx][idx2]["dx"]
			# 	dy = rel_matrix[idx][idx2]["dy"]
			# 	dz = rel_matrix[idx][idx2]["dz"]

			# 	yaw_deg = float(player_info.get("yaw", 0.0) or 0.0)
			# 	pitch_deg = float(player_info.get("pitch", 0.0) or 0.0)
			# 	yaw_rad = math.radians(yaw_deg)
			# 	pitch_rad = math.radians(pitch_deg)

			# 	# Angle between facing direction and target vector on XY plane.
			# 	xy_norm = math.hypot(dx, dy)
			# 	if xy_norm > 0:
			# 		fwd_x = math.cos(yaw_rad)
			# 		fwd_y = math.sin(yaw_rad)
			# 		dot_xy = (dx * fwd_x + dy * fwd_y) / xy_norm
			# 		dot_xy = max(-1.0, min(1.0, dot_xy))
			# 		d_theta_xy = math.degrees(math.acos(dot_xy))
			# 	else:
			# 		d_theta_xy = 0.0

			# 	# Angle between pitch and target elevation angle.
			# 	xy_plane_dist = math.hypot(dx, dy)
			# 	if xy_plane_dist > 0:
			# 		target_pitch = math.atan2(dz, xy_plane_dist)
			# 	else:
			# 		target_pitch = math.pi / 2 if dz > 0 else (-math.pi / 2 if dz < 0 else 0.0)
			# 	d_theta_z = abs(math.degrees(pitch_rad - target_pitch))

			# 	rel_matrix[idx][idx2]["d_theta_xy_ij"] = d_theta_xy
			# 	rel_matrix[idx][idx2]["d_theta_z_ij"] = d_theta_z
				
			# 	rel_data = rel_matrix[idx][idx2]
			# 	# result["MLP5_f"][sequence_id][idx] = [clip_and_scale(rel_data["dx"], map_config["ranges"]["x"]), clip_and_scale(rel_data["dy"], map_config["ranges"]["y"]), clip_and_scale(rel_data["dz"], map_config["ranges"]["z"]), math.log(clip_and_scale(rel_data["distance"], [0, 5000]) + 1), rel_data["j_is_teammate"], rel_data["j_is_enemy"], rel_data["j_is_projectile"], rel_data["j_is_spotted_by_i"], rel_data["i_is_spotted_by_j"], math.cos(math.radians(rel_data["d_theta_xy_ij"])), math.sin(math.radians(rel_data["d_theta_xy_ij"])), math.cos(math.radians(rel_data["d_theta_z_ij"])), math.sin(math.radians(rel_data["d_theta_z_ij"]))]
			# 	# result["MLP5_i"][sequence_id][idx] = idx2
			# 	# result["MLP5_mask"][sequence_id][idx] = True
			# 	mlp5_features.append([clip_and_scale(rel_data["dx"], map_config["ranges"]["x"]), clip_and_scale(rel_data["dy"], map_config["ranges"]["y"]), clip_and_scale(rel_data["dz"], map_config["ranges"]["z"]), math.log(clip_and_scale(rel_data["distance"], [0, 5000]) + 1), rel_data["j_is_teammate"], rel_data["j_is_enemy"], rel_data["j_is_projectile"], rel_data["j_is_spotted_by_i"], rel_data["i_is_spotted_by_j"], math.cos(math.radians(rel_data["d_theta_xy_ij"])), math.sin(math.radians(rel_data["d_theta_xy_ij"])), math.cos(math.radians(rel_data["d_theta_z_ij"])), math.sin(math.radians(rel_data["d_theta_z_ij"]))])
			# 	mlp5_index.append(idx2)
			
			# After processing all j for a given i, we can fill in the MLP5 features for that i.
			assert len(mlp5_features) <= 9, f"Number of related entities for player {idx} exceeds 9: {len(mlp5_features)}"
			result["MLP5_f"][sequence_id][idx] = mlp5_features + [[0 for _ in range(13)] for __ in range(9 - len(mlp5_features))]
			result["MLP5_i"][sequence_id][idx] = mlp5_index + [0 for _ in range(9 - len(mlp5_index))]
			result["MLP5_mask"][sequence_id][idx] = [True] * len(mlp5_features) + [False] * (9 - len(mlp5_features))


	return result

def process_one_round(round_json_data, json_name):
	results = []
	for i in range(len(round_json_data)):
		if random.random() > Kepp_rate:
			continue
		results.append(process_one_data(round_json_data[max(0, i - T_Window_Size + 1):i + 1], f"{json_name}_**_{0.5 + i * 0.25}s"))
	return results

def process_one_json(json_bytes: bytes, json_name: str):

	try:
		json_data = json.loads(json_bytes.decode("utf-8"))
	except json.JSONDecodeError as e:
		print(f"[ERROR] Failed to decode JSON in {json_name}: {e}")
		return []

	results = []

	round_groups = {}
	round_order = []
	for idx, entry in enumerate(json_data):
		round_id = entry.get("round")
		if round_id is None:
			print(f"[WARN] {json_name}: item {idx} missing round, skipping.")
			continue
		if round_id not in round_groups:
			round_groups[round_id] = []
			round_order.append(round_id)
		round_groups[round_id].append(entry)

	for round_id in round_order:
		round_data = round_groups[round_id]
		# ✅ 1. 时间过滤
		if not is_valid_round(round_data):
			continue

		try:
			round_results = process_one_round(
				round_data,
				f"{json_name}__round{round_id}",
			)
			results.extend(round_results)
		except Exception as exc:
			print(f"[WARN] {json_name}: round {round_id} failed: {exc}")
			continue

	return results


def main():

	from tqdm import tqdm

	args = parse_args()
	zip_root = Path(args.zip_root)
	output_dir = Path(args.output_dir)
	processed_path = Path(args.processed_list)

	if not zip_root.exists():
		raise FileNotFoundError(f"zip root not found: {zip_root}")

	processed = load_processed(processed_path)
	train_dir = output_dir / "train"
	test_dir = output_dir / "test"
	start_shard_train = find_start_shard(train_dir)
	start_shard_test = find_start_shard(test_dir)

	train_pattern = str(train_dir / "shards-%05d.tar")
	test_pattern = str(test_dir / "shards-%05d.tar")

	# Track which matches are assigned to test per map to avoid leakage.
	map_test_matches = {}
	# Cumulative per-map counters across all processed zips.
	map_occurrence_counts = {"train": {}, "test": {}}
	map_sample_counts = {"train": {}, "test": {}}

	only_append_to_train = getattr(args, "only_append_to_train", False)

	with wds.ShardWriter(
		train_pattern,
		start_shard=start_shard_train,
		maxsize=5 * 1024**3,
	) as train_sink, wds.ShardWriter(
		test_pattern,
		start_shard=start_shard_test,
		maxsize=5 * 1024**3,
	) as test_sink:
		numpy_keys = {
			"MLP1_f",
			"MLP1_i",
			"MLP1_mask",
			"MLP2_f",
			"MLP2_mask",
			"MLP3_f",
			"MLP3_i",
			"MLP3_mask",
			"MLP4_f",
			"MLP4_mask",
			"MLP5_f",
			"MLP5_i",
			"MLP5_mask",
			"EMB1_i",
			"EMB1_mask",
			"EMB2_i",
			"EMB2_mask",
			"DEAD_mask",
			"PAD_mask",
		}

		def get_dtype_for_key(key: str):
			"""根据key的后缀确定dtype"""
			if key.endswith("_f"):  # MLP的float输入
				return np.float32
			elif key.endswith("_i"):  # embedding的int输入
				return np.int32
			elif key.endswith("_mask"):  # 所有mask都是bool
				return bool
			else:
				return None

		def assert_shape(key, arr):
			if key.startswith("MLP1_f"):
				assert arr.shape == (1, 31, 3)
			elif key.startswith("MLP2_f"):
				assert arr.shape == (1, 31, 14)
			elif key.startswith("MLP5_f"):
				assert arr.shape == (1, 31, 9, 13)
			elif key.endswith("_mask"):
				if key == "EMB1_mask":  # EMB1_mask的shape是 (1, 31, 9)
					assert arr.shape == (1, 31, 9)
				elif key == "MLP5_mask":  # MLP5_mask的shape是 (1, 31, 9)
					assert arr.shape == (1, 31, 9)
				else:
					assert arr.shape == (1, 31)

		def encode_dict_fields(sample: dict) -> dict:
			encoded = {}
			for key, value in list(sample.items()):
				out_key = key
				if key in numpy_keys:
					out_key = f"{key}.npy.zst"
					dtype = get_dtype_for_key(key)
					arr = np.asarray(value, dtype=dtype)
					# =========================
					# 🔴 强制检查
					# =========================

					assert_shape(key, arr)

					if arr.dtype == object:
						raise ValueError(
							f"[FATAL] {key} has dtype=object! "
							f"Example value: {str(value)[:200]}"
						)

					# 检查 NaN / Inf
					if np.issubdtype(arr.dtype, np.floating):
						if not np.isfinite(arr).all():
							raise ValueError(f"[FATAL] {key} contains NaN or Inf")

					# 检查 shape（可选但很有用）
					if arr.ndim == 0:
						raise ValueError(f"[FATAL] {key} is scalar, expected tensor")

					# =========================
					# buf = io.BytesIO()
					# np.save(buf, arr, allow_pickle=False)
					# encoded[out_key] = buf.getvalue()
					buf = io.BytesIO()
					np.save(buf, arr, allow_pickle=False)
					compressed = cctx.compress(buf.getvalue())
					encoded[out_key] = compressed

					continue
				if key in {"meta"}:
					out_key = f"{key}.json.zst"
					raw = json.dumps(value).encode("utf-8")
					compressed = cctx.compress(raw)
					encoded[out_key] = compressed
					continue
				if isinstance(value, (dict, list)):
					encoded[out_key] = json.dumps(value).encode("utf-8")
				else:
					encoded[out_key] = value
			return encoded

		for zip_path in tqdm(iter_zip_files(zip_root, args.pattern)):
			# zip_results = []
			with zipfile.ZipFile(zip_path) as zf:
				for info in zf.infolist():
					if info.is_dir():
						continue
					if not info.filename.endswith(".json"):
						continue

					json_name = Path(info.filename).name
					if json_name in processed:
						continue

					json_bytes = zf.read(info)
					if not json_bytes:
						continue

					# print(f"Processing {json_name} from {zip_path.name}...")

					output_samples = process_one_json(json_bytes, json_name)


					if not output_samples:
						print(f"[WARN] No valid samples in {json_name}, skipping.")
						continue

					meta = output_samples[0].get("meta")
					map_name = meta.get("map_name")

					if map_name not in map_config["maps"]:
						print(f"⚠️ Unknown map '{map_name}' in {json_name}, skipping.")
						continue

					match_key = meta.get("source_json") or json_name


					is_test = False
					if not only_append_to_train:
						if map_name:
							assigned = map_test_matches.setdefault(map_name, set())
							if len(assigned) < N_Test_Map:
								assigned.add(match_key)
								is_test = True
							elif match_key in assigned:
								is_test = True

					if isinstance(output_samples, dict):
						output_samples = [output_samples]

					random.shuffle(output_samples)
					if only_append_to_train:
						for sample in output_samples:
							train_sink.write(encode_dict_fields(sample))
						# Update only train counters
						split_key = "train"
						map_occurrence_counts[split_key][map_name] = map_occurrence_counts[split_key].get(map_name, 0) + 1
						map_sample_counts[split_key][map_name] = map_sample_counts[split_key].get(map_name, 0) + len(output_samples)
					else:
						if is_test:
							for sample in output_samples:
								test_sink.write(encode_dict_fields(sample))
						else:
							for sample in output_samples:
								train_sink.write(encode_dict_fields(sample))

						# Update cumulative per-map counters.
						split_key = "test" if is_test else "train"
						map_occurrence_counts[split_key][map_name] = map_occurrence_counts[split_key].get(map_name, 0) + 1
						map_sample_counts[split_key][map_name] = map_sample_counts[split_key].get(map_name, 0) + len(output_samples)

					processed.add(json_name)
					append_processed(processed_path, json_name)

			# test_results = []
			# train_results = []
			# for is_test, output_samples in zip_results:
			# 	if is_test:
			# 		test_results.extend(output_samples)
			# 	else:
			# 		train_results.extend(output_samples)

			# random.shuffle(train_results)
			# random.shuffle(test_results)

			# for sample in train_results:
			# 	train_sink.write(encode_dict_fields(sample))
			# for sample in test_results:
			# 	test_sink.write(encode_dict_fields(sample))

			print("\nCumulative per-map counts after", zip_path.name)
			for split_key in ("train", "test"):
				print(f"[{split_key}]")
				for map_name in sorted(map_occurrence_counts[split_key].keys()):
					print(
						f"  {map_name}: matches={map_occurrence_counts[split_key][map_name]}, "
						f"samples={map_sample_counts[split_key].get(map_name, 0)}"
					)


if __name__ == "__main__":
	main()
