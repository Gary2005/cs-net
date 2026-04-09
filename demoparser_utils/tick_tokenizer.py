# 用tokens表示一个tick
import yaml
import math
import numpy as np

class TickTokenizer:

    """
    [BOS] [MAP] [time] [c4_planted] [c4_dropped] [c4_planted_duration] [c4_x] [c4_y] [c4_z] [player_i] [armor] [defuser] [x] [y] [z] [pitch] [yaw] [health] [is_blind] [inventory_1] ... [inventory_k] [team] ... [projectile_j] [duration] [x] [y] [z] [entity_projectile_j] [x] [y] [z] ... [EOS]
    projectile: 落地的道具
    entity_projectile: 道具实体
    armor: 0: 无护甲, 1: 半护甲, 2: 全护甲
    defuser: 0: 无拆弹器, 1: 有拆弹器
    """

    def __init__(self, config):
        self.config = config

        self.safe_mode = config.get("safe_mode", 1)

        self.weapons = config["weapons"]
        self.projectiles = config["projectiles"]
        self.entity_projectiles = config["entity_projectiles"]

        self.entity_projectiles2idx = {name: i for i, name in enumerate(self.entity_projectiles)}
        self.idx2entity_projectile = {i: name for i, name in enumerate(self.entity_projectiles)}

        self.weapon2idx = {name: i for i, name in enumerate(self.weapons)}
        self.idx2weapon = {i: name for i, name in enumerate(self.weapons)}

        self.projectile2idx = {name: i for i, name in enumerate(self.projectiles)}
        self.idx2projectile = {i: name for i, name in enumerate(self.projectiles)}

        self.map2idx = {name: i for i, name in enumerate(self.config["maps"])}

        self.time_grid = config["tokenizer"]["status"]["time_grid"]
        self.health_grid = config["tokenizer"]["status"]["health_grid"]
        self.x_grid = config["tokenizer"]["position"]["x"]["grid"]
        self.y_grid = config["tokenizer"]["position"]["y"]["grid"]
        self.z_grid = config["tokenizer"]["position"]["z"]["grid"]
        self.pitch_grid = config["tokenizer"]["view_angle"]["pitch"]["grid"]
        self.yaw_grid = config["tokenizer"]["view_angle"]["yaw"]["grid"]
        self.planted_duration_grid = config["tokenizer"]["planted_duration"]["grid"]

        self.x_range = config["tokenizer"]["position"]["x"]["range"]
        self.y_range = config["tokenizer"]["position"]["y"]["range"]
        self.z_range = config["tokenizer"]["position"]["z"]["range"]
        self.pitch_range = config["tokenizer"]["view_angle"]["pitch"]["range"]
        self.yaw_range = config["tokenizer"]["view_angle"]["yaw"]["range"]
        self.planted_duration_range = config["tokenizer"]["planted_duration"]["range"]

        self.x_n_grids = int(math.floor((self.x_range[1] - self.x_range[0]) / self.x_grid)) # grid id: 0 ~ n
        self.y_n_grids = int(math.floor((self.y_range[1] - self.y_range[0]) / self.y_grid))
        self.z_n_grids = int(math.floor((self.z_range[1] - self.z_range[0]) / self.z_grid))
        self.pitch_n_grids = int(math.floor((self.pitch_range[1] - self.pitch_range[0]) / self.pitch_grid))
        self.yaw_n_grids = int(math.floor((self.yaw_range[1] - self.yaw_range[0]) / self.yaw_grid))
        self.planted_duration_n_grids = int(math.floor((self.planted_duration_range[1] - self.planted_duration_range[0]) / self.planted_duration_grid))

        self.x_block = int(math.sqrt(self.x_n_grids))
        self.y_block = int(math.sqrt(self.y_n_grids))
        self.z_block = int(math.sqrt(self.z_n_grids))
        self.pitch_block = int(math.sqrt(self.pitch_n_grids))
        self.yaw_block = int(math.sqrt(self.yaw_n_grids))
        self.planted_duration_block = int(math.sqrt(self.planted_duration_n_grids))

        self.time_offset = 0
        self.time_total = self.config["tokenizer"]["status"]["max_time"]//self.time_grid + 1 

        self.player_id_offset = self.time_offset + self.time_total
        self.player_id_total = self.config["tokenizer"]["status"]["n_players"]

        self.armor_offset = self.player_id_offset + self.player_id_total
        self.armor_total = 3

        self.defuser_offset = self.armor_offset + self.armor_total
        self.defuser_total = 2

        self.x_offset = self.defuser_offset + self.defuser_total
        self.x_total = self.x_block + self.x_n_grids//self.x_block + 1

        self.y_offset = self.x_offset + self.x_total
        self.y_total = self.y_block + self.y_n_grids//self.y_block + 1

        self.z_offset = self.y_offset + self.y_total
        self.z_total = self.z_block + self.z_n_grids//self.z_block + 1

        self.pitch_offset = self.z_offset + self.z_total
        self.pitch_total = self.pitch_block + self.pitch_n_grids//self.pitch_block + 1

        self.yaw_offset = self.pitch_offset + self.pitch_total
        self.yaw_total = self.yaw_block + self.yaw_n_grids//self.yaw_block + 1

        self.health_offset = self.yaw_offset + self.yaw_total
        self.health_total = int(self.config["tokenizer"]["status"]["max_health"]/self.health_grid) + 1

        self.blind_offset = self.health_offset + self.health_total
        self.blind_total = 2 # 0, 1

        self.weapon_offset = self.blind_offset + self.blind_total
        self.weapon_total = len(self.weapons)

        self.team_offset = self.weapon_offset + self.weapon_total
        self.team_total = 2 # T, CT

        self.projectile_offset = self.team_offset + self.team_total
        self.projectile_total = len(self.projectiles)

        self.entity_projectile_offset = self.projectile_offset + self.projectile_total
        self.entity_projectile_total = len(self.entity_projectiles)

        self.map_token_offset = self.entity_projectile_offset + self.entity_projectile_total
        self.map_total = len(self.config["maps"])

        self.c4_planted_offset = self.map_token_offset + self.map_total
        self.c4_planted_total = 2

        self.c4_dropped_offset = self.c4_planted_offset + self.c4_planted_total
        self.c4_dropped_total = 2

        self.c4_planted_duration_offset = self.c4_dropped_offset + self.c4_dropped_total
        self.c4_planted_duration_total = self.planted_duration_block + self.planted_duration_n_grids//self.planted_duration_block + 1

        self.BOS = self.c4_planted_duration_offset + self.c4_planted_duration_total
        self.EOS = self.BOS + 1
        self.SEP = self.EOS + 1
        self.PAD = self.SEP + 1

    def grids_tokens(self, value, block_size, grid_size, range) -> tuple[int, int]:
        # clip to range

        if self.safe_mode:
            assert range[0] <= value <= range[1], f"Value {value} out of range {range}"

        if value < range[0]:
            value = range[0]
        if value > range[1]:
            value = range[1]

        value = value - range[0]
        value = value // grid_size
        value = int(value)
        block_id = value // block_size
        block_offset = value % block_size
        return (block_id + block_size, block_offset)
    
    def original_value(self, block_id, block_offset, block_size, grid_size, range) -> int:
        value = block_id * block_size + block_offset
        value = value * grid_size + range[0]
        return value
    

    def check_token_type(self, token) -> str:
        if token < self.player_id_offset:
            return "time"
        elif token < self.armor_offset:
            return "player_id"
        elif token < self.defuser_offset:
            return "armor"
        elif token < self.x_offset:
            return "defuser"
        elif token < self.y_offset:
            return "x"
        elif token < self.z_offset:
            return "y"
        elif token < self.pitch_offset:
            return "z"
        elif token < self.yaw_offset:
            return "pitch"
        elif token < self.health_offset:
            return "yaw"
        elif token < self.blind_offset:
            return "health"
        elif token < self.weapon_offset:
            return "is_blind"
        elif token < self.team_offset:
            return "weapon_name"
        elif token < self.projectile_offset:
            return "team"
        elif token < self.entity_projectile_offset:
            return "projectile"
        elif token < self.map_token_offset:
            return "entity_projectile"
        elif token < self.c4_planted_offset:
            return "map_token"
        elif token < self.c4_dropped_offset:
            return "c4_planted"
        elif token < self.c4_planted_duration_offset:
            return "c4_dropped"
        elif token < self.BOS:
            return "c4_planted_duration"
        elif token < self.EOS:
            return "BOS"
        elif token < self.SEP:
            return "EOS"
        elif token < self.SEP + 1:
            return "SEP"
        elif token < self.PAD + 1:
            return "PAD"
        else:
            raise ValueError(f"Unknown token type for token: {token}")
        

    def vocab_size(self) -> int:
        return self.PAD + 1
        
    def weapon_names_to_indices(self, names: list[str]) -> int:
        names_new = []
        for name in names:
            if name not in self.weapons:
                name = "knife"
            names_new.append(name)
        
        indices = [self.weapon2idx[name] for name in names_new]
        indices = sorted(indices)
        return indices
    
    def weapon_name_to_index(self, name: str) -> int:
        if name not in self.weapons:
            name = "knife"
        return self.weapon2idx[name]    
        
    def tokenize(self, info: dict) -> list[int]:
        self.steamid2idx = {}
        for i, player in enumerate(info["players_info"]):
            self.steamid2idx[player["steamid"]] = i

        tokens = []
        tokens.append(self.BOS)
        tokens.append(self.map_token_offset + self.map2idx[info["map_name"]])
        tokens.append(self.time_offset + int(min(max(0, info["round_seconds"]), self.config["tokenizer"]["status"]["max_time"]) // self.time_grid))

        tokens.append(self.c4_planted_offset + info["is_bomb_planted"])
        tokens.append(self.c4_dropped_offset + info["is_bomb_dropped"])
        if info["is_bomb_planted"]:
            # assert bomb_planted_duration is not null
            assert info["bomb_planted_duration"] is not None
            values = self.grids_tokens(info["bomb_planted_duration"], self.planted_duration_block, self.planted_duration_grid, self.planted_duration_range)
            tokens.extend([self.c4_planted_duration_offset + values[0], self.c4_planted_duration_offset + values[1]])



        center = self.config["maps"][info["map_name"]]["center"]

        c4_x_values = self.grids_tokens(info["bomb_position"][0] - center[0], self.x_block, self.x_grid, self.x_range)
        c4_y_values = self.grids_tokens(info["bomb_position"][1] - center[1], self.y_block, self.y_grid, self.y_range)
        c4_z_values = self.grids_tokens(info["bomb_position"][2] - center[2], self.z_block, self.z_grid, self.z_range)

        tokens.extend([self.x_offset + c4_x_values[0], self.x_offset + c4_x_values[1]])
        tokens.extend([self.y_offset + c4_y_values[0], self.y_offset + c4_y_values[1]])
        tokens.extend([self.z_offset + c4_z_values[0], self.z_offset + c4_z_values[1]])

        for player_info in info["players_info"]:
            if player_info["is_alive"] == False:
                continue
            player_idx = self.steamid2idx[player_info["steamid"]]
            tokens.append(self.player_id_offset + player_idx)

            has_armor = player_info["armor"] > 0
            has_helmet = player_info["has_helmet"]
            has_defuser = player_info["has_defuser"]

            armor_level = 0
            if has_armor:
                if has_helmet:
                    armor_level = 2
                else:
                    armor_level = 1
            tokens.append(self.armor_offset + armor_level)
            tokens.append(self.defuser_offset + (1 if has_defuser else 0))

            values = self.grids_tokens(player_info["X"] - center[0], self.x_block, self.x_grid, self.x_range)
            tokens.extend([self.x_offset + values[0], self.x_offset + values[1]])

            values = self.grids_tokens(player_info["Y"] - center[1], self.y_block, self.y_grid, self.y_range)
            tokens.extend([self.y_offset + values[0], self.y_offset + values[1]])

            values = self.grids_tokens(player_info["Z"] - center[2], self.z_block, self.z_grid, self.z_range)
            tokens.extend([self.z_offset + values[0], self.z_offset + values[1]])

            values = self.grids_tokens(player_info["pitch"], self.pitch_block, self.pitch_grid, self.pitch_range)
            tokens.extend([self.pitch_offset + values[0], self.pitch_offset + values[1]])

            values = self.grids_tokens(player_info["yaw"], self.yaw_block, self.yaw_grid, self.yaw_range)
            tokens.extend([self.yaw_offset + values[0], self.yaw_offset + values[1]])

            tokens.append(self.health_offset + player_info["health"] // self.health_grid)
            
            is_blind_token = 1 if player_info["flash_duration"] > 0 else 0
            tokens.append(self.blind_offset + is_blind_token)

            inventory_indices = self.weapon_names_to_indices(player_info["inventory"])
            for idx in inventory_indices:
                tokens.append(self.weapon_offset + idx)

            team_token = 0 if player_info["team_num"] == "T" else 1
            tokens.append(self.team_offset + team_token)

        for projectile_info in info["projectiles"]:
            projectile_token = self.projectile2idx[projectile_info["type"]]
            tokens.append(self.projectile_offset + projectile_token)
            # duration (shared with planted_duration)
            values = self.grids_tokens(projectile_info["duration"], self.planted_duration_block, self.planted_duration_grid, self.planted_duration_range)

            tokens.extend([self.c4_planted_duration_offset + values[0], self.c4_planted_duration_offset + values[1]])

            values = self.grids_tokens(projectile_info["position"][0] - center[0], self.x_block, self.x_grid, self.x_range)
            tokens.extend([self.x_offset + values[0], self.x_offset + values[1]])
            values = self.grids_tokens(projectile_info["position"][1] - center[1], self.y_block, self.y_grid, self.y_range)
            tokens.extend([self.y_offset + values[0], self.y_offset + values[1]])
            values = self.grids_tokens(projectile_info["position"][2] - center[2], self.z_block, self.z_grid, self.z_range)
            tokens.extend([self.z_offset + values[0], self.z_offset + values[1]])

        for entity_projectile_info in info["entity_grenades"]:
            entity_projectile_token = self.entity_projectiles2idx[entity_projectile_info["type"]]
            tokens.append(self.entity_projectile_offset + entity_projectile_token)
            values = self.grids_tokens(entity_projectile_info["position"][0] - center[0], self.x_block, self.x_grid, self.x_range)
            tokens.extend([self.x_offset + values[0], self.x_offset + values[1]])
            values = self.grids_tokens(entity_projectile_info["position"][1] - center[1], self.y_block, self.y_grid, self.y_range)
            tokens.extend([self.y_offset + values[0], self.y_offset + values[1]])
            values = self.grids_tokens(entity_projectile_info["position"][2] - center[2], self.z_block, self.z_grid, self.z_range)
            tokens.extend([self.z_offset + values[0], self.z_offset + values[1]])

        tokens.append(self.EOS)
        return tokens
    
    def detokenize(self, tokens: list[int]) -> dict:
        info = {
            "map_name": "",
            "round_seconds": 0,
            "players_info": [],
            "projectiles": [],
            "entity_grenades": []
        }

        # 辅助函数：从两个 token 中解码坐标数值
        def decode_grid_value(token1, token2, offset, block_size, grid_size, range_val: list) -> float:
            # token1 对应 (block_id + block_size)
            # token2 对应 (block_offset)
            val1 = token1 - offset
            val2 = token2 - offset
            
            block_id = val1 - block_size
            block_offset = val2
            
            return self.original_value(block_id, block_offset, block_size, grid_size, range_val)

        # 构建 map 的反向索引 (如果 __init__ 中没有 idx2map)
        idx2map = {v: k for k, v in self.map2idx.items()}

        idx = 0
        n = len(tokens)

        # 1. 跳过 BOS
        if idx < n and tokens[idx] == self.BOS:
            idx += 1

        # 2. 解析 Map
        if idx < n and self.map_token_offset <= tokens[idx] < self.map_token_offset + self.map_total:
            map_id = tokens[idx] - self.map_token_offset
            info["map_name"] = idx2map.get(map_id, "unknown")
            idx += 1

        # 3. 解析 Time
        if idx < n and self.time_offset <= tokens[idx] < self.player_id_offset:
            # time_token = offset + time // grid
            info["round_seconds"] = (tokens[idx] - self.time_offset) * self.time_grid
            idx += 1

        # 4. 解析 C4 状态
        if idx < n and self.c4_planted_offset <= tokens[idx] < self.c4_planted_offset + self.c4_planted_total:
            info["is_bomb_planted"] = (tokens[idx] - self.c4_planted_offset) == 1
            idx += 1
        if idx < n and self.c4_dropped_offset <= tokens[idx] < self.c4_dropped_offset + self.c4_dropped_total:
            info["is_bomb_dropped"] = (tokens[idx] - self.c4_dropped_offset) == 1
            idx += 1

        if idx < n and self.c4_planted_duration_offset <= tokens[idx] < self.c4_planted_duration_offset + self.c4_planted_duration_total:
            info["bomb_planted_duration"] = decode_grid_value(tokens[idx], tokens[idx+1], self.c4_planted_duration_offset, self.planted_duration_block, self.planted_duration_grid, self.planted_duration_range)
            idx += 2
        c4_position = [None, None, None]
        c4_position[0] = decode_grid_value(tokens[idx], tokens[idx+1], self.x_offset, self.x_block, self.x_grid, self.x_range) + self.config["maps"][info["map_name"]]["center"][0]
        idx += 2
        c4_position[1] = decode_grid_value(tokens[idx], tokens[idx+1], self.y_offset, self.y_block, self.y_grid, self.y_range) + self.config["maps"][info["map_name"]]["center"][1]
        idx += 2
        c4_position[2] = decode_grid_value(tokens[idx], tokens[idx+1], self.z_offset, self.z_block, self.z_grid, self.z_range) + self.config["maps"][info["map_name"]]["center"][2]
        idx += 2

        info["bomb_position"] = c4_position

        # 5. 循环解析实体 (Players, Projectiles, entity Projectiles)
        while idx < n:
            token = tokens[idx]
            
            # 遇到结束符或分隔符停止
            if token == self.EOS or token == self.SEP:
                break

            token_type = self.check_token_type(token)

            if token_type == "player_id":
                player = {}
                # 恢复 Player ID (注意：无法恢复原始 SteamID，只能恢复索引)
                p_idx = token - self.player_id_offset
                player["steamid"] = f"player_{p_idx}" # 使用占位符
                player["is_alive"] = True
                idx += 1

                # 解析 Armor
                armor_val = tokens[idx] - self.armor_offset
                if armor_val == 0:
                    player["armor"] = False
                    player["has_helmet"] = False
                elif armor_val == 1:
                    player["armor"] = True
                    player["has_helmet"] = False
                elif armor_val == 2:
                    player["armor"] = True
                    player["has_helmet"] = True
                idx += 1

                # 解析 Defuser
                defuser_val = tokens[idx] - self.defuser_offset
                player["has_defuser"] = defuser_val == 1
                idx += 1

                # 解析位置 X, Y, Z (每个坐标占 2 个 token)
                player["X"] = decode_grid_value(tokens[idx], tokens[idx+1], self.x_offset, self.x_block, self.x_grid, self.x_range) + self.config["maps"][info["map_name"]]["center"][0]
                idx += 2
                player["Y"] = decode_grid_value(tokens[idx], tokens[idx+1], self.y_offset, self.y_block, self.y_grid, self.y_range) + self.config["maps"][info["map_name"]]["center"][1]
                idx += 2
                player["Z"] = decode_grid_value(tokens[idx], tokens[idx+1], self.z_offset, self.z_block, self.z_grid, self.z_range) + self.config["maps"][info["map_name"]]["center"][2]
                idx += 2

                # 解析视角 Pitch, Yaw
                player["pitch"] = decode_grid_value(tokens[idx], tokens[idx+1], self.pitch_offset, self.pitch_block, self.pitch_grid, self.pitch_range)
                idx += 2
                player["yaw"] = decode_grid_value(tokens[idx], tokens[idx+1], self.yaw_offset, self.yaw_block, self.yaw_grid, self.yaw_range)
                idx += 2

                # 解析 Health
                player["health"] = (tokens[idx] - self.health_offset) * self.health_grid
                idx += 1

                # 解析 Blind
                is_blind = tokens[idx] - self.blind_offset
                player["flash_duration"] = 2.0 if is_blind == 1 else 0.0 # 只能恢复是否有致盲状态，无法恢复具体秒数
                idx += 1

                # 解析 Inventory (变长，直到遇到 Team Token)
                player["inventory"] = []
                while idx < n:
                    inv_token = tokens[idx]
                    if self.weapon_offset <= inv_token < self.team_offset:
                        w_idx = inv_token - self.weapon_offset
                        player["inventory"].append(self.idx2weapon[w_idx])
                        idx += 1
                    else:
                        break # 遇到 Team Token 或其他，跳出循环

                # 解析 Team
                if idx < n and self.team_offset <= tokens[idx] < self.team_offset + self.team_total:
                    team_val = tokens[idx] - self.team_offset
                    player["team_num"] = "T" if team_val == 0 else "CT"
                    idx += 1
                
                info["players_info"].append(player)

            elif token_type == "projectile":
                proj = {}
                p_type_idx = token - self.projectile_offset
                proj["type"] = self.idx2projectile.get(p_type_idx, "unknown")
                idx += 1

                # duration (shared with planted_duration)
                proj["duration"] = decode_grid_value(tokens[idx], tokens[idx+1], self.c4_planted_duration_offset, self.planted_duration_block, self.planted_duration_grid, self.planted_duration_range)
                idx += 2

                proj["position"] = [0.0, 0.0, 0.0]
                proj["position"][0] = decode_grid_value(tokens[idx], tokens[idx+1], self.x_offset, self.x_block, self.x_grid, self.x_range) + self.config["maps"][info["map_name"]]["center"][0]
                idx += 2
                proj["position"][1] = decode_grid_value(tokens[idx], tokens[idx+1], self.y_offset, self.y_block, self.y_grid, self.y_range) + self.config["maps"][info["map_name"]]["center"][1]
                idx += 2
                proj["position"][2] = decode_grid_value(tokens[idx], tokens[idx+1], self.z_offset, self.z_block, self.z_grid, self.z_range) + self.config["maps"][info["map_name"]]["center"][2]
                idx += 2
                
                info["projectiles"].append(proj)

            elif token_type == "entity_projectile":
                fly = {}
                f_type_idx = token - self.entity_projectile_offset
                fly["type"] = self.idx2entity_projectile.get(f_type_idx, "unknown")
                idx += 1

                fly["position"] = [0.0, 0.0, 0.0]
                fly["position"][0] = decode_grid_value(tokens[idx], tokens[idx+1], self.x_offset, self.x_block, self.x_grid, self.x_range) + self.config["maps"][info["map_name"]]["center"][0]
                idx += 2
                fly["position"][1] = decode_grid_value(tokens[idx], tokens[idx+1], self.y_offset, self.y_block, self.y_grid, self.y_range) + self.config["maps"][info["map_name"]]["center"][1]
                idx += 2
                fly["position"][2] = decode_grid_value(tokens[idx], tokens[idx+1], self.z_offset, self.z_block, self.z_grid, self.z_range) + self.config["maps"][info["map_name"]]["center"][2]
                idx += 2
                
                info["entity_grenades"].append(fly)

            else:
                raise ValueError(f"Unexpected token type during detokenization: {token_type}")
        
        return info

    def get_tokens_type(self, tokens: list[int]) -> list[str]:
        types = []
        for token in tokens:
            types.append(f"<{self.check_token_type(token)}>")
        return types
        


if __name__ == "__main__":
    with open("demoparser_utils/tokenizer.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    print(config)

    tokenizer = TickTokenizer(config)
    print(f"Vocab size: {tokenizer.vocab_size()}")
    print(f"Pad token id: {tokenizer.PAD}")

    import json
    test_json = json.load(open("test.json", "r"))
    print(len(test_json))

    tokens = tokenizer.tokenize(test_json[0])

    print(test_json[0])

    print(len(tokens))
    print(tokens)
    print(tokenizer.get_tokens_type(tokens))
    info_recovered = tokenizer.detokenize(tokens)
    print(info_recovered)