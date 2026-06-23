import random
from collections import deque
import numpy as np
import heapq

class Agent:
    MOVES = {
        0: (0, 0),    # STOP
        1: (-1, 0),   # LEFT
        2: (1, 0),    # RIGHT
        3: (0, -1),   # UP
        4: (0, 1),    # DOWN
    }
    team_id = "Talon"

    def __init__(self, agent_id: int):
        self.agent_id = int(agent_id)
        self._bomb_radius_memory = {}
        self._prev_player_positions = {}
        self._initial_box_count = 0

    def act(self, obs: dict) -> int:
        grid = obs["map"]
        players = obs["players"]
        bombs = obs["bombs"]
        self._current_bombs = bombs

        # Clear/initialize simulation cache for this step
        self._cache_effective_bombs = {}
        self._cache_grid_danger = {}

        # Verify alive status
        if self.agent_id >= len(players) or players[self.agent_id][2] != 1:
            return 0

        # Extract self information
        my_x, my_y, _, bombs_left, bomb_bonus = players[self.agent_id]
        my_pos = (int(my_x), int(my_y))
        if not hasattr(self, '_spawn_pos') or self._spawn_pos is None:
            self._spawn_pos = my_pos
        bomb_radius = max(1, int(bomb_bonus) + 1)
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        self._sync_bomb_radius_memory(bombs, players)

        # Track enemy velocities/directions
        self._enemy_velocities = {}
        for idx, p in enumerate(players):
            if idx == self.agent_id or p[2] != 1:
                continue
            curr_pos = (int(p[0]), int(p[1]))
            prev_pos = self._prev_player_positions.get(idx)
            if prev_pos is not None:
                dx = curr_pos[0] - prev_pos[0]
                dy = curr_pos[1] - prev_pos[1]
                self._enemy_velocities[idx] = (dx, dy)
            self._prev_player_positions[idx] = curr_pos

        # Enemies may overlap in the engine, so enemies are risk, not solid blockers
        enemies = [
            (int(p[0]), int(p[1]))
            for i, p in enumerate(players)
            if i != self.agent_id and p[2] == 1
        ]

        my_bombs = [(int(b[0]), int(b[1])) for b in bombs if len(b) > 3 and int(b[3]) == self.agent_id]
        nearest_enemy_dist = min([abs(my_pos[0] - ep[0]) + abs(my_pos[1] - ep[1]) for ep in enemies], default=999)
        combat_mode = (4 <= nearest_enemy_dist <= 6)
        is_consecutive = combat_mode and any(abs(my_pos[0] - bx) + abs(my_pos[1] - by) == 1 for bx, by in my_bombs)

        # Check for active enemy skirmish (2 or more enemies fighting within distance <= 4)
        skirmish_enemies = set()
        for i in range(len(enemies)):
            for j in range(i + 1, len(enemies)):
                dist = abs(enemies[i][0] - enemies[j][0]) + abs(enemies[i][1] - enemies[j][1])
                if dist <= 4:
                    skirmish_enemies.add(enemies[i])
                    skirmish_enemies.add(enemies[j])

        occupied = set()
        tactical_avoid = set()
        
        # Detonation and danger analysis
        effective_bombs = self._get_effective_bombs(grid, bombs, players)
        box_destroyed_time, blast_at_time, bombs_at_time = self._simulate_grid_and_danger(grid, effective_bombs, 18)
        
        danger_deadline = self._danger_deadlines(blast_at_time)
        danger_soon = set(danger_deadline)
        danger_now = {p for p, t in danger_deadline.items() if t <= 1}
        danger_avoid = {p for p, t in danger_deadline.items() if t <= 5}
        my_deadline = danger_deadline.get(my_pos, 999)

        is_camping_blast = False
        if danger_soon and my_deadline > 5:
            if any(self._next_pos(my_pos, a) in danger_soon for a in [1, 2, 3, 4]):
                is_camping_blast = True


        # Bottleneck cache and enemy distance precomputations
        self._bottleneck_cache = {}
        enemy_dists = {}
        escaping_enemies = {}
        for idx, p in enumerate(players):
            if idx == self.agent_id or p[2] != 1:
                continue
            ex, ey = int(p[0]), int(p[1])
            enemy_dists[idx] = self._compute_all_dists(grid, (ex, ey), extra_blocked=bomb_positions)
            escaping_path = self._find_escape_path_tiles(grid, (ex, ey), bombs, players)
            if escaping_path is not None:
                escaping_enemies[idx] = escaping_path

        # Cache variables for the spacing override check
        self._temp_tactical_avoid = tactical_avoid
        self._temp_enemy_dists = enemy_dists

        alive_count = 1 + len(enemies)
        box_count = int(np.sum(grid == 2))

        # ESCAPE MODE
        if my_deadline <= 4:
            escape = self._find_safe_path(grid, my_pos, bombs, players, occupied_at_t1=tactical_avoid, enemy_dists=enemy_dists)
            if escape is not None:
                return escape
            
            # Fallback: Ignore tactical avoid to prevent unnecessary self destruction
            escape_no_avoid = self._find_safe_path(grid, my_pos, bombs, players, occupied_at_t1=None, enemy_dists=enemy_dists)
            if escape_no_avoid is not None:
                return escape_no_avoid
                
            # Kamikaze
            if bombs_left > 0 and my_pos not in bomb_positions:
                kamikaze_bomb = {'pos': my_pos, 'timer': 7, 'radius': bomb_radius}
                my_blast_k = self._get_blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)
                for enemy_pos in enemies:
                    if enemy_pos in my_blast_k:
                        enemy_esc = self._find_safe_path(
                            grid, enemy_pos, bombs, players, simulated_bomb=kamikaze_bomb,
                            occupied_at_t1=None, require_permanent_safety=True,
                            enemy_dists=None
                        )
                        if enemy_esc is None:
                            return 5  # Guaranteed mutual kill

                # Even without guaranteed kill, if any enemy is in blast then pressure them
                if any(ep in my_blast_k for ep in enemies):
                    return 5

                # No enemy in blast but dying anyway then place bomb to destroy boxes/do chain react/deny resources
                boxes_in_blast = sum(1 for bx, by in my_blast_k if grid[bx, by] == 2)
                chain_bombs = any((int(b[0]), int(b[1])) in my_blast_k for b in bombs)
                if boxes_in_blast > 0 or chain_bombs:
                    return 5

            # Panic fallback: Pick the action that maximizes survival time (furthest detonation time). If there's a tie, choose the action that brings us closest to an enemy or stands on the same to drag them down with us or force a stats/double-death tie-break!
            best_a = 0
            max_det_time = -1
            best_tiebreak_score = -999999
            for a in [1, 2, 3, 4, 0]:
                nx, ny = self._next_pos(my_pos, a)
                if a != 0:
                    if not self._is_passable_at(grid, nx, ny, 1, box_destroyed_time) or (nx, ny) in bomb_positions:
                        continue
                t_det = 999
                for step in range(0, 18):
                    if (nx, ny) in blast_at_time.get(step, set()):
                        t_det = step
                        break
                
                # Compute tiebreak score: Stand on same tile or get as close as possible
                tiebreak_score = 0
                if enemies:
                    dists_to_enemies = [abs(nx - ep[0]) + abs(ny - ep[1]) for ep in enemies]
                    min_d = min(dists_to_enemies)
                    if min_d == 0:
                        tiebreak_score += 10000
                    tiebreak_score -= min_d * 10

                # Prefer moves that destroy boxes
                for bx, by in self._get_blast_tiles(grid, nx, ny, bomb_radius):
                    if grid[bx, by] == 2:
                        tiebreak_score += 1

                if t_det > max_det_time:
                    max_det_time = t_det
                    best_a = a
                    best_tiebreak_score = tiebreak_score
                elif t_det == max_det_time:
                    if tiebreak_score > best_tiebreak_score:
                        best_tiebreak_score = tiebreak_score
                        best_a = a
            return best_a

        # Count initial boxes at step 0
        if not hasattr(self, '_initial_box_count') or self._initial_box_count == 0:
            self._initial_box_count = max(1, box_count)

        pvp_gate_threshold = max(4, int(0.25 * self._initial_box_count))
        endgame_gate_threshold = max(2, int(0.15 * self._initial_box_count))

        # Late-game awareness: Adjust aggression
        late_game = box_count <= pvp_gate_threshold
        very_late_game = box_count <= endgame_gate_threshold

        # In very late game, if we have more kills/boxes then be more conservative about risky bomb placements.
        defensive_mode = very_late_game and alive_count <= 2

        # High value finish: Take guaranteed kills before resource collection.
        if bombs_left > 0 and my_pos not in bomb_positions:
            if self._should_place_finisher_bomb(
                grid, my_pos, bombs, players, tactical_avoid, enemy_dists,
                bomb_radius, bombs_left, alive_count, box_count
            ):
                return 5
        # Close combat bomb overload: Place consecutive bombs to pressure/trap enemy
        if bombs_left > 0 and my_pos not in bomb_positions:
            if self._should_overload_enemy_with_bombs(
                grid, my_pos, bombs, players, tactical_avoid, enemies, enemy_dists,
                bomb_radius, bombs_left
            ):
                return 5

        # Early game box destruction: Boxes dont respawn so prioritize destroying them in the early game
        if bombs_left > 0 and my_pos not in bomb_positions and my_deadline > 5 and not is_consecutive and not defensive_mode and not is_camping_blast:
            # Check if current position has 1+ boxes that are not already about to explode
            my_blast = self._get_blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)
            boxes_at_current = sum(1 for x, y in my_blast if grid[x, y] == 2 and ((x, y) not in box_destroyed_time or box_destroyed_time[(x, y)] >= 7))
            
            if boxes_at_current >= 1:
                # Do not spend a bomb on a 1-box spot if a better spot (hitting 2+ boxes) is reachable within 2 steps
                has_better_spot_nearby = False
                if boxes_at_current == 1:
                    for spot in self._box_bomb_spots(grid, occupied, bomb_radius):
                        if spot == my_pos:
                            continue
                        d = self._shortest_dist(grid, my_pos, spot)
                        if d > 2:
                            continue
                        s_blast = self._get_blast_tiles(grid, spot[0], spot[1], bomb_radius)
                        s_boxes = sum(1 for sx, sy in s_blast if grid[sx, sy] == 2 and ((sx, sy) not in box_destroyed_time or box_destroyed_time[(sx, sy)] >= 7 + d))
                        if s_boxes > 1:
                            has_better_spot_nearby = True
                            break
                            
                if not has_better_spot_nearby:
                    # The agent is on a box spot then place bomb there
                    simulated_bomb = {'pos': my_pos, 'timer': 7, 'radius': bomb_radius}
                    escape = self._find_safe_path(grid, my_pos, bombs, players, simulated_bomb=simulated_bomb, 
                                                 occupied_at_t1=tactical_avoid, require_permanent_safety=True, 
                                                 enemy_dists=enemy_dists)
                    if escape is not None:
                        return 5

        # Tie break optimization: Coordinated dual bomb box destruction
        if bombs_left >= 2 and my_pos not in bomb_positions and my_deadline > 5 and not is_camping_blast:
            if alive_count <= 2 or box_count <= int(0.40 * self._initial_box_count):
                combo = self._find_coordinated_dual_bomb_placement(
                    grid, my_pos, bombs, players, tactical_avoid, bomb_radius, 
                    bombs_left, box_count, danger_avoid
                )
                if combo and combo[2] >= 3:  # combo = (pos_B2, timing, total_boxes)
                    # Place first bomb for dual bomb combo
                    if my_pos not in bomb_positions:
                        return 5

        # Quick item grab: Pick up any item within 2 steps before speculative traps
        if my_deadline > 5:
            nearby_items = {
                (x, y) for x in range(grid.shape[0]) for y in range(grid.shape[1])
                if grid[x, y] in [3, 4] and abs(x - my_pos[0]) + abs(y - my_pos[1]) <= 2
            }
            if nearby_items:
                move = self._find_safe_path_to_targets(
                    grid, my_pos, nearby_items, bombs, players, max_depth=2,
                    occupied_at_t1=tactical_avoid, danger_soon=danger_avoid, enemy_dists=enemy_dists
                )
                if move is not None:
                    # Continue checking if we can bomb a box on the way
                    if bombs_left > 0 and my_pos not in bomb_positions and not is_consecutive and not is_camping_blast:
                        simulated_bomb = {'pos': my_pos, 'timer': 7, 'radius': bomb_radius}
                        esc = self._find_safe_path(grid, my_pos, bombs, players, simulated_bomb=simulated_bomb, occupied_at_t1=tactical_avoid, require_permanent_safety=True, enemy_dists=enemy_dists)
                        if esc is not None:
                            bh = sum(1 for x, y in self._get_blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius) if grid[x, y] == 2 and ((x, y) not in box_destroyed_time or box_destroyed_time[(x, y)] >= 7))
                            if bh >= 1:
                                return 5
                    return move

        # Triple bomb trap: Trigger sequential 3 bomb spaced trapping sequence
        if bombs_left >= 3 and my_pos not in bomb_positions and not defensive_mode:
            if self._should_place_triple_bomb_trap(
                grid, my_pos, bombs, players, tactical_avoid, enemy_dists,
                bomb_radius, bombs_left, alive_count, box_count
            ):
                return 5

        # Double bomb trap: trigger offensive trapping using sequential bombs
        if bombs_left >= 2 and my_pos not in bomb_positions and not defensive_mode:
            if self._should_place_double_bomb_trap(
                grid, my_pos, bombs, players, tactical_avoid, enemy_dists,
                bomb_radius, bombs_left, alive_count, box_count
            ):
                return 5

        # Item ambush: Place bomb to intercept enemies racing toward nearby items, but never bomb items we actually need (capacity when low bombs, radius when low radius)
        if bombs_left > 0 and my_pos not in bomb_positions and my_deadline > 5 and not is_consecutive:
            my_blast = self._get_blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)
            # Determine which item types we need and must not destroy
            wanted_values = set()
            if bombs_left <= 1:
                wanted_values.add(4)  # Capacity item
            if bomb_radius <= 2:
                wanted_values.add(3)  # Radius item
            # Abort ambush if any needed item is in our blast zone 
            blast_has_needed_item = any(
                grid[x, y] in wanted_values for x, y in my_blast
            )
            if not blast_has_needed_item:
                all_items = {
                    (x, y) for x in range(grid.shape[0]) for y in range(grid.shape[1])
                    if grid[x, y] in [3, 4]
                }
                ambush_items = all_items & my_blast
                if ambush_items and enemy_dists:
                    should_ambush = False
                    for item_pos in ambush_items:
                        d_us = self._shortest_dist(grid, my_pos, item_pos)
                        if d_us > 2:
                            continue  # Must be very close to collect and escape
                        for idx, dists in enemy_dists.items():
                            d_E = dists.get(item_pos, 999)
                            # Check if enemy approaching item within bomb timer range
                            if 2 <= d_E <= 7 and d_us < d_E and (d_E - d_us) <= 4:
                                # Skip ambush if the enemy is behind us or on the same side/line chasing the item.
                                d_E_to_us = dists.get(my_pos, 999)
                                ex, ey = int(players[idx][0]), int(players[idx][1])
                                on_same_line = (my_pos[0] == ex == item_pos[0]) or (my_pos[1] == ey == item_pos[1])
                                if d_E >= d_E_to_us or on_same_line:
                                    continue
                                should_ambush = True
                                break
                        if should_ambush:
                            break
                    if should_ambush:
                        simulated_bomb = {'pos': my_pos, 'timer': 7, 'radius': bomb_radius}
                        escape = self._find_safe_path(
                            grid, my_pos, bombs, players,
                            simulated_bomb=simulated_bomb,
                            occupied_at_t1=tactical_avoid,
                            require_permanent_safety=True,
                            enemy_dists=enemy_dists
                        )
                        if escape is not None:
                            return 5
                            
        # Agressive box farming
        if bombs_left > 0 and my_deadline > 5:
            box_farming_move = self._find_box_farming_action(
                grid, my_pos, bombs, players, occupied, enemy_dists,
                bomb_radius, bombs_left, alive_count, box_count, danger_avoid
            )
            if box_farming_move is not None:
                return box_farming_move

        # Item collection: Prioritize resource gathering safely
        # Items count toward draw state ranking tiebreaks (kills > boxes > items > bombs)
        preferred_values = set()
        if bombs_left <= 1:
            preferred_values.add(4)  # Capacity Item
        if bomb_radius <= 2:
            preferred_values.add(3)  # Radius Item

        if preferred_values:
            preferred_tiles = {
                (x, y) for x in range(grid.shape[0]) for y in range(grid.shape[1])
                if grid[x, y] in preferred_values
            }
            if preferred_tiles:
                move = self._find_safe_path_to_targets(grid, my_pos, preferred_tiles, bombs, players, occupied_at_t1=tactical_avoid, danger_soon=danger_avoid, enemy_dists=enemy_dists)
                if move is not None:
                    # Check if we can place a bomb to destroy a box on the way 
                    if bombs_left > 0 and my_pos not in bomb_positions and not is_consecutive and not is_camping_blast:
                        simulated_bomb = {'pos': my_pos, 'timer': 7, 'radius': bomb_radius}
                        escape_move = self._find_safe_path(grid, my_pos, bombs, players, simulated_bomb=simulated_bomb, occupied_at_t1=tactical_avoid, require_permanent_safety=True, enemy_dists=enemy_dists)
                        if escape_move is not None:
                            boxes_hit = 0
                            for x, y in self._get_blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius):
                                if grid[x, y] == 2:
                                    if (x, y) not in box_destroyed_time or box_destroyed_time[(x, y)] >= 7:
                                        boxes_hit += 1
                            if boxes_hit >= 1:
                                return 5
                    return move

        # Fallback: Collect any items when nearby (max 12 steps)
        any_item_tiles = {
            (x, y) for x in range(grid.shape[0]) for y in range(grid.shape[1])
            if grid[x, y] in [3, 4]
        }
        if any_item_tiles:
            move = self._find_safe_path_to_targets(grid, my_pos, any_item_tiles, bombs, players, max_depth=12, occupied_at_t1=tactical_avoid, danger_soon=danger_avoid, enemy_dists=enemy_dists)
            if move is not None:
                # Check if we can place a bomb to destroy a box on the way
                if bombs_left > 0 and my_pos not in bomb_positions and not is_consecutive and not is_camping_blast:
                    simulated_bomb = {'pos': my_pos, 'timer': 7, 'radius': bomb_radius}
                    escape_move = self._find_safe_path(grid, my_pos, bombs, players, simulated_bomb=simulated_bomb, occupied_at_t1=tactical_avoid, require_permanent_safety=True, enemy_dists=enemy_dists)
                    if escape_move is not None:
                        boxes_hit = 0
                        for x, y in self._get_blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius):
                            if grid[x, y] == 2:
                                if (x, y) not in box_destroyed_time or box_destroyed_time[(x, y)] >= 7:
                                    boxes_hit += 1
                        if boxes_hit >= 1:
                            item_move_after_bomb = self._find_safe_path_to_targets(
                                grid, my_pos, any_item_tiles, bombs, players, max_depth=12,
                                occupied_at_t1=tactical_avoid, danger_soon=danger_avoid, 
                                enemy_dists=enemy_dists, simulated_bomb=simulated_bomb
                            )
                            if item_move_after_bomb is not None:
                                return 5
                return move

        # Offensive tatic
        if bombs_left > 0 and my_pos not in bomb_positions:
            simulated_bomb = {'pos': my_pos, 'timer': 7, 'radius': bomb_radius}
            escape_move = self._find_safe_path(grid, my_pos, bombs, players, simulated_bomb=simulated_bomb, occupied_at_t1=tactical_avoid, require_permanent_safety=True, enemy_dists=enemy_dists)
            
            if escape_move is not None:
                my_blast = self._get_blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)
                # Check if enemy is trapped in the blast or pocket blocked by my_pos
                for idx, ep in enumerate(players):
                    if idx == self.agent_id or ep[2] != 1:
                        continue
                    enemy_pos = (int(ep[0]), int(ep[1]))
                    
                    # Direct blast trap check
                    is_blast_trapped = False
                    if enemy_pos in my_blast:
                        enemy_escape = self._find_safe_path(grid, enemy_pos, bombs, players, simulated_bomb=simulated_bomb, occupied_at_t1=occupied, require_permanent_safety=True, enemy_dists=enemy_dists)
                        if enemy_escape is None:
                            is_blast_trapped = True
                    
                    # Pocket block trap check
                    is_pocket_trapped = False
                    q = deque([(enemy_pos, 0)])
                    seen = {enemy_pos}
                    while q:
                        curr, d = q.popleft()
                        if d > 0:
                            if my_pos == curr and self._is_bottleneck(grid, enemy_pos, curr, max_size=8):
                                is_pocket_trapped = True
                                break
                        if d < 3:
                            for a in [1, 2, 3, 4]:
                                nx, ny = self._next_pos(curr, a)
                                npos = (nx, ny)
                                if npos not in seen and self._passable(grid, nx, ny):
                                    seen.add(npos)
                                    q.append((npos, d + 1))
                    
                    # Corridor end-block trap: If enemy is in a straight corridor and our bomb covers one end and also check if the other end is also blocked
                    is_corridor_trapped = False
                    if not is_blast_trapped and not is_pocket_trapped:
                        if enemy_pos in my_blast or any(abs(enemy_pos[0] - bx) + abs(enemy_pos[1] - by) <= 1 for bx, by in my_blast):
                            # Check if enemy is in a corridor (<=2 open neighbors)
                            enemy_exits = self._open_neighbor_count(grid, enemy_pos)
                            if enemy_exits <= 2:
                                enemy_escape_sim = self._find_safe_path(
                                    grid, enemy_pos, bombs, players, simulated_bomb=simulated_bomb,
                                    occupied_at_t1=None, require_permanent_safety=True,
                                    enemy_dists=None
                                )
                                if enemy_escape_sim is None:
                                    is_corridor_trapped = True

                    if is_blast_trapped or is_pocket_trapped or is_corridor_trapped:
                        return 5

        # Offensive navigation (walk to enemy pocket exit to block them)
        if bombs_left > 0 and my_deadline > 5 and box_count <= pvp_gate_threshold:
            trap_targets = set()
            for idx, ep in enumerate(players):
                if idx == self.agent_id or ep[2] != 1:
                    continue
                enemy_pos = (int(ep[0]), int(ep[1]))

                # Find all passable tiles within distance 3 from enemy_pos
                possible_exits = set()
                q = deque([(enemy_pos, 0)])
                seen = {enemy_pos}
                while q:
                    curr, d = q.popleft()
                    if d > 0:
                        possible_exits.add(curr)
                    if d < 3:
                        for a in [1, 2, 3, 4]:
                            nx, ny = self._next_pos(curr, a)
                            npos = (nx, ny)
                            if npos not in seen and self._passable(grid, nx, ny):
                                seen.add(npos)
                                q.append((npos, d + 1))
                for B in possible_exits:
                    if self._is_bottleneck(grid, enemy_pos, B, max_size=8):
                        trap_targets.add(B)
            if trap_targets:
                move = self._find_safe_path_to_targets(grid, my_pos, trap_targets, bombs, players, max_depth=8, occupied_at_t1=tactical_avoid, danger_soon=danger_avoid, enemy_dists=enemy_dists)
                if move is not None:
                    return move

        # Offensive sniper mode (fight from a futher place)
        if bombs_left > 0 and len(skirmish_enemies) >= 2 and box_count <= pvp_gate_threshold:
            my_dists = self._compute_all_dists(grid, my_pos, extra_blocked=bomb_positions)
            sniper_candidates = []
            for spot, dist in my_dists.items():
                if dist == 0 or dist > 8:
                    continue
                if spot in occupied or spot in bomb_positions:
                    continue
                if danger_avoid and spot in danger_avoid:
                    continue
                if not self._passable(grid, spot[0], spot[1]):
                    continue
                # Standing aside check: At least distance 2 from all enemies (to stay safe)
                if any(abs(spot[0] - ep[0]) + abs(spot[1] - ep[1]) < 2 for ep in enemies):
                    continue

                # Check if placing bomb here reaches any skirmishing enemy
                blast = self._get_blast_tiles(grid, spot[0], spot[1], bomb_radius)
                hit_enemies = skirmish_enemies & blast
                if hit_enemies:
                    # Score the sniper position (prefer shorter distance and hitting more enemies)
                    score = 4.0 * len(hit_enemies) - 0.3 * dist
                    sniper_candidates.append((score, spot))
            
            if sniper_candidates:
                validated_sniper = []
                for score, spot in sorted(sniper_candidates, reverse=True)[:5]:
                    simulated_bomb = {'pos': spot, 'timer': 7, 'radius': bomb_radius}
                    post_bomb_escape = self._find_safe_path(
                        grid, spot, bombs, players, simulated_bomb=simulated_bomb,
                        occupied_at_t1=occupied, require_permanent_safety=True,
                        enemy_dists=enemy_dists
                    )
                    if post_bomb_escape is not None:
                        validated_sniper.append((score, spot))
                
                if validated_sniper:
                    best_sniper_spot = max(validated_sniper, key=lambda x: x[0])[1]
                    if my_pos == best_sniper_spot:
                        return 5
                    else:
                        move = self._find_safe_path_to_targets(
                            grid, my_pos, {best_sniper_spot}, bombs, players, max_depth=8,
                            occupied_at_t1=tactical_avoid, danger_soon=danger_avoid, enemy_dists=enemy_dists
                        )
                        if move is not None:
                            return move

        # Smart bomb placing - Place second bombs with optimal spacing
        if bombs_left >= 2 and my_pos not in bomb_positions and my_deadline > 5:
            spacing_pos = self._find_optimal_spacing_position(grid, my_pos, bombs, bomb_radius, bombs_left)
            if spacing_pos and spacing_pos != my_pos:
                # This position has good spacing from our existing bombs
                simulated_bomb = {'pos': spacing_pos, 'timer': 7, 'radius': bomb_radius}
                escape = self._find_safe_path(grid, spacing_pos, bombs, players, simulated_bomb=simulated_bomb,
                                             occupied_at_t1=tactical_avoid, require_permanent_safety=True,
                                             enemy_dists=enemy_dists)
                if escape is not None:
                    move = self._find_safe_path_to_targets(grid, my_pos, {spacing_pos}, bombs, players,
                                                          max_depth=6, occupied_at_t1=tactical_avoid,
                                                          danger_soon=danger_avoid, enemy_dists=enemy_dists)
                    if move is not None:
                        return move

        # Farming boxes and pressure enemy
        if bombs_left > 0 and my_pos not in bomb_positions and not is_consecutive and not is_camping_blast:
            simulated_bomb = {'pos': my_pos, 'timer': 7, 'radius': bomb_radius}
            escape_move = self._find_safe_path(grid, my_pos, bombs, players, simulated_bomb=simulated_bomb, occupied_at_t1=tactical_avoid, require_permanent_safety=True, enemy_dists=enemy_dists)
            
            if escape_move is not None and self._should_place_value_bomb(
                grid, my_pos, bombs, players, occupied, enemies, enemy_dists,
                bomb_radius, bombs_left, alive_count, box_count, box_destroyed_time, escaping_enemies,
                defensive_mode=defensive_mode
            ):
                return 5

        # Chain bombing
        if bombs_left >= 2 and my_deadline > 5:
            # Check if we have a bomb at our current position (just placed)
            my_bombs = [(int(b[0]), int(b[1])) for b in bombs if len(b) > 3 and int(b[3]) == self.agent_id]
            if my_pos in my_bombs:
                # We just placed a bomb, immediately move to next multi box spot
                next_box_move = self._find_next_box_action(
                    grid, my_pos, bombs, players, occupied, enemy_dists,
                    bomb_radius, bombs_left - 1, alive_count, box_count, danger_avoid
                )
                if next_box_move is not None:
                    return next_box_move

        # Offensive navigation
        if bombs_left > 0 and enemies:
            attack_move = self._find_attack_lane_action(
                grid, my_pos, enemies, bombs, players, tactical_avoid, bomb_radius,
                bombs_left, alive_count, box_count, danger_avoid, enemy_dists, escaping_enemies
            )
            if attack_move is not None:
                return attack_move

        # Predictive enemy trapping: Use velocity to intercept escaping enemies
        if bombs_left > 0 and alive_count <= 3 and my_deadline > 5:
            # Get the primary threat (closest or most threatening enemy)
            if enemies:
                primary_threat_idx = None
                best_threat = 0.0
                for idx in range(len(players)):
                    if idx != self.agent_id and players[idx][2] == 1:
                        threat = self._analyze_enemy_threat_level(grid, idx, players, my_pos, bomb_radius, bombs_left)
                        if threat > best_threat:
                            best_threat = threat
                            primary_threat_idx = idx
                
                if primary_threat_idx is not None and best_threat > 0.3:
                    # Predict enemy's escape path
                    escaping_path = self._predict_enemy_next_moves(primary_threat_idx, players, grid, bombs)
                    if escaping_path and len(escaping_path) >= 2:
                        # Find interception points
                        intercepts = self._find_escape_interception_points(grid, escaping_path[0], escaping_path, bomb_radius)
                        if intercepts:
                            best_intercept_pos = intercepts[0][0]
                            # Check if we can safely reach and bomb this position
                            if my_pos != best_intercept_pos:
                                simulated_bomb = {'pos': best_intercept_pos, 'timer': 7, 'radius': bomb_radius}
                                escape = self._find_safe_path(grid, best_intercept_pos, bombs, players, 
                                                             simulated_bomb=simulated_bomb, occupied_at_t1=tactical_avoid,
                                                             require_permanent_safety=True, enemy_dists=enemy_dists)
                                if escape is not None:
                                    # Navigate to this interception point
                                    move = self._find_safe_path_to_targets(grid, my_pos, {best_intercept_pos}, bombs, 
                                                                          players, max_depth=6, occupied_at_t1=tactical_avoid,
                                                                          danger_soon=danger_avoid, enemy_dists=enemy_dists)
                                    if move is not None:
                                        return move
##########################################################################################################################################
        # 1v1 endgame priority: Navigate to enemy before farming when few boxes remain
        if alive_count <= 2 and box_count <= endgame_gate_threshold and enemies:
            # Check if we should play aggressive in this 1v1
            if self._should_play_aggressive(grid, players, my_pos, alive_count, box_count, bombs_left, bomb_radius, enemies):
                # Try to find optimal positioning first
                if bombs_left > 0:
                    # Find spot that's good for 1v1: not too close, not too far, good escapes
                    best_1v1_spot = None
                    best_score = -999
                    my_dists = self._compute_all_dists(grid, my_pos, extra_blocked=bomb_positions)
                    
                    for pos, dist in my_dists.items():
                        if dist > 6:
                            continue
                        if pos in danger_avoid:
                            continue
                        
                        # Score this position for 1v1
                        def_score = self._evaluate_defensive_position(grid, pos, enemies, bomb_radius, danger_avoid)
                        if def_score > best_score:
                            best_score = def_score
                            best_1v1_spot = pos
                    
                    if best_1v1_spot and best_1v1_spot != my_pos:
                        move = self._find_safe_path_to_targets(grid, my_pos, {best_1v1_spot}, bombs, players, 
                                                              max_depth=6, occupied_at_t1=tactical_avoid,
                                                              danger_soon=danger_avoid, enemy_dists=enemy_dists)
                        if move is not None:
                            return move
            
            # Conservative 1v1: Navigate to enemy
            move = self._find_safe_path_to_targets(grid, my_pos, set(enemies), bombs, players, occupied_at_t1=tactical_avoid, danger_soon=danger_avoid, enemy_dists=enemy_dists)
            if move is not None:
                if bombs_left > 0 and my_pos not in bomb_positions and not is_camping_blast:
                    simulated_bomb = {'pos': my_pos, 'timer': 7, 'radius': bomb_radius}
                    esc = self._find_safe_path(grid, my_pos, bombs, players, simulated_bomb=simulated_bomb, occupied_at_t1=tactical_avoid, require_permanent_safety=True, enemy_dists=enemy_dists)
                    if esc is not None:
                        bh = sum(1 for x, y in self._get_blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius) if grid[x, y] == 2 and ((x, y) not in box_destroyed_time or box_destroyed_time[(x, y)] >= 7))
                        if bh >= 1:
                            return 5
                return move

        # Navigation to box spots
        if bombs_left > 0:
            box_spots = self._box_bomb_spots(grid, occupied, bomb_radius)
            
            # Late-game box farming gating: Focus on combat/items unless box is very close
            if box_count <= endgame_gate_threshold and box_spots and enemies:
                nearest_enemy_dist = min(abs(my_pos[0] - ep[0]) + abs(my_pos[1] - ep[1]) for ep in enemies)
                if nearest_enemy_dist <= 6:
                    close_spots = set()
                    for spot in box_spots:
                        d = self._shortest_dist(grid, my_pos, spot)
                        if d <= 3:
                            close_spots.add(spot)
                    box_spots = close_spots

            if box_spots:
                move = self._find_best_box_spot_action(grid, my_pos, box_spots, bombs, players, occupied, bomb_radius, danger_soon=danger_avoid, enemy_dists=enemy_dists)
                if move is not None:
                    return move

        # Fallback escape: Move to safety if in danger soon
        if my_deadline <= 6:
            escape = self._find_safe_path(grid, my_pos, bombs, players, occupied_at_t1=tactical_avoid, enemy_dists=enemy_dists)
            if escape is not None:
                return escape

        # Camp the edge of blasts to pick up items instantly
        if danger_soon:
            edge_tiles = {
                (x, y) for x in range(grid.shape[0]) for y in range(grid.shape[1])
                if self._is_passable_at(grid, x, y, 1, box_destroyed_time)
                and danger_deadline.get((x, y), 999) > 5
                and any(self._next_pos((x, y), a) in danger_soon for a in [1, 2, 3, 4])
            }
            if edge_tiles:
                if my_pos in edge_tiles:
                    return 0  # Stay and camp
                move = self._find_safe_path_to_targets(grid, my_pos, edge_tiles, bombs, players, occupied_at_t1=tactical_avoid, danger_soon=danger_avoid, enemy_dists=enemy_dists)
                if move is not None:
                    return move

        # Navigation to enemies
        if enemies:
            move = self._find_safe_path_to_targets(grid, my_pos, set(enemies), bombs, players, occupied_at_t1=tactical_avoid, danger_soon=danger_avoid, enemy_dists=enemy_dists)
            if move is not None:
                # Check if we can place a bomb to destroy a box on the way
                if bombs_left > 0 and my_pos not in bomb_positions and not is_consecutive and not is_camping_blast:
                    simulated_bomb = {'pos': my_pos, 'timer': 7, 'radius': bomb_radius}
                    escape_move = self._find_safe_path(grid, my_pos, bombs, players, simulated_bomb=simulated_bomb, occupied_at_t1=tactical_avoid, require_permanent_safety=True, enemy_dists=enemy_dists)
                    if escape_move is not None:
                        boxes_hit = 0
                        for x, y in self._get_blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius):
                            if grid[x, y] == 2:
                                if (x, y) not in box_destroyed_time or box_destroyed_time[(x, y)] >= 7:
                                    boxes_hit += 1
                        if boxes_hit >= 1:
                            enemy_move_after_bomb = self._find_safe_path_to_targets(
                                grid, my_pos, set(enemies), bombs, players, 
                                occupied_at_t1=tactical_avoid, danger_soon=danger_avoid, 
                                enemy_dists=enemy_dists, simulated_bomb=simulated_bomb
                            )
                            if enemy_move_after_bomb is not None:
                                return 5
                return move

        # Idle move: Move toward nearest enemy when possible, otherwise stay safe
        valid_moves = []
        for a in [1, 2, 3, 4, 0]:
            nx, ny = self._next_pos(my_pos, a)
            if a != 0:
                if not self._is_passable_at(grid, nx, ny, 1, box_destroyed_time):
                    continue
                if (nx, ny) in bomb_positions:
                    continue
            if danger_deadline.get((nx, ny), 999) > 1:
                valid_moves.append(a)
        
        if not valid_moves:
            return 0
        
        # Chain bombing priority: Prefer moving toward boxes over staying still
        if bombs_left > 0 and len(valid_moves) > 1 and not defensive_mode:
            # Check if any valid moves take us closer to the nearest box
            nearest_box_dist = 999
            for x in range(grid.shape[0]):
                for y in range(grid.shape[1]):
                    if grid[x, y] == 2:
                        d = abs(my_pos[0] - x) + abs(my_pos[1] - y)
                        nearest_box_dist = min(nearest_box_dist, d)
            
            if nearest_box_dist < 999:
                best_a = 0
                best_dist = nearest_box_dist
                for a in valid_moves:
                    if a == 0:  # Staying still
                        continue
                    nx, ny = self._next_pos(my_pos, a)
                    new_dist = min(abs(nx - bx) + abs(ny - by) 
                                  for bx in range(grid.shape[0]) 
                                  for by in range(grid.shape[1]) 
                                  if grid[bx, by] == 2)
                    if new_dist < best_dist:
                        best_dist = new_dist
                        best_a = a
                
                # If we found a move closer to boxes, prefer it
                if best_a != 0:
                    return best_a
        
        # Prefer moving toward enemies or staying still/safer in defensive mode
        if enemies and len(valid_moves) > 1:
            nearest_enemy = min(enemies, key=lambda ep: abs(my_pos[0] - ep[0]) + abs(my_pos[1] - ep[1]))
            best_a = 0
            best_dist = abs(my_pos[0] - nearest_enemy[0]) + abs(my_pos[1] - nearest_enemy[1])
            
            if defensive_mode:
                # Maintain distance and prefer positions with open exits in defensive mode
                best_score = -999
                for a in valid_moves:
                    nx, ny = self._next_pos(my_pos, a)
                    d = abs(nx - nearest_enemy[0]) + abs(ny - nearest_enemy[1])
                    exits = self._open_neighbor_count(grid, (nx, ny)) if a != 0 else self._open_neighbor_count(grid, my_pos)
                    # Score based on open exits and distance
                    score = exits * 2.0
                    if d >= 3:
                        score += 1.0  # Stay at safe distance
                    elif d <= 1:
                        score -= 3.0  # Too close
                    if a == 0:
                        score += 0.5  # Slight preference for staying put
                    if score > best_score:
                        best_score = score
                        best_a = a
                return best_a
            else:
                for a in valid_moves:
                    if a == 0:
                        continue
                    nx, ny = self._next_pos(my_pos, a)
                    d = abs(nx - nearest_enemy[0]) + abs(ny - nearest_enemy[1])
                    if d < best_dist:
                        best_dist = d
                        best_a = a
                if best_a != 0:
                    return best_a
        
        if 0 in valid_moves:
            return 0
        return random.choice(valid_moves)

    # =========================================================================
    # CORE RECURSIVE BLAST AND PROPAGATION SYSTEMS
    # =========================================================================

    def _next_pos(self, pos, action):
        dx, dy = self.MOVES[action]
        return pos[0] + dx, pos[1] + dy

    def _in_bounds(self, grid, x, y):
        return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

    def _passable(self, grid, x, y):
        return self._in_bounds(grid, x, y) and grid[x, y] in [0, 3, 4]

    def _get_blast_tiles(self, grid, bx, by, radius):
        tiles = {(bx, by)}
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for r in range(1, radius + 1):
                x, y = bx + dx * r, by + dy * r
                if not self._in_bounds(grid, x, y):
                    break
                cell = grid[x, y]
                if cell == 1:  # Wall blocks blast
                    break
                tiles.add((x, y))
                if cell == 2:  # Box blocks blast but is hit
                    break
        return tiles

    def _sync_bomb_radius_memory(self, bombs, players):
        if len(bombs) == 0:
            self._bomb_radius_memory.clear()
            return

        active = set()
        for b in bombs:
            bx, by = int(b[0]), int(b[1])
            owner_id = int(b[3]) if len(b) > 3 else -1
            key = (bx, by, owner_id)
            active.add(key)
            if key not in self._bomb_radius_memory:
                radius = 1
                if 0 <= owner_id < len(players):
                    radius = max(1, int(players[owner_id][4]) + 1)
                self._bomb_radius_memory[key] = radius

        for key in list(self._bomb_radius_memory):
            if key not in active:
                del self._bomb_radius_memory[key]

    def _remembered_bomb_radius(self, bx, by, owner_id, players):
        key = (int(bx), int(by), int(owner_id))
        if key in self._bomb_radius_memory:
            return self._bomb_radius_memory[key]
        if 0 <= owner_id < len(players):
            return max(1, int(players[owner_id][4]) + 1)
        return 1

    def _player_has_bombs(self, players, idx):
        ep = players[idx]
        if ep[3] > 0:
            return True
        if hasattr(self, '_current_bombs'):
            return any(len(b) > 3 and int(b[3]) == idx for b in self._current_bombs)
        return False

    def _danger_deadlines(self, blast_at_time):
        deadlines = {}
        for t, tiles in blast_at_time.items():
            for tile in tiles:
                if t < deadlines.get(tile, 999):
                    deadlines[tile] = t
        return deadlines

    def _simulate_accurate_bombs_and_danger(self, grid, active_bombs, players, simulated_bomb=None, max_depth=18):
        # Initialize bombs with starting timers and radii
        bombs_list = []
        for b in active_bombs:
            if isinstance(b, dict):
                # If b is already a dict, it's already in the correct format (e.g. from effective_bombs wrapper)
                bombs_list.append({
                    'pos': b['pos'],
                    'timer': b['timer'],
                    'radius': b['radius'],
                    'placed_at': b.get('placed_at', 0),
                    'detonated': False
                })
            else:
                bx, by, timer = int(b[0]), int(b[1]), int(b[2])
                owner_id = int(b[3]) if len(b) > 3 else -1
                radius = self._remembered_bomb_radius(bx, by, owner_id, players)
                bombs_list.append({
                    'pos': (bx, by),
                    'timer': timer,
                    'radius': radius,
                    'placed_at': 0,
                    'detonated': False
                })
        if simulated_bomb:
            if isinstance(simulated_bomb, list):
                for sb in simulated_bomb:
                    bombs_list.append({
                        'pos': sb['pos'],
                        'timer': sb['timer'],
                        'radius': sb['radius'],
                        'placed_at': sb.get('placed_at', 0),
                        'detonated': False
                    })
            else:
                bombs_list.append({
                    'pos': simulated_bomb['pos'],
                    'timer': simulated_bomb['timer'],
                    'radius': simulated_bomb['radius'],
                    'placed_at': simulated_bomb.get('placed_at', 0),
                    'detonated': False
                })

        # Track active boxes
        destroyed_boxes = {} # (x, y) -> step
        blast_at_time = {t: set() for t in range(max_depth + 1)}
        bombs_at_time = {t: set() for t in range(max_depth + 1)}

        # Run step-by-step simulation
        for t in range(1, max_depth + 1):
            # Record bomb positions at this step
            for b in bombs_list:
                placed_at = b.get('placed_at', 0)
                # A bomb blocks entering starting from placed_at + 1 (unless placed_at is 0)
                block_start = placed_at + 1 if placed_at > 0 else 0
                if t >= block_start:
                    if not b['detonated'] and b['timer'] >= t:
                        bombs_at_time[t].add(b['pos'])

            # Find all bombs that detonate at step t
            detonating = []
            for b in bombs_list:
                placed_at = b.get('placed_at', 0)
                if t >= placed_at:
                    if not b['detonated'] and b['timer'] <= t:
                        detonating.append(b)

            if not detonating:
                continue

            # Process detonations at step t using a queue
            queue = deque(detonating)
            for b in detonating:
                b['detonated'] = True

            while queue:
                curr = queue.popleft()
                bx, by = curr['pos']
                r = curr['radius']
                
                # Add to blast tiles at step t
                blast_at_time[t].add((bx, by))

                # Propagate blast in 4 directions
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    for step_r in range(1, r + 1):
                        x, y = bx + dx * step_r, by + dy * step_r
                        if not self._in_bounds(grid, x, y):
                            break
                        cell = grid[x, y]
                        if cell == 1:  # Wall blocks
                            break
                        
                        blast_at_time[t].add((x, y))

                        # Check if blast hits and detonates another bomb
                        for other in bombs_list:
                            if not other['detonated'] and other['pos'] == (x, y):
                                other['timer'] = t
                                other['detonated'] = True
                                queue.append(other)

                        # Check if blast hits a box
                        if cell == 2:
                            # Ignore if already destroyed
                            if (x, y) in destroyed_boxes and destroyed_boxes[(x, y)] < t:
                                continue
                            else:
                                # Box is hit and blocks the blast
                                if (x, y) not in destroyed_boxes:
                                    destroyed_boxes[(x, y)] = t
                                break

        # Initialize t = 0 state
        blast_at_time[0] = set()
        bombs_at_time[0] = {b['pos'] for b in bombs_list if b['timer'] > 0}

        # Convert bombs_list to effective_bombs format
        effective_bombs = [{
            'pos': b['pos'],
            'timer': b['timer'],
            'radius': b['radius']
        } for b in bombs_list]

        return effective_bombs, destroyed_boxes, blast_at_time, bombs_at_time

    def _get_effective_bombs(self, grid, active_bombs, players, simulated_bomb=None):
        if simulated_bomb is None:
            cache_key = None
        elif isinstance(simulated_bomb, list):
            cache_key = tuple(sorted((sb['pos'], sb['timer'], sb['radius']) for sb in simulated_bomb))
        else:
            cache_key = (simulated_bomb['pos'], simulated_bomb['timer'], simulated_bomb['radius'])

        if hasattr(self, '_cache_effective_bombs') and cache_key in self._cache_effective_bombs:
            return self._cache_effective_bombs[cache_key]

        effective_bombs, _, _, _ = self._simulate_accurate_bombs_and_danger(grid, active_bombs, players, simulated_bomb)
        if hasattr(self, '_cache_effective_bombs'):
            self._cache_effective_bombs[cache_key] = effective_bombs
        return effective_bombs

    def _simulate_grid_and_danger(self, grid, effective_bombs, max_depth=18):
        cache_key = (tuple(sorted((b['pos'], b['timer'], b['radius']) for b in effective_bombs)), max_depth)
        if hasattr(self, '_cache_grid_danger') and cache_key in self._cache_grid_danger:
            return self._cache_grid_danger[cache_key]

        _, box_destroyed_time, blast_at_time, bombs_at_time = self._simulate_accurate_bombs_and_danger(grid, effective_bombs, [], max_depth=max_depth)
        if hasattr(self, '_cache_grid_danger'):
            self._cache_grid_danger[cache_key] = (box_destroyed_time, blast_at_time, bombs_at_time)
        return box_destroyed_time, blast_at_time, bombs_at_time

    def _is_passable_at(self, grid, x, y, nt, box_destroyed_time):
        if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]):
            return False
        cell = grid[x, y]
        if cell == 1:
            return False
        if cell == 2:
            return (x, y) in box_destroyed_time and box_destroyed_time[(x, y)] < nt
        return True

    # =========================================================================
    # CORE TIME-SPACE PATHFINDING ENGINE (BFS)
    # =========================================================================

    def _find_safe_path(self, grid, start_pos, active_bombs, players, simulated_bomb=None, max_depth=16, occupied_at_t1=None, require_permanent_safety=False, enemy_dists=None):
        # Build simulated bombs list including simulated enemy bombs if require_permanent_safety is True
        sim_bombs = []
        if simulated_bomb:
            if isinstance(simulated_bomb, list):
                sim_bombs.extend(simulated_bomb)
            else:
                sim_bombs.append(simulated_bomb)
        
        if require_permanent_safety:
            for idx, p_info in enumerate(players):
                if idx != self.agent_id and p_info[2] == 1: # Active enemy
                    if self._player_has_bombs(players, idx): # Enemy has bombs left
                        ex, ey = int(p_info[0]), int(p_info[1])
                        e_radius = max(1, int(p_info[4]) + 1)
                        sim_bombs.append({
                            'pos': (ex, ey),
                            'timer': 7,
                            'radius': e_radius,
                            'placed_at': 0
                        })
        
        effective_bombs = self._get_effective_bombs(grid, active_bombs, players, sim_bombs if sim_bombs else None)
        box_destroyed_time, blast_at_time, bombs_at_time = self._simulate_grid_and_danger(grid, effective_bombs, max_depth)
        
        max_timer = max((b['timer'] for b in effective_bombs), default=0)
        eb = {b['pos'] for b in effective_bombs}
        actual_bomb_positions = {(int(b[0]), int(b[1])) if not isinstance(b, dict) else b['pos'] for b in active_bombs}

        # Identify boxes currently targeted by our bombs (simulated or active on map)
        our_boxes_hit = set()
        our_bombs_list = []
        if simulated_bomb:
            if isinstance(simulated_bomb, list):
                our_bombs_list.extend(simulated_bomb)
            else:
                our_bombs_list.append(simulated_bomb)
        for b in active_bombs:
            if not isinstance(b, dict):
                owner_id = int(b[3]) if len(b) > 3 else -1
                if owner_id == self.agent_id:
                    our_bombs_list.append({
                        'pos': (int(b[0]), int(b[1])),
                        'radius': self._remembered_bomb_radius(int(b[0]), int(b[1]), owner_id, players)
                    })
        for ob in our_bombs_list:
            ob_pos = ob['pos']
            ob_radius = ob.get('radius')
            if ob_radius is None:
                ob_radius = self._remembered_bomb_radius(ob_pos[0], ob_pos[1], self.agent_id, players)
            for tx, ty in self._get_blast_tiles(grid, ob_pos[0], ob_pos[1], ob_radius):
                if grid[tx, ty] == 2:
                    our_boxes_hit.add((tx, ty))

        # Recompute enemy distances treating simulated bombs as obstacles
        if require_permanent_safety and simulated_bomb and enemy_dists:
            sim_blocked = set()
            if isinstance(simulated_bomb, list):
                for sb in simulated_bomb:
                    sim_blocked.add(sb['pos'])
            else:
                sim_blocked.add(simulated_bomb['pos'])
                
            blocked_enemy_dists = {}
            for idx, p_info in enumerate(players):
                if idx == self.agent_id or p_info[2] != 1:
                    continue
                ex, ey = int(p_info[0]), int(p_info[1])
                blocked_enemy_dists[idx] = self._compute_all_dists(grid, (ex, ey), extra_blocked=sim_blocked | actual_bomb_positions)
            effective_enemy_dists = blocked_enemy_dists
        else:
            effective_enemy_dists = enemy_dists

        def is_permanently_safe(p, t):
            if not self._is_passable_at(grid, p[0], p[1], t, box_destroyed_time):
                return False
            for step in range(t, max_timer + 1):
                if p in blast_at_time.get(step, set()):
                    return False
            if require_permanent_safety and effective_enemy_dists:
                sim_timer = 7
                if isinstance(simulated_bomb, dict):
                    sim_timer = simulated_bomb.get('timer', 7)
                elif isinstance(simulated_bomb, list) and simulated_bomb:
                    sim_timer = max((sb.get('timer', 7) for sb in simulated_bomb), default=7)
                if t <= sim_timer:
                    for idx, dists in effective_enemy_dists.items():
                        ep = players[idx]
                        if self._player_has_bombs(players, idx):  # Enemy has bombs
                            d_E = dists.get(p, 999)
                            if d_E <= t + 1:
                                return False
            return True

        # Dijkstra priority queue: (cost, pos, t, first_action)
        pq = [(0.0, start_pos, 0, None)]
        seen = {}  # (pos, t) -> min_cost

        best_incomplete_path = None
        max_survival_time = -1
        min_cost_to_max_survival = 999999.0

        while pq:
            cost, pos, t, first_action = heapq.heappop(pq)

            if seen.get((pos, t), 999999.0) <= cost:
                continue
            seen[(pos, t)] = cost

            if is_permanently_safe(pos, t):
                return first_action

            if t > max_survival_time or (t == max_survival_time and cost < min_cost_to_max_survival):
                max_survival_time = t
                min_cost_to_max_survival = cost
                best_incomplete_path = first_action

            if t >= max_depth:
                continue

            for a in [1, 2, 3, 4, 0]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                nt = t + 1

                if a != 0:
                    if not self._is_passable_at(grid, nx, ny, nt, box_destroyed_time):
                        continue

                # Obstacle check: solid bomb
                if a != 0 and npos in bombs_at_time.get(nt, set()):
                    continue

                # Danger check: blast explosion
                if npos in blast_at_time.get(nt, set()):
                    continue

                # Player collision check at t = 1
                if nt == 1 and occupied_at_t1 and npos in occupied_at_t1:
                    continue

                # Calculate transition cost
                cost_step = 1.0 if a != 0 else 0.05
                
                # Check for risk penalty from ticking bombs
                penalty = 0.0
                # Find if this tile will explode soon
                t_det = 999
                for step in range(nt, max_timer + 1):
                    if npos in blast_at_time.get(step, set()):
                        t_det = step
                        break
                
                if t_det != 999:
                    diff = t_det - nt
                    if diff <= 3:
                        penalty = 8.0 / (diff + 1)
                
                # Dead-end avoidance: Penalize narrow corridors
                if a != 0:
                    n_exits = 0
                    for ca in [1, 2, 3, 4]:
                        cx, cy = self._next_pos(npos, ca)
                        if self._is_passable_at(grid, cx, cy, nt, box_destroyed_time) and (cx, cy) not in bombs_at_time.get(nt, set()):
                            n_exits += 1
                    if n_exits <= 1:
                        penalty += 2.0

                # Item collection bonus: Prefer escaping through items
                if grid[nx, ny] in [3, 4]:
                    penalty -= 8.0

                # Center preference: Avoid being pushed to map edges
                edge_dist = min(nx, ny, grid.shape[0] - 1 - nx, grid.shape[1] - 1 - ny)
                if edge_dist <= 1:
                    penalty += 0.5

                # Box proximity guidance: Prefer staying close to targeted boxes
                if our_boxes_hit:
                    min_dist_to_box = min(abs(nx - bx) + abs(ny - by) for bx, by in our_boxes_hit)
                    penalty += 1.5 * min_dist_to_box

                # Farming value guidance: Prefer safe tiles adjacent to boxes
                adjacent_boxes = 0
                for ca in [1, 2, 3, 4]:
                    cx, cy = self._next_pos(npos, ca)
                    if self._in_bounds(grid, cx, cy) and grid[cx, cy] == 2:
                        if (cx, cy) not in our_boxes_hit:
                            adjacent_boxes += 1
                if adjacent_boxes > 0:
                    penalty -= 2.0 * adjacent_boxes


                # Enemy blast lane avoidance: Penalize being in line of fire
                if enemy_dists:
                    for idx, dists in enemy_dists.items():
                        ep = players[idx]
                        if not self._player_has_bombs(players, idx):
                            continue
                        ex_p, ey_p = int(ep[0]), int(ep[1])
                        er = max(1, int(ep[4]) + 1)
                        if (nx == ex_p and abs(ny - ey_p) <= er) or (ny == ey_p and abs(nx - ex_p) <= er):
                            penalty += 1.0
                            break

                # Trap zone penalty/check
                if enemy_dists:
                    if self._is_trap_zone(grid, npos, nt, enemy_dists, players, max_size=3, extra_blocked=eb):
                        if require_permanent_safety:
                            continue
                        penalty += 20.0
                    elif self._is_trap_zone(grid, npos, nt, enemy_dists, players, max_size=6, extra_blocked=eb):
                        if require_permanent_safety:
                            continue
                        penalty += 4.0
                
                next_cost = cost + max(0.01, cost_step + penalty)
                
                if seen.get((npos, nt), 999999.0) > next_cost:
                    heapq.heappush(pq, (next_cost, npos, nt, a if first_action is None else first_action))

        if require_permanent_safety:
            return None
        return best_incomplete_path

    def _find_safe_path_to_targets(self, grid, start_pos, targets, active_bombs, players, max_depth=12, occupied_at_t1=None, danger_soon=None, enemy_dists=None, simulated_bomb=None):
        return self._find_safe_path_to_targets_internal(grid, start_pos, targets, active_bombs, players, max_depth, occupied_at_t1, danger_soon, enemy_dists, simulated_bomb)

    def _find_safe_path_to_targets_internal(self, grid, start_pos, targets, active_bombs, players, max_depth=12, occupied_at_t1=None, danger_soon=None, enemy_dists=None, simulated_bomb=None):
        """
        Dijkstra pathfinding to target tiles that guarantees we can safely escape to permanent safety afterwards,
        prioritizing paths that collect items.
        """
        if not targets:
            return None

        effective_bombs = self._get_effective_bombs(grid, active_bombs, players, simulated_bomb)
        box_destroyed_time, blast_at_time, bombs_at_time = self._simulate_grid_and_danger(grid, effective_bombs, max_depth + 6)
        
        max_timer = max((b['timer'] for b in effective_bombs), default=0)
        eb = {b['pos'] for b in effective_bombs}

        def is_permanently_safe(p, t):
            if not self._is_passable_at(grid, p[0], p[1], t, box_destroyed_time):
                return False
            for step in range(t, max_timer + 1):
                if p in blast_at_time.get(step, set()):
                    return False
            return True

        # Dijkstra priority queue: stores (cost, pos, t, first_action)
        pq = [(0.0, start_pos, 0, None)]
        seen = {}  # (pos, t) -> min_cost

        while pq:
            cost, pos, t, first_action = heapq.heappop(pq)

            if seen.get((pos, t), 999999.0) <= cost:
                continue
            seen[(pos, t)] = cost

            if pos in targets:
                # Verify escape to safety from target location
                if is_permanently_safe(pos, t):
                    return first_action
                
                # Run a fast escape solver from the target state
                sub_q = deque([(pos, t)])
                sub_seen = {(pos, t)}
                escaped = False
                while sub_q:
                    sp, st = sub_q.popleft()
                    if is_permanently_safe(sp, st):
                        escaped = True
                        break
                    if st - t >= 6:
                        continue
                    for sa in [1, 2, 3, 4, 0]:
                        snx, sny = self._next_pos(sp, sa)
                        snpos = (snx, sny)
                        snt = st + 1
                        if (snpos, snt) in sub_seen:
                            continue
                        if sa != 0 and not self._is_passable_at(grid, snx, sny, snt, box_destroyed_time):
                            continue
                        if snpos in bombs_at_time.get(snt, set()) or snpos in blast_at_time.get(snt, set()):
                            continue
                        if enemy_dists and self._is_trap_zone(grid, snpos, snt, enemy_dists, players, max_size=5, extra_blocked=eb):
                            continue
                        sub_seen.add((snpos, snt))
                        sub_q.append((snpos, snt))
                if escaped:
                    return first_action

            if t >= max_depth:
                continue

            for a in [1, 2, 3, 4, 0]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                nt = t + 1

                if a != 0:
                    if not self._is_passable_at(grid, nx, ny, nt, box_destroyed_time):
                        continue

                if danger_soon and npos in danger_soon:
                    continue

                if (a != 0 and npos in bombs_at_time.get(nt, set())) or npos in blast_at_time.get(nt, set()):
                    continue

                if nt == 1 and occupied_at_t1 and npos in occupied_at_t1:
                    continue

                if enemy_dists and self._is_trap_zone(grid, npos, nt, enemy_dists, players, max_size=5, extra_blocked=eb):
                    continue

                cost_step = 1.0 if a != 0 else 0.05
                if grid[nx, ny] in [3, 4]:
                    cost_step -= 2.0

                new_cost = cost + max(0.01, cost_step)
                if seen.get((npos, nt), 999999.0) > new_cost:
                    heapq.heappush(pq, (new_cost, npos, nt, a if first_action is None else first_action))

        return None

    # =========================================================================
    # STRATEGIC UTILITIES
    # =========================================================================

    def _box_bomb_spots(self, grid, occupied, bomb_radius):
        spots = set()
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if not self._passable(grid, x, y) or (x, y) in occupied:
                    continue
                # Check if bomb at (x, y) hits any box within radius
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    for r in range(1, bomb_radius + 1):
                        tx, ty = x + dx * r, y + dy * r
                        if not self._in_bounds(grid, tx, ty):
                            break
                        cell = grid[tx, ty]
                        if cell == 1:  # Wall blocks blast
                            break
                        if cell == 2:  # Box blocks blast
                            spots.add((x, y))
                            break
                    if (x, y) in spots:
                        break
        return spots

    def _shortest_dist(self, grid, start, target):
        start = tuple(start)
        target = tuple(target)
        q = deque([(start, 0)])
        seen = {start}
        while q:
            pos, d = q.popleft()
            if pos == target:
                return d
            for a in [1, 2, 3, 4]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                if npos in seen:
                    continue
                if not self._passable(grid, nx, ny):
                    continue
                seen.add(npos)
                q.append((npos, d + 1))
        return 999

    def _should_place_finisher_bomb(self, grid, my_pos, bombs, players, occupied, enemy_dists,
                                    bomb_radius, bombs_left, alive_count, box_count):
        simulated_bomb = {'pos': my_pos, 'timer': 7, 'radius': bomb_radius}
        escape_move = self._find_safe_path(
            grid, my_pos, bombs, players, simulated_bomb=simulated_bomb,
            occupied_at_t1=occupied, require_permanent_safety=True,
            enemy_dists=enemy_dists
        )
        if escape_move is None:
            return False

        base_bombs = self._get_effective_bombs(grid, bombs, players)
        base_timer_by_pos = {b['pos']: b['timer'] for b in base_bombs}
        effective_bombs = self._get_effective_bombs(grid, bombs, players, simulated_bomb)
        blast_window = set()
        for b in effective_bombs:
            pos = b['pos']
            base_timer = base_timer_by_pos.get(pos, 999)
            if pos == my_pos or b['timer'] < base_timer:
                blast_window |= self._get_blast_tiles(grid, pos[0], pos[1], b['radius'])

        my_blast = self._get_blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)
        pressure_ok = alive_count <= 2 or box_count <= int(0.45 * self._initial_box_count) or bomb_radius >= 4 or bombs_left >= 2

        for idx, ep in enumerate(players):
            if idx == self.agent_id or ep[2] != 1:
                continue
            enemy_pos = (int(ep[0]), int(ep[1]))
            if enemy_pos not in blast_window:
                continue

            enemy_escape = self._find_safe_path(
                grid, enemy_pos, bombs, players, simulated_bomb=simulated_bomb,
                occupied_at_t1=None, require_permanent_safety=True,
                enemy_dists=None
            )
            if enemy_escape is None:
                return True

            if pressure_ok and enemy_pos in my_blast:
                safe_replies = self._count_safe_first_steps_after_bomb(
                    grid, enemy_pos, bombs, players, simulated_bomb
                )
                if safe_replies <= 1:
                    return True

        return False

    def _get_valid_spacing_paths(self, grid, start_pos, bombs, players, B1, max_len=4):
        # Generate straight corridor paths in cardinal directions
        effective_B1 = self._get_effective_bombs(grid, bombs, players, B1)
        box_destroyed_time, blast_at_time, bombs_at_time = self._simulate_grid_and_danger(grid, effective_B1, 8)
        
        paths = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            path = [start_pos]
            for step in range(1, max_len + 1):
                nx, ny = start_pos[0] + dx * step, start_pos[1] + dy * step
                npos = (nx, ny)
                if not self._passable(grid, nx, ny):
                    break
                
                # Ensure npos is safe at time t=step under B1
                if npos in bombs_at_time.get(step, set()) or npos in blast_at_time.get(step, set()):
                    break
                
                path.append(npos)
            if len(path) > 1:
                paths.append(path)
        return paths

    def _should_place_double_bomb_trap(self, grid, my_pos, bombs, players, occupied, enemy_dists,
                                       bomb_radius, bombs_left, alive_count, box_count):
        if bombs_left < 2:
            return False

        # Ensure we can safely place first bomb and escape
        B1 = {'pos': my_pos, 'timer': 7, 'radius': bomb_radius, 'placed_at': 0}
        escape_move = self._find_safe_path(
            grid, my_pos, bombs, players, simulated_bomb=B1,
            occupied_at_t1=occupied, require_permanent_safety=True,
            enemy_dists=enemy_dists
        )
        if escape_move is None:
            return False

        paths = self._get_valid_spacing_paths(grid, my_pos, bombs, players, B1, max_len=3)
        if not paths:
            return False

        # Find enemies nearby
        max_check_dist = bomb_radius + 4
        for idx, ep in enumerate(players):
            if idx == self.agent_id or ep[2] != 1:
                continue
            enemy_pos = (int(ep[0]), int(ep[1]))
            
            dist_to_enemy = abs(my_pos[0] - enemy_pos[0]) + abs(my_pos[1] - enemy_pos[1])
            if dist_to_enemy > max_check_dist:
                continue

            # Check if they can escape B1
            enemy_escape_B1 = self._find_safe_path(
                grid, enemy_pos, bombs, players, simulated_bomb=B1,
                occupied_at_t1=None, require_permanent_safety=True,
                enemy_dists=None
            )
            if enemy_escape_B1 is None:
                continue

            # Search through all paths to place B2
            found_trap = False
            for path in paths:
                if found_trap:
                    break
                # Prioritize t = 2 (perfect spacing), then t = 3, then t = 1 (consecutive)
                t_options = []
                for val in [2, 3, 1]:
                    if val < len(path):
                        t_options.append(val)

                for t in t_options:
                    B2_pos = path[t]
                    B2 = {'pos': B2_pos, 'timer': 7 - t, 'radius': bomb_radius, 'placed_at': t}

                    # Check if enemy has no escape under both bombs
                    enemy_escape_both = self._find_safe_path(
                        grid, enemy_pos, bombs, players, simulated_bomb=[B1, B2],
                        occupied_at_t1=None, require_permanent_safety=True,
                        enemy_dists=None
                    )
                    if enemy_escape_both is None:
                        # Can we safely escape from both bombs?
                        our_escape_both = self._find_safe_path(
                            grid, my_pos, bombs, players, simulated_bomb=[B1, B2],
                            occupied_at_t1=occupied, require_permanent_safety=True,
                            enemy_dists=enemy_dists
                        )
                        if our_escape_both is not None:
                            return True
                        found_trap = True
                        break
        return False

    def _should_place_triple_bomb_trap(self, grid, my_pos, bombs, players, occupied, enemy_dists,
                                       bomb_radius, bombs_left, alive_count, box_count):
        if bombs_left < 3:
            return False

        # Ensure we can safely place B1 and escape
        B1 = {'pos': my_pos, 'timer': 7, 'radius': bomb_radius, 'placed_at': 0}
        escape_move = self._find_safe_path(
            grid, my_pos, bombs, players, simulated_bomb=B1,
            occupied_at_t1=occupied, require_permanent_safety=True,
            enemy_dists=enemy_dists
        )
        if escape_move is None:
            return False

        paths = self._get_valid_spacing_paths(grid, my_pos, bombs, players, B1, max_len=4)
        if not paths:
            return False

        # Find enemies nearby
        max_check_dist = bomb_radius + 4
        for idx, ep in enumerate(players):
            if idx == self.agent_id or ep[2] != 1:
                continue
            enemy_pos = (int(ep[0]), int(ep[1]))
            
            dist_to_enemy = abs(my_pos[0] - enemy_pos[0]) + abs(my_pos[1] - enemy_pos[1])
            if dist_to_enemy > max_check_dist:
                continue

            # Check if they can escape B1
            enemy_escape_B1 = self._find_safe_path(
                grid, enemy_pos, bombs, players, simulated_bomb=B1,
                occupied_at_t1=None, require_permanent_safety=True,
                enemy_dists=None
            )
            if enemy_escape_B1 is None:
                continue

            # Search through all valid paths and spacing combinations
            found_trap = False
            for path in paths:
                if found_trap:
                    break
                # Generate spacing combinations
                spacing_options = []
                for t1 in range(1, len(path) - 1):
                    for t2 in range(t1 + 1, len(path)):
                        spacing_options.append((t1, t2))

                def score_option(opt):
                    val1, val2 = opt
                    if val1 == 2 and val2 == 4:
                        return 10  # Perfect double-spaced intersections
                    if val1 == 2 or (val2 - val1) == 2:
                        return 5   # Partial spaced intersections
                    return 1       # Consecutive or wide

                spacing_options.sort(key=score_option, reverse=True)
                # Limit spacing options for performance
                spacing_options = spacing_options[:3]

                for t1, t2 in spacing_options:
                    B2_pos = path[t1]
                    B3_pos = path[t2]
                    B2 = {'pos': B2_pos, 'timer': 7 - t1, 'radius': bomb_radius, 'placed_at': t1}
                    B3 = {'pos': B3_pos, 'timer': 7 - t2, 'radius': bomb_radius, 'placed_at': t2}

                    # Check if enemy is already trapped by B1 and B2
                    enemy_escape_B12 = self._find_safe_path(
                        grid, enemy_pos, bombs, players, simulated_bomb=[B1, B2],
                        occupied_at_t1=None, require_permanent_safety=True,
                        enemy_dists=None
                    )
                    if enemy_escape_B12 is None:
                        continue  # Handled by double-bomb

                    # Check if enemy is trapped under all three bombs
                    enemy_escape_both = self._find_safe_path(
                        grid, enemy_pos, bombs, players, simulated_bomb=[B1, B2, B3],
                        occupied_at_t1=None, require_permanent_safety=True,
                        enemy_dists=None
                    )
                    if enemy_escape_both is None:
                        # Can we safely escape from all three bombs?
                        our_escape_both = self._find_safe_path(
                            grid, my_pos, bombs, players, simulated_bomb=[B1, B2, B3],
                            occupied_at_t1=occupied, require_permanent_safety=True,
                            enemy_dists=enemy_dists
                        )
                        if our_escape_both is not None:
                            return True
                        found_trap = True
                        break
        return False

    def _count_safe_first_steps_after_bomb(self, grid, start_pos, bombs, players, simulated_bomb):
        effective_bombs = self._get_effective_bombs(grid, bombs, players, simulated_bomb)
        box_destroyed_time, blast_at_time, bombs_at_time = self._simulate_grid_and_danger(grid, effective_bombs, 8)
        count = 0
        for a in [1, 2, 3, 4, 0]:
            nx, ny = self._next_pos(start_pos, a)
            npos = (nx, ny)
            if a != 0:
                if not self._is_passable_at(grid, nx, ny, 1, box_destroyed_time):
                    continue
                if npos in bombs_at_time.get(1, set()):
                    continue
            if npos in blast_at_time.get(1, set()):
                continue
            count += 1
        return count

    def _should_place_value_bomb(self, grid, my_pos, bombs, players, occupied, enemies, enemy_dists,
                                 bomb_radius, bombs_left, alive_count, box_count, box_destroyed_time, escaping_enemies=None,
                                 defensive_mode=False):
        score, info = self._bomb_site_score(
            grid, my_pos, bombs, players, enemies, enemy_dists,
            bomb_radius, bombs_left, alive_count, box_count, travel_dist=0,
            box_destroyed_time=box_destroyed_time, escaping_enemies=escaping_enemies
        )
        if info['guaranteed'] or info['chain_kill_pressure']:
            return True

        # Early game box destruction
        early_game = box_count >= int(0.60 * self._initial_box_count)
        mid_game = int(0.35 * self._initial_box_count) < box_count < int(0.65 * self._initial_box_count)
        
        # Tie break optimization: Heavy bonus for multi box destruction
        if info['boxes'] >= 3:
            # Three or more boxes
            threshold = -5.0
        elif info['boxes'] == 2:
            # Two boxes
            if early_game or mid_game:
                threshold = -3.0
            else:
                threshold = -2.0
        elif info['boxes'] == 1:
            # Single box
            if early_game:
                threshold = 4.1
            elif mid_game:
                threshold = 3.3
            else:
                threshold = 1.15
        else:
            threshold = 4.25
        
        # Gameplay phase adjustments
        if alive_count <= 2:
            if info['direct_enemy']:
                threshold = -2.0  # Always pressure final opponent
            else:
                threshold -= 1.8
                # In 1v1, heavily prioritize box destruction
                if info['boxes'] >= 1:
                    threshold -= 2.0
        elif box_count <= int(0.50 * self._initial_box_count):
            threshold -= 0.5
            # Late game: Multi box is good for tie breaks
            if info['boxes'] >= 2:
                threshold -= 1.5
        
        if bombs_left >= 2:
            threshold -= 0.4
            # With spare bombs, be more aggressive
            if early_game and info['boxes'] >= 1:
                threshold -= 1.0

        # Spam overload: Lower threshold if near enemy with spare bombs
        near_enemy = any(abs(my_pos[0] - ep[0]) + abs(my_pos[1] - ep[1]) <= bomb_radius + 4 for ep in enemies)
        if bombs_left >= 2 and near_enemy:
            threshold -= 1.5

        # Defensive mode: Raise threshold when winning
        if defensive_mode:
            threshold += 2.0
            # Only bomb for guaranteed kills or multi boxes
            if not info['direct_enemy'] and info['boxes'] <= 1:
                return False

        # Spacing check for value bombs
        my_bombs = [(int(b[0]), int(b[1])) for b in bombs if len(b) > 3 and int(b[3]) == self.agent_id]
        for bx, by in my_bombs:
            if abs(my_pos[0] - bx) + abs(my_pos[1] - by) == 1:
                if not info['guaranteed'] and info['boxes'] <= 1:
                    return False

        if score < threshold:
            return False

        # Do not spend bomb if a better site is reachable nearby
        if not info['direct_enemy'] and not info['chain_kill_pressure'] and info['boxes'] <= 2:
            for spot in self._box_bomb_spots(grid, occupied, bomb_radius):
                if spot == my_pos:
                    continue
                d = self._shortest_dist(grid, my_pos, spot)
                if d > 2:
                    continue
                s_score, s_info = self._bomb_site_score(
                    grid, spot, bombs, players, enemies, enemy_dists,
                    bomb_radius, bombs_left, alive_count, box_count,
                    travel_dist=d, box_destroyed_time=box_destroyed_time,
                    escaping_enemies=escaping_enemies
                )
                # Be more forgiving of nearby spots in early game
                min_improvement = 0.5 if early_game else (1.5 + 0.4 * d)
                
                box_count_better = s_info['boxes'] > info['boxes']
                intersection_better = (s_info['boxes'] == info['boxes'] and s_score - score >= min_improvement)
                if box_count_better or intersection_better:
                    return False
        return True

    def _bomb_site_score(self, grid, pos, bombs, players, enemies, enemy_dists,
                         bomb_radius, bombs_left, alive_count, box_count,
                         travel_dist=0, box_destroyed_time=None, escaping_enemies=None):
        simulated_bomb = {'pos': pos, 'timer': 7, 'radius': bomb_radius}
        blast = self._get_blast_tiles(grid, pos[0], pos[1], bomb_radius)
        det_time = 7 + travel_dist

        boxes_hit = 0
        item_tiles = []
        for tx, ty in blast:
            cell = grid[tx, ty]
            if cell == 2:
                if box_destroyed_time is None or (tx, ty) not in box_destroyed_time or box_destroyed_time[(tx, ty)] >= det_time:
                    boxes_hit += 1
            elif cell in [3, 4]:
                item_tiles.append((tx, ty, int(cell)))

        # Prioritize box destruction in early game
        early_game_bonus = 0.0
        if box_count >= int(0.60 * self._initial_box_count):
            # Early game: Boxes abundant, prioritize destruction heavily
            early_game_bonus = boxes_hit * 2.0
        elif int(0.35 * self._initial_box_count) < box_count < int(0.65 * self._initial_box_count):
            # Mid game
            early_game_bonus = boxes_hit * 1.2

        # Weight multi box destruction for tie breaks
        if boxes_hit == 0:
            score = -1.0 - 0.15 * travel_dist
        elif boxes_hit == 1:
            score = 2.2 + early_game_bonus - 0.25 * travel_dist
        elif boxes_hit == 2:
            # Two boxes
            score = 6.5 + early_game_bonus - 0.25 * travel_dist
        elif boxes_hit >= 3:
            # Three or more boxes
            score = 9.0 + (boxes_hit - 3) * 3.5 + early_game_bonus - 0.25 * travel_dist

        # Intersection and explosion directions optimization
        explosion_dirs = self._explosion_directions_count(grid, pos)
        open_exits = self._open_neighbor_count(grid, pos)
        
        if explosion_dirs == 4:
            score += 1.2
        elif explosion_dirs == 3:
            score += 0.6
            
        if open_exits == 4:
            score += 0.8
        elif open_exits == 3:
            score += 0.4
        elif open_exits <= 1:
            score -= 0.3

        # Blast coverage size bonus (rewards open sightlines/corridors)
        score += 0.05 * len(blast)
        
        direct_enemy = False
        guaranteed = False

        # Space-time escape interception matching
        if escaping_enemies:
            for idx, path in escaping_enemies.items():
                t_idx = min(det_time, len(path) - 1)
                target_pos = path[t_idx]
                if target_pos in blast:
                    score += 8.0
                    guaranteed = True
                    direct_enemy = True



        for enemy_pos in enemies:
            if enemy_pos in blast:
                direct_enemy = True
                safe_replies = self._count_safe_first_steps_after_bomb(
                    grid, enemy_pos, bombs, players, simulated_bomb
                )
                score += 4.0
                if safe_replies <= 1:
                    score += 7.0
                    guaranteed = True
                elif safe_replies <= 2:
                    score += 3.0
            else:
                nearest_blast = min((abs(enemy_pos[0] - bx) + abs(enemy_pos[1] - by) for bx, by in blast), default=99)
                if nearest_blast == 1:
                    score += 0.8

            if bombs_left >= 2:
                d_enemy = abs(pos[0] - enemy_pos[0]) + abs(pos[1] - enemy_pos[1])
                if d_enemy <= bomb_radius + 3:
                    score += 1.0
                    if self._open_neighbor_count(grid, enemy_pos) <= 2:
                        score += 1.0

            if alive_count <= 2:
                d_enemy = abs(pos[0] - enemy_pos[0]) + abs(pos[1] - enemy_pos[1])
                if d_enemy <= bomb_radius + 4:
                    score += 2.0
                    if self._open_neighbor_count(grid, enemy_pos) <= 2:
                        score += 1.5

        base_bombs = self._get_effective_bombs(grid, bombs, players)
        base_timer_by_pos = {b['pos']: b['timer'] for b in base_bombs}
        effective_bombs = self._get_effective_bombs(grid, bombs, players, simulated_bomb)
        chain_kill_pressure = False
        chain_boxes = 0  # Count boxes destroyed by chain reactions
        
        for b in effective_bombs:
            bpos = b['pos']
            if bpos == pos:
                continue
            if b['timer'] >= base_timer_by_pos.get(bpos, 999):
                continue
            acc_blast = self._get_blast_tiles(grid, bpos[0], bpos[1], b['radius'])
            score += 1.2
            
            # Count boxes destroyed by triggered bombs
            for cx, cy in acc_blast:
                if grid[cx, cy] == 2 and (cx, cy) not in blast:
                    chain_boxes += 1
            
            for enemy_pos in enemies:
                if enemy_pos in acc_blast:
                    chain_kill_pressure = True
                    score += 5.0
                    if bombs_left >= 2:
                        # Spam overload bonus for chaining bombs to expand threat zones
                        score += 3.0
        
        # Bonus for chain reaction box destruction
        if chain_boxes >= 1:
            score += 3.5 + chain_boxes * 1.5

        for ix, iy, cell in item_tiles:
            if (cell == 4 and bombs_left <= 1) or (cell == 3 and bomb_radius <= 2):
                score -= 3.0
            for idx, dists in enemy_dists.items():
                d_enemy = dists.get((ix, iy), 999)
                if d_enemy <= 3:
                    score += 1.4

        if alive_count <= 2:
            score += 1.4 if direct_enemy else 0.4 * boxes_hit
        elif box_count <= int(0.50 * self._initial_box_count) and (direct_enemy or chain_kill_pressure):
            score += 1.0

        return score, {
            'boxes': boxes_hit,
            'direct_enemy': direct_enemy,
            'guaranteed': guaranteed,
            'chain_kill_pressure': chain_kill_pressure,
        }

    def _find_attack_lane_action(self, grid, my_pos, enemies, bombs, players, occupied, bomb_radius,
                                 bombs_left, alive_count, box_count, danger_soon, enemy_dists, escaping_enemies=None):
        if bomb_radius < 2:
            return None

        # PVP gate: Only join attack lanes when box count is low
        pvp_gate_threshold = max(4, int(0.25 * self._initial_box_count))
        if box_count > pvp_gate_threshold:
            return None

        has_material = bombs_left >= 2 or bomb_radius >= 3
        fight_phase = alive_count <= 2 or box_count <= int(0.60 * self._initial_box_count)
        if not (has_material or fight_phase):
            return None
        if alive_count > 2 and not ((bombs_left >= 2 and bomb_radius >= 3) or box_count <= int(0.40 * self._initial_box_count)):
            return None

        max_depth = 9 if alive_count <= 2 or box_count <= int(0.33 * self._initial_box_count) else 7
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        my_dists = self._compute_all_dists(grid, my_pos, extra_blocked=bomb_positions)
        candidates = []

        for spot, dist in my_dists.items():
            if dist == 0 or dist > max_depth:
                continue
            if spot in occupied or spot in bomb_positions:
                continue
            if danger_soon and spot in danger_soon:
                continue
            if not self._passable(grid, spot[0], spot[1]):
                continue

            blast = self._get_blast_tiles(grid, spot[0], spot[1], bomb_radius)
            direct_seed = any(ep in blast for ep in enemies)
            chain_seed = any((int(b[0]), int(b[1])) in blast for b in bombs)
            if not direct_seed and not chain_seed:
                if alive_count > 2:
                    continue
                nearest_blast = min(
                    (abs(ep[0] - bx) + abs(ep[1] - by) for ep in enemies for bx, by in blast),
                    default=99
                )
                if nearest_blast > 1:
                    continue

            site_score, info = self._bomb_site_score(
                grid, spot, bombs, players, enemies, enemy_dists,
                bomb_radius, bombs_left, alive_count, box_count, travel_dist=dist,
                escaping_enemies=escaping_enemies
            )
            if not (info['direct_enemy'] or info['chain_kill_pressure'] or (alive_count <= 2 and site_score >= 4.0)):
                continue

            exits = self._open_neighbor_count(grid, spot)
            if exits <= 1 and alive_count > 2:
                continue

            nearest_enemy = min(abs(spot[0] - ep[0]) + abs(spot[1] - ep[1]) for ep in enemies)
            score = site_score - 0.35 * dist + 0.25 * exits
            if info['direct_enemy']:
                score += 2.5
            if info['chain_kill_pressure']:
                score += 2.5
            if alive_count <= 2:
                score += 2.0
            if bombs_left >= 2:
                score += 1.0
            if nearest_enemy <= 1:
                score -= 1.2

            candidates.append((score, spot))

        if not candidates:
            return None

        validated = []
        for score, spot in sorted(candidates, reverse=True)[:8]:
            simulated_bomb = {'pos': spot, 'timer': 7, 'radius': bomb_radius}
            post_bomb_escape = self._find_safe_path(
                grid, spot, bombs, players, simulated_bomb=simulated_bomb,
                occupied_at_t1=occupied, require_permanent_safety=True,
                enemy_dists=enemy_dists
            )
            if post_bomb_escape is not None:
                validated.append((score, spot))

        if not validated:
            return None

        best_score = max(score for score, _ in validated)
        targets = {spot for score, spot in validated if score >= best_score - 0.75}
        return self._find_safe_path_to_targets(
            grid, my_pos, targets, bombs, players, max_depth=max_depth,
            occupied_at_t1=occupied, danger_soon=None, enemy_dists=enemy_dists
        )

    def _open_neighbor_count(self, grid, pos):
        count = 0
        for a in [1, 2, 3, 4]:
            nx, ny = self._next_pos(pos, a)
            if self._passable(grid, nx, ny):
                count += 1
        return count

    def _explosion_directions_count(self, grid, pos) -> int:
        count = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            tx, ty = pos[0] + dx, pos[1] + dy
            if 0 <= tx < grid.shape[0] and 0 <= ty < grid.shape[1]:
                if grid[tx, ty] != 1:  # Not a wall
                    count += 1
        return count

    def _find_best_box_spot_action(self, grid, my_pos, box_spots, bombs, players, occupied, bomb_radius, danger_soon=None, enemy_dists=None):
        return self._find_best_box_spot_action_internal(grid, my_pos, box_spots, bombs, players, occupied, bomb_radius, danger_soon, enemy_dists)

    def _find_best_box_spot_action_internal(self, grid, my_pos, box_spots, bombs, players, occupied, bomb_radius, danger_soon=None, enemy_dists=None):
        effective_bombs = self._get_effective_bombs(grid, bombs, players)
        box_destroyed_time, blast_at_time, bombs_at_time = self._simulate_grid_and_danger(grid, effective_bombs, 18)
        
        max_timer = max((b['timer'] for b in effective_bombs), default=0)
        eb = {b['pos'] for b in effective_bombs}

        def is_permanently_safe(p, t):
            if not self._is_passable_at(grid, p[0], p[1], t, box_destroyed_time):
                return False
            for step in range(t, max_timer + 1):
                if p in blast_at_time.get(step, set()):
                    return False
            return True

        pq = [(0.0, my_pos, 0, None)]
        seen = {}  # (pos, t) -> min_cost
        
        reachable_spots = {}
        
        while pq:
            cost, pos, t, first_action = heapq.heappop(pq)
            
            if seen.get((pos, t), 999999.0) <= cost:
                continue
            seen[(pos, t)] = cost
            
            if pos in box_spots and first_action is not None:
                if pos not in reachable_spots:
                    escaped = False
                    if is_permanently_safe(pos, t):
                        escaped = True
                    else:
                        sub_q = deque([(pos, t)])
                        sub_seen = {(pos, t)}
                        while sub_q:
                            sp, st = sub_q.popleft()
                            if is_permanently_safe(sp, st):
                                escaped = True
                                break
                            if st - t >= 6:
                                continue
                            for sa in [1, 2, 3, 4, 0]:
                                snx, sny = self._next_pos(sp, sa)
                                snpos = (snx, sny)
                                snt = st + 1
                                if (snpos, snt) in sub_seen:
                                    continue
                                if sa != 0 and not self._is_passable_at(grid, snx, sny, snt, box_destroyed_time):
                                    continue
                                if snpos in bombs_at_time.get(snt, set()) or snpos in blast_at_time.get(snt, set()):
                                    continue
                                if enemy_dists and self._is_trap_zone(grid, snpos, snt, enemy_dists, players, max_size=5, extra_blocked=eb):
                                    continue
                                sub_seen.add((snpos, snt))
                                sub_q.append((snpos, snt))
                    if escaped:
                        reachable_spots[pos] = (first_action, t)
            
            if t >= 12:  # Max search depth for target pathfinding
                continue
                
            for a in [1, 2, 3, 4, 0]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                nt = t + 1
                
                if a != 0:
                    if not self._is_passable_at(grid, nx, ny, nt, box_destroyed_time):
                        continue
                if danger_soon and npos in danger_soon:
                    continue
                if (a != 0 and npos in bombs_at_time.get(nt, set())) or npos in blast_at_time.get(nt, set()):
                    continue
                if nt == 1 and occupied and npos in occupied:
                    continue
                if enemy_dists and self._is_trap_zone(grid, npos, nt, enemy_dists, players, max_size=5, extra_blocked=eb):
                    continue
                
                cost_step = 1.0 if a != 0 else 0.05
                if grid[nx, ny] in [3, 4]:
                    cost_step -= 2.0
                    
                new_cost = cost + max(0.01, cost_step)
                if seen.get((npos, nt), 999999.0) > new_cost:
                    heapq.heappush(pq, (new_cost, npos, nt, a if first_action is None else first_action))
                
        if not reachable_spots:
            return None
            
        best_action = None
        best_score = -1000
        for spot, (action, dist) in reachable_spots.items():
            # Estimate box hits using simulated box states
            boxes_hit = 0
            blast = self._get_blast_tiles(grid, spot[0], spot[1], bomb_radius)
            for tx, ty in blast:
                if grid[tx, ty] == 2:
                    if (tx, ty) not in box_destroyed_time or box_destroyed_time[(tx, ty)] > dist:
                        boxes_hit += 1
            score = 2.0 * boxes_hit - 0.5 * dist + 0.05 * len(blast)
            
            spawn_has_boxes = self._spawn_area_has_boxes(grid)
            spawn_x, spawn_y = self._spawn_pos if hasattr(self, '_spawn_pos') and self._spawn_pos is not None else (0, 0)
            if spawn_has_boxes and abs(spot[0] - spawn_x) + abs(spot[1] - spawn_y) <= 6:
                score += 15.0
            
            # Intersection optimization
            explosion_dirs = self._explosion_directions_count(grid, spot)
            n_exits = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                sx, sy = spot[0] + dx, spot[1] + dy
                if self._passable(grid, sx, sy):
                    n_exits += 1
                    
            if explosion_dirs == 4:
                score += 1.2
            elif explosion_dirs == 3:
                score += 0.6
                
            if n_exits == 4:
                score += 0.8
            elif n_exits == 3:
                score += 0.4
            elif n_exits <= 1:
                score -= 0.5
            
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _find_escape_path_tiles(self, grid, start_pos, active_bombs, players):
        effective_bombs = self._get_effective_bombs(grid, active_bombs, players)
        box_destroyed_time, blast_at_time, bombs_at_time = self._simulate_grid_and_danger(grid, effective_bombs, 18)
        
        max_timer = max((b['timer'] for b in effective_bombs), default=0)

        def is_permanently_safe(p, t):
            if not self._is_passable_at(grid, p[0], p[1], t, box_destroyed_time):
                return False
            for step in range(t, max_timer + 1):
                if p in blast_at_time.get(step, set()):
                    return False
            return True

        if is_permanently_safe(start_pos, 0):
            return None  # Not currently in danger

        pq = [(0.0, start_pos, 0, [start_pos])]
        seen = {}

        while pq:
            cost, pos, t, path = heapq.heappop(pq)

            if seen.get((pos, t), 999999.0) <= cost:
                continue
            seen[(pos, t)] = cost

            if is_permanently_safe(pos, t):
                return path

            if t >= 16:
                continue

            for a in [1, 2, 3, 4, 0]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                nt = t + 1

                if a != 0:
                    if not self._is_passable_at(grid, nx, ny, nt, box_destroyed_time):
                        continue

                # Obstacle check: solid bomb
                if a != 0 and npos in bombs_at_time.get(nt, set()):
                    continue

                # Danger check: blast explosion
                if npos in blast_at_time.get(nt, set()):
                    continue

                # Cost structure: 1.0 for moving, 0.05 for waiting
                cost_step = 1.0 if a != 0 else 0.05
                next_cost = cost + cost_step

                if seen.get((npos, nt), 999999.0) > next_cost:
                    heapq.heappush(pq, (next_cost, npos, nt, path + [npos]))

        return None

    def _compute_all_dists(self, grid, start, extra_blocked=None):
        dists = {start: 0}
        q = deque([start])
        while q:
            pos = q.popleft()
            d = dists[pos]
            for a in [1, 2, 3, 4]:
                nx, ny = self._next_pos(pos, a)
                npos = (nx, ny)
                if extra_blocked and npos in extra_blocked:
                    continue
                if not self._passable(grid, nx, ny):
                    continue
                if npos not in dists:
                    dists[npos] = d + 1
                    q.append(npos)
        return dists

    def _is_bottleneck(self, grid, pos, B, max_size=6, extra_blocked=None):
        eb = frozenset(extra_blocked) if extra_blocked else frozenset()
        cache_key = (pos, B, max_size, eb)
        if cache_key in self._bottleneck_cache:
            return self._bottleneck_cache[cache_key]
        
        q = deque([pos])
        seen = {pos, B}
        if extra_blocked:
            seen.update(extra_blocked)
        count = 0
        while q:
            curr = q.popleft()
            count += 1
            if count > max_size:
                self._bottleneck_cache[cache_key] = False
                return False
            for a in [1, 2, 3, 4]:
                nx, ny = self._next_pos(curr, a)
                npos = (nx, ny)
                if npos in seen:
                    continue
                if not self._passable(grid, nx, ny):
                    continue
                seen.add(npos)
                q.append(npos)
        self._bottleneck_cache[cache_key] = True
        return True

    def _is_trap_zone(self, grid, pos, t_P, enemy_dists, players, max_size=3, extra_blocked=None):
        for a in [1, 2, 3, 4]:
            B = self._next_pos(pos, a)
            if not self._passable(grid, B[0], B[1]):
                continue
            
            if self._is_bottleneck(grid, pos, B, max_size, extra_blocked):
                for idx, dists in enemy_dists.items():
                    ep = players[idx]
                    if not self._player_has_bombs(players, idx):
                        continue
                    d_E = dists.get(B, 999)
                    # Factor in enemy bomb radius: Very high radius enemies are more dangerous
                    enemy_radius = max(1, int(ep[4]) + 1)
                    # Extend threshold for dangerous enemies
                    effective_threshold = t_P + 2 + (1 if enemy_radius >= 4 else 0)
                    if d_E <= effective_threshold:
                        return True
        return False

    def _check_spacing_override(self, grid, my_pos, bombs, players, occupied, enemy_dists, bomb_radius):
        my_bombs = []
        for b in bombs:
            if len(b) > 3 and int(b[3]) == self.agent_id:
                my_bombs.append((int(b[0]), int(b[1])))
        
        if not my_bombs:
            return None
            
        player_positions = {(int(p[0]), int(p[1])) for i, p in enumerate(players) if p[2] == 1 and i != self.agent_id}
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        
        for bx, by in my_bombs:
            dx = my_pos[0] - bx
            dy = my_pos[1] - by
            dist = abs(dx) + abs(dy)
            if dist == 1:
                # We are adjacent to our own active bomb
                spaced_pos = (my_pos[0] + dx, my_pos[1] + dy)
                if self._passable(grid, spaced_pos[0], spaced_pos[1]) and spaced_pos not in bomb_positions and spaced_pos not in player_positions:
                    # Check if we can safely move to spaced_pos
                    move = self._find_safe_path_to_targets(
                        grid, my_pos, {spaced_pos}, bombs, players, max_depth=2,
                        occupied_at_t1=occupied, danger_soon=None, enemy_dists=enemy_dists
                    )
                    if move is not None:
                        return move
        return None

    def _is_small_pocket(self, grid, pos, t, box_destroyed_time, max_size=4):
        # Flood fill to find size of reachable pocket at time t
        q = deque([pos])
        seen = {pos}
        count = 0
        while q:
            curr = q.popleft()
            count += 1
            if count > max_size:
                return False
            for a in [1, 2, 3, 4]:
                nx, ny = self._next_pos(curr, a)
                npos = (nx, ny)
                if npos in seen:
                    continue
                if self._is_passable_at(grid, nx, ny, t, box_destroyed_time):
                    seen.add(npos)
                    q.append(npos)
        return True

    # =========================================================================
    # ADVANCED MULTI-BOX DESTRUCTION AND TIE-BREAK SYSTEMS
    # =========================================================================

    def _find_multi_box_destruction_spots(self, grid, my_pos, bomb_radius, max_spots=20):
        """Find all passable positions and score them by number of boxes they can destroy."""
        spots_by_boxes = {}  # boxes_destroyed -> [list of (x, y)]
        
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if not self._passable(grid, x, y):
                    continue
                
                blast = self._get_blast_tiles(grid, x, y, bomb_radius)
                boxes_hit = sum(1 for bx, by in blast if grid[bx, by] == 2)
                
                if boxes_hit > 0:
                    if boxes_hit not in spots_by_boxes:
                        spots_by_boxes[boxes_hit] = []
                    spots_by_boxes[boxes_hit].append((x, y))
        
        # Sort by number of boxes (highest first) and flatten
        result = []
        for box_count in sorted(spots_by_boxes.keys(), reverse=True):
            for spot in spots_by_boxes[box_count][:max_spots]:
                result.append((spot, box_count))
        
        return result[:max_spots]

    def _analyze_box_density(self, grid, bomb_radius):
        """Analyze the density of boxes in different regions of the map."""
        density_map = {}  # pos -> box_density_score
        window_size = bomb_radius * 2 + 1
        
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if not self._passable(grid, x, y):
                    continue
                
                # Count boxes in a window around position
                box_count = 0
                for dx in range(-window_size, window_size + 1):
                    for dy in range(-window_size, window_size + 1):
                        nx, ny = x + dx, y + dy
                        if self._in_bounds(grid, nx, ny) and grid[nx, ny] == 2:
                            distance = abs(dx) + abs(dy)
                            # Weight closer boxes more
                            weight = max(0, window_size - distance)
                            box_count += weight
                
                if box_count > 0:
                    density_map[(x, y)] = box_count
        
        return density_map

    def _evaluate_multi_box_destruction(self, grid, pos, bombs, players, bomb_radius, 
                                        bombs_left, box_destroyed_time, simulated_bomb=None):
        """Score a position based on multi-box destruction potential including chain reactions."""
        
        # Direct boxes destroyed
        my_blast = self._get_blast_tiles(grid, pos[0], pos[1], bomb_radius)
        direct_boxes = sum(1 for bx, by in my_blast if grid[bx, by] == 2)
        
        # Chain reaction boxes
        chain_boxes = 0
        chain_damage_positions = set()
        
        simulated_bomb_data = simulated_bomb if simulated_bomb else {'pos': pos, 'timer': 7, 'radius': bomb_radius}
        effective_bombs = self._get_effective_bombs(grid, bombs, players, simulated_bomb_data)
        
        base_bombs = self._get_effective_bombs(grid, bombs, players)
        base_timer_by_pos = {b['pos']: b['timer'] for b in base_bombs}
        
        for b in effective_bombs:
            bpos = b['pos']
            if bpos == pos:
                continue
            # Check if our bomb triggers this bomb early
            if bpos in my_blast:
                # Bomb in blast radius
                base_timer = base_timer_by_pos.get(bpos, 999)
                if b['timer'] < base_timer:
                    # Bomb triggered early
                    chain_blast = self._get_blast_tiles(grid, bpos[0], bpos[1], b['radius'])
                    for cx, cy in chain_blast:
                        if grid[cx, cy] == 2 and (cx, cy) not in my_blast:
                            chain_boxes += 1
                            chain_damage_positions.add((cx, cy))
        
        total_boxes = direct_boxes + chain_boxes
        
        # Score multi box destruction
        score = 0.0
        if direct_boxes == 1:
            score = 1.0
        elif direct_boxes == 2:
            score = 2.8
        elif direct_boxes >= 3:
            score = 4.5 + (direct_boxes - 3) * 1.5
        
        # Bonus for chain reactions
        if chain_boxes > 0:
            score += 2.5 + chain_boxes * 2.0
        
        # Bonus for clearing clusters
        if direct_boxes >= 2:
            score += 1.5
        
        return score, {
            'direct_boxes': direct_boxes,
            'chain_boxes': chain_boxes,
            'total_boxes': total_boxes,
            'score': score
        }

    def _should_place_dual_box_bomb(self, grid, my_pos, bombs, players, occupied, enemy_dists,
                                    bomb_radius, bombs_left, alive_count, box_count):
        """Check if we should place a bomb focused on box destruction using dual bomb strategy."""
        
        if bombs_left < 2:
            return False
        
        # First bomb placement
        B1 = {'pos': my_pos, 'timer': 7, 'radius': bomb_radius, 'placed_at': 0}
        escape_move = self._find_safe_path(
            grid, my_pos, bombs, players, simulated_bomb=B1,
            occupied_at_t1=occupied, require_permanent_safety=True,
            enemy_dists=enemy_dists
        )
        if escape_move is None:
            return False
        
        # Find complementary positions for second bomb
        my_blast_B1 = self._get_blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)
        boxes_from_B1 = sum(1 for x, y in my_blast_B1 if grid[x, y] == 2)
        
        if boxes_from_B1 == 0:
            return False
        
        # Get valid spacing paths for second bomb
        paths = self._get_valid_spacing_paths(grid, my_pos, bombs, players, B1, max_len=4)
        if not paths:
            return False
        
        # Find position that destroys additional boxes
        best_complementary_boxes = 0
        for path in paths:
            for pos_B2 in path[1:]:  # Skip starting position
                blast_B2 = self._get_blast_tiles(grid, pos_B2[0], pos_B2[1], bomb_radius)
                # Count boxes destroyed by B2 unique to B1
                complementary_boxes = sum(1 for x, y in blast_B2 
                                         if grid[x, y] == 2 and (x, y) not in my_blast_B1)
                best_complementary_boxes = max(best_complementary_boxes, complementary_boxes)
        
        # We want total of at least 4 boxes from both bombs combined
        return boxes_from_B1 + best_complementary_boxes >= 4

    def _find_coordinated_dual_bomb_placement(self, grid, my_pos, bombs, players, occupied, 
                                              bomb_radius, bombs_left, box_count, danger_avoid):
        """Find the best dual-bomb placement to maximize box destruction for tie-breaks."""
        
        if bombs_left < 2:
            return None
        
        B1 = {'pos': my_pos, 'timer': 7, 'radius': bomb_radius, 'placed_at': 0}
        
        # Check we can escape first bomb
        escape_move = self._find_safe_path(
            grid, my_pos, bombs, players, simulated_bomb=B1,
            occupied_at_t1=occupied, require_permanent_safety=True,
            enemy_dists={}
        )
        if escape_move is None:
            return None
        
        # Get our blast area
        my_blast_B1 = self._get_blast_tiles(grid, my_pos[0], my_pos[1], bomb_radius)
        boxes_from_B1 = sum(1 for x, y in my_blast_B1 if grid[x, y] == 2)
        
        # Find valid paths for second bomb
        paths = self._get_valid_spacing_paths(grid, my_pos, bombs, players, B1, max_len=3)
        if not paths:
            return None
        
        best_combo = None
        best_total_boxes = 0
        
        for path in paths:
            for t, pos_B2 in enumerate(path[1:], 1):
                if pos_B2 in danger_avoid:
                    continue
                
                # Calculate boxes from second bomb not in first bomb blast
                blast_B2 = self._get_blast_tiles(grid, pos_B2[0], pos_B2[1], bomb_radius)
                boxes_B2_unique = sum(1 for x, y in blast_B2 
                                     if grid[x, y] == 2 and (x, y) not in my_blast_B1)
                
                total_boxes = boxes_from_B1 + boxes_B2_unique
                
                if total_boxes >= 3 and total_boxes > best_total_boxes:
                    best_total_boxes = total_boxes
                    best_combo = (pos_B2, t, total_boxes)
        
        return best_combo  # (position_for_B2, timing, total_boxes_destroyed)

    def _estimate_chain_box_destruction(self, grid, pos, bombs, players, bomb_radius):
        """Estimate how many boxes will be destroyed through chain reactions at a position."""
        
        my_blast = self._get_blast_tiles(grid, pos[0], pos[1], bomb_radius)
        direct_boxes = sum(1 for x, y in my_blast if grid[x, y] == 2)
        
        # Simulate bomb placement and estimate chain reactions
        simulated_bomb = {'pos': pos, 'timer': 7, 'radius': bomb_radius}
        effective_bombs = self._get_effective_bombs(grid, bombs, players, simulated_bomb)
        
        base_bombs = self._get_effective_bombs(grid, bombs, players)
        base_timer_by_pos = {b['pos']: b['timer'] for b in base_bombs}
        
        chain_boxes = 0
        for b in effective_bombs:
            bpos = b['pos']
            if bpos == pos:
                continue
            # Check if bomb in blast is triggered early
            if bpos in my_blast:
                base_timer = base_timer_by_pos.get(bpos, 999)
                if b['timer'] < base_timer:
                    # Count boxes in triggered bomb blast
                    chain_blast = self._get_blast_tiles(grid, bpos[0], bpos[1], b['radius'])
                    chain_boxes += sum(1 for x, y in chain_blast 
                                      if grid[x, y] == 2 and (x, y) not in my_blast)
        
        return direct_boxes + chain_boxes
 
    def _should_prioritize_box_farming(self, box_count, initial_box_count, alive_count, 
                                      bombs_left, bomb_radius, enemies):
        """Determine if we should prioritize box farming over other strategies.
        
        Prioritize destroying boxes throughout the entire game.
        """
        
        # Early game: Farm boxes aggressively
        if box_count >= int(0.60 * initial_box_count):
            if bombs_left >= 1 and bomb_radius >= 2:
                return True
        
        # Mid game
        if int(0.35 * initial_box_count) < box_count < int(0.65 * initial_box_count):
            if bombs_left >= 1:
                return True
        
        # Late game
        if box_count <= int(0.35 * initial_box_count):
            return True
        
        # Focus on farming in 1v1 to win via tie break
        if alive_count <= 2 and box_count <= int(0.50 * initial_box_count):
            return True
        
        # Farm if we have many bombs but low radius
        if bombs_left >= 3 and bomb_radius <= 2:
            return True
        
        # Prioritize if we can get 2+ boxes at current position
        return False

    def _spawn_area_has_boxes(self, grid) -> bool:
        if not hasattr(self, '_spawn_pos') or self._spawn_pos is None:
            return False
        sx, sy = self._spawn_pos
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x, y] == 2:
                    if abs(x - sx) + abs(y - sy) <= 6:
                        return True
        return False

    def _find_next_box_action(self, grid, my_pos, bombs, players, occupied, enemy_dists,
                             bomb_radius, bombs_left, alive_count, box_count, danger_avoid):
        """Immediately pursue next box spot after placing a bomb.
        
        Chain bomb placements for efficiency.
        """
        
        if bombs_left <= 0:
            return None
        
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        # Find nearby multi-box spots
        my_dists = self._compute_all_dists(grid, my_pos, extra_blocked=bomb_positions)
        
        next_targets = []
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if not self._passable(grid, x, y) or (x, y) in occupied:
                    continue
                
                if (x, y) == my_pos:
                    continue
                
                blast = self._get_blast_tiles(grid, x, y, bomb_radius)
                boxes_here = sum(1 for bx, by in blast if grid[bx, by] == 2)
                
                # Only pursue 2+ box spots
                if boxes_here < 2:
                    continue
                
                dist = my_dists.get((x, y), 999)
                
                # Chain bombing search range
                max_dist = 12
                if dist > max_dist:
                    continue
                
                if (x, y) in danger_avoid:
                    continue
                
                # Score based on distance and boxes
                score = 5.0 * boxes_here - 0.4 * dist + 0.05 * len(blast)
                
                spawn_has_boxes = self._spawn_area_has_boxes(grid)
                spawn_x, spawn_y = self._spawn_pos if hasattr(self, '_spawn_pos') and self._spawn_pos is not None else (0, 0)
                if spawn_has_boxes and abs(x - spawn_x) + abs(y - spawn_y) <= 6:
                    score += 15.0
                
                # Reward intersections with more explosion directions
                explosion_dirs = self._explosion_directions_count(grid, (x, y))
                n_exits = self._open_neighbor_count(grid, (x, y))
                
                if explosion_dirs == 4:
                    score += 1.2
                elif explosion_dirs == 3:
                    score += 0.6
                    
                if n_exits == 4:
                    score += 0.8
                elif n_exits == 3:
                    score += 0.4
                elif n_exits <= 1:
                    score -= 0.5
                    
                next_targets.append((score, (x, y)))
        
        if not next_targets:
            return None
        
        # Get top targets
        targets = {spot for score, spot in sorted(next_targets, reverse=True)[:4]}
        
        # Use aggressive pathfinding
        return self._find_safe_path_to_targets(
            grid, my_pos, targets, bombs, players, max_depth=10,
            occupied_at_t1=occupied, danger_soon=danger_avoid, enemy_dists=enemy_dists
        )

    def _find_box_farming_action(self, grid, my_pos, bombs, players, occupied, enemy_dists,
                                bomb_radius, bombs_left, alive_count, box_count, danger_avoid):
        """Dedicated box farming action that focuses on box destruction throughout the game."""
        
        # Always look for multi-box spots if we have bombs
        if bombs_left <= 0:
            return None
        
        # Find multi-box spots
        multi_box_spots = self._find_multi_box_destruction_spots(grid, my_pos, bomb_radius, max_spots=12)
        
        if not multi_box_spots:
            return None
        
        bomb_positions = {(int(b[0]), int(b[1])) for b in bombs}
        # Score each spot by distance and box count
        candidates = []
        my_dists = self._compute_all_dists(grid, my_pos, extra_blocked=bomb_positions)
        
        for spot, box_count_at_spot in multi_box_spots:
            # Prioritize spots with 2+ boxes
            if box_count_at_spot < 2:
                continue
            
            if spot == my_pos:
                continue
            
            dist = my_dists.get(spot, 999)
            
            # Set search distance based on game phase
            max_dist = 20 if box_count >= int(0.50 * self._initial_box_count) else 15
            if dist >= max_dist:
                continue
            
            if spot in occupied or spot in danger_avoid:
                continue
            
            # Weight box spots by box count and distance
            base_score = 4.0 if box_count_at_spot == 2 else (6.0 + (box_count_at_spot - 3) * 2.0)
            blast = self._get_blast_tiles(grid, spot[0], spot[1], bomb_radius)
            score = base_score - 0.25 * dist + 0.05 * len(blast)
            
            spawn_has_boxes = self._spawn_area_has_boxes(grid)
            spawn_x, spawn_y = self._spawn_pos if hasattr(self, '_spawn_pos') and self._spawn_pos is not None else (0, 0)
            if spawn_has_boxes and abs(spot[0] - spawn_x) + abs(spot[1] - spawn_y) <= 6:
                score += 15.0

            # Intersection optimization
            explosion_dirs = self._explosion_directions_count(grid, spot)
            n_exits = self._open_neighbor_count(grid, spot)
            
            if explosion_dirs == 4:
                score += 1.2
            elif explosion_dirs == 3:
                score += 0.6
                
            if n_exits == 4:
                score += 0.8
            elif n_exits == 3:
                score += 0.4
            elif n_exits <= 1:
                score -= 0.5

            candidates.append((score, spot))
        
        if not candidates:
            return None
        
        # Target the top spots
        targets = {spot for score, spot in sorted(candidates, reverse=True)[:6]}
        return self._find_safe_path_to_targets(
            grid, my_pos, targets, bombs, players, max_depth=14,
            occupied_at_t1=occupied, danger_soon=danger_avoid, enemy_dists=enemy_dists
        )

    def _find_box_corridor_chains(self, grid, bomb_radius):
        """Find corridors with multiple boxes in sequence."""
        chains = []  # List of (corridor_positions, total_boxes, box_positions)
        
        # Scan horizontal corridors
        for y in range(grid.shape[1]):
            x = 0
            while x < grid.shape[0]:
                if grid[x, y] == 2:  # Found box
                    # Start of potential chain
                    chain_start = x
                    boxes_in_chain = []
                    while x < grid.shape[0]:
                        if grid[x, y] == 2:
                            boxes_in_chain.append((x, y))
                            x += 1
                        elif grid[x, y] in [0, 3, 4]:  # Passable
                            x += 1
                        else:  # Wall
                            break
                    
                    if len(boxes_in_chain) >= 2:
                        # Analyze optimal bombing positions for chain
                        corridor = list(range(chain_start, x))
                        chains.append({
                            'direction': 'horizontal',
                            'corridor_range': (chain_start, x),
                            'y': y,
                            'boxes': boxes_in_chain,
                            'box_count': len(boxes_in_chain)
                        })
                else:
                    x += 1
        
        # Scan vertical corridors
        for x in range(grid.shape[0]):
            y = 0
            while y < grid.shape[1]:
                if grid[x, y] == 2:  # Found box
                    # Start of potential chain
                    chain_start = y
                    boxes_in_chain = []
                    while y < grid.shape[1]:
                        if grid[x, y] == 2:
                            boxes_in_chain.append((x, y))
                            y += 1
                        elif grid[x, y] in [0, 3, 4]:  # Passable
                            y += 1
                        else:  # Wall
                            break
                    
                    if len(boxes_in_chain) >= 2:
                        # Found a chain
                        corridor = list(range(chain_start, y))
                        chains.append({
                            'direction': 'vertical',
                            'corridor_range': (chain_start, y),
                            'x': x,
                            'boxes': boxes_in_chain,
                            'box_count': len(boxes_in_chain)
                        })
                else:
                    y += 1
        
        return chains

    def _detect_box_hotspots(self, grid, bomb_radius, initial_box_count, current_box_count):
        """Identify areas with high box density.
        
        Returns list of hotspot regions sorted by potential.
        """
        hotspots = []
        visited = set()
        
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if (x, y) in visited or grid[x, y] != 2:
                    continue
                
                # BFS to find connected box clusters
                cluster = set()
                q = deque([(x, y)])
                visited.add((x, y))
                
                while q:
                    cx, cy = q.popleft()
                    cluster.add((cx, cy))
                    
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = cx + dx, cy + dy
                        if (nx, ny) not in visited and self._in_bounds(grid, nx, ny):
                            if grid[nx, ny] == 2:
                                visited.add((nx, ny))
                                q.append((nx, ny))
                
                # Analyze this cluster
                if len(cluster) >= 2:
                    # Find best bombing positions for cluster
                    best_hit_count = 0
                    for bx, by in cluster:
                        blast = self._get_blast_tiles(grid, bx, by, bomb_radius)
                        hit_count = sum(1 for x, y in blast if (x, y) in cluster)
                        best_hit_count = max(best_hit_count, hit_count)
                    
                    hotspots.append({
                        'cluster': cluster,
                        'size': len(cluster),
                        'best_hit_count': best_hit_count,
                        'centroid': (sum(x for x, y in cluster) // len(cluster),
                                   sum(y for x, y in cluster) // len(cluster))
                    })
        
        # Sort by number of boxes we can hit
        hotspots.sort(key=lambda h: h['best_hit_count'], reverse=True)
        return hotspots

    def _score_corridor_bombing(self, grid, chain_info, bomb_radius):
        """Score how many boxes can be destroyed with coordinated bombs in a corridor."""
        boxes = chain_info['boxes']
        box_count = len(boxes)
        
        if chain_info['direction'] == 'horizontal':
            y = chain_info['y']
            # Try placing bombs at different positions
            # Simulate bomb placement along corridor
            best_coverage = 0
            for bx in range(chain_info['corridor_range'][0], chain_info['corridor_range'][1]):
                blast = self._get_blast_tiles(grid, bx, y, bomb_radius)
                hit_count = sum(1 for bx, by in blast if grid[bx, by] == 2)
                best_coverage = max(best_coverage, hit_count)
            return best_coverage
        else:  # vertical
            x = chain_info['x']
            best_coverage = 0
            for by in range(chain_info['corridor_range'][0], chain_info['corridor_range'][1]):
                blast = self._get_blast_tiles(grid, x, by, bomb_radius)
                hit_count = sum(1 for bx, by in blast if grid[bx, by] == 2)
                best_coverage = max(best_coverage, hit_count)
            return best_coverage

    # =========================================================================
    # ENHANCED ENEMY PREDICTION AND TACTICAL SYSTEMS
    # =========================================================================

    def _predict_enemy_next_moves(self, enemy_idx, players, grid, bombs):
        """Predict where an enemy will move in the next few steps using their velocity."""
        if enemy_idx >= len(players) or players[enemy_idx][2] != 1:
            return None
        
        enemy_pos = (int(players[enemy_idx][0]), int(players[enemy_idx][1]))
        velocity = self._enemy_velocities.get(enemy_idx, (0, 0))
        
        if velocity == (0, 0):
            return [enemy_pos]  # Stays in place
        
        # Predict next 3 steps
        predicted_path = [enemy_pos]
        curr_pos = enemy_pos
        
        for step in range(3):
            next_x = curr_pos[0] + velocity[0]
            next_y = curr_pos[1] + velocity[1]
            next_pos = (next_x, next_y)
            
            # Check if next position is passable
            if not self._in_bounds(grid, next_x, next_y) or not self._passable(grid, next_x, next_y):
                # Stop or change direction
                break
            
            predicted_path.append(next_pos)
            curr_pos = next_pos
        
        return predicted_path

    def _find_escape_interception_points(self, grid, enemy_pos, escaping_path, bomb_radius):
        """Find positions where we can place a bomb to intercept an enemy on their escape route."""
        if not escaping_path or len(escaping_path) < 2:
            return []
        
        intercept_points = []
        
        # Check positions along escape path
        for i, path_pos in enumerate(escaping_path[:6]):  # Check first 6 steps of escape
            # Check positions to blast path position
            for dx in range(-bomb_radius, bomb_radius + 1):
                for dy in range(-bomb_radius, bomb_radius + 1):
                    bx, by = path_pos[0] + dx, path_pos[1] + dy
                    if not self._in_bounds(grid, bx, by) or not self._passable(grid, bx, by):
                        continue
                    
                    blast = self._get_blast_tiles(grid, bx, by, bomb_radius)
                    if path_pos in blast:
                        # Intercept enemy on escape route
                        time_until_hit = i + 1
                        score = 10.0 / (time_until_hit + 1)  # Earlier is better
                        intercept_points.append(((bx, by), score, time_until_hit))
        
        return sorted(intercept_points, key=lambda x: x[1], reverse=True)

    def _evaluate_defensive_position(self, grid, my_pos, enemies, bomb_radius, danger_avoid):
        """Evaluate how safe a position is for defensive bomb placement."""
        if not enemies:
            return 1.0
        
        # Calculate minimum distance to enemies
        min_dist_to_enemy = min(abs(my_pos[0] - ep[0]) + abs(my_pos[1] - ep[1]) for ep in enemies)
        
        # Calculate escape routes
        escape_routes = self._open_neighbor_count(grid, my_pos)
        
        # Check if position is in danger soon
        if my_pos in danger_avoid:
            return 0.3
        
        # Score based on distance
        distance_score = 1.0
        if min_dist_to_enemy <= 2:
            distance_score = 0.2  # Too close
        elif min_dist_to_enemy <= 4:
            distance_score = 1.0  # Good distance
        elif min_dist_to_enemy <= 6:
            distance_score = 0.8  # Acceptable
        else:
            distance_score = 0.5  # Too far to threaten
        
        # Bonus for escape routes
        route_bonus = min(escape_routes * 0.2, 1.0)
        
        return distance_score * (1.0 + route_bonus)

    def _find_optimal_spacing_position(self, grid, my_pos, bombs, bomb_radius, bombs_left):
        """Find optimal position for next bomb to maximize spacing from previous bombs."""
        if bombs_left <= 1 or len(bombs) == 0:
            return None
        
        my_bombs = [(int(b[0]), int(b[1])) for b in bombs if len(b) > 3 and int(b[3]) == self.agent_id]
        if not my_bombs:
            return None
        
        # Find positions 2-3 steps away from our bombs
        ideal_spacing_positions = []
        
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if not self._passable(grid, x, y):
                    continue
                
                pos = (x, y)
                
                # Calculate minimum distance to any of our bombs
                min_bomb_dist = min(abs(x - bx) + abs(y - by) for bx, by in my_bombs)
                
                # Spacing of 2-3 steps avoids triggering each other
                if 2 <= min_bomb_dist <= 3:
                    # Calculate boxes hit
                    blast = self._get_blast_tiles(grid, x, y, bomb_radius)
                    boxes_here = sum(1 for bx, by in blast if grid[bx, by] == 2)
                    
                    if boxes_here >= 1:
                        score = boxes_here * 2.0 + min(min_bomb_dist - 2, 1)
                        ideal_spacing_positions.append((score, pos))
        
        if not ideal_spacing_positions:
            return None
        
        # Return best position for next bomb
        return max(ideal_spacing_positions, key=lambda x: x[0])[1]

    def _analyze_enemy_threat_level(self, grid, enemy_idx, players, my_pos, bomb_radius, bombs_left):
        """Analyze how threatening a specific enemy is to our current position."""
        if enemy_idx >= len(players) or players[enemy_idx][2] != 1:
            return 0.0
        
        enemy_pos = (int(players[enemy_idx][0]), int(players[enemy_idx][1]))
        
        # Distance threat
        dist = abs(my_pos[0] - enemy_pos[0]) + abs(my_pos[1] - enemy_pos[1])
        distance_threat = max(0.0, 1.0 - (dist / 10.0))
        
        # Check if enemy can directly blast us
        blast_threat = 0.0
        for potential_bomb_radius in range(1, bomb_radius + 2):
            blast = self._get_blast_tiles(grid, enemy_pos[0], enemy_pos[1], potential_bomb_radius)
            if my_pos in blast:
                blast_threat = 1.0
                break
        
        # Check bomb advantage
        enemy_bombs = players[enemy_idx][3]  # bombs_left for this player
        bomb_threat = 0.0
        if enemy_bombs > bombs_left:
            bomb_threat = min((enemy_bombs - bombs_left) / 5.0, 0.5)
        
        # Combined threat
        threat = (distance_threat * 0.4) + (blast_threat * 0.4) + (bomb_threat * 0.2)
        return min(threat, 1.0)

    def _should_play_aggressive(self, grid, players, my_pos, alive_count, box_count, bombs_left, bomb_radius, enemies):
        """Determine if we should play aggressive or conservative."""
        # 1v1 endgame with few boxes
        if alive_count <= 2 and box_count <= int(0.40 * self._initial_box_count):
            return True
        
        # Bomb and radius advantage
        if bombs_left >= 2 and bomb_radius >= 3:
            return True
        
        # Low threat enemies
        if enemies:
            alive_enemy_threats = []
            for idx, p in enumerate(players):
                if idx != self.agent_id and p[2] == 1:
                    alive_enemy_threats.append(
                        self._analyze_enemy_threat_level(grid, idx, players, my_pos, bomb_radius, bombs_left)
                    )
            avg_enemy_threat = sum(alive_enemy_threats) / len(alive_enemy_threats) if alive_enemy_threats else 0.0
            if avg_enemy_threat < 0.3:
                return True
        
        # Default: Play conservative
        return False

    def _find_minimum_viable_farm_spot(self, grid, my_pos, bomb_radius, max_distance=8):
        """Find closest spot with 2+ boxes for desperate farming."""
        my_dists = self._compute_all_dists(grid, my_pos)
        best_spots = []
        
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if not self._passable(grid, x, y):
                    continue
                
                pos = (x, y)
                dist = my_dists.get(pos, 999)
                if dist > max_distance:
                    continue
                
                blast = self._get_blast_tiles(grid, x, y, bomb_radius)
                boxes_here = sum(1 for bx, by in blast if grid[bx, by] == 2)
                
                if boxes_here >= 2:
                    score = boxes_here * 3.0 - dist * 0.5
                    best_spots.append((score, pos, dist))
        
        if not best_spots:
            return None
        
        return max(best_spots, key=lambda x: x[0])[1]

    def _count_valid_escape_routes(self, grid, start_pos, bombs, players, simulated_bomb) -> int:
        effective_bombs = self._get_effective_bombs(grid, bombs, players, simulated_bomb)
        box_destroyed_time, blast_at_time, bombs_at_time = self._simulate_grid_and_danger(grid, effective_bombs, 12)
        
        count = 0
        for a in [1, 2, 3, 4, 0]:
            nx, ny = self._next_pos(start_pos, a)
            npos = (nx, ny)
            if a != 0:
                if not self._is_passable_at(grid, nx, ny, 1, box_destroyed_time):
                    continue
                if npos in bombs_at_time.get(1, set()):
                    continue
            if npos in blast_at_time.get(1, set()):
                continue
                
            # Verify escape to safety
            if self._find_safe_path(grid, npos, bombs, players, simulated_bomb=simulated_bomb, max_depth=10, occupied_at_t1=None, require_permanent_safety=True, enemy_dists=None) is not None:
                count += 1
        return count
        
    def _should_overload_enemy_with_bombs(self, grid, my_pos, bombs, players, occupied, enemies, enemy_dists, bomb_radius, bombs_left) -> bool:
        # We need 2+ bombs available
        if bombs_left < 2:
            return False
            
        # We need 1+ active bomb of our own
        my_active_bombs = [(int(b[0]), int(b[1])) for b in bombs if len(b) > 3 and int(b[3]) == self.agent_id]
        if not my_active_bombs:
            return False
            
        # Check if enemy is in close range
        close_enemies = []
        for ep in enemies:
            dist = abs(my_pos[0] - ep[0]) + abs(my_pos[1] - ep[1])
            if dist <= 3:
                close_enemies.append(ep)
                
        if not close_enemies:
            return False
            
        # Ensure we can safely place the bomb and escape
        simulated_bomb = {'pos': my_pos, 'timer': 7, 'radius': bomb_radius}
        escape_move = self._find_safe_path(
            grid, my_pos, bombs, players, simulated_bomb=simulated_bomb,
            occupied_at_t1=occupied, require_permanent_safety=True,
            enemy_dists=enemy_dists
        )
        if escape_move is None:
            return False
            
        # Check if the bomb restricts the options of the closest close enemy
        primary_enemy = min(close_enemies, key=lambda ep: abs(my_pos[0] - ep[0]) + abs(my_pos[1] - ep[1]))
        
        safe_exits_before = self._count_valid_escape_routes(grid, primary_enemy, bombs, players, None)
        safe_exits_after = self._count_valid_escape_routes(grid, primary_enemy, bombs, players, simulated_bomb)
        
        # Overload conditions: restricts enemy escape routes
        if safe_exits_before >= 1 and safe_exits_after < safe_exits_before and safe_exits_after <= 1:
            return True
            
        return False

