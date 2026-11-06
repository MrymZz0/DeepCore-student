"""
visualizer.py
Sci-Fi Tactical Game UI for DeepCore AI Search.
Natively loads drill_vehicle.png, teleporter_pad.png, rock_wall.png, ore_rubble.png,
pit.png, toxic_slime.png, bridge.png, boulder.png, and miner.png.
[TA PROVIDED - DO NOT MODIFY]
"""

from dataclasses import dataclass
import math
import os
import random
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple
import pygame

from .environment import DeepCoreEnv, Action, BUILTIN_MAPS
from .state import DeepCoreState, SearchMetrics, Pos


# =============================================================================
# Vector Icon Helper
# =============================================================================
def draw_vector_icon(surf: pygame.Surface, kind: str, cx: int, cy: int, color: Tuple[int, int, int]) -> None:
    if kind == "play":
        pygame.draw.polygon(surf, color, [(cx - 4, cy - 6), (cx - 4, cy + 6), (cx + 6, cy)])
    elif kind == "pause":
        pygame.draw.rect(surf, color, (cx - 5, cy - 6, 3, 12), border_radius=1)
        pygame.draw.rect(surf, color, (cx + 2, cy - 6, 3, 12), border_radius=1)
    elif kind == "star":
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            r = 6.5 if i % 2 == 0 else 2.8
            pts.append((cx + math.cos(ang) * r, cy + math.sin(ang) * r))
        pygame.draw.polygon(surf, color, pts)
    elif kind == "dpad":
        pygame.draw.rect(surf, color, (cx - 2, cy - 6, 4, 12), border_radius=1)
        pygame.draw.rect(surf, color, (cx - 6, cy - 2, 12, 4), border_radius=1)
    elif kind == "chart":
        pygame.draw.rect(surf, color, (cx - 6, cy + 1, 3, 5))
        pygame.draw.rect(surf, color, (cx - 1, cy - 5, 3, 11))
        pygame.draw.rect(surf, color, (cx + 4, cy - 2, 3, 8))
    elif kind == "reset":
        pygame.draw.arc(surf, color, (cx - 6, cy - 6, 12, 12), 0.6, math.pi * 1.85, 2)
        pygame.draw.polygon(surf, color, [(cx + 2, cy - 6), (cx + 7, cy - 6), (cx + 5, cy - 1)])
    elif kind == "wave":
        pygame.draw.circle(surf, color, (cx, cy), 2)
        pygame.draw.circle(surf, color, (cx, cy), 5, 1)
        pygame.draw.circle(surf, color, (cx, cy), 8, 1)
    elif kind == "path":
        pygame.draw.line(surf, color, (cx - 5, cy + 4), (cx, cy - 4), 2)
        pygame.draw.line(surf, color, (cx, cy - 4), (cx + 5, cy + 2), 2)
        pygame.draw.circle(surf, color, (cx - 5, cy + 4), 2)
        pygame.draw.circle(surf, color, (cx + 5, cy + 2), 2)
    elif kind == "heat":
        for dx in (-4, 1):
            for dy in (-4, 1):
                pygame.draw.rect(surf, color, (cx + dx, cy + dy, 4, 4), border_radius=1)


@dataclass
class PillButton:
    rect: pygame.Rect
    label: str
    group: str
    value: str = ""
    active: bool = False
    style: str = "default"
    icon: str = ""


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    color: Tuple[int, int, int]
    life: float
    max_life: float


class DeepCoreWorkbench:
    WIN_W = 1220
    WIN_H = 780
    TOP_BAR_H = 98
    BOT_BAR_H = 58

    BG_CAVERN = (11, 14, 20)
    BAR_BG = (19, 24, 34)
    BAR_BORDER = (44, 56, 78)
    PILL_BG = (30, 38, 54)
    PILL_HOVER = (44, 56, 78)
    TEXT_MAIN = (240, 245, 252)

    ACCENT_BLUE = (52, 152, 245)
    ACCENT_PURPLE = (155, 92, 245)
    ACCENT_GREEN = (40, 198, 120)
    ACCENT_AMBER = (245, 180, 45)
    ACCENT_RED = (235, 75, 75)

    ALGO_PILLS = [
        ("manual", "Manual", "dpad"),
        ("bfs", "BFS", ""),
        ("dfs", "DFS", ""),
        ("ids", "IDS", ""),
        ("bidirectional", "Bidir", ""),
        ("ucs", "UCS", ""),
        ("gbfs", "Greedy", ""),
        ("astar", "A*", ""),
        ("wastar", "W-A*", "star"),
        ("idastar", "IDA*", "star"),
        ("sma", "SMA*", "star"),
    ]

    HEUR_PILLS = [
        ("naive", "H0: Naive"),
        ("h1", "H1: MST+Pod"),
        ("h2", "H2: Master"),
    ]

    WEIGHT_PILLS = [1.5, 2.0, 3.0]
    SPEED_PILLS = [1.0, 2.0, 4.0, 8.0]

    def __init__(
        self,
        env: Optional[DeepCoreEnv] = None,
        metrics: Optional[SearchMetrics] = None,
        manual_mode: bool = True,
        initial_level: str = "level1",
        initial_algo: str = "manual",
        initial_heur: str = "naive",
        initial_weight: float = 1.5,
        runner_callback: Optional[Callable] = None,
        heuristic_lookup: Optional[Dict[str, Callable[[DeepCoreState, DeepCoreEnv], float]]] = None,
    ):
        pygame.init()
        pygame.display.set_caption("DeepCore - Autonomous Mine Rescue")
        self.screen = pygame.display.set_mode((self.WIN_W, self.WIN_H), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.f_title = pygame.font.SysFont(["Segoe UI", "Arial"], 15, bold=True)
        self.f_pill = pygame.font.SysFont(["Segoe UI", "Arial"], 13, bold=True)
        self.f_ui = pygame.font.SysFont(["Segoe UI", "Arial"], 13)
        self.f_mono = pygame.font.SysFont(["Consolas", "Courier New"], 13, bold=True)
        self.f_small = pygame.font.SysFont(["Consolas", "Courier New"], 11, bold=True)

        self.runner_callback = runner_callback
        self.heuristic_lookup = heuristic_lookup or {}

        self.level_keys = list(BUILTIN_MAPS.keys())
        self.selected_level = initial_level if initial_level in self.level_keys else "level1"
        self.selected_algo = initial_algo
        self.selected_heur = initial_heur
        self.selected_weight = initial_weight

        self.show_heatmap = True
        self.show_path = True
        self.wave_mode = False
        self.show_compare_modal = False

        self.speed_idx = 1
        self.playing = False
        self.last_step_ms = 0
        self.tween_t = 1.0
        self.path_cursor = 0
        self.wave_cursor = 0

        self.is_searching = False
        self.search_start_time = 0.0
        self.search_job_id = 0
        self.pending_result: Optional[SearchMetrics] = None

        self.raw_sprites: Dict[str, pygame.Surface] = {}
        self.sprites: Dict[str, pygame.Surface] = {}
        self._load_custom_sprites()

        self.env: DeepCoreEnv = DeepCoreEnv.from_file_or_builtin(self.selected_level)
        self.metrics: Optional[SearchMetrics] = metrics
        self.compare_rows: List[SearchMetrics] = []
        self.status_msg = "WASD/Arrows: Play Manually | Click 'BFS' to test sample solver!"
        self.status_col = self.ACCENT_GREEN

        self.trajectory: List[Tuple[DeepCoreState, str, int, int]] = []
        self.explored_rank: Dict[Pos, int] = {}

        self.manual_state: DeepCoreState = self.env.get_initial_state()
        self.prev_bot_pos: Pos = self.manual_state.bot_pos
        self.manual_cost: int = 0
        self.manual_undo: List[Tuple[DeepCoreState, int, str]] = []
        self.manual_steps: int = 0
        self.bot_dir: str = "S"

        self.pills: List[PillButton] = []
        self.particles: List[Particle] = []
        self.hover_tile: Optional[Pos] = None

        self._load_level(self.selected_level)
        if self.selected_algo != "manual":
            self.run_active_algorithm()

    # -------------------------------------------------------------------------
    # Custom Sprite Loader
    # -------------------------------------------------------------------------
    def _load_custom_sprites(self) -> None:
        self.raw_sprites.clear()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_dirs = [
            os.path.join(base_dir, "assets"),
            os.path.normpath(os.path.join(base_dir, "..", "..", "assets")),
            os.path.join(os.getcwd(), "assets"),
            "assets",
        ]
        asset_dir = next((d for d in candidate_dirs if os.path.isdir(d)), None)
        if not asset_dir:
            return

        aliases = {
            "bot": ["drill_vehicle", "bot", "robot", "player"],
            "wall": ["rock_wall", "wall", "walls", "brick"],
            "rubble": ["ore_rubble", "rubble", "debris", "stones"],
            "chasm": ["pit", "chasm", "hole", "lava"],
            "elevator": ["teleporter_pad", "elevator", "portal", "exit"],
            "sludge": ["toxic_slime", "sludge", "toxic", "acid"],
            "bridge": ["bridge", "plank", "planks"],
            "boulder": ["boulder", "rock", "stone"],
            "miner": ["miner", "npc", "worker"],
            "floor": ["floor", "ground", "tile"],
        }

        for canonical, file_names in aliases.items():
            for fname in file_names:
                for ext in (".png", ".jpg", ".jpeg", ".webp"):
                    path = os.path.join(asset_dir, fname + ext)
                    if os.path.isfile(path):
                        try:
                            self.raw_sprites[canonical] = pygame.image.load(path).convert_alpha()
                            break
                        except Exception as err:
                            print(f"[Warning] Failed loading {path}: {err}")
                if canonical in self.raw_sprites:
                    break

        sheet_path = os.path.join(asset_dir, "Sci-Fi Mining Rescue Icon Grid.png")
        if os.path.isfile(sheet_path):
            try:
                sheet = pygame.image.load(sheet_path).convert_alpha()
                sw, sh = sheet.get_size()
                tw, th = sw // 3, sh // 3
                grid_map = {
                    (0, 0): "bot",      (0, 1): "elevator", (0, 2): "miner",
                    (1, 0): "boulder",  (1, 1): "chasm",    (1, 2): "bridge",
                    (2, 0): "sludge",   (2, 1): "rubble",   (2, 2): "wall",
                }
                for (row, col), key in grid_map.items():
                    if key not in self.raw_sprites:
                        sub = sheet.subsurface(pygame.Rect(col * tw, row * th, tw, th)).copy()
                        self.raw_sprites[key] = sub
            except Exception as err:
                print(f"[Warning] Failed slicing grid sheet: {err}")

    def _rescale_sprites(self) -> None:
        ts = self.tile_size
        self.sprites = {
            name: pygame.transform.smoothscale(img, (ts, ts))
            for name, img in self.raw_sprites.items()
        }

    # -------------------------------------------------------------------------
    # Diagnostics & Safe Runners
    # -------------------------------------------------------------------------
    def _eval_heuristic_display(self, h_key: str, state: DeepCoreState) -> str:
        fn = self.heuristic_lookup.get(h_key)
        if fn is None:
            return "N/A"
        try:
            val = float(fn(state, self.env))
            return "INF" if math.isinf(val) else f"{val:.0f}"
        except NotImplementedError:
            return "TODO"
        except Exception:
            return "ERR"

    def _describe_tile_safe(self, pos: Pos, state: DeepCoreState) -> Tuple[str, str]:
        k = state.carried
        surcharge = 2 * k
        if pos in self.env.walls:
            return "Cavern Rock Wall (#)", "Impassable"
        if pos in state.chasms:
            return "Bottomless Chasm (O)", "Push a Boulder (B) here to build a bridge!"
        if pos in state.rubble:
            return "Collapsed Ore Rubble (@)", "Walk into @ (or Shift+Dir) to Laser-Drill (Cost=10)"
        if pos in state.boulders:
            return "Heavy Ore Boulder (B)", f"Push Cost = 6 + 2({k}) = {6 + surcharge}"
        if pos == self.env.elevator_pos:
            return "Extraction Teleporter Pad (E)", f"Step Cost = {1 + surcharge} (Goal once all Miners rescued)"
        if pos in state.miners:
            base = 5 if pos in self.env.sludge else 1
            return "Trapped Miner (M)", f"Step Cost = {base + surcharge} (Rescues Miner)"
        if pos in self.env.initial_chasms and pos not in state.chasms:
            return "Reinforced Boulder Bridge (=)", f"Walk Cost = 1 + 2({k}) = {1 + surcharge}"
        if pos in self.env.sludge:
            return "Toxic Slime Pool (~)", f"Walk Cost = 5 + 2({k}) = {5 + surcharge}"
        return "Clear Grated Tunnel (.)", f"Walk Cost = 1 + 2({k}) = {1 + surcharge}"

    def _invoke_student_runner(self, algo: str, heur: str, weight: float) -> SearchMetrics:
        if self.runner_callback is None:
            return SearchMetrics(algorithm_name=algo.upper(), solved=False, status_message="Runner callback missing")
        try:
            return self.runner_callback(self.env, algo, heur, weight)
        except TypeError:
            return self.runner_callback(self.env, algo, heur)

    # -------------------------------------------------------------------------
    # Layout & Level State
    # -------------------------------------------------------------------------
    def _load_level(self, lvl_key: str) -> None:
        self.search_job_id += 1
        self.is_searching = False
        self.selected_level = lvl_key
        self.env = DeepCoreEnv.from_file_or_builtin(lvl_key)
        self.metrics = None
        self.compare_rows.clear()
        self.explored_rank.clear()
        self.particles.clear()
        self._load_custom_sprites()
        self._reset_manual()
        self._update_layout()

    def _reset_manual(self) -> None:
        self.manual_state = self.env.get_initial_state()
        self.prev_bot_pos = self.manual_state.bot_pos
        self.manual_cost = 0
        self.manual_steps = 0
        self.manual_undo.clear()
        self.trajectory = [(self.manual_state, "START", 0, 0)]
        self.path_cursor = 0
        self.wave_cursor = 0
        self.tween_t = 1.0
        self.playing = False
        self.bot_dir = "S"
        self.status_msg = (
            f"{self.env.level_name} ({self.env.rows}x{self.env.cols}, "
            f"{self.env.total_miners} Miners, {len(self.env.initial_state.boulders)} Boulders): Ready!"
        )
        self.status_col = self.ACCENT_BLUE

    def _update_layout(self) -> None:
        self.WIN_W, self.WIN_H = self.screen.get_size()
        avail_w = self.WIN_W - 60
        avail_h = self.WIN_H - self.TOP_BAR_H - self.BOT_BAR_H - 60
        ts_w = avail_w // max(1, self.env.cols)
        ts_h = avail_h // max(1, self.env.rows)
        self.tile_size = max(30, min(68, ts_w, ts_h))

        gw = self.env.cols * self.tile_size
        gh = self.env.rows * self.tile_size
        self.ox = (self.WIN_W - gw) // 2
        self.oy = self.TOP_BAR_H + 34 + (avail_h - gh) // 2
        self._rescale_sprites()

    def _spawn_burst(self, pos: Pos, color: Tuple[int, int, int], count: int = 18) -> None:
        ts = self.tile_size
        cx = self.ox + pos[1] * ts + ts * 0.5
        cy = self.oy + pos[0] * ts + ts * 0.5
        for _ in range(count):
            ang = random.uniform(0, math.tau)
            spd = random.uniform(40, 160)
            life = random.uniform(0.25, 0.55)
            self.particles.append(
                Particle(cx, cy, math.cos(ang) * spd, math.sin(ang) * spd, random.uniform(2.5, 5.0), color, life, life)
            )

    # -------------------------------------------------------------------------
    # Search Execution (Threaded)
    # -------------------------------------------------------------------------
    def run_active_algorithm(self) -> None:
        if self.selected_algo == "manual":
            self.search_job_id += 1
            self.is_searching = False
            self._reset_manual()
            return

        self.search_job_id += 1
        job_id = self.search_job_id
        self.is_searching = True
        self.playing = False
        self.search_start_time = time.perf_counter()
        self.pending_result = None
        algo, heur, weight = self.selected_algo, self.selected_heur, self.selected_weight
        env_ref = self.env

        def worker() -> None:
            res = self._invoke_student_runner(algo, heur, weight)
            if self.search_job_id == job_id and self.env is env_ref:
                self.pending_result = res

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _apply_completed_search(self, res: SearchMetrics) -> None:
        self.is_searching = False
        self.metrics = res
        self.explored_rank.clear()
        for i, pos in enumerate(res.explored_order):
            if pos not in self.explored_rank:
                self.explored_rank[pos] = i + 1

        self.trajectory = [(self.env.get_initial_state(), "START", 0, 0)]
        if res.solved and res.solution_actions:
            curr = self.env.get_initial_state()
            cum = 0
            for act_str in res.solution_actions:
                curr, sc = self.env.transition(curr, Action(act_str))
                cum += sc
                self.trajectory.append((curr, str(act_str), sc, cum))

        self.wave_cursor = 0 if self.wave_mode else len(res.explored_order)
        self.path_cursor = 0
        self.tween_t = 0.0
        self.playing = bool(res.solved)
        self.last_step_ms = pygame.time.get_ticks()

        if res.solved:
            self.status_msg = (
                f"[SOLVED] {res.algorithm_name} | Cost: {res.total_cost} | "
                f"Steps: {len(res.solution_actions)} | Expanded: {res.nodes_expanded:,} | "
                f"Generated: {res.nodes_generated:,} | Memory: {res.max_memory_size:,} | "
                f"Time: {res.runtime_sec:.2f} s"
            )
            self.status_col = self.ACCENT_GREEN
        else:
            msg = getattr(res, "status_message", "Not implemented yet in solver.py / heuristics.py")
            self.status_msg = f"[TODO / UNSOLVED] {msg}"
            self.status_col = self.ACCENT_AMBER

    def run_comparison_suite(self) -> None:
        self.compare_rows.clear()
        suite = [
            ("bfs", "naive", 1.0),
            ("dfs", "naive", 1.0),
            ("ids", "naive", 1.0),
            ("bidirectional", "naive", 1.0),
            ("ucs", "naive", 1.0),
            ("gbfs", "naive", 1.0),
            ("astar", "naive", 1.0),
            ("astar", "h1", 1.0),
            ("astar", "h2", 1.0),
            ("wastar", "h2", self.selected_weight),
            ("idastar", "h2", 1.0),
            ("sma", "h2", 1.0),
        ]
        for a, h, w in suite:
            self.compare_rows.append(self._invoke_student_runner(a, h, w))
        self.show_compare_modal = True

    # -------------------------------------------------------------------------
    # UI Control Ribbon Construction
    # -------------------------------------------------------------------------
    def _build_controls(self) -> None:
        self.pills.clear()

        x = 14
        y1 = 12
        for i, lk in enumerate(self.level_keys[:5]):
            r = pygame.Rect(x, y1, 42, 32)
            self.pills.append(PillButton(r, f"L{i+1}", "LVL", lk, active=(self.selected_level == lk), style="primary"))
            x += 47

        x += 12
        for aid, albl, ico in self.ALGO_PILLS:
            pad = 34 if ico else 20
            w = max(58, self.f_pill.size(albl)[0] + pad)
            r = pygame.Rect(x, y1, w, 32)
            st = "bonus" if aid in ("wastar", "idastar", "sma") else "primary"
            self.pills.append(PillButton(r, albl, "ALGO", aid, active=(self.selected_algo == aid), style=st, icon=ico))
            x += w + 6

        x = 14
        y2 = 54
        uses_h = self.selected_algo in ("gbfs", "astar", "wastar", "idastar", "sma")
        for hid, hlbl in self.HEUR_PILLS:
            w = self.f_pill.size(hlbl)[0] + 22
            r = pygame.Rect(x, y2, w, 30)
            self.pills.append(PillButton(r, hlbl, "HEUR", hid, active=(uses_h and self.selected_heur == hid), style="primary"))
            x += w + 6

        x += 10
        for wv in self.WEIGHT_PILLS:
            lbl = f"W={wv:g}"
            r = pygame.Rect(x, y2, 56, 30)
            is_act = self.selected_algo == "wastar" and abs(self.selected_weight - wv) < 1e-4
            self.pills.append(PillButton(r, lbl, "WEIGHT", str(wv), active=is_act, style="bonus"))
            x += 62

        rx = self.WIN_W - 14
        run_lbl = "Stop Search" if self.is_searching else "Solve / Replay"
        run_ico = "pause" if self.is_searching else "play"
        for lbl, grp, st, ico in [
            ("Reset (R)", "ACT_RESET", "default", "reset"),
            ("Compare All", "ACT_COMPARE", "warn", "chart"),
            (run_lbl, "ACT_RUN", "action", run_ico),
        ]:
            bw = self.f_pill.size(lbl)[0] + 38
            rx -= bw
            self.pills.append(PillButton(pygame.Rect(rx, y2, bw, 30), lbl, grp, style=st, icon=ico))
            rx -= 8

        by = self.WIN_H - self.BOT_BAR_H + 14
        bx = 16
        for lbl, grp, ico in [
            ("|<", "T_START", ""),
            ("< Step", "T_PREV", ""),
            ("Pause" if self.playing else "Play", "T_PLAY", "pause" if self.playing else "play"),
            ("Step >", "T_NEXT", ""),
            (">|", "T_END", ""),
        ]:
            pad = 36 if ico else 20
            bw = max(44, self.f_pill.size(lbl)[0] + pad)
            self.pills.append(PillButton(pygame.Rect(bx, by, bw, 30), lbl, grp, active=(grp == "T_PLAY" and self.playing), icon=ico))
            bx += bw + 6

        bx += 12
        for i, spd in enumerate(self.SPEED_PILLS):
            lbl = f"{spd:g}x"
            self.pills.append(PillButton(pygame.Rect(bx, by, 44, 30), lbl, "SPEED", str(i), active=(self.speed_idx == i)))
            bx += 50

        brx = self.WIN_W - 16
        for lbl, grp, active, ico in [
            ("Search Wave", "TOG_WAVE", self.wave_mode, "wave"),
            ("Path Line", "TOG_PATH", self.show_path, "path"),
            ("Heatmap", "TOG_HEAT", self.show_heatmap, "heat"),
        ]:
            bw = self.f_pill.size(lbl)[0] + 36
            brx -= bw
            self.pills.append(PillButton(pygame.Rect(brx, by, bw, 30), lbl, grp, active=active, icon=ico))
            brx -= 8

    # -------------------------------------------------------------------------
    # Interactions & Manual Controls
    # -------------------------------------------------------------------------
    def _on_pill_click(self, p: PillButton) -> None:
        g = p.group
        if g == "LVL":
            self._load_level(p.value)
            if self.selected_algo != "manual":
                self.run_active_algorithm()
        elif g == "ALGO":
            self.selected_algo = p.value
            self.run_active_algorithm()
        elif g == "HEUR":
            self.selected_heur = p.value
            if self.selected_algo == "manual":
                self.selected_algo = "astar"
            self.run_active_algorithm()
        elif g == "WEIGHT":
            self.selected_weight = float(p.value)
            self.selected_algo = "wastar"
            self.run_active_algorithm()
        elif g == "ACT_RUN":
            if self.is_searching:
                self.search_job_id += 1
                self.is_searching = False
                self.status_msg = "Search stopped by user."
                self.status_col = self.ACCENT_AMBER
            else:
                if self.selected_algo == "manual":
                    self.selected_algo = "bfs"
                self.run_active_algorithm()
        elif g == "ACT_COMPARE":
            self.run_comparison_suite()
        elif g == "ACT_RESET":
            self._load_level(self.selected_level)
        elif g == "T_START":
            self.playing = False
            self.path_cursor = 0
            self.wave_cursor = 0
            self.tween_t = 1.0
        elif g == "T_END":
            self.playing = False
            self.path_cursor = max(0, len(self.trajectory) - 1)
            if self.metrics:
                self.wave_cursor = len(self.metrics.explored_order)
            self.tween_t = 1.0
        elif g == "T_PREV":
            self.playing = False
            self.path_cursor = max(0, self.path_cursor - 1)
            self.tween_t = 1.0
        elif g == "T_NEXT":
            self.playing = False
            self.path_cursor = min(len(self.trajectory) - 1, self.path_cursor + 1)
            self.tween_t = 0.0
        elif g == "T_PLAY":
            if self.selected_algo != "manual":
                if self.path_cursor >= len(self.trajectory) - 1:
                    self.path_cursor = 0
                self.playing = not self.playing
                self.last_step_ms = pygame.time.get_ticks()
        elif g == "SPEED":
            self.speed_idx = int(p.value)
        elif g == "TOG_HEAT":
            self.show_heatmap = not self.show_heatmap
        elif g == "TOG_PATH":
            self.show_path = not self.show_path
        elif g == "TOG_WAVE":
            self.wave_mode = not self.wave_mode
            if self.metrics:
                self.wave_cursor = 0 if self.wave_mode else len(self.metrics.explored_order)
                self.playing = True

    def _on_manual_move(self, d: str, shift_held: bool) -> None:
        if self.selected_algo != "manual":
            self.search_job_id += 1
            self.is_searching = False
            self.selected_algo = "manual"
            self._reset_manual()

        self.bot_dir = d
        candidates = (
            [Action(f"DRILL_{d}")]
            if shift_held
            else [Action(f"MOVE_{d}"), Action(f"PUSH_{d}"), Action(f"DRILL_{d}")]
        )
        legal = set(self.env.get_legal_actions(self.manual_state))
        for act in candidates:
            if act in legal:
                self.prev_bot_pos = self.manual_state.bot_pos
                self.manual_undo.append((self.manual_state, self.manual_cost, self.bot_dir))
                prev_chasms = len(self.manual_state.chasms)
                prev_rubble = len(self.manual_state.rubble)

                self.manual_state, sc = self.env.transition(self.manual_state, act)
                self.manual_cost += sc
                self.manual_steps += 1
                self.tween_t = 0.0

                if len(self.manual_state.rubble) < prev_rubble:
                    dr, dc = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}[d]
                    self._spawn_burst((self.prev_bot_pos[0] + dr, self.prev_bot_pos[1] + dc), (255, 140, 50), 24)
                if len(self.manual_state.chasms) < prev_chasms:
                    self._spawn_burst(self.manual_state.bot_pos, (100, 210, 255), 22)

                if self.env.is_goal(self.manual_state):
                    self.status_msg = f"[VICTORY!] All Miners Rescued! Total Cost: {self.manual_cost} in {self.manual_steps} steps!"
                    self.status_col = self.ACCENT_GREEN
                    self._spawn_burst(self.manual_state.bot_pos, (50, 255, 140), 35)
                else:
                    self.status_msg = f"Executed {act.value} (+{sc} cost) | Total Cost: {self.manual_cost} | Press Z to Undo"
                    self.status_col = self.ACCENT_BLUE
                break

    # -------------------------------------------------------------------------
    # Arena & Sprite Renderer (Custom Image + Procedural Fallback)
    # -------------------------------------------------------------------------
    def _draw_arena(self, state: DeepCoreState, prev_pos: Pos, last_act: str, mouse_pos: Tuple[int, int]) -> None:
        ts = self.tile_size
        ticks = pygame.time.get_ticks()
        self.hover_tile = None
        total_exp = max(1, len(self.metrics.explored_order)) if self.metrics else 1

        board_rect = pygame.Rect(self.ox - 8, self.oy - 8, self.env.cols * ts + 16, self.env.rows * ts + 16)
        pygame.draw.rect(self.screen, (6, 8, 12), board_rect.move(0, 6), border_radius=12)
        pygame.draw.rect(self.screen, (24, 30, 44), board_rect, border_radius=12)
        pygame.draw.rect(self.screen, self.BAR_BORDER, board_rect, 2, border_radius=12)

        for r in range(self.env.rows):
            for c in range(self.env.cols):
                pos = (r, c)
                x = self.ox + c * ts
                y = self.oy + r * ts
                rect = pygame.Rect(x, y, ts, ts)
                if rect.collidepoint(mouse_pos):
                    self.hover_tile = pos

                # 1. Impassable Rock Walls (#)
                if pos in self.env.walls:
                    if "wall" in self.sprites:
                        self.screen.blit(self.sprites["wall"], (x, y))
                    else:
                        pygame.draw.rect(self.screen, (28, 34, 46), rect)
                        top_face = pygame.Rect(x + 2, y + 2, ts - 4, ts - 8)
                        front_lip = pygame.Rect(x + 2, y + ts - 8, ts - 4, 6)
                        pygame.draw.rect(self.screen, (42, 50, 66), front_lip, border_radius=2)
                        pygame.draw.rect(self.screen, (68, 78, 98), top_face, border_radius=4)
                    continue

                # 2. Base Floor Layer (.)
                if "floor" in self.sprites:
                    self.screen.blit(self.sprites["floor"], (x, y))
                else:
                    pygame.draw.rect(self.screen, (34, 41, 56), rect)
                    pygame.draw.rect(self.screen, (43, 51, 68), rect.inflate(-2, -2), border_radius=3)

                # 3. Terrain Overlays: Chasm (O), Bridge (=), Sludge (~)
                if pos in state.chasms:
                    pygame.draw.rect(self.screen, (10, 8, 14), rect.inflate(-6, -6), border_radius=8)
                    if "chasm" in self.sprites:
                        self.screen.blit(self.sprites["chasm"], (x, y))
                    else:
                        pygame.draw.rect(self.screen, (26, 20, 28), rect)
                        inner = rect.inflate(-4, -4)
                        pygame.draw.rect(self.screen, (12, 10, 16), inner, border_radius=5)
                        glow = int(110 + 50 * math.sin(ticks * 0.005 + r + c))
                        pygame.draw.rect(self.screen, (glow, 45, 20), inner.inflate(-8, -8), border_radius=4)
                        pygame.draw.rect(self.screen, (210, 145, 35), inner, 2, border_radius=5)

                elif pos in self.env.initial_chasms and pos not in state.chasms:
                    if "bridge" in self.sprites:
                        self.screen.blit(self.sprites["bridge"], (x, y))
                    else:
                        pygame.draw.rect(self.screen, (24, 28, 38), rect)
                        for py in range(y + 4, y + ts - 4, max(7, ts // 5)):
                            plank = pygame.Rect(x + 3, py, ts - 6, max(4, ts // 7))
                            pygame.draw.rect(self.screen, (168, 120, 72), plank, border_radius=2)

                elif pos in self.env.sludge:
                    if "sludge" in self.sprites:
                        self.screen.blit(self.sprites["sludge"], (x, y))
                    else:
                        pygame.draw.rect(self.screen, (22, 62, 46), rect)
                        pygame.draw.rect(self.screen, (32, 86, 62), rect.inflate(-3, -3), border_radius=4)
                        w_off = math.sin(ticks * 0.006 + r * 2 + c) * 3
                        pygame.draw.circle(self.screen, (75, 220, 135), (int(x + ts * 0.35 + w_off), int(y + ts * 0.4)), max(2, ts // 11))

                # 4. Search Expansion Heatmap
                if self.show_heatmap and self.selected_algo != "manual" and pos in self.explored_rank:
                    rnk = self.explored_rank[pos]
                    limit = self.wave_cursor if self.wave_mode else total_exp
                    if rnk <= limit:
                        ratio = rnk / total_exp
                        ov = pygame.Surface((ts - 2, ts - 2), pygame.SRCALPHA)
                        ov.fill((int(40 + 215 * ratio), int(200 - 130 * ratio), int(250 - 170 * ratio), 72))
                        self.screen.blit(ov, (x + 1, y + 1))

                # 5. Extraction Elevator (E)
                if pos == self.env.elevator_pos:
                    if "elevator" in self.sprites:
                        self.screen.blit(self.sprites["elevator"], (x, y))
                    else:
                        er = rect.inflate(-ts // 6, -ts // 6)
                        pygame.draw.rect(self.screen, (20, 88, 128), er, border_radius=6)
                        pygame.draw.rect(self.screen, (70, 215, 255), er, 2, border_radius=6)
                        lbl = self.f_mono.render("E", True, (235, 255, 255))
                        self.screen.blit(lbl, lbl.get_rect(center=rect.center))

                # 6. Collapsed Rubble (@)
                if pos in state.rubble:
                    if "rubble" in self.sprites:
                        self.screen.blit(self.sprites["rubble"], (x, y))
                    else:
                        cx, cy = rect.center
                        rad = int(ts * 0.34)
                        pts = [(cx, cy - rad), (cx + rad, cy + 4), (cx + rad // 2, cy + rad - 2), (cx - rad // 2, cy + rad - 2), (cx - rad, cy + 4)]
                        pygame.draw.polygon(self.screen, (198, 112, 48), pts)
                        rlbl = self.f_small.render("@", True, (255, 245, 220))
                        self.screen.blit(rlbl, rlbl.get_rect(center=(cx, cy + 1)))

                # 7. Pushable Boulder (B)
                if pos in state.boulders:
                    if "boulder" in self.sprites:
                        self.screen.blit(self.sprites["boulder"], (x, y))
                    else:
                        b_rect = rect.inflate(-ts // 5, -ts // 5)
                        pygame.draw.rect(self.screen, (18, 22, 30), b_rect.move(2, 3), border_radius=6)
                        pygame.draw.rect(self.screen, (88, 98, 114), b_rect, border_radius=6)
                        top_b = pygame.Rect(b_rect.x, b_rect.y, b_rect.width, b_rect.height - 4)
                        pygame.draw.rect(self.screen, (138, 150, 170), top_b, border_radius=6)
                        pygame.draw.rect(self.screen, (60, 68, 82), b_rect, 2, border_radius=6)
                        blbl = self.f_small.render("B", True, (25, 32, 42))
                        self.screen.blit(blbl, blbl.get_rect(center=top_b.center))

                # 8. Stranded Miner (M)
                if pos in state.miners:
                    bob = int(2 * math.sin(ticks * 0.008 + r + c))
                    if "miner" in self.sprites:
                        self.screen.blit(self.sprites["miner"], (x, y + bob))
                    else:
                        cx, cy = rect.centerx, rect.centery + bob
                        pygame.draw.circle(self.screen, (248, 192, 36), (cx, cy), int(ts * 0.28))
                        mlbl = self.f_small.render("M", True, (30, 22, 5))
                        self.screen.blit(mlbl, mlbl.get_rect(center=(cx, cy)))

        # 9. Solution Path Vector Line
        if self.show_path and len(self.trajectory) > 1 and self.selected_algo != "manual":
            pts = [
                (self.ox + st.bot_pos[1] * ts + ts // 2, self.oy + st.bot_pos[0] * ts + ts // 2)
                for st, _, _, _ in self.trajectory
            ]
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, (55, 195, 255), False, pts, 3)

        # 10. DeepCore Rescue Robot with Directional Rotation
        t = min(1.0, self.tween_t)
        ir = prev_pos[0] + (state.bot_pos[0] - prev_pos[0]) * t
        ic = prev_pos[1] + (state.bot_pos[1] - prev_pos[1]) * t
        bx = int(self.ox + ic * ts)
        by = int(self.oy + ir * ts)
        bot_r = pygame.Rect(bx, by, ts, ts).inflate(-ts // 5, -ts // 5)

        # drill_vehicle.png drill points South by default; Pygame rotates counter-clockwise
        rot_angles = {"S": 0, "E": 90, "N": 180, "W": 270}
        angle = rot_angles.get(self.bot_dir, 0)

        if "bot" in self.sprites:
            bot_surf = pygame.transform.rotate(self.sprites["bot"], angle)
            rot_rect = bot_surf.get_rect(center=(bx + ts // 2, by + ts // 2))
            self.screen.blit(bot_surf, rot_rect)
        else:
            pygame.draw.rect(self.screen, (12, 15, 22), bot_r.move(2, 4), border_radius=7)
            pygame.draw.rect(self.screen, (175, 42, 34), bot_r, border_radius=7)
            top_bot = pygame.Rect(bot_r.x, bot_r.y, bot_r.width, bot_r.height - 4)
            pygame.draw.rect(self.screen, (235, 76, 60), top_bot, border_radius=7)
            pygame.draw.circle(self.screen, (85, 240, 255), top_bot.center, max(5, ts // 7))

        if state.carried > 0:
            bdg = pygame.Rect(bot_r.right - 14, bot_r.top - 5, 20, 16)
            pygame.draw.rect(self.screen, self.ACCENT_AMBER, bdg, border_radius=4)
            bt = self.f_small.render(f"+{state.carried}", True, (15, 18, 24))
            self.screen.blit(bt, bt.get_rect(center=bdg.center))

        if last_act.startswith("DRILL_") and self.tween_t < 0.85:
            dk = last_act.split("_")[1]
            dr, dc = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}[dk]
            p1 = (bx + ts // 2, by + ts // 2)
            p2 = (bx + ts // 2 + dc * ts, by + ts // 2 + dr * ts)
            pygame.draw.line(self.screen, (255, 95, 35), p1, p2, 6)
            pygame.draw.line(self.screen, (255, 245, 200), p1, p2, 2)

        if self.hover_tile:
            hr, hc = self.hover_tile
            pygame.draw.rect(self.screen, self.ACCENT_AMBER, (self.ox + hc * ts, self.oy + hr * ts, ts, ts), 2, border_radius=4)

        if self.is_searching:
            elapsed = time.perf_counter() - self.search_start_time
            box = pygame.Rect(self.WIN_W // 2 - 230, self.WIN_H // 2 - 42, 460, 84)
            pygame.draw.rect(self.screen, (16, 22, 34), box, border_radius=10)
            pygame.draw.rect(self.screen, self.ACCENT_AMBER, box, 2, border_radius=10)
            msg1 = self.f_title.render(f"SEARCHING ({self.selected_algo.upper()}) ON {self.env.level_name}...", True, self.ACCENT_AMBER)
            msg2 = self.f_mono.render(f"Elapsed Time: {elapsed:.1f} s  (Click 'Stop Search' above to cancel)", True, self.TEXT_MAIN)
            self.screen.blit(msg1, msg1.get_rect(center=(box.centerx, box.y + 26)))
            self.screen.blit(msg2, msg2.get_rect(center=(box.centerx, box.y + 56)))

    # -------------------------------------------------------------------------
    # HUD & Modal Telemetry
    # -------------------------------------------------------------------------
    def _draw_hud_bars(self, state: DeepCoreState, last_act: str, cum_cost: int, mouse_pos: Tuple[int, int]) -> None:
        pygame.draw.rect(self.screen, self.BAR_BG, (0, 0, self.WIN_W, self.TOP_BAR_H))
        pygame.draw.line(self.screen, self.BAR_BORDER, (0, self.TOP_BAR_H), (self.WIN_W, self.TOP_BAR_H), 2)

        by = self.WIN_H - self.BOT_BAR_H
        pygame.draw.rect(self.screen, self.BAR_BG, (0, by, self.WIN_W, self.BOT_BAR_H))
        pygame.draw.line(self.screen, self.BAR_BORDER, (0, by), (self.WIN_W, by), 2)

        for p in self.pills:
            hov = p.rect.collidepoint(mouse_pos) and not self.show_compare_modal
            if p.style == "action":
                bg = (235, 85, 85) if self.is_searching else ((55, 218, 138) if hov else self.ACCENT_GREEN)
                fg = (12, 18, 24)
                bd = (255, 180, 180) if self.is_searching else (160, 255, 200)
            elif p.style == "warn":
                bg = (255, 198, 70) if hov else self.ACCENT_AMBER
                fg = (15, 18, 24)
                bd = (255, 230, 150)
            elif p.active:
                bg = self.ACCENT_PURPLE if p.style == "bonus" else self.ACCENT_BLUE
                fg = (255, 255, 255)
                bd = (175, 220, 255)
            else:
                bg = self.PILL_HOVER if hov else self.PILL_BG
                fg = self.TEXT_MAIN
                bd = self.BAR_BORDER

            pygame.draw.rect(self.screen, bg, p.rect, border_radius=7)
            pygame.draw.rect(self.screen, bd, p.rect, 1, border_radius=7)

            lbl = self.f_pill.render(p.label, True, fg)
            if p.icon:
                draw_vector_icon(self.screen, p.icon, p.rect.x + 14, p.rect.centery, fg)
                self.screen.blit(lbl, lbl.get_rect(midleft=(p.rect.x + 25, p.rect.centery)))
            else:
                self.screen.blit(lbl, lbl.get_rect(center=p.rect.center))

        h0_v = self._eval_heuristic_display("naive", state)
        h1_v = self._eval_heuristic_display("h1", state)
        h2_v = self._eval_heuristic_display("h2", state)

        tele_text = (
            f"Cost g(n): {cum_cost}  |  "
            f"Payload: {state.carried}/{self.env.total_miners} (+{2*state.carried}/step)  |  "
            f"H0: {h0_v}  |  H1: {h1_v}  |  H2: {h2_v}  |  "
            f"Action: {last_act}"
        )
        t_surf = self.f_mono.render(tele_text, True, self.ACCENT_AMBER)
        self.screen.blit(t_surf, (18, self.TOP_BAR_H + 10))

        if self.hover_tile:
            tn, tc = self._describe_tile_safe(self.hover_tile, state)
            r_txt = f"[{self.hover_tile[0]},{self.hover_tile[1]}] {tn} - {tc}"
            r_col = self.TEXT_MAIN
        else:
            r_txt = self.status_msg
            r_col = self.status_col
        rs = self.f_ui.render(r_txt, True, r_col)
        self.screen.blit(rs, (self.WIN_W - rs.get_width() - 18, self.TOP_BAR_H + 10))

    def _draw_compare_modal(self) -> None:
        overlay = pygame.Surface((self.WIN_W, self.WIN_H), pygame.SRCALPHA)
        overlay.fill((8, 10, 16, 215))
        self.screen.blit(overlay, (0, 0))

        mw, mh = 860, 520
        mx, my = (self.WIN_W - mw) // 2, (self.WIN_H - mh) // 2
        mrect = pygame.Rect(mx, my, mw, mh)
        pygame.draw.rect(self.screen, self.BAR_BG, mrect, border_radius=12)
        pygame.draw.rect(self.screen, self.ACCENT_BLUE, mrect, 2, border_radius=12)

        title = self.f_title.render(
            f"STUDENT ALGORITHM & HEURISTIC BENCHMARK - {self.env.level_name} (Click or ESC to close)",
            True,
            self.ACCENT_AMBER,
        )
        self.screen.blit(title, (mx + 24, my + 20))

        headers = "Algorithm              Status   Steps   Total Cost   Expanded   Generated   Frontier   Memory   Time (ms)"
        self.screen.blit(self.f_mono.render(headers, True, self.ACCENT_BLUE), (mx + 24, my + 58))
        pygame.draw.line(self.screen, self.BAR_BORDER, (mx + 20, my + 80), (mx + mw - 20, my + 80), 1)

        ry = my + 92
        for row in self.compare_rows:
            status_lbl = "SOLVED" if row.solved else "TODO"
            steps = str(len(row.solution_actions)) if row.solved else "-"
            cost = str(row.total_cost) if row.solved else "-"
            col = self.ACCENT_GREEN if row.solved else self.ACCENT_AMBER
            line = (
                f"{row.algorithm_name:<22} {status_lbl:<8} {steps:>5}   "
                f"{cost:>10}   {row.nodes_expanded:>8}   {row.nodes_generated:>9}   "
                f"{row.max_frontier_size:>8}   {row.max_memory_size:>6}   "
                f"{row.runtime_sec * 1000:>8.2f}"
            )
            self.screen.blit(self.f_mono.render(line, True, col), (mx + 24, ry))
            ry += 34

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------
    def run(self) -> None:
        running = True
        while running:
            now = pygame.time.get_ticks()
            dt = self.clock.tick(60) / 1000.0
            mouse_pos = pygame.mouse.get_pos()

            if self.pending_result is not None:
                res = self.pending_result
                self.pending_result = None
                self._apply_completed_search(res)

            self._build_controls()

            spd = 4.0 * self.SPEED_PILLS[self.speed_idx]
            if self.tween_t < 1.0:
                self.tween_t = min(1.0, self.tween_t + dt * spd * 1.6)

            for pt in self.particles[:]:
                pt.x += pt.vx * dt
                pt.y += pt.vy * dt
                pt.life -= dt
                if pt.life <= 0:
                    self.particles.remove(pt)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self._update_layout()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.show_compare_modal:
                        self.show_compare_modal = False
                        continue
                    for p in self.pills:
                        if p.rect.collidepoint(event.pos):
                            self._on_pill_click(p)
                            break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.show_compare_modal:
                            self.show_compare_modal = False
                        elif self.is_searching:
                            self.search_job_id += 1
                            self.is_searching = False
                        else:
                            running = False
                    elif event.key == pygame.K_r:
                        self._load_level(self.selected_level)
                        if self.selected_algo != "manual":
                            self.run_active_algorithm()
                    elif event.key == pygame.K_z and self.selected_algo == "manual" and self.manual_undo:
                        self.manual_state, self.manual_cost, self.bot_dir = self.manual_undo.pop()
                        self.prev_bot_pos = self.manual_state.bot_pos
                        self.manual_steps = max(0, self.manual_steps - 1)
                        self.tween_t = 1.0
                    elif event.key == pygame.K_SPACE and self.selected_algo != "manual":
                        self.playing = not self.playing
                    else:
                        d_map = {
                            pygame.K_UP: "N", pygame.K_w: "N",
                            pygame.K_DOWN: "S", pygame.K_s: "S",
                            pygame.K_LEFT: "W", pygame.K_a: "W",
                            pygame.K_RIGHT: "E", pygame.K_d: "E",
                        }
                        if event.key in d_map:
                            self._on_manual_move(d_map[event.key], bool(pygame.key.get_mods() & pygame.KMOD_SHIFT))

            if self.playing and self.selected_algo != "manual" and not self.is_searching:
                step_ms = max(25.0, 1000.0 / spd)
                if now - self.last_step_ms >= step_ms:
                    self.last_step_ms = now
                    if self.wave_mode and self.metrics and self.wave_cursor < len(self.metrics.explored_order):
                        self.wave_cursor = min(len(self.metrics.explored_order), self.wave_cursor + max(1, int(spd)))
                    else:
                        self.wave_mode = False
                        if self.path_cursor < len(self.trajectory) - 1:
                            self.path_cursor += 1
                            self.tween_t = 0.0
                        else:
                            self.playing = False

            if self.selected_algo == "manual":
                st, prev_p, act_lbl, cum_c = self.manual_state, self.prev_bot_pos, "MANUAL", self.manual_cost
            else:
                idx = max(0, min(self.path_cursor, len(self.trajectory) - 1))
                p_idx = max(0, idx - 1)
                st, act_lbl, _, cum_c = self.trajectory[idx]
                prev_p = self.trajectory[p_idx][0].bot_pos

            self.screen.fill(self.BG_CAVERN)
            self._draw_arena(st, prev_p, act_lbl, mouse_pos)

            for pt in self.particles:
                pygame.draw.circle(self.screen, pt.color, (int(pt.x), int(pt.y)), int(pt.radius * (pt.life / pt.max_life)))

            self._draw_hud_bars(st, act_lbl, cum_c, mouse_pos)
            if self.show_compare_modal:
                self._draw_compare_modal()

            pygame.display.flip()

        pygame.quit()


DeepCoreVisualizer = DeepCoreWorkbench
