import pygame
import numpy as np
import math
import random

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 500
# Bottom UI bar (buttons + status). Drones should never enter this area.
UI_BAR_HEIGHT = 80
MAP_HEIGHT = SCREEN_HEIGHT - UI_BAR_HEIGHT

CELL_SIZE = 4
GRID_W = SCREEN_WIDTH // CELL_SIZE
# IMPORTANT: grid only represents the explorable map area (not the UI bar)
GRID_H = MAP_HEIGHT // CELL_SIZE

# Colors
COLOR_UNKNOWN = (40, 40, 40)
COLOR_FREE = (255, 255, 255)
COLOR_OBSTACLE = (0, 0, 0)
COLOR_LASER = (255, 255, 0)

# Drone colors with their transparent coverage colors
DRONE_COLORS = [
    ((255, 0, 0), (255, 0, 0, 30)),      # Red
    ((0, 255, 0), (0, 255, 0, 30)),      # Green
    ((0, 0, 255), (0, 0, 255, 30)),      # Blue
    ((255, 165, 0), (255, 165, 0, 30)),  # Orange
    ((128, 0, 128), (128, 0, 128, 30)),  # Purple
    ((0, 255, 255), (0, 255, 255, 30)),  # Cyan
    ((255, 255, 0), (255, 255, 0, 30)),  # Yellow
    ((255, 192, 203), (255, 192, 203, 30)), # Pink
]

# Simulation Physics
SENSOR_RANGE = 60
FOV_ANGLE = 360
SPEED = 2.0
REPULSION_RADIUS = 50

# LiDAR sampling density (higher reduces tiny missed pockets)
LIDAR_RAYS = 48
RAY_STEP = max(1, CELL_SIZE // 2)

# Physical spacing (prevents drones overlapping)
DRONE_BODY_RADIUS = 12
MIN_DRONE_SEPARATION = DRONE_BODY_RADIUS * 2

# States
STATE_EXPLORER = "EXPLORER"
STATE_COLLECTOR = "COLLECTOR"

# Game states
STATE_SETUP = "SETUP"
STATE_RUNNING = "RUNNING"
STATE_PAUSED = "PAUSED"

# Button dimensions
BUTTON_WIDTH = 120
BUTTON_HEIGHT = 40
BUTTON_MARGIN = 10

# ==========================================
# CLASS 1: MAP MANAGER
# ==========================================
class MapManager:
    def __init__(self):
        self.grid = np.zeros((GRID_W, GRID_H), dtype=int)
        self.obstacles = []  # Store obstacle positions for manual placement

    def clear(self):
        self.grid.fill(0)
        self.obstacles = []

    def add_obstacle(self, cx, cy, radius):
        """Add a circular obstacle at grid coordinates"""
        self.obstacles.append((cx, cy, radius))
        gx, gy = int(cx // CELL_SIZE), int(cy // CELL_SIZE)
        r = int(radius // CELL_SIZE)
        
        x_range = slice(max(0, gx-r), min(GRID_W, gx+r))
        y_range = slice(max(0, gy-r), min(GRID_H, gy+r))
        
        mask_h = y_range.stop - y_range.start
        mask_w = x_range.stop - x_range.start
        
        if mask_w > 0 and mask_h > 0:
            y, x = np.ogrid[-r:r, -r:r]
            mask = x**2 + y**2 <= r**2
            # Adjust mask to fit the actual slice size
            mask = mask[:mask_h, :mask_w]
            self.grid[x_range, y_range][mask] = 2

    def remove_obstacle_at(self, x, y):
        """Remove obstacle at screen coordinates"""
        gx, gy = int(x // CELL_SIZE), int(y // CELL_SIZE)
        if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
            if self.grid[gx, gy] == 2:
                # Find and remove from obstacles list
                for i, (ox, oy, r) in enumerate(self.obstacles):
                    dist = math.sqrt((x - ox)**2 + (y - oy)**2)
                    if dist <= r:
                        self.obstacles.pop(i)
                        self._rebuild_grid()
                        return True
        return False

    def _rebuild_grid(self):
        """Rebuild grid from obstacles list"""
        self.grid.fill(0)
        for cx, cy, radius in self.obstacles:
            gx, gy = int(cx // CELL_SIZE), int(cy // CELL_SIZE)
            r = int(radius // CELL_SIZE)
            x_range = slice(max(0, gx-r), min(GRID_W, gx+r))
            y_range = slice(max(0, gy-r), min(GRID_H, gy+r))
            mask_h = y_range.stop - y_range.start
            mask_w = x_range.stop - x_range.start
            if mask_w > 0 and mask_h > 0:
                y, x = np.ogrid[-r:r, -r:r]
                mask = x**2 + y**2 <= r**2
                mask = mask[:mask_h, :mask_w]
                self.grid[x_range, y_range][mask] = 2

    def update_map(self, x, y, state):
        gx, gy = int(x // CELL_SIZE), int(y // CELL_SIZE)
        if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
            if self.grid[gx, gy] != 2:
                self.grid[gx, gy] = state

    def get_frontiers(self):
        """Return frontier cells (Unknown adjacent to Free).

        Previous implementation randomly sampled cells, which can miss very small
        frontier pockets. This version computes the full frontier mask with numpy
        and then (optionally) subsamples for performance.
        """
        g = self.grid
        unknown = (g == 0)
        free = (g == 1)

        # 8-neighborhood adjacency using shifted masks (helps catch tiny corner pockets)
        free_left = np.zeros_like(free)
        free_left[1:, :] = free[:-1, :]
        free_right = np.zeros_like(free)
        free_right[:-1, :] = free[1:, :]
        free_up = np.zeros_like(free)
        free_up[:, 1:] = free[:, :-1]
        free_down = np.zeros_like(free)
        free_down[:, :-1] = free[:, 1:]

        free_ul = np.zeros_like(free)
        free_ul[1:, 1:] = free[:-1, :-1]
        free_ur = np.zeros_like(free)
        free_ur[:-1, 1:] = free[1:, :-1]
        free_dl = np.zeros_like(free)
        free_dl[1:, :-1] = free[:-1, 1:]
        free_dr = np.zeros_like(free)
        free_dr[:-1, :-1] = free[1:, 1:]

        frontier_mask = unknown & (
            free_left | free_right | free_up | free_down |
            free_ul | free_ur | free_dl | free_dr
        )
        xs, ys = np.where(frontier_mask)
        if xs.size == 0:
            return []

        # Subsample if many frontiers (keeps FPS stable)
        max_points = 1200
        if xs.size > max_points:
            idx = np.random.choice(xs.size, size=max_points, replace=False)
            xs = xs[idx]
            ys = ys[idx]

        return list(zip(xs * CELL_SIZE, ys * CELL_SIZE))

# ==========================================
# CLASS 2: DRONE (X-Quadrotor shape)
# ==========================================
class Drone:
    def __init__(self, id, start_x, start_y, global_map, color_pair):
        self.id = id
        self.pos = np.array([start_x, start_y], dtype=float)
        self.vel = np.array([0.0, 0.0])
        self.map = global_map
        self.state = STATE_EXPLORER
        self.target = None
        self.body_color = color_pair[0]
        self.coverage_color = color_pair[1]
        self.angle = 0  # Rotation angle for the X shape

    def draw(self, screen):
        """Draw X-shaped quadrotor drone"""
        x, y = int(self.pos[0]), int(self.pos[1])
        
        # Draw coverage area (transparent circle)
        coverage_surface = pygame.Surface((SENSOR_RANGE * 2, SENSOR_RANGE * 2), pygame.SRCALPHA)
        pygame.draw.circle(coverage_surface, self.coverage_color, 
                          (SENSOR_RANGE, SENSOR_RANGE), SENSOR_RANGE)
        screen.blit(coverage_surface, (x - SENSOR_RANGE, y - SENSOR_RANGE))
        
        # Draw sensor range outline
        pygame.draw.circle(screen, self.body_color, (x, y), SENSOR_RANGE, 1)
        
        # Draw X-shaped drone body
        arm_length = 15
        rotor_radius = 5
        center_radius = 6
        
        # Calculate arm endpoints (45-degree X shape)
        arms = [
            (math.cos(math.pi/4), math.sin(math.pi/4)),      # Top-right
            (math.cos(3*math.pi/4), math.sin(3*math.pi/4)),  # Top-left
            (math.cos(5*math.pi/4), math.sin(5*math.pi/4)),  # Bottom-left
            (math.cos(7*math.pi/4), math.sin(7*math.pi/4)),  # Bottom-right
        ]
        
        # Draw arms
        for dx, dy in arms:
            end_x = x + dx * arm_length
            end_y = y + dy * arm_length
            pygame.draw.line(screen, (50, 50, 50), (x, y), (end_x, end_y), 4)
            # Draw rotor at the end
            pygame.draw.circle(screen, (30, 30, 30), (int(end_x), int(end_y)), rotor_radius)
            pygame.draw.circle(screen, (100, 100, 100), (int(end_x), int(end_y)), rotor_radius - 2)
        
        # Draw center body
        pygame.draw.circle(screen, self.body_color, (x, y), center_radius)
        pygame.draw.circle(screen, (0, 0, 0), (x, y), center_radius, 2)
        
        # Draw direction indicator (small dot showing facing)
        if np.linalg.norm(self.vel) > 0.1:
            angle = math.atan2(self.vel[1], self.vel[0])
            indicator_x = x + math.cos(angle) * (center_radius - 2)
            indicator_y = y + math.sin(angle) * (center_radius - 2)
            pygame.draw.circle(screen, (255, 255, 255), (int(indicator_x), int(indicator_y)), 2)

    def sense(self):
        for i in range(LIDAR_RAYS):
            angle = (i / LIDAR_RAYS) * 2 * math.pi
            for r in range(0, SENSOR_RANGE, RAY_STEP):
                rx = self.pos[0] + math.cos(angle) * r
                ry = self.pos[1] + math.sin(angle) * r
                gx, gy = int(rx // CELL_SIZE), int(ry // CELL_SIZE)
                if not (0 <= gx < GRID_W and 0 <= gy < GRID_H):
                    break
                cell_val = self.map.grid[gx, gy]
                if cell_val == 2:
                    break
                else:
                    self.map.grid[gx, gy] = 1

    def plan(self, all_drones):
        separation_force = np.array([0.0, 0.0])
        for other in all_drones:
            if other.id != self.id:
                dist = np.linalg.norm(self.pos - other.pos)
                if dist < REPULSION_RADIUS:
                    # Stronger repulsion when very close, softer when far
                    push_dir = (self.pos - other.pos) / (dist + 1e-6)
                    strength = 6.0 * (1.0 - (dist / REPULSION_RADIUS))
                    separation_force += push_dir * strength

        frontiers = self.map.get_frontiers()
        if not frontiers:
            self.vel = np.array([0.0, 0.0])
            return

        closest_dist = float('inf')
        best_target = None
        for f in frontiers:
            f_pos = np.array(f)
            dist = np.linalg.norm(self.pos - f_pos)
            if dist < closest_dist:
                closest_dist = dist
                best_target = f_pos

        if closest_dist < SENSOR_RANGE:
            self.state = STATE_COLLECTOR
        else:
            self.state = STATE_EXPLORER

        if best_target is not None:
            # Clamp target to reachable movement bounds (avoids edge/corner "stuck" behavior)
            best_target = best_target.astype(float)
            best_target[0] = np.clip(best_target[0], DRONE_BODY_RADIUS, SCREEN_WIDTH - DRONE_BODY_RADIUS)
            best_target[1] = np.clip(best_target[1], DRONE_BODY_RADIUS, MAP_HEIGHT - DRONE_BODY_RADIUS)

            direction = (best_target - self.pos)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            final_vector = (direction * SPEED) + separation_force
            if np.linalg.norm(final_vector) > SPEED:
                final_vector = (final_vector / np.linalg.norm(final_vector)) * SPEED
            self.vel = final_vector

    def update(self):
        self.pos += self.vel
        # Keep drones inside the map area (exclude bottom UI bar)
        self.pos[0] = np.clip(self.pos[0], DRONE_BODY_RADIUS, SCREEN_WIDTH - DRONE_BODY_RADIUS)
        self.pos[1] = np.clip(self.pos[1], DRONE_BODY_RADIUS, MAP_HEIGHT - DRONE_BODY_RADIUS)

# ==========================================
# CLASS 3: BUTTON
# ==========================================
class Button:
    def __init__(self, x, y, width, height, text, color, hover_color, text_color=(255, 255, 255)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False
        self.font = pygame.font.Font(None, 28)

    def draw(self, screen):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2, border_radius=8)
        
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False

# ==========================================
# CLASS 4: SIMULATION MANAGER
# ==========================================
class SimulationManager:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Multi-UAV Decentralized Exploration")
        self.clock = pygame.time.Clock()
        self.running = True
        
        self.map = MapManager()
        self.drones = []
        self.simulation_state = STATE_SETUP
        self.num_drones = 3  # Default number of drones
        
        # Setup mode variables
        self.placing_obstacle = False
        self.obstacle_start_pos = None
        self.temp_obstacle_radius = 0
        
        # Create buttons
        button_y = SCREEN_HEIGHT - 60
        self.start_pause_btn = Button(SCREEN_WIDTH//2 - 60, button_y, 120, 40, 
                                      "Start", (0, 150, 0), (0, 200, 0))
        self.reset_btn = Button(SCREEN_WIDTH//2 + 70, button_y, 120, 40, 
                               "Reset", (150, 100, 0), (200, 150, 0))
        self.exit_btn = Button(SCREEN_WIDTH//2 - 190, button_y, 120, 40, 
                              "Exit", (150, 0, 0), (200, 0, 0))
        
        # Input box for drone count
        self.input_rect = pygame.Rect(SCREEN_WIDTH//2 - 50, 100, 100, 40)
        self.input_active = False
        self.input_text = str(self.num_drones)
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

    def spawn_drones(self):
        self.drones = []
        center_x, center_y = SCREEN_WIDTH // 2, MAP_HEIGHT // 2
        for i in range(self.num_drones):
            angle = (i / self.num_drones) * 2 * math.pi
            offset_x = math.cos(angle) * 30
            offset_y = math.sin(angle) * 30
            color_pair = DRONE_COLORS[i % len(DRONE_COLORS)]
            self.drones.append(Drone(i, center_x + offset_x, center_y + offset_y, 
                                    self.map, color_pair))

    def _resolve_drone_collisions(self):
        """Hard-constraint separation so drones never overlap."""
        n = len(self.drones)
        if n < 2:
            return

        # A couple iterations helps when many drones are bunched up
        for _ in range(2):
            for i in range(n):
                for j in range(i + 1, n):
                    a = self.drones[i]
                    b = self.drones[j]
                    delta = a.pos - b.pos
                    dist = float(np.linalg.norm(delta))
                    min_dist = float(MIN_DRONE_SEPARATION)
                    if dist < 1e-6:
                        # Perfect overlap: nudge deterministically
                        delta = np.array([1.0, 0.0])
                        dist = 1.0
                    if dist < min_dist:
                        push_dir = delta / dist
                        push = (min_dist - dist) * 0.5
                        a.pos += push_dir * push
                        b.pos -= push_dir * push
                        # Re-clamp after pushing
                        a.pos[0] = np.clip(a.pos[0], DRONE_BODY_RADIUS, SCREEN_WIDTH - DRONE_BODY_RADIUS)
                        a.pos[1] = np.clip(a.pos[1], DRONE_BODY_RADIUS, MAP_HEIGHT - DRONE_BODY_RADIUS)
                        b.pos[0] = np.clip(b.pos[0], DRONE_BODY_RADIUS, SCREEN_WIDTH - DRONE_BODY_RADIUS)
                        b.pos[1] = np.clip(b.pos[1], DRONE_BODY_RADIUS, MAP_HEIGHT - DRONE_BODY_RADIUS)

    def handle_setup_events(self, event):
        if event.type == pygame.KEYDOWN:
            if self.input_active:
                if event.key == pygame.K_RETURN:
                    self.input_active = False
                    try:
                        val = int(self.input_text)
                        self.num_drones = max(1, min(val, 8))  # Limit 1-8 drones
                    except:
                        self.input_text = str(self.num_drones)
                elif event.key == pygame.K_BACKSPACE:
                    self.input_text = self.input_text[:-1]
                else:
                    if event.unicode.isdigit():
                        self.input_text += event.unicode
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check input box
            if self.input_rect.collidepoint(event.pos):
                self.input_active = True
            else:
                self.input_active = False
            
            # Left click to add obstacle
            if event.button == 1:
                if not any(btn.rect.collidepoint(event.pos) for btn in 
                          [self.start_pause_btn, self.reset_btn, self.exit_btn]):
                    if event.pos[1] < MAP_HEIGHT:  # Not clicking on buttons/UI bar
                        self.placing_obstacle = True
                        self.obstacle_start_pos = event.pos
                        self.temp_obstacle_radius = 0
            
            # Right click to remove obstacle
            if event.button == 3:
                self.map.remove_obstacle_at(event.pos[0], event.pos[1])
        
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.placing_obstacle:
                if self.temp_obstacle_radius > 5:  # Minimum size
                    self.map.add_obstacle(self.obstacle_start_pos[0], 
                                         self.obstacle_start_pos[1], 
                                         self.temp_obstacle_radius)
                self.placing_obstacle = False
                self.obstacle_start_pos = None
                self.temp_obstacle_radius = 0
        
        if event.type == pygame.MOUSEMOTION and self.placing_obstacle:
            dx = event.pos[0] - self.obstacle_start_pos[0]
            dy = event.pos[1] - self.obstacle_start_pos[1]
            self.temp_obstacle_radius = int(math.sqrt(dx*dx + dy*dy))

        # Button events
        if self.start_pause_btn.handle_event(event):
            if self.simulation_state == STATE_SETUP:
                self.spawn_drones()
                self.simulation_state = STATE_RUNNING
                self.start_pause_btn.text = "Pause"
                self.start_pause_btn.color = (150, 150, 0)
                self.start_pause_btn.hover_color = (200, 200, 0)
            elif self.simulation_state == STATE_RUNNING:
                self.simulation_state = STATE_PAUSED
                self.start_pause_btn.text = "Resume"
                self.start_pause_btn.color = (0, 150, 0)
                self.start_pause_btn.hover_color = (0, 200, 0)
            elif self.simulation_state == STATE_PAUSED:
                self.simulation_state = STATE_RUNNING
                self.start_pause_btn.text = "Pause"
                self.start_pause_btn.color = (150, 150, 0)
                self.start_pause_btn.hover_color = (200, 200, 0)
        
        if self.reset_btn.handle_event(event):
            self.reset_simulation()
        
        if self.exit_btn.handle_event(event):
            self.running = False

    def reset_simulation(self):
        self.map.clear()
        self.drones = []
        self.simulation_state = STATE_SETUP
        self.start_pause_btn.text = "Start"
        self.start_pause_btn.color = (0, 150, 0)
        self.start_pause_btn.hover_color = (0, 200, 0)
        self.placing_obstacle = False
        self.obstacle_start_pos = None
        self.temp_obstacle_radius = 0

    def handle_running_events(self, event):
        if self.start_pause_btn.handle_event(event):
            self.simulation_state = STATE_PAUSED
            self.start_pause_btn.text = "Resume"
            self.start_pause_btn.color = (0, 150, 0)
            self.start_pause_btn.hover_color = (0, 200, 0)
        
        if self.reset_btn.handle_event(event):
            self.reset_simulation()
        
        if self.exit_btn.handle_event(event):
            self.running = False

    def handle_paused_events(self, event):
        if self.start_pause_btn.handle_event(event):
            self.simulation_state = STATE_RUNNING
            self.start_pause_btn.text = "Pause"
            self.start_pause_btn.color = (150, 150, 0)
            self.start_pause_btn.hover_color = (200, 200, 0)
        
        if self.reset_btn.handle_event(event):
            self.reset_simulation()
        
        if self.exit_btn.handle_event(event):
            self.running = False

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            if self.simulation_state == STATE_SETUP:
                self.handle_setup_events(event)
            elif self.simulation_state == STATE_RUNNING:
                self.handle_running_events(event)
            elif self.simulation_state == STATE_PAUSED:
                self.handle_paused_events(event)

    def update_physics(self):
        if self.simulation_state == STATE_RUNNING:
            for drone in self.drones:
                drone.sense()
                drone.plan(self.drones)
                drone.update()
            self._resolve_drone_collisions()

    def draw_setup_screen(self):
        self.screen.fill((30, 30, 30))

        # Draw map first so UI overlays remain visible
        self.draw_map()
        
        # Draw title
        title = self.font.render("Setup Mode: Place Obstacles & Configure", True, (255, 255, 255))
        self.screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 20))
        
        # Draw instructions
        instructions = [
            "Left Click + Drag: Create circular obstacle",
            "Right Click: Remove obstacle",
        ]
        for i, text in enumerate(instructions):
            surf = self.small_font.render(text, True, (200, 200, 200))
            self.screen.blit(surf, (50, 60 + i * 25))
        
        # Draw input box
        pygame.draw.rect(self.screen, (255, 255, 255) if self.input_active else (200, 200, 200),
                self.input_rect, 2)
        input_surf = self.font.render(self.input_text, True, (255, 255, 255))
        self.screen.blit(input_surf, (self.input_rect.x + 10, self.input_rect.y + 8))

        # Label for input
        label = self.small_font.render( "Enter drones Num",True, (255, 255, 255))
        self.screen.blit(label, (self.input_rect.x - 150, self.input_rect.y + 10))
        
        # Draw temporary obstacle being placed
        if self.placing_obstacle and self.obstacle_start_pos:
            pygame.draw.circle(self.screen, (100, 100, 100, 128), 
                             self.obstacle_start_pos, self.temp_obstacle_radius, 2)
            pygame.draw.circle(self.screen, (150, 150, 150, 100), 
                             self.obstacle_start_pos, self.temp_obstacle_radius)
        
        # Draw buttons
        self.start_pause_btn.draw(self.screen)
        self.reset_btn.draw(self.screen)
        self.exit_btn.draw(self.screen)

    def draw_map(self):
        # Create RGB array
        visual_map = np.zeros((GRID_W, GRID_H, 3), dtype=np.uint8)
        visual_map[self.map.grid == 0] = COLOR_UNKNOWN
        visual_map[self.map.grid == 1] = COLOR_FREE
        visual_map[self.map.grid == 2] = COLOR_OBSTACLE
        
        surf = pygame.surfarray.make_surface(visual_map)
        surf = pygame.transform.scale(surf, (SCREEN_WIDTH, MAP_HEIGHT))
        self.screen.blit(surf, (0, 0))

    def draw_simulation(self):
        self.screen.fill((0, 0, 0))
        
        # Draw map
        self.draw_map()
        
        # Draw drones
        for drone in self.drones:
            drone.draw(self.screen)
        
        # Draw status
        status_text = "RUNNING" if self.simulation_state == STATE_RUNNING else "PAUSED"
        status_color = (0, 255, 0) if self.simulation_state == STATE_RUNNING else (255, 165, 0)
        status_surf = self.font.render(status_text, True, status_color)
        self.screen.blit(status_surf, (10, SCREEN_HEIGHT - 75))
        
        # Draw drone count
        count_surf = self.small_font.render(f"Drones: {len(self.drones)}", True, (255, 255, 255))
        self.screen.blit(count_surf, (10, SCREEN_HEIGHT - 50))
        
        # Draw buttons
        self.start_pause_btn.draw(self.screen)
        self.reset_btn.draw(self.screen)
        self.exit_btn.draw(self.screen)

    def draw(self):
        if self.simulation_state == STATE_SETUP:
            self.draw_setup_screen()
        else:
            self.draw_simulation()
        
        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_events()
            self.update_physics()
            self.draw()
            self.clock.tick(30)
        
        pygame.quit()

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    sim = SimulationManager()
    sim.run()