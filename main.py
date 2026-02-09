import pygame
import numpy as np
import math
import random

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
CELL_SIZE = 4  # Size of each grid cell in pixels
GRID_W = SCREEN_WIDTH // CELL_SIZE
GRID_H = SCREEN_HEIGHT // CELL_SIZE

# Colors
COLOR_UNKNOWN = (40, 40, 40)   # Dark Grey
COLOR_FREE = (255, 255, 255)   # White
COLOR_OBSTACLE = (0, 0, 0)     # Black
COLOR_DRONE = (0, 255, 0)      # Green
COLOR_LASER = (255, 255, 0)    # Yellow

# Simulation Physics
NUM_DRONES = 5
SENSOR_RANGE = 60    # Range in pixels
FOV_ANGLE = 360      # Field of view (360 for LiDAR)
SPEED = 2.0
REPULSION_RADIUS = 50 # How far drones stay apart (The "Voronoi" effect)

# States from the Paper
STATE_EXPLORER = "EXPLORER"
STATE_COLLECTOR = "COLLECTOR"

# ==========================================
# CLASS 1: MAP MANAGER (The Environment)
# ==========================================
class MapManager:
    def __init__(self):
        # 0: Unknown, 1: Free, 2: Obstacle
        self.grid = np.zeros((GRID_W, GRID_H), dtype=int)
        self.generate_forest()

    def generate_forest(self):
        """Generates random 'trees' (obstacles)"""
        # Create random clusters of trees
        for _ in range(40):
            cx, cy = random.randint(0, GRID_W), random.randint(0, GRID_H)
            radius = random.randint(2, 6)
            y, x = np.ogrid[-radius:radius, -radius:radius]
            mask = x**2 + y**2 <= radius**2
            
            # Apply mask to grid (handling boundaries)
            x_range = slice(max(0, cx-radius), min(GRID_W, cx+radius))
            y_range = slice(max(0, cy-radius), min(GRID_H, cy+radius))
            
            # Create a localized mask of the correct shape
            mask_h, mask_w = self.grid[x_range, y_range].shape
            if mask_w > 0 and mask_h > 0:
                self.grid[x_range, y_range] = 2 # Mark as Obstacle

    def update_map(self, x, y, state):
        """Updates a specific cell (called by Raycaster)"""
        gx, gy = int(x // CELL_SIZE), int(y // CELL_SIZE)
        if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
            # Only update if currently Unknown (0) or verifying Free (1)
            # We don't overwrite Obstacles (2) unless we want dynamic updates
            if self.grid[gx, gy] != 2: 
                self.grid[gx, gy] = state

    def get_frontiers(self):
        """
        Paper Concept: Frontiers are boundaries between Free and Unknown space.
        Returns a list of (x, y) coordinates of frontier cells.
        """
        # Create binary masks
        free_mask = (self.grid == 1)
        unknown_mask = (self.grid == 0)
        
        # Simple convolution-like check (inefficient but works for python sim)
        # Find Unknown cells that have at least one Free neighbor
        # In a real thesis, use cv2.findContours or scipy.signal.convolve2d
        frontiers = []
        
        # Optimization: Randomly sample instead of full scan every frame
        # to save FPS.
        for _ in range(200): 
            rx, ry = random.randint(1, GRID_W-2), random.randint(1, GRID_H-2)
            if self.grid[rx, ry] == 0: # If unknown
                # Check neighbors
                if (self.grid[rx+1, ry] == 1 or self.grid[rx-1, ry] == 1 or
                    self.grid[rx, ry+1] == 1 or self.grid[rx, ry-1] == 1):
                    frontiers.append((rx * CELL_SIZE, ry * CELL_SIZE))
                    
        return frontiers

# ==========================================
# CLASS 2: DRONE (The Agent)
# ==========================================
class Drone:
    def __init__(self, id, start_x, start_y, global_map):
        self.id = id
        self.pos = np.array([start_x, start_y], dtype=float)
        self.vel = np.array([0.0, 0.0])
        self.map = global_map
        self.state = STATE_EXPLORER
        self.target = None
        self.color = COLOR_DRONE

    def sense(self):
        """
        Simulates LiDAR Raycasting.
        Updates the map clearing Unknown -> Free.
        Stops at Obstacles.
        """
        num_rays = 20
        for i in range(num_rays):
            angle = (i / num_rays) * 2 * math.pi
            
            # Simple Raycast Step
            for r in range(0, SENSOR_RANGE, CELL_SIZE):
                rx = self.pos[0] + math.cos(angle) * r
                ry = self.pos[1] + math.sin(angle) * r
                
                gx, gy = int(rx // CELL_SIZE), int(ry // CELL_SIZE)
                
                if not (0 <= gx < GRID_W and 0 <= gy < GRID_H):
                    break # Out of bounds
                
                cell_val = self.map.grid[gx, gy]
                
                if cell_val == 2: # Obstacle
                    break # Laser hit tree
                else:
                    self.map.grid[gx, gy] = 1 # Mark Free

    def plan(self, all_drones):
        """
        The CORE LOGIC from the Paper.
        1. Explorer Mode: Go to nearest Frontier.
        2. Collector Mode: Go to small leftover holes.
        3. Separation: Avoid other drones (Decentralized Coordination).
        """
        
        # --- 1. COORDINATION (Separation Force) ---
        # Eq 1 in Paper: Keeps drones spread out
        separation_force = np.array([0.0, 0.0])
        for other in all_drones:
            if other.id != self.id:
                dist = np.linalg.norm(self.pos - other.pos)
                if dist < REPULSION_RADIUS:
                    push = (self.pos - other.pos) / (dist + 0.1)
                    separation_force += push * 3.0 # Strength of repulsion

        # --- 2. TARGET SELECTION (State Machine) ---
        frontiers = self.map.get_frontiers()
        
        if not frontiers:
            # Map done or no frontiers found, hover in place
            self.vel = np.array([0.0, 0.0])
            return

        # Find nearest frontier
        closest_dist = float('inf')
        best_target = None
        
        for f in frontiers:
            f_pos = np.array(f)
            dist = np.linalg.norm(self.pos - f_pos)
            if dist < closest_dist:
                closest_dist = dist
                best_target = f_pos

        # LOGIC: Switch State
        # If the target is very close but we haven't scanned it yet, we are collecting
        # If the target is far, we are exploring.
        if closest_dist < SENSOR_RANGE:
            self.state = STATE_COLLECTOR
            self.color = (0, 255, 255) # Cyan
        else:
            self.state = STATE_EXPLORER
            self.color = (0, 255, 0) # Green

        # --- 3. MOVEMENT (Potential Field) ---
        if best_target is not None:
            direction = (best_target - self.pos)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            
            # Combine forces: Attraction to Target + Repulsion from Drones
            final_vector = (direction * SPEED) + separation_force
            
            # Normalize to Max Speed
            if np.linalg.norm(final_vector) > SPEED:
                final_vector = (final_vector / np.linalg.norm(final_vector)) * SPEED
            
            self.vel = final_vector

    def update(self):
        # Update Position
        self.pos += self.vel
        
        # Boundary Check
        self.pos[0] = np.clip(self.pos[0], 0, SCREEN_WIDTH)
        self.pos[1] = np.clip(self.pos[1], 0, SCREEN_HEIGHT)

# ==========================================
# CLASS 3: SIMULATION MANAGER (Main Loop)
# ==========================================
class SimulationManager:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Multi-UAV Decentralized Exploration (Paper Replica)")
        self.clock = pygame.time.Clock()
        self.running = True
        
        self.map = MapManager()
        self.drones = []
        
        # Spawn Drones in the center
        for i in range(NUM_DRONES):
            self.drones.append(Drone(i, SCREEN_WIDTH//2 + i*10, SCREEN_HEIGHT//2, self.map))

    def run(self):
        while self.running:
            self.handle_events()
            self.update_physics()
            self.draw()
            self.clock.tick(30) # 30 FPS

        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update_physics(self):
        for drone in self.drones:
            drone.sense()          # LiDAR Raycast
            drone.plan(self.drones)# Decision Logic
            drone.update()         # Move

    def draw(self):
        self.screen.fill((0, 0, 0))
        
        # 1. Draw Map Surface
        # Create a surface from the grid array
        # This is a bit slow in python, optimized approach:
        # Construct an RGB array
        visual_map = np.zeros((GRID_W, GRID_H, 3), dtype=np.uint8)
        visual_map[self.map.grid == 0] = COLOR_UNKNOWN
        visual_map[self.map.grid == 1] = COLOR_FREE
        visual_map[self.map.grid == 2] = COLOR_OBSTACLE
        
        # Scale it up to screen size
        surf = pygame.surfarray.make_surface(visual_map)
        surf = pygame.transform.scale(surf, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.screen.blit(surf, (0, 0))
        
        # 2. Draw Drones
        for drone in self.drones:
            # Body
            pygame.draw.circle(self.screen, drone.color, (int(drone.pos[0]), int(drone.pos[1])), 6)
            # Sensor Range Ring
            pygame.draw.circle(self.screen, (50, 255, 50), (int(drone.pos[0]), int(drone.pos[1])), SENSOR_RANGE, 1)

        pygame.display.flip()

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    sim = SimulationManager()
    sim.run()