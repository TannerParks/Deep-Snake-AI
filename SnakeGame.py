import math
from collections import namedtuple
import pygame
import random
from math import exp
import numpy as np
from collections import deque

# Colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

BLOCK_SIZE = 20  # Size of a block
SPEED = 20  # Speed of the game

# Speeds for the game
SPEED_SLOW = 1
SPEED_SLOWEST = 0.5
SPEED_NORMAL = 20
SPEED_FAST = 800
SPEED_LIST = [1, 20, 200, 800]
PAUSE = False
DEBUG = False
COLLISION_OFF = False

WIDTH = 600 # Width and height of the window
HEIGHT = 600
START_X = 300   # Starting x and y coordinates for the snake
START_Y = 300

BOARD_NORMALIZE = (WIDTH * HEIGHT / (BLOCK_SIZE ** 2))  # Total spaces on the board, use to normalize values
MANHATTAN_NORMALIZE = (WIDTH + HEIGHT) / BLOCK_SIZE # Use to normalize manhattan distance values

Point = namedtuple('Point', ['x', 'y'])


class Fruit:
    def __init__(self, window, snake, grid):
        self.window = window
        self.snake = snake  # Reference to the snake object, ensures the Fruit class has the most up-to-date snake position
        self.grid = grid
        self.x = None
        self.y = None
        self.move_fruit()

    def draw_fruit(self):
        """Draws the fruit on the board."""
        pygame.draw.rect(self.window, RED, [self.x, self.y, BLOCK_SIZE, BLOCK_SIZE])

    def move_fruit(self):
        """Moves the fruit to a new location. If the location is occupied, move to a new location. This is done by generating a 
        list of all possible points and removing the ones that are occupied by the snake. Then, a random point is chosen from the 
        remaining points. We use this in favor of a while loop to prevent the game from freezing if the snake is too long."""
        #all_points = [(x, y) for x in range(0, WIDTH, BLOCK_SIZE) for y in range(0, HEIGHT, BLOCK_SIZE)]
        #snake_positions = set(zip(self.snake.x, self.snake.y))
        #free_points = [pt for pt in all_points if pt not in snake_positions]
        free_points = []
        for y in range(HEIGHT // BLOCK_SIZE):
            for x in range(WIDTH // BLOCK_SIZE):
                if self.grid[y][x] == 0:
                    free_points.append((x * BLOCK_SIZE, y * BLOCK_SIZE))

        # If there are free points, choose one at random
        if free_points:
            self.x, self.y = random.choice(free_points)
        else:
            print("No free points")
            # TODO: End game


class Snake:
    def __init__(self, window, length):
        self.window = window
        self.length = length
        self.x = [START_X] * length # Starting x and y coordinates for the snake
        self.y = [START_Y] * length
        self.direction = "right"  # Default starting direction
        self.last_tail_x = START_X  # Store last tail position
        self.last_tail_y = START_Y

    def move_up(self):
        if self.direction != "down":
            self.direction = "up"

    def move_down(self):
        if self.direction != "up":
            self.direction = "down"

    def move_left(self):
        if self.direction != "right":
            self.direction = "left"

    def move_right(self):
        if self.direction != "left":
            self.direction = "right"

    def grow(self):
        """Increases the length of the snake by one."""
        self.length += 1
        self.x.append(self.last_tail_x)
        self.y.append(self.last_tail_y)

    def draw_snake(self):
        """Draws the snake."""
        #self.window.fill(BLACK)
        for i in range(self.length):
            if i == 0:
                # Head of the snake
                pygame.draw.rect(self.window, BLACK, [self.x[i], self.y[i], BLOCK_SIZE, BLOCK_SIZE])  # Outline
                pygame.draw.rect(self.window, (0, 150, 20),
                                 [self.x[i] + 2, self.y[i] + 2, 16, 16])  # Slightly different green to make head easier to see in training

                # Eye positions based on direction
                match self.direction:
                    case "up":
                        eye1 = (self.x[i] + 6, self.y[i] + 6)
                        eye2 = (self.x[i] + 14, self.y[i] + 6)
                    case "down":
                        eye1 = (self.x[i] + 6, self.y[i] + 14)
                        eye2 = (self.x[i] + 14, self.y[i] + 14)
                    case "left":
                        eye1 = (self.x[i] + 6, self.y[i] + 6)
                        eye2 = (self.x[i] + 6, self.y[i] + 14)
                    case "right":
                        eye1 = (self.x[i] + 14, self.y[i] + 6)
                        eye2 = (self.x[i] + 14, self.y[i] + 14)

                # Draw the eyes
                pygame.draw.circle(self.window, WHITE, eye1, 3)
                pygame.draw.circle(self.window, WHITE, eye2, 3)
            else:
                pygame.draw.rect(self.window, BLACK, [self.x[i], self.y[i], BLOCK_SIZE, BLOCK_SIZE])  # Outline
                pygame.draw.rect(self.window, GREEN,
                                 [self.x[i] + BLOCK_SIZE * 0.1, self.y[i] + BLOCK_SIZE * 0.1, BLOCK_SIZE * 0.8,
                                  BLOCK_SIZE * 0.8])

    def AI_update_direction(self, action: list[int]):
        """Updates the direction of the snake based on the action taken."""
        directions = ["right", "down", "left", "up"]
        idx = directions.index(self.direction)

        # Change the direction based on the action
        match action:
            case [1, 0, 0]:  # Continue in same direction
                self.direction = directions[idx]
                #print("straight")
            case [0, 1, 0]:  # Make a right turn
                self.direction = directions[(idx + 1) % 4]
                #print("right")
            case [0, 0, 1]:  # Make a left turn
                self.direction = directions[(idx - 1) % 4]
                #print("left")
            case _:
                pass

    def move_snake(self):
        """Moves the snake by updating the x and y coordinates of each block. The head of the snake moves in the direction it is going."""
        self.last_tail_x = self.x[-1]
        self.last_tail_y = self.y[-1]

        # Shift blocks to position of the block in front of it
        for i in range(self.length - 1, 0, -1):
            self.x[i] = self.x[i - 1]
            self.y[i] = self.y[i - 1]

        # Move the head of the snake in the direction it is facing
        match self.direction:
            case "up":
                self.y[0] -= BLOCK_SIZE
            case "down":
                self.y[0] += BLOCK_SIZE
            case "left":
                self.x[0] -= BLOCK_SIZE
            case "right":
                self.x[0] += BLOCK_SIZE


class Game:
    def __init__(self, human=True):
        self.human = human
        pygame.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()

        if DEBUG:
            self.debug_overlays = []
            self.debug_info = {}


        self.snake = None
        self.fruit = None
        self.grid = [[0 for _ in range(WIDTH // BLOCK_SIZE)] for _ in range(HEIGHT // BLOCK_SIZE)]
        self.score = 0
        self.frame_iteration = 0
        self.games_played = 0
        self.fruit_direction_changes = 0                    # TODO: Testing
        self.prev_fruit_direction = None                    # TODO: Testing
        self.recent_fruit_times = deque([], maxlen=20)
        self.dynamic_timeout = 0
        self.running = True
        self.reward = 0
        self.prev_dist_to_fruit = None
        self.directional_fruit_reachability = None
        self.directional_dangers = None
        self.directional_accessible_areas = None
        self.prev_directional_accessible_areas = None
        self.prev_accessible_area = None
        self.directional_escape_exists = None
        self.directional_escape_times = None
        self.directional_escape_candidates = None
        self.prev_had_escape = None
        self.tight_space_counter = None
        self.directional_tail_reachability = None
        self.prev_tail_reachable = None
        self.tail_distance = None
        self.prev_tail_distance = None
        self.tail_loop_counter = None
        self.tail_access_counter = None
        self.prev_max_density = None
        self.local_density = None
        self.approach_density = None
        self.state = None

        # TODO: Initialize statistics tracking
        self.area_utilization_samples = []
        self.recent_rewards_buffer = []
        self.max_reward_buffer_size = 50
        self.process_took_too_long_flag = False
        self.log_file = self.generate_log_filename()
        self.initialize_stats_logging()

        self.reset()

    def reset(self):
        """Resets the game and variables to start a new game or replay."""
        # TODO: Logging
        #if self.games_played > 0:
        #    timeout_occurred = self.process_took_too_long_flag
        #    self.log_game_stats(timeout=timeout_occurred)
        #    #self.process_took_too_long_flag = False

        self.snake = Snake(self.window, 1)
        self.update_grid()
        self.fruit = Fruit(self.window, self.snake, self.grid)
        if DEBUG:
            self.debug_overlays = []
            self.debug_info = {}
        self.score = 0
        self.frame_iteration = 0
        self.games_played += 1
        pygame.display.set_caption(f"Snake Game {self.games_played}      {self.log_file}")
        self.fruit_direction_changes = 0                    # TODO: Testing
        self.prev_fruit_direction = None                    # TODO: Testing
        self.recent_fruit_times.clear()
        self.dynamic_timeout = 250
        self.running = True
        self.reward = 0
        self.prev_dist_to_fruit = self.check_distance_to_fruit()
        self.directional_dangers = self.distance_to_collision(self.snake.direction)
        (self.directional_accessible_areas, self.directional_tail_reachability,
         self.directional_fruit_reachability, self.directional_escape_exists,
         self.directional_escape_times, self.directional_escape_candidates) = self.flood_fill_directional(self.snake.direction)
        self.prev_directional_accessible_areas = None
        self.prev_accessible_area = None
        self.prev_had_escape = None
        self.tight_space_counter = 0
        self.prev_tail_reachable = None
        self.tail_distance = 0
        self.prev_tail_distance = None
        self.tail_loop_counter = 0
        self.tail_access_counter = 0
        self.local_density = self.get_local_density()
        self.prev_max_density = None
        self.approach_density = None
        self.update_game_state()
        self.state = self.get_state()

        return self.state

    def update_grid(self):
        """Update grid with current snake position"""
        if self.collision():  # Don't update grid if snake is in collision state
            return

        # Clear grid
        for y in range(HEIGHT // BLOCK_SIZE):
            for x in range(WIDTH // BLOCK_SIZE):
                self.grid[y][x] = 0

        # Mark snake positions
        for i in range(self.snake.length):
            x = self.snake.x[i] // BLOCK_SIZE
            y = self.snake.y[i] // BLOCK_SIZE
            self.grid[y][x] = 1

    def display_score(self):
        """Displays the number of fruits eaten."""
        font = pygame.font.SysFont("Verdana", 25)
        score = font.render(f"SCORE: {self.score}", True, (200, 200, 200))
        self.window.blit(score, (0, 0))

    def handle_events(self):
        """Handles events such as key presses and window closing."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                # If the game is being played by a human, handle key presses
                if self.human:
                    self.handle_keydown(event.key)
                else:
                    self.AI_handle_keydown(event.key)

    def handle_keydown(self, key):
        """Handles key presses."""
        global SPEED
        global DEBUG
        global COLLISION_OFF
        global PAUSE

        match key:
            case pygame.K_UP | pygame.K_w:
                self.snake.move_up()
            case pygame.K_DOWN | pygame.K_s:
                self.snake.move_down()
            case pygame.K_LEFT | pygame.K_a:
                self.snake.move_left()
            case pygame.K_RIGHT | pygame.K_d:
                self.snake.move_right()
            case pygame.K_ESCAPE:
                COLLISION_OFF = False
                self.running = False
            case pygame.K_1:
                SPEED -= 0.5 if SPEED > 0 else 1
            case pygame.K_2:
                SPEED += 0.5
            case pygame.K_SPACE:
                if SPEED > 1:
                    SPEED = SPEED_SLOW
                else:
                    SPEED = SPEED_NORMAL
            case pygame.K_g:
                self.snake.grow()
            case pygame.K_e:
                self.process_ate_fruit()
            case pygame.K_f:
                PAUSE = not PAUSE
            case pygame.K_r:
                self.fruit.move_fruit()
            case pygame.K_NUMLOCK:
                if not DEBUG:
                    DEBUG = True
                else:
                    DEBUG = False
            case pygame.K_c:
                COLLISION_OFF = True if not COLLISION_OFF else False
                print(f"Collision turned off: {COLLISION_OFF}")
            case pygame.K_EQUALS:
                print("RESETTING")
                self.reset()

    def AI_handle_keydown(self, key):
        """Handles key presses available while training the AI (these are user pressed)"""
        global SPEED
        global DEBUG
        global PAUSE

        match key:
            case pygame.K_SPACE:
                if SPEED > 1:
                    SPEED = SPEED_SLOWEST
                else:
                    SPEED = SPEED_NORMAL
            case pygame.K_1:
                if SPEED != SPEED_SLOWEST:
                    SPEED = SPEED_SLOWEST
            case pygame.K_2:
                if SPEED != SPEED_SLOW:
                    SPEED = SPEED_SLOW
            case pygame.K_3:
                if SPEED != SPEED_NORMAL:
                    SPEED = SPEED_NORMAL
            case pygame.K_4:
                if SPEED != SPEED_FAST:
                    SPEED = SPEED_FAST
            case pygame.K_f:
                PAUSE = not PAUSE
            case pygame.K_o:
                self.running = False
                print("Game ended")
            case pygame.K_NUMLOCK:
                if not DEBUG:
                    DEBUG = True
                else:
                    DEBUG = False

    def update_game_state(self):
        """Updates the game state by moving the snake, drawing the snake and fruit, and displaying the score."""
        self.snake.move_snake()
        self.update_grid()
        self.window.fill(BLACK) # Clear screen at start of frame
        self.fruit.draw_fruit()
        self.snake.draw_snake()
        self.display_score()
        pygame.display.update()

    def snake_ate_fruit(self):
        """Returns true if the snake has eaten a fruit."""
        return self.snake.x[0] == self.fruit.x and self.snake.y[0] == self.fruit.y

    def update_dynamic_timeout(self):
        """Update the timeout dynamically based on recent fruit collection times."""
        base_timeout = 250  # Minimum timeout
        length_factor = 2 + (self.snake.length / BOARD_NORMALIZE) * 3

        if len(self.recent_fruit_times) >= 3:  # Ensure we have enough data
            # Calculate median time between fruits
            time_diffs = np.diff(list(self.recent_fruit_times))

            # Give more weight to recent times using exponential weights
            weights = np.exp(np.linspace(0, 1, len(time_diffs)))
            weighted_avg_time = np.average(time_diffs, weights=weights)

            dynamic_component = weighted_avg_time * 2 * length_factor

            self.dynamic_timeout = self.frame_iteration + dynamic_component
        else:
            # Default timeout scales with snake length
            self.dynamic_timeout = self.frame_iteration + (base_timeout * length_factor)

    def process_ate_fruit(self):
        """Processes the snake eating a fruit by increasing the score, length, and moving the fruit."""
        self.score += 1
        self.snake.grow()  # Add a block to the snake
        self.update_grid()
        self.fruit.move_fruit()
        self.recent_fruit_times.append(self.frame_iteration)
        self.update_dynamic_timeout()

    def check_L2_distance_to_fruit(self):
        """Calculates the Euclidean distance between the snake and the fruit then normalizes it between 1 and 0."""
        snake_head = (self.snake.x[0], self.snake.y[0])
        fruit = (self.fruit.x, self.fruit.y)
        dist_to_fruit = round(math.dist(snake_head, fruit))

        max_distance = math.sqrt(WIDTH**2 + HEIGHT**2)
        normalized_distance = dist_to_fruit / max_distance

        return normalized_distance

    def check_distance_to_fruit(self):
        """Calculate Manhattan distance to fruit"""
        manhattan_dist = (abs(self.fruit.x - self.snake.x[0]) + abs(self.fruit.y - self.snake.y[0])) / BLOCK_SIZE

        return manhattan_dist

    def collision(self, point=None):
        """Checks if a point has collided with anything by making a set of the snake's body and checking if the point is in it or
        checking if the point is out of bounds"""
        if point is None:
            point = Point(self.snake.x[0], self.snake.y[0]) # Default to the head of the snake
        snake_body = set(zip(self.snake.x[1:], self.snake.y[1:]))

        if (point.x > WIDTH - BLOCK_SIZE) or (point.x < 0) or (point.y < 0) or (point.y > HEIGHT - BLOCK_SIZE):
            return 1
        if point in snake_body:
            return 1
        return 0

    def process_collision(self):
        """Processes the snake colliding with something."""
        self.running = False

    def flood_fill(self, point=None, growing=False):
        """Flood fill algorithm to find the number of accessible points from a given point, whether the tail and
        fruit is accessible, and possible escape routes within the area."""
        if self.collision():  # Return 0, 0, 0 if the snake is in a collision state
            return 0, 0, 0, 0, 0, 0

        if point is None:
            point = Point(self.snake.x[0], self.snake.y[0])  # Default to the head of the snake

        # Create a copy of the grid to simulate the next move
        # If not growing, the tail will move so mark the current tail cell as free
        grid_copy = [row[:] for row in self.grid]
        tail_position = (self.snake.x[-1], self.snake.y[-1])
        #if not growing: # TODO Simulation
        #    tail_x = tail_position[0] // BLOCK_SIZE
        #    tail_y = tail_position[1] // BLOCK_SIZE
        #    grid_copy[tail_y][tail_x] = 0  # Simulate tail movement

        queue = deque([point])
        accessible_points = 0
        visited = set()

        fruit_position = (self.fruit.x, self.fruit.y)
        # Determine what the tail will be after moving.
        #if growing: # TODO Simulation
        #    next_tail_position = tail_position
        #else:
        #    if self.snake.length > 1:
        #        next_tail_position = (self.snake.x[-2], self.snake.y[-2])
        #    else:
        #        next_tail_position = tail_position

        tail_reachable = 0
        fruit_reachable = 0

        while queue:
            pt = queue.popleft()
            x = pt.x // BLOCK_SIZE
            y = pt.y // BLOCK_SIZE

            # Out of bounds or already visited?
            if pt in visited or x < 0 or x >= WIDTH // BLOCK_SIZE or y < 0 or y >= HEIGHT // BLOCK_SIZE:
                continue

            # Check if we reached the tail (or where it will be)
            #if (pt.x, pt.y) == tail_position or (pt.x, pt.y) == next_tail_position: # TODO Simulation
            #    tail_reachable = 1
            if (pt.x, pt.y) == tail_position:
                tail_reachable = 1

            # Check if we've reached the fruit
            if (pt.x, pt.y) == fruit_position:
                fruit_reachable = 1

            # If the cell is occupied by the snake, block it.
            # Since we already freed the tail in grid_copy (if not growing), we can simply check:
            if grid_copy[y][x] == 1:
                continue

            visited.add(pt)
            accessible_points += 1

            # Add the adjacent points.
            queue.append(Point(pt.x + BLOCK_SIZE, pt.y))
            queue.append(Point(pt.x - BLOCK_SIZE, pt.y))
            queue.append(Point(pt.x, pt.y + BLOCK_SIZE))
            queue.append(Point(pt.x, pt.y - BLOCK_SIZE))

        escape_time, candidate_point, has_escape = self._find_escape_coordinates(visited, accessible_points)

        return accessible_points, tail_reachable, fruit_reachable, has_escape, escape_time, candidate_point

    def flood_fill_directional(self, direction):
        """Performs a flood fill algorithm in each direction to find the number of accessible points in each direction."""
        # Mapping relative directions to vector adjustments
        match direction:
            case "up":
                directions = {'straight': (0, -BLOCK_SIZE), 'left': (-BLOCK_SIZE, 0), 'right': (BLOCK_SIZE, 0)}
            case "down":
                directions = {'straight': (0, BLOCK_SIZE), 'left': (BLOCK_SIZE, 0), 'right': (-BLOCK_SIZE, 0)}
            case "left":
                directions = {'straight': (-BLOCK_SIZE, 0), 'left': (0, BLOCK_SIZE), 'right': (0, -BLOCK_SIZE)}
            case "right":
                directions = {'straight': (BLOCK_SIZE, 0), 'left': (0, -BLOCK_SIZE), 'right': (0, BLOCK_SIZE)}

        accessible_points = {"straight": 0, "left": 0, "right": 0}
        tail_reachability = {"straight": 0, "left": 0, "right": 0}
        fruit_reachability = {"straight": 0, "left": 0, "right": 0}
        escape_exists = {"straight": 0, "left": 0, "right": 0}
        escape_times = {"straight": None, "left": None, "right": None}
        escape_candidates = {"straight": None, "left": None, "right": None}

        # Check if the snake's head is on the fruit (means the tail won't move on the next turn)
        growing = (self.snake.x[0] == self.fruit.x and self.snake.y[0] == self.fruit.y)

        for rel_dir, (dx, dy) in directions.items():
            # Calculate new starting point based on relative direction
            (x, y) = (self.snake.x[0] + dx, self.snake.y[0] + dy)
            accessible, tail_reachable, fruit_reachable, has_escape, escape_time, candidate_point = self.flood_fill(Point(x, y), growing=growing)

            accessible_points[rel_dir] = accessible
            tail_reachability[rel_dir] = tail_reachable
            fruit_reachability[rel_dir] = fruit_reachable
            escape_exists[rel_dir] = 1 if has_escape else 0
            escape_times[rel_dir] = escape_time
            escape_candidates[rel_dir] = candidate_point

        #print("\n")

        return accessible_points, tail_reachability, fruit_reachability, escape_exists, escape_times, escape_candidates

    def _find_escape_coordinates(self, visited_points, accessible_area):
        """Identifies snake segments that will become escape routes."""
        if accessible_area == 0:
            return 0, None, False
        # Only check segments up to accessible area
        max_segments_to_check = min(self.snake.length, accessible_area)

        #is_corridor = self._is_corridor_area(visited_points)
        head_point = Point(self.snake.x[0], self.snake.y[0])
        tail_point = Point(self.snake.x[-1], self.snake.y[-1])

        # Check each tail segment starting from the end
        for i in range(1, max_segments_to_check):
            tail_idx = -i
            segment_x, segment_y = self.snake.x[tail_idx], self.snake.y[tail_idx]
            candidate_point = Point(segment_x, segment_y)

            # Check if any adjacent cell is in the visited area (this means it's accessible)
            for dx, dy in [(BLOCK_SIZE, 0), (-BLOCK_SIZE, 0), (0, BLOCK_SIZE), (0, -BLOCK_SIZE)]:
                adjacent_x, adjacent_y = segment_x + dx, segment_y + dy
                adjacent_point = Point(adjacent_x, adjacent_y)

                # Skip candidate if it's too close to the head (this is to avoid times when the next move covers the escape)
                # If the candidate point is the tail then we don't skip since it'll be opened on the following turn
                if abs(adjacent_x - head_point.x) + abs(adjacent_y - head_point.y) <= BLOCK_SIZE and (candidate_point != tail_point):
                    continue

                if adjacent_point in visited_points:
                    if self._is_viable_candidate(adjacent_point, visited_points):
                        #print(f"Escape Found at: {adjacent_point}\tSegment: {i}")
                        return i, candidate_point, True

        return 0, None, False

    def _is_viable_candidate(self, candidate_point, visited_points):
        """Determines if an escape candidate has enough room to be usable."""
        # If the candidate is the tail, return true
        tail_point = Point(self.snake.x[-1], self.snake.y[-1])
        if candidate_point == tail_point:
            return True

        # Check orthogonal neighbors (direct movement options)
        orthogonal_neighbors = [
            Point(candidate_point.x + dx, candidate_point.y + dy)
            for dx, dy in [(BLOCK_SIZE, 0), (-BLOCK_SIZE, 0), (0, BLOCK_SIZE), (0, -BLOCK_SIZE)]
        ]

        # Check diagonal neighbors (additional space detection)
        diagonal_neighbors = [
            Point(candidate_point.x + dx, candidate_point.y + dy)
            for dx, dy in [(BLOCK_SIZE, BLOCK_SIZE), (-BLOCK_SIZE, BLOCK_SIZE),
                           (BLOCK_SIZE, -BLOCK_SIZE), (-BLOCK_SIZE, -BLOCK_SIZE)]
        ]

        orthogonal_accessible = sum(1 for n in orthogonal_neighbors if n in visited_points)
        diagonal_accessible = sum(1 for n in diagonal_neighbors if n in visited_points)

        # A viable escape needs either good orthogonal movement options or enough combined space for maneuvering
        return orthogonal_accessible >= 3 or (orthogonal_accessible >= 2 and diagonal_accessible >= 1)

    def get_escape_direction(self, candidate_point):
        """Given a candidate point and the current head position, return a one-hot vector for cardinal directions."""
        head = Point(self.snake.x[0], self.snake.y[0])
        direction = [0, 0, 0, 0]  # up, right, down, left
        if candidate_point.y < head.y:
            direction[0] = 1
        if candidate_point.x > head.x:
            direction[1] = 1
        if candidate_point.y > head.y:
            direction[2] = 1
        if candidate_point.x < head.x:
            direction[3] = 1
        return direction

    def get_nearest_escape(self, escape_times, escape_exists, candidate_points):
        """Given dictionaries for each relative direction, pick the candidate with the smallest escape time."""
        best_time = float('inf')
        best_candidate = None
        for d in ["straight", "left", "right"]:
            if escape_exists[d]:
                if escape_times[d] is not None and escape_times[d] < best_time:
                    best_time = escape_times[d]
                    best_candidate = candidate_points[d]
        return best_candidate

    def _is_corridor_area(self, visited_points):
        """Determines if an area is a narrow corridor with limited maneuvering space."""
        # Sample points to check connectivity
        sample_size = min(20, len(visited_points))
        sample_points = random.sample(list(visited_points), sample_size) if len(visited_points) > sample_size else list(visited_points)

        corridor_points = 0

        for point in sample_points:
            # Count adjacent points
            adjacent_count = 0
            for dx, dy in [(BLOCK_SIZE, 0), (-BLOCK_SIZE, 0), (0, BLOCK_SIZE), (0, -BLOCK_SIZE)]:
                if Point(point.x + dx, point.y + dy) in visited_points:
                    adjacent_count += 1

            # Points with 2 or fewer neighbors indicate corridor-like structure
            if adjacent_count <= 2:
                corridor_points += 1

        # If more than 70% of sampled points have corridor characteristics
        return corridor_points / max(1, len(sample_points)) > 0.7

    def took_too_long(self):
        """Encourages the AI to find a solution faster by penalizing it for taking too long (time limit is dynamic)."""
        return self.frame_iteration > self.dynamic_timeout

    def process_took_too_long(self):
        """Processes the game taking too long."""
        #if self.snake.length >= 500:
        #    #print(f"[DEBUG] Timeout turned off!")
        #    return
        print(f"Took too long, game {self.games_played} ending, Score {self.score}")
        self.process_took_too_long_flag = True
        self.running = False

    def distance_to_collision(self, rel_dir):
        """Calculates the distance between the snake and possible collisions in each direction."""
        match rel_dir:
            case "up":
                directions = {'straight': (0, -BLOCK_SIZE), 'left': (-BLOCK_SIZE, 0), 'right': (BLOCK_SIZE, 0)}
            case "down":
                directions = {'straight': (0, BLOCK_SIZE), 'left': (BLOCK_SIZE, 0), 'right': (-BLOCK_SIZE, 0)}
            case "left":
                directions = {'straight': (-BLOCK_SIZE, 0), 'left': (0, BLOCK_SIZE), 'right': (0, -BLOCK_SIZE)}
            case "right":
                directions = {'straight': (BLOCK_SIZE, 0), 'left': (0, -BLOCK_SIZE), 'right': (0, BLOCK_SIZE)}

        dangers = {'straight': 0, 'left': 0, 'right': 0}

        for rel_dir, (dx, dy) in directions.items():
            (x, y) = (self.snake.x[0] + dx, self.snake.y[0] + dy)
            distance = 0
            while self.collision(Point(x, y)) == 0: # Keep going until a collision would happen
                x += dx
                y += dy
                distance += 1

            # Maximum distance calculations need to consider the game board's dimensions
            if dx != 0:  # Moving horizontally
                max_distance = WIDTH / BLOCK_SIZE
            else:  # Moving vertically
                max_distance = HEIGHT / BLOCK_SIZE

            dangers[rel_dir] = distance/max_distance

        return dangers

    def check_distance_to_tail(self):
        """Calculates distance between head and tail"""
        manhattan_dist = (abs(self.snake.x[-1] - self.snake.x[0]) + abs(self.snake.y[-1] - self.snake.y[0])) / BLOCK_SIZE

        return manhattan_dist

    def get_local_density(self, radius=5):
        """Calculate actual density of snake segments in each direction."""
        densities = {'left': 0, 'right': 0, 'up': 0, 'down': 0}
        total_spaces = {'left': 0, 'right': 0, 'up': 0, 'down': 0}

        head_x = self.snake.x[0] // BLOCK_SIZE
        head_y = self.snake.y[0] // BLOCK_SIZE

        # Check each direction
        for y in range(max(0, head_y - radius), min(HEIGHT // BLOCK_SIZE, head_y + radius + 1)):
            for x in range(max(0, head_x - radius), min(WIDTH // BLOCK_SIZE, head_x + radius + 1)):
                if x == head_x and y == head_y:
                    continue  # Skip head position

                # Determine direction from head
                x_diff = x - head_x
                y_diff = y - head_y

                if abs(x_diff) > abs(y_diff):
                    direction = 'left' if x_diff < 0 else 'right'
                else:
                    direction = 'up' if y_diff < 0 else 'down'

                total_spaces[direction] += 1
                if self.grid[y][x] == 1:  # Snake body present
                    densities[direction] += 1

        #print(f"{densities}\n{total_spaces}\n{[densities[d] / max(1, total_spaces[d]) for d in ['left', 'right', 'up', 'down']]}\n")

        # Return actual density (occupied/total) for each direction
        return [densities[d] / max(1, total_spaces[d]) for d in ['left', 'right', 'up', 'down']]

    def get_fruit_accessibility_bonus(self):
        """Computes a bonus factor for fruit reward based on the fruit's proximity to walls.
        A fruit near a wall (harder to reach) gets a higher bonus."""
        wall_margin = BLOCK_SIZE * 2  # Define how close to a wall is considered 'difficult'
        bonus = 1.0

        # Check horizontal proximity to walls
        if self.fruit.x < wall_margin or self.fruit.x > WIDTH - wall_margin:
            bonus += 0.1
        # Check vertical proximity to walls
        if self.fruit.y < wall_margin or self.fruit.y > HEIGHT - wall_margin:
            bonus += 0.1

        return bonus

    def get_space_management_reward(self, current_area, current_tail_reachable):
        """Calculate the reward/penalty for space management such as partitions, finding space, going into a tight space, etc."""
        # Skip calculation if no previous area recorded
        if self.prev_accessible_area is None or current_area < 1 or abs(current_area - self.prev_accessible_area) <= 1:
            return 0, False

        # Calculate normalized metrics
        snake_length = self.snake.length
        remaining_space = BOARD_NORMALIZE - snake_length
        normalized_snake_length = snake_length / BOARD_NORMALIZE
        epsilon = 1e-6  # Avoid division by zero

        # Determine if we're in a tight space
        critical_ratio = 1.5  # We want at least 1.5x snake length in accessible area
        minimum_desired_area = min(remaining_space, snake_length * critical_ratio)
        in_tight_space = current_area < minimum_desired_area

        space_reward = 0

        # ---Lost space penalty---
        base_penalty = 12
        min_penalty = 3

        # Scale by snake length with exponential growth for longer snakes (shared by penalty and recovery)
        length_factor = 1 + (normalized_snake_length ** 1.5) * 4

        if current_area < self.prev_accessible_area and abs(self.prev_accessible_area - current_area) > 1:
            # Calculate percentage of area lost
            area_loss_ratio = (self.prev_accessible_area - current_area) / self.prev_accessible_area

            # Apply a curve to better differentiate between small and large losses
            curved_loss = area_loss_ratio ** 0.4

            # Add space constraint factor (increases penalty when snake had limited space)
            space_ratio = min(1.0, self.prev_accessible_area / (remaining_space + epsilon))
            space_constraint_factor = 1 + (1 - space_ratio) ** 0.8

            # Add a small-partition boost for very small partitions
            small_partition_boost = 1.5 if area_loss_ratio < 0.1 else 1.0

            # Calculate final penalty with minimum threshold
            partition_penalty = -max(min_penalty,
                                     base_penalty * curved_loss * length_factor * space_constraint_factor * small_partition_boost)

            self.debug_info["Partition Penalty"] = partition_penalty
            space_reward += partition_penalty

        # ---Space Recovery Bonus---
        if self.prev_accessible_area < minimum_desired_area and current_area > self.prev_accessible_area:
            recovery_ratio = current_area / max(1, self.prev_accessible_area)
            significant_recovery = recovery_ratio > 1.2  # 20% improvement threshold

            if significant_recovery:
                # Calculate the equivalent penalty for losing this much space
                # This simulates what the penalty would be if we were going from current_area to prev_accessible_area
                area_loss_ratio = (current_area - self.prev_accessible_area) / current_area
                curved_loss = area_loss_ratio ** 0.4
                space_ratio = min(1.0, current_area / (remaining_space + epsilon))
                space_constraint_factor = 1 + (1 - space_ratio) ** 0.8
                small_partition_boost = 1.5 if area_loss_ratio < 0.1 else 1.0
                equivalent_penalty = max(min_penalty,
                                         base_penalty * curved_loss * length_factor * space_constraint_factor * small_partition_boost)

                # Apply a 65% cap to ensure recovery is always less than equivalent penalty
                max_recovery_cap = 0.65 * equivalent_penalty

                # Calculate standard recovery bonus
                tight_space_factor = max(0.2, min(1.0, self.prev_accessible_area / minimum_desired_area))
                gain_ratio = min(1.0, (current_area - self.prev_accessible_area) / max(minimum_desired_area - self.prev_accessible_area, 1))
                curved_gain = gain_ratio ** 0.4
                calculated_bonus = 8 * curved_gain * length_factor * (tight_space_factor ** 0.5)

                # Use the smaller of the calculated bonus and the capped value
                recovery_bonus = min(calculated_bonus, max_recovery_cap)
                self.debug_info["Space Recovery Bonus"] = recovery_bonus

        return space_reward, in_tight_space

    def get_reward(self):
        """Get the reward for the current state."""
        reward = 0
        current_accessible_area = max(self.directional_accessible_areas.values())
        remaining_area = BOARD_NORMALIZE - self.snake.length
        normalized_snake_length = self.snake.length / BOARD_NORMALIZE
        self.debug_info = {}    # Reset debug_info on new pass

        # --- Fruit Reward ---
        # Reward the AI for eating the fruit (scaled with its length)
        avg_fruit_interval = np.mean(np.diff(list(self.recent_fruit_times))) if len(self.recent_fruit_times) >= 2 else 0
        fruit_reward = 0
        if self.snake_ate_fruit():
            self.tail_loop_counter = 0  # Reset loop counter when fruit has been eaten (avoids penalizing it for a loop after it's eaten)
            fruit_accessibility_bonus = self.get_fruit_accessibility_bonus()
            fruit_reward = 100 + min(100, 0.25 * self.snake.length)
            fruit_reward *= fruit_accessibility_bonus

            if self.prev_accessible_area is not None and current_accessible_area < self.prev_accessible_area and abs(self.prev_accessible_area - current_accessible_area) > 2:
                # Risk bonus for partition causing fruits
                risk_bonus = min(25, 0.25 * self.snake.length)
                fruit_reward += risk_bonus
                self.debug_info['Partition Risk Bonus'] = risk_bonus

            reward += fruit_reward
            self.process_ate_fruit()
        self.debug_info['Fruit Reward'] = fruit_reward

        # --- Distance Reward ---
        # Reward the AI for getting closer to the fruit
        current_dist_to_fruit = self.check_distance_to_fruit()
        current_tail_reachable = any(self.directional_tail_reachability.values())
        access_to_fruit = any(self.directional_fruit_reachability.values())
        has_escape = any(self.directional_escape_exists.values()) or current_tail_reachable

        distance_reward = 0

        if current_dist_to_fruit > self.prev_dist_to_fruit and access_to_fruit:
            distance_reward = -1/(1 + 0.01 * self.snake.length)
        elif current_dist_to_fruit < self.prev_dist_to_fruit and access_to_fruit:
            distance_reward = 1/(1 + 0.01 * self.snake.length)

        if has_escape:
            distance_reward *= 2

        reward += distance_reward
        self.debug_info['Distance Reward'] = distance_reward

        # --- Space Management Rewards/Penalties ---
        space_reward, in_tight_space = self.get_space_management_reward(current_accessible_area, current_tail_reachable)
        reward += space_reward

        #print(f"Reward: {space_reward}\n{space_reward_debug}\n")

        # --- Density Rewards/Penalties (Adaptive) ---
        max_density = max(self.local_density)
        avg_density = sum(self.local_density) / 4
        dense_directions = sum(1 for d in self.local_density if d > 0.5)
        organization_reward = 0

        # Reward organized density patterns
        if max_density > 0.7 and self.snake.length > 10:  # Only trigger if there's significant density
            density_diff = max_density - avg_density
            organization_reward = 5 * density_diff

            # If tail is reachable, apply an extra 1.25x boost for good configurations
            if current_tail_reachable and dense_directions in [1, 2, 3]:
                organization_reward *= 1.5
            elif not current_tail_reachable and dense_directions == 4:
                # Penalize if dense in all directions AND no tail access
                organization_reward = -3 * (1 - density_diff)

            # Scale reward with length since organization matters more for longer snakes
            if self.snake.length > 100:
                length_scale = 1 + (np.log2(self.snake.length / 100) / 10) # ~1.0-1.3x multiplier
                organization_reward *= length_scale

            if in_tight_space: # Encourage density in tight spaces, also discourages spiraling in tight spaces
                if any(self.directional_escape_exists.values()):
                    organization_reward *= 1.5

            reward += organization_reward
            self.debug_info['Organization Reward'] = organization_reward

        self.prev_max_density = max_density

        # --- Tail Reachability Reward/Penalty ---
        min_length_threshold = 15
        tail_access_change = 0

        # Calculate reward components
        tail_access_change = 0
        tail_consistency_reward = 0

        if self.snake.length > min_length_threshold:
            # Update counter with memory effect
            if current_tail_reachable:
                self.tail_access_counter = min(self.tail_access_counter + 1, 10)
            else:
                # Faster decay when tail becomes inaccessible (immediate feedback)
                self.tail_access_counter = max(self.tail_access_counter - 2, -10)

            # Base reward based on snake length
            base_reward_scale = 2 + (normalized_snake_length * 2)

            # Reward for consistently having tail access
            if current_tail_reachable:
                # Reward factor grows with the positive side of tail_access_counter
                consistency_factor = (1+(self.tail_access_counter / 10))
                tail_consistency_reward = base_reward_scale * consistency_factor
            else:
                # Penalty factor grows with how negative the counter is
                inaccessible_factor = -(1-(self.tail_access_counter / 10))
                tail_consistency_reward = base_reward_scale * inaccessible_factor


            # 2. Reward for changes in accessibility
            if current_tail_reachable and not self.prev_tail_reachable:
                # Reward for regaining access (recovery)
                tail_access_change = base_reward_scale * 2
            elif not current_tail_reachable and self.prev_tail_reachable:
                # Significant penalty for losing access
                tail_access_change = -base_reward_scale * 3

            #print(f"Counter: {self.tail_access_counter}\tConsistency: {tail_consistency_reward}\tAccess: {tail_access_change}")

        reward += (tail_access_change + tail_consistency_reward)
        self.debug_info['Tail Access Reward'] = tail_access_change
        self.debug_info['Tail Consistency Reward'] = tail_consistency_reward
        self.prev_tail_reachable = current_tail_reachable

        # --- Escape Route Rewards/Penalties ---
        escape_reward = 0

        if self.prev_had_escape is not None:
            # 1. Base reward for having an escape route in tight spaces (meant to encourage stalling for an exit)
            if current_accessible_area < remaining_area and has_escape:     # FIXME: Rewarding in regular play space (why is the ai improving with this)
                if abs(current_accessible_area - self.prev_accessible_area) <= 1:
                    self.tight_space_counter += 1
                else:
                    self.tight_space_counter = 0

                alpha = 0.1  # tuning parameter for decay rate
                reward_candidate = 3 * math.exp(-alpha * self.tight_space_counter)
                escape_reward = max(reward_candidate, 1)
                print(f"Counter: {self.tight_space_counter}\t\tReward: {escape_reward}")
                self.debug_info['Escape Tight Space Reward'] = max(reward_candidate, 1)

            # 2. Penalty for losing last escape route
            if self.prev_had_escape and not has_escape:
                escape_loss_penalty = -50.0 * (1 + 0.5 * normalized_snake_length)
                #print("ESCAPE LOSS FOOL")
                escape_reward += escape_loss_penalty
                self.debug_info['Escape Loss Penalty'] = escape_loss_penalty

            # 3. Extra penalty for entering tight space with no escape
            if in_tight_space and not has_escape and self.tight_space_counter <= 1:
                no_escape_penalty = -10.0 * (1 + normalized_snake_length)
                escape_reward += no_escape_penalty
                self.debug_info['No Escape Penalty'] = no_escape_penalty

            # 4. Reward for finding escape (possible since we have an imperfect escape route finder)
            if not self.prev_had_escape and has_escape:
                escape_found_reward = 20.0 * (1 + normalized_snake_length)
                #print("ESCAPE FOUND GENIUS")
                escape_reward += escape_found_reward
                self.debug_info["Escape Found Reward"] = escape_found_reward

            reward += escape_reward

        # --- Collision Penalty ---
        #collision_penalty = 0
        #if self.collision():
        #    # Collisions become more costly the longer the snake is
        #    collision_penalty = -130 * (1 + normalized_snake_length)
        #    reward += collision_penalty
        #    self.process_collision()
        #self.debug_info['Collision Penalty'] = collision_penalty

        # --- Tail Loop Penalty ---
        # Compute normalized Manhattan distance between head and tail
        self.tail_distance = self.check_distance_to_tail()
        normalized_tail_distance = self.tail_distance / MANHATTAN_NORMALIZE
        min_length_for_loop = 3
        tail_loop_threshold = 2  # If the head is within 2 blocks of the tail, consider it a loop event

        # If the head is very close to the tail, increment the loop counter
        if self.tail_distance < tail_loop_threshold and self.snake.length > min_length_for_loop:
            self.tail_loop_counter += 1
        else:
            self.tail_loop_counter = 0

        #print(f"Tail Distance: {self.tail_distance}\t\tLength: {self.snake.length}\t\tCounter: {self.tail_loop_counter}")

        # Compare to the average fruit capture frame gap and whether there's enough space move from the tail
        tail_loop_penalty = 0

        if avg_fruit_interval > 0 and self.tail_loop_counter > avg_fruit_interval * 0.5 and current_accessible_area > self.tail_distance:
            tail_loop_penalty = -5 # * ((self.tail_loop_counter - avg_fruit_interval) / self.tail_loop_counter)
            #print(f"PENALTY: {tail_loop_penalty}")

        reward += tail_loop_penalty
        self.debug_info['Tail Loop Penalty'] = tail_loop_penalty

        # --- Head-Tail Slack Reward/Penalty ---        # TODO: TESTING
        #slack_reward = 0

        #if self.tail_distance > self.prev_tail_distance:
        #    slack_reward += 1

        # --- Timeout Penalty ---
        # End game after a certain amount of time to encourage faster solutions
        timeout_penalty = 0
        over_time = self.frame_iteration - self.dynamic_timeout

        if self.took_too_long():
            timeout_penalty = -(1 + 0.002 * over_time)
            #if self.snake.length <= 10 or self.tail_distance <= 5:
            #    timeout_penalty *= 2
            reward += timeout_penalty

            # Force a game over if way past limit
            if over_time > 500:
                #print(f"Overtime exceeded!\t\tFrame Iteration: {self.frame_iteration}\t\tDynamic Timeout: {self.dynamic_timeout}, Length: {self.snake.length}, Score: {self.score}, Reward: {reward}")
                timeout_terminal_penalty = -130 * (1 + normalized_snake_length)
                reward += timeout_terminal_penalty  # TODO: Overwrite reward with terminal penalty? (like collision)
                self.debug_info['Timeout Terminal Penalty'] = timeout_terminal_penalty
                self.process_took_too_long()
        self.debug_info['Timeout Penalty'] = timeout_penalty

        # --- Collision Penalty ---
        collision_penalty = 0
        if self.collision():
            # Collisions become more costly the longer the snake is
            collision_penalty = -130 * (1 + normalized_snake_length)
            reward = collision_penalty  # Overwrite all other rewards and penalties
            self.process_collision()
        self.debug_info['Collision Penalty'] = collision_penalty

        if DEBUG:
            # Check if we're near fruit but not getting it
            head_x = self.snake.x[0] // BLOCK_SIZE
            head_y = self.snake.y[0] // BLOCK_SIZE
            fruit_x = self.fruit.x // BLOCK_SIZE
            fruit_y = self.fruit.y // BLOCK_SIZE
            manhattan_to_fruit = abs(head_x - fruit_x) + abs(head_y - fruit_y)

            # Count non-zero rewards
            non_zero_rewards = sum(1 for value in self.debug_info.values() if abs(value) > 0.01)
            score_magnitude = sum(abs(score) for score in self.debug_info.values())

            # Define interesting conditions to log
            should_log = (
                    abs(reward) > 10 or  # Large total reward
                    non_zero_rewards >= 4 or  # Many active reward components
                    score_magnitude > 15 or
                    #'Cutoff Penalty' in debug_info or  # Any cutoff penalty
                    ('Collision Penalty' in self.debug_info and self.debug_info['Collision Penalty'] < 0)  # Collision
                    #('Fruit Reward' in debug_info and debug_info['Fruit Reward'] > 0)  # Fruit eaten
            )

            if should_log:
                print("\n====== Reward Breakdown ======")
                print(f"Move: {self.frame_iteration}\t\tLength: {self.snake.length}")  # Add frame counter if available
                print(f"Average Fruit Interval: {avg_fruit_interval}")
                print(f"Time Since Last Fruit: {self.frame_iteration - list(self.recent_fruit_times)[-1]}")

                # Sort rewards by absolute magnitude for easier reading
                sorted_rewards = sorted(self.debug_info.items(), key=lambda x: abs(x[1]), reverse=True)

                for key, value in sorted_rewards:
                    if abs(value) > 0.01:  # Only print non-zero rewards
                        print(f"{key}: {value:.2f}")

                print(f"Total Reward: {reward:.2f}")
                print(f"Accessible Area: {self.prev_accessible_area} → {current_accessible_area}")
                print(f"Tail Distance: {self.tail_distance}")
                print(f"Loop Counter: {self.tail_loop_counter}")
                print("=" * 30)

        self.debug_info["Previous Area"] = self.prev_accessible_area if self.prev_accessible_area is not None else current_accessible_area
        self.debug_info["Current Area"] = current_accessible_area

        self.prev_dist_to_fruit = current_dist_to_fruit
        self.prev_accessible_area = current_accessible_area
        self.prev_directional_accessible_areas = self.directional_accessible_areas.copy()
        self.prev_had_escape = has_escape
        self.track_reward_debug_info(self.debug_info)

        return reward

    def get_state(self):
        """Get the current state of the game like the position of the snake, the position of the food, etc."""
        # Update the distance to collision and accessible areas
        self.directional_dangers = self.distance_to_collision(rel_dir=self.snake.direction)
        (self.directional_accessible_areas, self.directional_tail_reachability,
         self.directional_fruit_reachability, self.directional_escape_exists,
         self.directional_escape_times, self.directional_escape_candidates) = self.flood_fill_directional(self.snake.direction)
        self.local_density = self.get_local_density(radius=5)

        max_norm_value = max(20, self.snake.length / 5)
        escape_time_straight = 0 if not self.directional_escape_exists["straight"] else min(1.0, self.directional_escape_times["straight"] / self.snake.length)
        escape_time_left = 0 if not self.directional_escape_exists["left"] else min(1.0, self.directional_escape_times["left"] / self.snake.length)
        escape_time_right = 0 if not self.directional_escape_exists["right"] else min(1.0, self.directional_escape_times["right"] / self.snake.length)

        best_candidate = self.get_nearest_escape(self.directional_escape_times, self.directional_escape_exists, self.directional_escape_candidates)
        if best_candidate is not None:
            escape_direction_vector = self.get_escape_direction(best_candidate)
        else:
            escape_direction_vector = [0, 0, 0, 0]

        # TODO: Logging
        # Track area utilization each time we update accessible areas
        self.track_area_utilization()

        #print(f"Area: {self.directional_accessible_areas}\nTail: {self.directional_tail_reachability}\nFruit: {self.directional_fruit_reachability}\nON FRUIT: {(self.snake.x[0], self.snake.y[0]) == (self.fruit.x, self.fruit.y)}\nTail: {(self.snake.x[-1], self.snake.y[-1])}\nNext Tail: {(self.snake.x[-2], self.snake.y[-2]) if self.snake.length > 1 else None}\n")
        #print(f"Area: {self.directional_accessible_areas}\nTail: {self.directional_tail_reachability}\nFruit: {self.directional_fruit_reachability}\n")

        # Get the reward for the current state
        self.reward = self.get_reward()

        state = [
            # Danger Straight
            self.directional_dangers["straight"],

            # Danger Right
            self.directional_dangers["right"],

            # Danger Left
            self.directional_dangers["left"],

            # Direction
            self.snake.direction == "left",
            self.snake.direction == "right",
            self.snake.direction == "up",
            self.snake.direction == "down",

            # Fruit direction
            self.fruit.x < self.snake.x[0], # Left
            self.fruit.x > self.snake.x[0], # Right
            self.fruit.y < self.snake.y[0], # Up
            self.fruit.y > self.snake.y[0],  # Down

            # Distance from fruit (previous distance is updated to current distance in reward function)
            self.prev_dist_to_fruit / MANHATTAN_NORMALIZE,

            # Size of the normalized snake
            self.snake.length / BOARD_NORMALIZE,

            # Normalized density of where the snake's body is in a radius of 5
            *self.get_local_density(radius=5),

            self.tail_distance / MANHATTAN_NORMALIZE,

            # Tail Reachability
            self.directional_tail_reachability["straight"],
            self.directional_tail_reachability["left"],
            self.directional_tail_reachability["right"],

            # Fruit Reachability
            self.directional_fruit_reachability["straight"],
            self.directional_fruit_reachability["left"],
            self.directional_fruit_reachability["right"],

            # Open spaces
            self.directional_accessible_areas["straight"] / BOARD_NORMALIZE,
            self.directional_accessible_areas["left"] / BOARD_NORMALIZE,
            self.directional_accessible_areas["right"] / BOARD_NORMALIZE,

            self.directional_escape_exists["straight"],
            self.directional_escape_exists["left"],
            self.directional_escape_exists["right"],

            escape_time_straight,
            escape_time_left,
            escape_time_right,

            *escape_direction_vector
        ]

        #print(state)
        #print(len(state))

        return np.array(state, dtype=float)

    def play(self, action=None):
        """Starts running the base of the game and allows for key presses."""
        self.frame_iteration += 1
        self.reward = 0
        self.handle_events()

        # If the AI is playing, it will pass in an action
        if action is not None:
            self.snake.AI_update_direction(action)
        
        # Update the game state
        self.update_game_state()

        # Update the state based on the new game state
        self.state = self.get_state()

        self.clock.tick(SPEED)

        return self.state, self.reward, self.running, self.score

# ----------------------------------------------------------------------------------------------------------------------
    # TODO: Logging
    def generate_log_filename(self):
        """Generate a unique filename for each training session."""
        import time
        timestamp = time.strftime("%Y%m%d-%H%M%S")  # Format: YYYYMMDD-HHMMSS
        return f"training_log_{timestamp}"

    def initialize_stats_logging(self):
        """Initialize the stats logging file with headers."""
        import os

        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs', exist_ok=True)

        # Create the CSV with a simple header
        with open(f'logs/{self.log_file}.csv', 'w') as f:
            f.write('game_number,score,total_reward,moves_per_fruit,timeout,random_death,avg_area_utilization,' +
                    ','.join([f'Frame_{i}' for i in range(1, self.max_reward_buffer_size + 1)]) + '\n')

        # Initialize tracking variables
        self.recent_rewards_buffer = []
        self.max_reward_buffer_size = 50
        self.area_utilization_samples = []
        self.process_took_too_long_flag = False

    def track_area_utilization(self):
        """Track area utilization throughout the game."""
        if hasattr(self, 'directional_accessible_areas') and self.directional_accessible_areas:
            current_area = max(self.directional_accessible_areas.values())
            available_spaces = BOARD_NORMALIZE - self.snake.length
            if available_spaces > 0:
                utilization = current_area / available_spaces
                self.area_utilization_samples.append(utilization)

    def track_reward_debug_info(self, debug_info):
        """Track reward components from the last several frames."""
        # Filter out zero rewards for cleaner output
        filtered_debug_info = {k: v for k, v in debug_info.items() if abs(v) > 0.01}

        # Add current frame's reward info to buffer
        self.recent_rewards_buffer.append(filtered_debug_info)

        # Keep only the most recent frames
        if len(self.recent_rewards_buffer) > self.max_reward_buffer_size:
            self.recent_rewards_buffer.pop(0)

    def format_frame_rewards(self, rewards_dict):
        """Format the rewards dictionary into a readable string."""
        if not rewards_dict:
            return "No rewards"

        # Sort by reward name for consistency
        items = sorted(rewards_dict.items())

        # Format each reward as "key: value"
        formatted_rewards = [f"{key}: {value:.2f}" for key, value in items]

        # Join with newlines for readability
        return "\n".join(formatted_rewards)

    def log_game_stats(self, timeout=False, random_death=False, total_reward=0):
        """Log game statistics to a CSV file after each game, with compact representation of frame rewards."""
        import numpy as np
        import os

        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)

        # Calculate metrics
        moves_per_fruit = self.frame_iteration / self.score if self.score > 0 else self.frame_iteration
        avg_area_utilization = np.mean(self.area_utilization_samples) if self.area_utilization_samples else 0

        # Prepare frame data (most recent first)
        frame_data = list(reversed(self.recent_rewards_buffer))

        # Format each frame's rewards
        formatted_frames = []
        for i in range(self.max_reward_buffer_size):
            if i < len(frame_data):
                formatted_frames.append(self.format_frame_rewards(frame_data[i]))
            else:
                formatted_frames.append("")  # Empty for missing frames

        # Write to CSV
        with open(f'logs/{self.log_file}.csv', 'a') as f:
            # Basic stats
            f.write(f'{self.games_played - 1},{self.score},{total_reward},{moves_per_fruit:.2f},{timeout},{random_death},{avg_area_utilization:.4f}')

            # Add each formatted frame
            for frame in formatted_frames:
                # Escape any commas and quotes in the frame text for CSV
                escaped_frame = f'"{frame.replace("\"", "\"\"")}"'
                f.write(f',{escaped_frame}')

            f.write('\n')

        # Reset tracking variables for next game
        self.area_utilization_samples = []
        self.recent_rewards_buffer = []
        self.process_took_too_long_flag = False


if __name__ == "__main__":
    game = Game()
    while game.running:
        #game.handle_events()
        #if PAUSE:
        #    game.clock.tick(1)  # Limit frame rate while paused
        #    continue
        game.play()

    pygame.quit()