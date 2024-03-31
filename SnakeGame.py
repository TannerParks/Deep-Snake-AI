import math
from collections import namedtuple
import pygame
import random
from math import dist
import numpy as np
from collections import deque

# Colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

BLOCK_SIZE = 20  # Size of a block
SPEED = 20  # Speed of the game
SPEED_LIST = [5, 20, 400, 800]  # List of speeds for the game

WIDTH = 600 # Width and height of the window
HEIGHT = 600
START_X = 300   # Starting x and y coordinates for the snake
START_Y = 300

Point = namedtuple('Point', ['x', 'y'])

class Fruit:
    def __init__(self, window, snake):
        self.window = window
        self.snake = snake  # Reference to the snake object, ensures the Fruit classs has the most up-to-date snake position
        self.x = None
        self.y = None
        self.move_fruit()

    def draw_fruit(self):
        """Draws the fruit on the board."""
        pygame.draw.rect(self.window, RED, [self.x, self.y, BLOCK_SIZE, BLOCK_SIZE])
        #print(self.snake.x, self.snake.y)

    def move_fruit(self):
        """Moves the fruit to a new location. If the location is occupied, move to a new location. This is done by generating a 
        list of all possible points and removing the ones that are occupied by the snake. Then, a random point is chosen from the 
        remaining points. We use this in favor of a while loop to prevent the game from freezing if the snake is too long."""
        all_points = [(x, y) for x in range(0, WIDTH, BLOCK_SIZE) for y in range(0, HEIGHT, BLOCK_SIZE)]
        snake_positions = set(zip(self.snake.x, self.snake.y))
        free_points = [pt for pt in all_points if pt not in snake_positions]

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
        self.x.append(0)  # Appends new set of (x, y) coords to corresponding lists
        self.y.append(0)

    def draw_snake(self):
        """Draws the snake."""
        self.window.fill(BLACK)
        for i in range(self.length):
            pygame.draw.rect(self.window, BLACK, [self.x[i], self.y[i], BLOCK_SIZE, BLOCK_SIZE])  # Outlines snake segments
            pygame.draw.rect(self.window, GREEN, [self.x[i] + 2, self.y[i] + 2, 16, 16])

    # FIXME: Can remove this function and just call move_snake() or AI_update_direction directly
    def slither(self, action):
        self.AI_update_direction(action)
        self.move_snake()
    
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
        self.reset()   

    def reset(self):
        """Resets the game to start a new game or replay."""
        self.snake = Snake(self.window, 1)
        self.fruit = Fruit(self.window, self.snake)
        self.score = 0
        self.update_game_state()
        self.frame_iteration = 0
        self.running = True
        self.reward = 0
        self.prev_dist_to_fruit = self.check_distance_to_fruit()
        self.directional_dangers = self.distance_to_collision(self.snake.direction)
        self.state = self.get_state()

        #print(self.state)

        return self.state

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
                #self.force_quit = True
            elif event.type == pygame.KEYDOWN:
                # If the game is being played by a human, handle key presses
                if self.human:
                    self.handle_keydown(event.key)

    def handle_keydown(self, key):
        """Handles key presses."""
        global SPEED 
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
                self.running = False
            case pygame.K_SPACE:            ####### TODO: For debugging purposes #######
                if SPEED == SPEED_LIST[1]:
                    SPEED = SPEED_LIST[0]
                else:
                    SPEED = SPEED_LIST[1]
                print(f"Speed: {SPEED}")
            case pygame.K_f:
                if SPEED == SPEED_LIST[0] or SPEED == SPEED_LIST[1] or SPEED == SPEED_LIST[2]:
                    SPEED = SPEED_LIST[3]
                else:
                    SPEED = SPEED_LIST[2]
                print(f"Speed: {SPEED}")
            case pygame.K_e:
                self.process_ate_fruit()
    
    def update_game_state(self):
        """Updates the game state by moving the snake, drawing the snake and fruit, and displaying the score."""
        #self.snake.slither(action)
        self.snake.move_snake()
        self.snake.draw_snake()
        self.fruit.draw_fruit()
        self.display_score()
        pygame.display.update()
    
    def snake_ate_fruit(self):
        """Checks if the snake has eaten a fruit."""
        return self.snake.x[0] == self.fruit.x and self.snake.y[0] == self.fruit.y
    
    def process_ate_fruit(self):
        """Processes the snake eating a fruit by increasing the score, length, and moving the fruit."""
        self.score += 1
        self.prev_dist_to_fruit = math.inf
        self.snake.grow()  # Add a block to the snake
        self.fruit.move_fruit()
    
    def check_distance_to_fruit(self):
        """Calculates the distance between the snake and the fruit."""
        snake_head = (self.snake.x[0], self.snake.y[0])
        fruit = (self.fruit.x, self.fruit.y)
        dist_to_fruit = round(math.dist(snake_head, fruit))

        max_distance = math.sqrt(WIDTH**2 + HEIGHT**2)
        normalized_distance = dist_to_fruit / max_distance

        return normalized_distance
        
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
    
    def flood_fill(self, point=None):
        """Flood fill algorithm to check if the snake can reach the fruit."""
        if self.collision():    # Return 0 if the snake is in a collision state
            return 0
        
        if point is None:
            point = Point(self.snake.x[0], self.snake.y[0]) # Default to the head of the snake

        grid = [[0 for _ in range(WIDTH // BLOCK_SIZE)] for _ in range(HEIGHT // BLOCK_SIZE)]
        queue = deque([point])
        accessible_points = 0
        visited = set()

        # Mark the snake's body on the grid
        for i in range(self.snake.length):
            x = self.snake.x[i] // BLOCK_SIZE
            y = self.snake.y[i] // BLOCK_SIZE
            grid[y][x] = 1

        while queue:
            pt = queue.popleft()
            x = pt.x // BLOCK_SIZE
            y = pt.y // BLOCK_SIZE

            # Check if the point is out of bounds or if it's already been visited
            if pt in visited or x < 0 or x >= WIDTH // BLOCK_SIZE or y < 0 or y >= HEIGHT // BLOCK_SIZE or grid[y][x] == 1:
                continue
            
            visited.add(pt)
            accessible_points += 1

            # Add the points around the current point to the queue
            queue.append(Point(pt.x + BLOCK_SIZE, pt.y))
            queue.append(Point(pt.x - BLOCK_SIZE, pt.y))
            queue.append(Point(pt.x, pt.y + BLOCK_SIZE))
            queue.append(Point(pt.x, pt.y - BLOCK_SIZE))
        
        # Normalize the number of accessible points
        normalized_accessible_points = accessible_points / (WIDTH * HEIGHT / (BLOCK_SIZE**2))
            
        return normalized_accessible_points

    def took_too_long(self):
        """Encourages the AI to find a solution faster by penalizing it for taking too long (time limit is based on the snake's length)"""
        return self.frame_iteration > 100 * self.snake.length
    
    def process_took_too_long(self):
        """Processes the game taking too long."""
        self.running = False

    def distance_to_collision(self, direction):
        """Calculates the distance between the snake and possible collisions in each direction."""
        match direction:
            case "up":
                directions = {'straight': (0, -BLOCK_SIZE), 'left': (-BLOCK_SIZE, 0), 'right': (BLOCK_SIZE, 0)}
            case "down":
                directions = {'straight': (0, BLOCK_SIZE), 'left': (BLOCK_SIZE, 0), 'right': (-BLOCK_SIZE, 0)}
            case "left":
                directions = {'straight': (-BLOCK_SIZE, 0), 'left': (0, BLOCK_SIZE), 'right': (0, -BLOCK_SIZE)}
            case "right":
                directions = {'straight': (BLOCK_SIZE, 0), 'left': (0, -BLOCK_SIZE), 'right': (0, BLOCK_SIZE)}
        
        dangers = {'straight': 0, 'left': 0, 'right': 0}

        for direction, (dx, dy) in directions.items():
            (x, y) = (self.snake.x[0], self.snake.y[0])
            distance = 0
            while self.collision(Point(x, y)) == 0:
                x += dx
                y += dy
                distance += 1
            
            # Maximum distance calculations need to consider the game board's dimensions
            if dx != 0:  # Moving horizontally
                max_distance = WIDTH / BLOCK_SIZE
            else:  # Moving vertically
                max_distance = HEIGHT / BLOCK_SIZE

            dangers[direction] = distance/max_distance
        
        return dangers
        
    def get_state(self):
        """Get the current state of the game like the position of the snake, the position of the food, etc."""
        # Get points around the snake's head
        pt_up = Point(self.snake.x[0], self.snake.y[0] - BLOCK_SIZE)
        pt_down = Point(self.snake.x[0], self.snake.y[0] + BLOCK_SIZE)
        pt_left = Point(self.snake.x[0] - BLOCK_SIZE, self.snake.y[0])
        pt_right = Point(self.snake.x[0] + BLOCK_SIZE, self.snake.y[0])

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

            # Distance from fruit
            self.prev_dist_to_fruit,

            # Open spaces
            self.flood_fill(pt_right),
            self.flood_fill(pt_left),
            self.flood_fill(pt_up),
            self.flood_fill(pt_down)
        ]

        return np.array(state, dtype=float)

    def get_reward(self):
        """Get the reward for the current state."""
        reward = 0
        # Check the distance to the fruit
        if self.snake_ate_fruit():
            reward += 50
            self.process_ate_fruit()
        
        # Reward the AI for getting closer to the fruit
        current_dist_to_fruit = self.check_distance_to_fruit()
        if current_dist_to_fruit > self.prev_dist_to_fruit:
            reward -= 1
        else:
            reward += 1
        self.prev_dist_to_fruit = current_dist_to_fruit

        # Check for collisions and if the snake ate a fruit
        if self.collision():
            reward -= 50
            self.process_collision()

        # End game after a certain amount of time to encourage faster solutions
        if self.took_too_long():
            reward -= 10
            self.process_took_too_long()

        return reward      

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

        self.directional_dangers = self.distance_to_collision(direction=self.snake.direction)

        # Get the reward for the current state
        self.reward = self.get_reward()

        self.clock.tick(SPEED)

        # Update the state based on the new game state
        self.state = self.get_state()

        return self.state, self.reward, self.running, self.score


if __name__ == "__main__":
    game = Game()
    while game.running:
        game.play()
        #game.reset()

    pygame.quit()