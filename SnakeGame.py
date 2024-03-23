import math
from collections import namedtuple
import pygame
import random
from math import dist
import numpy as np

# Colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

BLOCK_SIZE = 20  # Size of a block
SPEED = 20  # Speed of the game

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
        self.init_game()

    def init_game(self):
        """Initializes the game's main components."""
        pygame.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.reset_game_state()    

    def reset_game_state(self):
        """Resets the game state to start a new game or replay."""
        self.snake = Snake(self.window, 1)
        self.fruit = Fruit(self.window, self.snake)
        self.score = 0
        self.frame_iteration = 0
        self.running = True
        self.force_quit = False
        self.reward = 0

    def distance(self): # TODO: Make this distance(snake, object) so the AI can check how close it is to multiple things
        fruit = (self.fruit.x, self.fruit.y)
        snake = (self.snake.x[0], self.snake.y[0])
        dis = round(math.dist(snake, fruit))
        return dis

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
            case pygame.K_UP:
                self.snake.move_up()
            case pygame.K_DOWN:
                self.snake.move_down()
            case pygame.K_LEFT:
                self.snake.move_left()
            case pygame.K_RIGHT:
                self.snake.move_right()
            case pygame.K_ESCAPE:
                self.running = False
                #self.force_quit = True
            case pygame.K_SPACE:    # TODO: Just for debugging purposes
                if SPEED == 20:
                    SPEED = 5
                else:
                    SPEED = 20
    
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
        self.reward = 50
        self.score += 1
        self.snake.grow()  # Add a block to the snake
        self.fruit.move_fruit()
        
    def collision(self, point=None):
        """Checks if a point has collided with anything by making a set of the snake's body and checking if the point is in it or 
        checking if the point is out of bounds"""
        if point is None:
            point = Point(self.snake.x[0], self.snake.y[0]) # Default to the head of the snake
        snake_body = set(zip(self.snake.x[1:], self.snake.y[1:]))

        if (point.x > WIDTH - BLOCK_SIZE) or (point.x < 0) or (point.y < 0) or (point.y > HEIGHT - BLOCK_SIZE):
            print("Hit wall")
            return 1
        if point in snake_body:
            print("Hit snake")
            return 1
        return 0
    
    def process_collision(self):
        """Processes the snake colliding with something."""
        self.reward = -50
        self.running = False

    def took_too_long(self):
        """Made to encourage the AI towards a faster solution."""
        if self.frame_iteration > 100 * self.snake.length:
            print("Took too long")
            self.running = False

    def play(self, action=None):
        """Starts running the base of the game and allows for key presses."""
        self.frame_iteration += 1
        self.handle_events()

        # If the AI is playing, it will pass in an action
        if action is not None:
            self.snake.AI_update_direction(action)

        self.update_game_state()

        # Check for collisions and if the snake ate a fruit
        if self.collision():
            self.process_collision()
        if self.snake_ate_fruit():
            self.process_ate_fruit()
            
        # End game after a certain amount of time to encourage faster solutions
        #self.took_too_long()   # FIXME: AI only

        self.clock.tick(SPEED)

        return self.reward, self.running, self.score


if __name__ == "__main__":
    game = Game()
    while game.running:
        game.play()

    pygame.quit()