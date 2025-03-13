import pgzrun
import random

WIDTH = 800
HEIGHT = 600

player = Actor("player", (WIDTH / 2, HEIGHT - 70))
obstacles = []

# Adjust the obstacle speed and spawn likelihood by changing these values
OBSTACLE_SPEED = 3

# Increase this value for fewer obstacles
SPAWN_LIKELIHOOD = 60  


def draw():
   screen.clear()
   player.draw()
   for obstacle in obstacles:
       obstacle.draw()

def update():
   global obstacles

   # Move the player with arrow keys
   if keyboard.left:
       player.x -= 5
   elif keyboard.right:
       player.x += 5

   # Spawn new obstacles
   if random.randint(1, SPAWN_LIKELIHOOD) == 1:
       obstacles.append(Actor("obstacle", (random.randint(50, WIDTH - 50), 0)))

   # Move obstacles and check for collisions
   for obstacle in obstacles:
       obstacle.y += OBSTACLE_SPEED  # Adjusted obstacle speed
       if obstacle.colliderect(player):
           print("Game Over!")
           pgzrun.quit()

   # Remove off-screen obstacles
   obstacles = [obstacle for obstacle in obstacles if obstacle.y < HEIGHT]

pgzrun.go()
