import pygame
import sys
import math

# Initialize Pygame
pygame.init()

# Set up display
width, height = 400, 200
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Animated Logo')

# Set up colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
blue = (0, 0, 255)

# Set up fonts
font_size = 40
font_A = pygame.font.SysFont(None, font_size)
font_N = pygame.font.SysFont(None, font_size)
font_P = pygame.font.SysFont(None, font_size)

# Function to draw the letters
def draw_letters(angle, scale):
    letter_A = font_A.render('A', True, black)
    letter_N = font_N.render('N', True, red)
    letter_P = font_P.render('P', True, blue)

    rotated_A = pygame.transform.rotate(letter_A, angle)
    rotated_N = pygame.transform.rotate(letter_N, angle)
    rotated_P = pygame.transform.rotate(letter_P, angle)

    scaled_A = pygame.transform.scale(rotated_A, (int(rotated_A.get_width() * scale), int(rotated_A.get_height() * scale)))
    scaled_N = pygame.transform.scale(rotated_N, (int(rotated_N.get_width() * scale), int(rotated_N.get_height() * scale)))
    scaled_P = pygame.transform.scale(rotated_P, (int(rotated_P.get_width() * scale), int(rotated_P.get_height() * scale)))

    screen.blit(scaled_A, (50, 50))
    screen.blit(scaled_N, (150, 50))
    screen.blit(scaled_P, (250, 50))

# Main loop
clock = pygame.time.Clock()

angle = 0
scale = 1.0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill(white)

    draw_letters(angle, scale)

    angle += 2
    scale = 1.5 + 1.5 * math.sin(math.radians(angle))  # Oscillating scale for a bouncing effect

    pygame.display.flip()
    clock.tick(60)
