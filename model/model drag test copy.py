import pygame
from pygame.locals import *
import numpy as np
import OpenGL.GL as gl
from OpenGL.GL import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random

score = 0
cube_position = np.array([random.uniform(-5, 5), random.uniform(2, -4), 0.0])
cube_size = 0.5

# Initialization
pygame.init()
screen = pygame.display.set_mode((1200, 800), DOUBLEBUF | OPENGL)
gluPerspective(45, (1200 / 800), 0.1, 50.0)
glTranslatef(0.0, 0.0, -10)
glEnable(GL_DEPTH_TEST)

# Simulation parameters
mass = 100.0
time_step = 0.016
gravitational_force = np.array([0.0, -9.8, 0.0])  # Gravity
elasticity = 0.8
damping = 0.2

# Ball properties
ball_position = np.array([0.0, 0.0, 0.0])
ball_previous_position = ball_position - np.array([0.0, 0.0, 0.0])
ball_velocity = np.array([0.0, 0.0, 0.0])

# Paddle properties
paddle_position = np.array([0.0, 4.0, 0.0])
paddle_size = [3.0, 0.5, 1.0]

# Rope properties
num_particles = 50  # Number of particles in the rope
rope_length = 4  # Total length of the rope
segment_length = rope_length / (num_particles - 1)  # Length of each segment
particles = []  # Particle positions
prev_particles = []  # Previous positions for Verlet integration

# Initialize particles
start_position = np.array([0.0, 0.0, 0.0])
for i in range(num_particles):
    position = start_position + np.array([0.0, -i * segment_length, 0.0])
    particles.append(position)
    prev_particles.append(position - np.array([0.0, 0.01, 0.0]))  # Slight offset for Verlet integration

def reset_game():
    global score, ball_position, ball_previous_position, ball_velocity, cube_position
    score = 0
    ball_position = np.array([0.0, 0.0, 0.0])
    ball_previous_position = ball_position - np.array([0.0, 0.0, 0.0])
    ball_velocity = np.array([0.0, 0.0, 0.0])
    cube_position = np.array([random.uniform(-5, 5), random.uniform(2, -4), 0.0])

def draw_text(position, text, font_size=18):
    font = pygame.font.SysFont("Arial", font_size)
    text_surface = font.render(text, True, (255, 255, 255), (0, 0, 0))
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    glWindowPos2d(position[0], position[1])
    glDrawPixels(text_surface.get_width(), text_surface.get_height(), gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, text_data)


# Drawing functions
def draw_sphere(position, radius, slices=20, stacks=20):
    glPushMatrix()
    glTranslatef(*position)
    for i in range(stacks):
        lat0 = np.pi * (-0.5 + float(i) / stacks)
        z0 = np.sin(lat0)
        zr0 = np.cos(lat0)

        lat1 = np.pi * (-0.5 + float(i + 1) / stacks)
        z1 = np.sin(lat1)
        zr1 = np.cos(lat1)

        glBegin(GL_QUADS)
        for j in range(slices + 1):
            lng = 2 * np.pi * float(j) / slices
            x = np.cos(lng)
            y = np.sin(lng)

            glVertex3f(x * zr0 * radius, y * zr0 * radius, z0 * radius)
            glVertex3f(x * zr1 * radius, y * zr1 * radius, z1 * radius)
        glEnd()
    glPopMatrix()

def draw_cube(size):
    half_size = size / 2.0
    vertices = [
        [-half_size, -half_size, -half_size],  # 0
        [half_size, -half_size, -half_size],   # 1
        [half_size, half_size, -half_size],    # 2
        [-half_size, half_size, -half_size],   # 3
        [-half_size, -half_size, half_size],   # 4
        [half_size, -half_size, half_size],    # 5
        [half_size, half_size, half_size],     # 6
        [-half_size, half_size, half_size],    # 7
    ]
    faces = [
        (0, 1, 2, 3),  # Back
        (4, 5, 6, 7),  # Front
        (0, 1, 5, 4),  # Bottom
        (3, 2, 6, 7),  # Top
        (0, 3, 7, 4),  # Left
        (1, 2, 6, 5)   # Right
    ]
    glBegin(GL_QUADS)
    for face in faces:
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()

def draw_paddle(paddle_position, deformation):
    glPushMatrix()
    glTranslatef(
        paddle_position[0],
        paddle_position[1] - deformation,
        paddle_position[2]
    )
    glScalef(
        3, 0.2 + deformation, 1
    )  # Scale the cube to the shape of the paddle with deformation
    draw_cube(1)
    glPopMatrix()

def draw_rope(particles):
    glBegin(GL_LINES)
    for i in range(len(particles) - 1):
        glVertex3f(*particles[i])
        glVertex3f(*particles[i + 1])
    glEnd()

# Physics functions
def verlet_integration(current, previous, acceleration, dt):
    next_position = 2 * current - previous + acceleration * (dt ** 2)
    return next_position

def enforce_constraints():
    # Pierwsza cząsteczka liny podąża za paddle
    
    

    # Enforce the distance constraint between adjacent particles
    for _ in range(5):  # Iterate multiple times for stability
        for i in range(1, len(particles)):
            p1 = particles[i - 1]
            p2 = particles[i]
            direction = p2 - p1
            distance = np.linalg.norm(direction)
            if distance > 0:
                correction = (distance - segment_length) * (direction / distance) * 0.5
                particles[i - 1] += correction
                particles[i] -= correction

    # Connect the last particle to the ball
    particles[0] = np.copy(paddle_position)
    particles[-1] = ball_position

# Main loop
running = True
clock = pygame.time.Clock()
is_dragging = False
is_dragging = False
pull_vector = np.array([0.0, 0.0, 0.0])  # Naciąg liny
elasticity_constant = 0.001  # Siła sprężystości liny

while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            world_mouse_pos = np.array([
                (mouse_x / 1200) * 2 - 1,
                -(mouse_y / 800) * 2 + 1,
                0.0
            ]) * 5  # Adjust the scaling factor to match the world coordinates
            if np.linalg.norm(world_mouse_pos[:2] - ball_position[:2]) < 0.5:
                is_dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if is_dragging:
                ball_velocity = -elasticity_constant * pull_vector
                ball_previous_position = ball_position - ball_velocity * time_step
                is_dragging = False

    keys = pygame.key.get_pressed()
    if keys[K_UP]:
        if paddle_position[1] < 8.0:
            paddle_position[1] += 0.1
    if keys[K_DOWN]:
        if paddle_position[1] > 3.0:
            paddle_position[1] -= 0.1
    if keys[K_LEFT]:
        if paddle_position[0] > -5.0:
            paddle_position[0] -= 0.1
    if keys[K_RIGHT]:
        if paddle_position[0] < 5.0:
            paddle_position[0] += 0.1
    if keys[K_a]:  # Ruch piłki w lewo
        ball_position[0] -= 1
        ball_position[1] -= 0.3
    if keys[K_d]:  # Ruch piłki w prawo
        ball_position[0] += 1
        ball_position[1] -= 0.3
    if keys[K_w]:  #skracanie liny
        if rope_length > 1:
            rope_length -= 1
    if keys[K_s]:  #wydłużanie liny
        ball_position[1] -= 0.3
    if keys[K_ESCAPE]:
        running = False

    # Mouse dragging logic
    if is_dragging:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        ball_position[:2] = np.array([
            (mouse_x / 1200) * 2 - 1,
            -(mouse_y / 800) * 2 + 1
        ]) * 10
    if not is_dragging:
        # Oblicz przyspieszenie (tylko grawitacja działa na piłkę w tym przykładzie)
        acceleration = gravitational_force / mass

        # Zapisujemy aktualną pozycję jako poprzednią PRZED jej zmianą
        temp_position = np.copy(ball_position)

        # Zastosuj Verlet Integration
        ball_position = 2 * ball_position - ball_previous_position + acceleration * (time_step ** 2)

        # Zapisz poprzednią pozycję
        ball_previous_position = temp_position

        # Zastosuj tłumienie (damping)
        ball_velocity = (ball_position - ball_previous_position) / time_step
        ball_velocity *= damping

        # Zapewnij, że piłka pozostaje w zakresie liny
        direction_to_rope = ball_position - particles[-2]
        distance_to_rope = np.linalg.norm(direction_to_rope)
        if distance_to_rope > segment_length:
            correction = (distance_to_rope - segment_length) * (direction_to_rope / distance_to_rope)
            ball_position -= correction

        # Sprawdź kolizję piłki z sześcianem
        if np.linalg.norm(ball_position - cube_position) < 0.5 + (cube_size / 2):
            score += 1
            cube_position = np.array([random.uniform(-5, 5), random.uniform(2, -4), 0.0])

        # Sprawdź, czy piłka opuściła ekran
        if ball_position[0] < -6 or ball_position[0] > 6:
            reset_game()

        # Enforce rope constraint
        enforce_constraints()

       

    # Rendering
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_sphere(ball_position, 0.5)
    draw_paddle(paddle_position, deformation=0.0)
    draw_rope(particles)

    # Narysuj sześcian
    glPushMatrix()
    glTranslatef(*cube_position)
    draw_cube(cube_size)
    glPopMatrix()

    # Wyświetl wynik
    draw_text((10, 750), f"Score: {score}")

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
