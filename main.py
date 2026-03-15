# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:03:24 2026

@author: q88546sh

Thus the second iteration of our Boids Program, using a quiver plot to represent our Boids

Documentation of modules and links to sources used heavily to be listed below:
    
https://numpy.org/doc/

To get animations to render properly navigate in spyder to
tools -> preferences -> graphics , then set the back end to Qt5.
Restart the kernel and the animation will open in another window upon running the program.

https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html
http://www.kfish.org/boids/pseudocode.html
https://github-pages.ucl.ac.uk/rsd-engineeringcourse/ch02data/084Boids.html
"""

# Packages

import numpy as np
from matplotlib import pyplot as plt, animation # to visualise we'll use FuncAnimation which generates an initial plot then updates its data for each frame
from scipy.spatial import KDTree


# Parameters

parameters2D = {
    'size': 50,
    'seed': 17,
    'steps': 200,
    'ndim': 2,
    'population': 200,
    'inner_radius': 3,
    'outer_radius': 10,
    'border_distance': 10,
    'cohesion_strength': 0.005,
    'seperation_strength': 0.1,
    'alignment_strength': 0.3,
    'border_strength': 0.5
}



# Initialise the System

rng = np.random.default_rng(seed=17) # set seed to avoid issues with reproducibility

def initialise_boid_states(rng, num_boids = 200, min_position = (0,0), max_position = (2000,2000), min_velocity = (-10,-10), max_velocity = (10,10)):
    positions = rng.uniform(low = min_position, high = max_position, size = (num_boids,2))
    velocities = rng.uniform(low = min_velocity, high = max_velocity, size = (num_boids,2))
    return positions, velocities

positions, velocities = initialise_boid_states(rng)

def get_unit_vector(vector):
    # This can absolutely be optimised by using an alpha-max beta-min algorithm
    return vector / (vector**2).sum(-1)[...,np.newaxis]**0.5
    
def plot_boids(positions, velocities, figsize= (8,8), xlim = (0,2000), ylim = (0,2000)):
    fig, ax = plt.subplots(figsize = (8,8))
    ax.set(xlim=xlim,ylim=ylim, xlabel = "x coordinate",ylabel = "y coordinate")
    velocity_unit_vectors =get_unit_vector(velocities)
    arrows = ax.quiver(
        positions[:,0], # horizontal coordinates for arrow origins
        positions[:,1], # vertical coordinates for arrow origins
        velocity_unit_vectors[:,0], # horizontal component of arrow vector
        velocity_unit_vectors[:,1], # vertical component of arrow vector
        scale = 50, # arrow size, can fiddle around with visual fixes later
        color = 'k', # again TBD
        angles = 'xy', # specifies arrow directions
        pivot = 'middle' # sets the middle of the arrows at our specified origin coords
        )
    return fig, ax, arrows

#fig, ax, arrows = plot_boids(positions, velocities)

# Define Model Dynamics

def simulate_timestep(positions,velocities,forces,timestep):
    positions += timestep*velocities
    velocities += timestep* sum(force(positions,velocities) for force in forces)

def cohesion_force(positions, velocities, cohesion_strength = 0.001):
    # Should later be updated to include only boids within the visual range, not globally
    return cohesion_strength * (positions.mean(axis=0)[np.newaxis]-positions)

def cohesion_force2(positions, velocities, cohesion_strength=0.025, visual_range=100):
    tree = KDTree(positions)
    forces = np.zeros_like(positions)
    for i, pos in enumerate(positions):
        idxs = tree.query_ball_point(pos, visual_range)
        idxs = [j for j in idxs if j != i]  # remove self
        if len(idxs) == 0:
            continue
        avg_pos = positions[idxs].mean(axis=0)
        forces[i] = cohesion_strength * (avg_pos - pos)
    return forces      

def separation_force(positions, velocities, separation_strength=0.05, protected_range = 30 ):
    displacements = positions[np.newaxis]- positions[:,np.newaxis] # This method is horrendously computationally expensive, replace with a KDTree at first convenience
    are_close = (displacements**2).sum(-1)**0.5 <= protected_range # condition for two boids to be within separation distance
    return separation_strength* np.where(are_close[...,None], displacements,0).sum(0)

def alignment_force(positions,velocities, alignment_strength = 0.05, visual_range = 100):
    displacements = positions[np.newaxis] - positions[:, np.newaxis]
    velocity_differences = velocities[np.newaxis] - velocities[:, np.newaxis]
    are_close = (displacements**2).sum(-1)**0.5 <= visual_range
    return -alignment_strength* np.where(are_close[...,None], velocity_differences, 0).mean(0)

def border_avoidance(positions, velocities, border_strength = 0.5, border_distance = 100, ndim = 2):
    bforce = np.zeros_like(positions)
    for i in range(ndim):
        if positions[:,i] < border_distance:
            bforce += border_strength
        elif positions[:,i] > 2000 - border_distance: # I know this is super inelegant, bear with me
            bforce -= border_strength
    return bforce

def border_avoidance2(positions, velocities, border_strength = 0.5, border_distance = 100, maximum = 2000, ndim = 2):
    combined_forces = np.zeros_like(velocities)
    for i in range(ndim):
        coordinate = positions[:,i]
        too_small = coordinate <= border_distance # Conditions for detecting when the boid is within the margin of 250 px
        too_large = coordinate >= maximum - border_distance
        dist_small = border_distance - coordinate
        dist_large = coordinate - (maximum - border_distance)
        # Apply the forces
        small_forces = border_strength * np.where(too_small, dist_small, 0)  # Act on boids with a coordinate smaller than the 'lower' margin
        large_forces = -border_strength * np.where(too_large,dist_large, 0) # Act on boids with a coordinate larger than the 'higher' margin
        combined_forces[:,i] = small_forces + large_forces
    return combined_forces

# Generate an Animation

def animate_flock(positions,velocities, forces = (),timestep = 1, num_steps = 100):
    fig, ax, arrows = plot_boids(positions,velocities) # Import initialised matplotlib objects
    def update_frame(frame_index):
        simulate_timestep(positions,velocities,forces,timestep)
        velocity_unit_vectors = get_unit_vector(velocities)
        arrows.set_offsets(positions)
        arrows.set_UVC(velocity_unit_vectors[:,0],velocity_unit_vectors[:,1])
        return [arrows]
    # Close the matplotlib figure object to avoid displaying the static figure
    #plt.close(fig)
    return animation.FuncAnimation(fig, update_frame, num_steps, interval = 50)

ani = animate_flock(positions, velocities, [cohesion_force2, separation_force, alignment_force, border_avoidance2])
plt.show()

