# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 17:31:51 2026

@author: q88546sh


Thus the Third and (Hopefully) final iteration of our Boids Model.

http://www.kfish.org/boids/pseudocode.html
https://numpy.org/doc/
https://agentpy.readthedocs.io/en/latest/guide.html

please work

"""

# Packages

import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def norm(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

class Boid(ap.Agent):
    # this all looks scary but TLDR we're initialising a type of object with particular rules that govern how agentpy will update it
    def setup(self):
        self.velocity = norm(self.model.nprandom.random(self.p.ndim) - 0.5) # generate a vector of length self.p.ndim (2 or 3D) with entries in [0,1),
        # then subtract 0.5 -  shifting the range to [-0.5,0.5) ;  the vector points in a random direction about the origin - then normalise
    def setup_pos(self,space):
        self.space = space # reminds agentpy to attach the agent to the model's 'space' object
        self.neighbors = space.neighbors # assigns the method space.neighbors to the object
        self.pos = space.positions[self]
    def update_velocity(self):
        pos = self.pos #  creates a local reference to the agent's pos vector
        ndim = self.p.ndim # pulls ndim into another local variable (these local references are optimisations; they avoid calling self.pos when looping over many timesteps)
        # Cohesion
        nbs = self.neighbors(self,distance=self.p.visual_range)
        nbs_len = len(nbs)
        nbs_pos_array = np.array(nbs.pos)
        nbs_vec_array = np.array(nbs.velocity)
        if nbs_len >0:
            centre = np.sum(nbs_pos_array,0)/ nbs_len
            v1 = (centre-pos)*self.p.cohesion_strength
        else:
            v1 = np.zeros(ndim)
        # Separation
        v2 = np.zeros(ndim)
        for nb in self.neighbors(self,distance=self.p.protected_radius):
            v2 -= nb.pos - pos
        v2 *= self.p.separation_strength
        # Alignment
        if nbs_len > 0:
            average_v = np.sum(nbs_vec_array, 0) / nbs_len
            v3 = (average_v - self.velocity) * self.p.alignment_strength
        else:
            v3 = np.zeros(ndim)
        # Borders - I'm wondering about changing this to some kind of reward function
        v4 = np.zeros(ndim)
        d = self.p.border_distance # again, unpack parameters to local variables
        s = self.p.border_strength
        for i in range(ndim):
            if pos[i] < d:
                v4[i] +=s
            elif pos[i] > self.space.shape[i] - d:
                v4[i] -= s
        # Update velocity
        self.velocity += v1 + v2 + v3 + v4
        self.velocity = norm(self.velocity)
    def update_position(self):
        self.space.move_by(self,self.velocity)
        
class BoidsModel(ap.Model):
    def setup(self):
        # initialise the agents and the model's other key objects
        self.space  = ap.Space(self,shape = [self.p.size]*self.p.ndim)
        self.agents = ap.AgentList(self,self.p.population, Boid)
        self.space.add_agents(self.agents, random = True)
        self.agents.setup_pos(self.space)
    def step(self):
        # characterises what happens in each somulation step
        self.agents.update_velocity()
        self.agents.update_position()
        
# Animation / Visualisation        
def animation_plot_single(m,ax):
    # takes args m := model, ax := parameter dictionary
    pop = m.p.population
    ndim = m.p.ndim #store ndim as a local variable
    ax.clear()
    ax.set_title(f"plenty ({pop}) of fish in the sea {ndim}D, t={m.t}")
    pos = m.space.positions.values()
    pos = np.array(list(m.space.positions.values())).T # transform
    ax.scatter(*pos, s= 5, c = 'black')
    ax.set_xlim(0,m.p.size)
    ax.set_ylim(0,m.p.size)
    if ndim == 3:
        ax.set_zlim(0,m.p.size)
    #ax.set_axis_off()

def animate_model(model, parameters):
    m = model(parameters)
    m.run(1)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d' if parameters['ndim']==3 else None)

    def update(frame):
        m.step()
        animation_plot_single(m, ax)

    anim = FuncAnimation(fig, update, frames=parameters['steps'], interval=10)
    plt.show()
    return anim

def animate_model2(model, parameters):
    # a second method. hopefully better fps
    m = model(parameters)
    m.run(1)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d' if parameters['ndim']==3 else None)

    # initialise scatter
    pos = np.array(list(m.space.positions.values())).T
    scat = ax.scatter(*pos, s=5, c='black')
    # generate axes
    ax.set_xlim(0, m.p.size)
    ax.set_ylim(0, m.p.size)
    if parameters['ndim'] == 3:
        ax.set_zlim(0, m.p.size)

    def update(frame):
        # by updating the axes and not calling axclear each frame it should in theory allow us to get more frames
        m.step()
        pos = np.array(list(m.space.positions.values())).T
        scat._offsets3d = (pos[0], pos[1], pos[2]) if parameters['ndim']==3 else pos
        ax.set_title(f"plenty ({m.p.population}) of fish in the sea {parameters['ndim']}D")

    anim = FuncAnimation(fig, update, frames=parameters['steps'], interval=10)
    plt.show()
    return anim

def animate_model_quiver(model, parameters):
    m = model(parameters)
    m.run(1)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    # --- initial positions and velocities ---
    pos = np.array(list(m.space.positions.values()))
    vel = np.array([a.velocity for a in m.agents])

    # --- create initial quiver object ---
    Q = ax.quiver(
        pos[:,0], pos[:,1], pos[:,2],
        vel[:,0], vel[:,1], vel[:,2],
        length=1.0, normalize=True, color='black'
    )

    ax.set_xlim(0, m.p.size)
    ax.set_ylim(0, m.p.size)
    ax.set_zlim(0, m.p.size)

    def update(frame):
        m.step()

        pos = np.array(list(m.space.positions.values()))
        vel = np.array([a.velocity for a in m.agents])

        # Update quiver data (3D version)
        Q.set_segments([])  # clear old segments
        Q._segments3d = [
            [[x, y, z], [x+u, y+v, z+w]]
            for (x,y,z), (u,v,w) in zip(pos, vel)
        ]

        ax.set_title(f"Boids (quiver) {m.p.ndim}D")

    anim = FuncAnimation(fig, update, frames=parameters['steps'], interval=10)
    plt.show()
    return anim


# Parameters


parameters3D = {
    'size': 100,
    'seed': 123,
    'steps': 500,
    'ndim': 3,
    'population': 500,
    'protected_radius': 3,
    'visual_range': 10,
    'border_distance': 10,
    'cohesion_strength': 0.005,
    'separation_strength': 0.1,
    'alignment_strength': 0.3,
    'border_strength': 0.5
}
# NB Check these names and values pass a sanity check later

# this HAS to be stored to an object either here (anim) or inside animate_model - else matplotlib garbage-collects the animation
#anim = animate_model(BoidsModel, parameters3D) # the least efficient version it seems
#anim = animate_model2(BoidsModel,parameters3D) # uncomment/ run this for pointlike boids
anim2 = animate_model_quiver(BoidsModel, parameters3D) # run this for a quiver plot (3d only)


# Parameter Sweeps and Data Analysis (Seaborn/Pandas)

# see https://agentpy.readthedocs.io/en/latest/agentpy_forest_fire.html
# tutorial on setting up a sample and running a parameter sweep