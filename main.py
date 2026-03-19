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

def safe_normalize(v, speed=1.0):
    """Normalize vector v to a given speed. Returns zero vector if v is zero."""
    n = np.linalg.norm(v)
    if n == 0:
        return np.zeros_like(v)
    return (v / n) * speed

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
        pos = self.space.positions[self]  # read fresh pos from space (avoids stale-reference risk)
        ndim = self.p.ndim  # local variable avoids repeated attribute lookups
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
        # Two-stage velocity update: compute next_velocity now, apply later.
        # This prevents asynchronous update artefacts where boids updated earlier
        # in the loop influence the velocity computation of boids updated later.
        self.next_velocity = norm(self.velocity + v1 + v2 + v3 + v4)

    def apply_velocity(self):
        """Stage 2: assign the pre-computed next_velocity to self.velocity."""
        self.velocity = self.next_velocity

    def update_position(self):
        self.space.move_by(self, self.velocity)
        
# ---------------------------------------------------------------------------
# Predator agent
# ---------------------------------------------------------------------------
class Predator(ap.Agent):
    """A predator that chases nearby boids by steering toward their local centre."""

    def setup(self):
        # Random initial direction, normalised to predator_speed
        self.velocity = safe_normalize(
            self.model.nprandom.random(self.p.ndim) - 0.5,
            speed=self.p.predator_speed
        )
        self.next_velocity = self.velocity.copy()

    def setup_pos(self, space):
        self.space = space
        self.pos = space.positions[self]

    def update_velocity(self, boids):
        """Steer toward the centre of nearby prey (boids).
        If no prey are nearby, maintain current heading with border handling.
        """
        pos = self.space.positions[self]  # always read fresh position
        ndim = self.p.ndim
        new_vel = self.velocity.copy()  # work on a copy; don't corrupt current velocity

        # --- find boids within predator visual range ---
        if len(boids) > 0:
            nearby = self.space.neighbors(self, distance=self.p.predator_visual_range)
            nearby_boids = [a for a in nearby if isinstance(a, Boid)]
        else:
            nearby_boids = []

        if len(nearby_boids) > 0:
              # Steer toward centre of nearby prey
              valid_boids = [b for b in nearby_boids if b in self.space.positions]
              if len(valid_boids) > 0:
                 centre = np.mean([self.space.positions[b] for b in valid_boids], axis=0)
                 desired_dir = safe_normalize(centre - pos)
                 new_vel += desired_dir * self.p.predator_turn_strength
              # else: no valid boids to steer toward
        # else: keep current heading (no turn adjustment)

        # --- border avoidance (same rules as boids) ---
        v_border = np.zeros(ndim)
        d = self.p.border_distance
        s = self.p.border_strength
        for i in range(ndim):
            if pos[i] < d:
                v_border[i] += s
            elif pos[i] > self.space.shape[i] - d:
                v_border[i] -= s
        new_vel += v_border

        # Renormalise to predator speed
        self.next_velocity = safe_normalize(new_vel, speed=self.p.predator_speed)

    def apply_velocity(self):
        """Apply the pre-computed next_velocity."""
        self.velocity = self.next_velocity

    def update_position(self):
        self.space.move_by(self, self.velocity)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BoidsModel(ap.Model):
    def setup(self):
        # Initialise the agents and the model's other key objects
        self.space = ap.Space(self, shape=[self.p.size]*self.p.ndim)
        self.agents = ap.AgentList(self, self.p.population, Boid)
        self.space.add_agents(self.agents, random=True)
        self.agents.setup_pos(self.space)

        # --- Predators (added to the same space) ---
        n_pred = self.p.n_predators
        self.predators = ap.AgentList(self, n_pred, Predator)
        if len(self.predators) > 0:
            self.space.add_agents(self.predators, random=True)
            self.predators.setup_pos(self.space)

    def step(self):
        # --- Stage 1: compute next velocities (all agents read current state) ---
        if len(self.agents) > 0:
            self.agents.update_velocity()   # computes next_velocity for each boid
        if len(self.predators) > 0:
            for pred in self.predators:
                pred.update_velocity(self.agents)  # pass boid list so predator can hunt

        # --- Stage 2: apply computed velocities ---
        if len(self.agents) > 0:
            self.agents.apply_velocity()
        if len(self.predators) > 0:
            self.predators.apply_velocity()

        # --- Stage 3: move all agents ---
        if len(self.agents) > 0:
            self.agents.update_position()
        if len(self.predators) > 0:
            self.predators.update_position()

        # --- Stage 4: predation (capture) ---
        self._predation_step()

    def _predation_step(self):
        """Each predator eats at most one boid per timestep (the closest
        boid within capture radius).  Boids already marked for removal
        by another predator in this step are skipped."""
        if len(self.predators) == 0 or len(self.agents) == 0:
            return

        eaten = set()  # boid ids already claimed this step

        for pred in self.predators:
            pred_pos = self.space.positions[pred]
            # Find boids within capture radius
            nearby = self.space.neighbors(pred, distance=self.p.predator_capture_radius)
            candidates = [a for a in nearby
                          if isinstance(a, Boid) and id(a) not in eaten]
            if len(candidates) == 0:
                continue
            # Pick the closest boid
            dists = [np.linalg.norm(self.space.positions[b] - pred_pos)
                     for b in candidates]
            closest = candidates[int(np.argmin(dists))]
            eaten.add(id(closest))

        # Remove all eaten boids at once (safe batch removal)
        boids_to_remove = [b for b in self.agents if id(b) in eaten]
        for boid in boids_to_remove:
            # Remove from space positions dict
            if boid in self.space.positions:
                del self.space.positions[boid]
            # Remove from boid agent list
            if boid in self.agents:
                self.agents.remove(boid)
        
# Animation / Visualisation        
def animation_plot_single(m, ax):
    # takes args m := model, ax := parameter dictionary
    ndim = m.p.ndim
    n_boids = len(m.agents)
    n_pred = len(m.predators)
    ax.clear()
    ax.set_title(f"Boids: {n_boids}  Predators: {n_pred}  {ndim}D, t={m.t}")
    # --- Boid positions (black) ---
    if n_boids > 0:
        boid_pos = np.array([m.space.positions[b] for b in m.agents]).T
        ax.scatter(*boid_pos, s=5, c='black')
    # --- Predator positions (red) ---
    if n_pred > 0:
        pred_pos = np.array([m.space.positions[p] for p in m.predators]).T
        ax.scatter(*pred_pos, s=80, c='red', marker='^')
    ax.set_xlim(0, m.p.size)
    ax.set_ylim(0, m.p.size)
    if ndim == 3:
        ax.set_zlim(0, m.p.size)

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
    # Second method — now uses ax.clear() each frame because predation
    # changes the number of boids, so we cannot simply update scatter offsets.
    m = model(parameters)
    m.run(1)

    fig = plt.figure(figsize=(7,7))
    ndim = parameters['ndim']
    ax = fig.add_subplot(111, projection='3d' if ndim == 3 else None)

    def update(frame):
        m.step()
        ax.clear()
        n_boids = len(m.agents)
        n_pred = len(m.predators)
        # Boids (black scatter)
        if n_boids > 0:
            boid_pos = np.array([m.space.positions[b] for b in m.agents]).T
            ax.scatter(*boid_pos, s=5, c='black')
        # Predators (red scatter)
        if n_pred > 0:
            pred_pos = np.array([m.space.positions[p] for p in m.predators]).T
            ax.scatter(*pred_pos, s=80, c='red', marker='^')
        ax.set_xlim(0, m.p.size)
        ax.set_ylim(0, m.p.size)
        if ndim == 3:
            ax.set_zlim(0, m.p.size)
        ax.set_title(f"Boids: {n_boids}  Predators: {n_pred}  {ndim}D, t={m.t}")

    anim = FuncAnimation(fig, update, frames=parameters['steps'], interval=10)
    plt.show()
    return anim

def animate_model_quiver(model, parameters):
    # Quiver for boids (black) + scatter for predators (red).
    # Uses ax.clear() each frame because predation changes the boid count.
    m = model(parameters)
    m.run(1)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(0, m.p.size)
    ax.set_ylim(0, m.p.size)
    ax.set_zlim(0, m.p.size)

    def update(frame):
        m.step()
        ax.clear()
        ax.set_xlim(0, m.p.size)
        ax.set_ylim(0, m.p.size)
        ax.set_zlim(0, m.p.size)

        n_boids = len(m.agents)
        n_pred = len(m.predators)

        # --- Boid quiver (black) ---
        if n_boids > 0:
            boid_pos = np.array([m.space.positions[b] for b in m.agents])
            boid_vel = np.array([b.velocity for b in m.agents])
            ax.quiver(
                boid_pos[:, 0], boid_pos[:, 1], boid_pos[:, 2],
                boid_vel[:, 0], boid_vel[:, 1], boid_vel[:, 2],
                length=1.0, normalize=True, color='black'
            )

        # --- Predator scatter (red) ---
        if n_pred > 0:
            pred_pos = np.array([m.space.positions[p] for p in m.predators])
            ax.scatter(
                pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2],
                s=80, c='red', marker='^'
            )

        ax.set_title(f"Boids: {n_boids}  Predators: {n_pred}  (quiver) {m.p.ndim}D")

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
    'border_strength': 0.5,
    # ---- Predator parameters ----
    'n_predators': 1,                # number of predators in the simulation
    'predator_speed': 1.5,           # faster than boids (boid speed ~ 1.0 after norm)
    'predator_visual_range': 20,     # how far a predator can see prey
    'predator_capture_radius': 2,    # distance at which a predator eats a boid
    'predator_turn_strength': 0.3,   # how sharply predator steers toward prey
}
# NB Check these names and values pass a sanity check later

# this HAS to be stored to an object either here (anim) or inside animate_model - else matplotlib garbage-collects the animation
#anim = animate_model(BoidsModel, parameters3D) # the least efficient version it seems
#anim = animate_model2(BoidsModel,parameters3D) # uncomment/ run this for pointlike boids
anim2 = animate_model_quiver(BoidsModel, parameters3D) # run this for a quiver plot (3d only)


# Parameter Sweeps and Data Analysis (Seaborn/Pandas)

# see https://agentpy.readthedocs.io/en/latest/agentpy_forest_fire.html
# tutorial on setting up a sample and running a parameter sweep