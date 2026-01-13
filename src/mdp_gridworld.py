"""
GridWorld MDP Implementation
============================
Defines the Markov Decision Process for a grid-based navigation environment.

MDP Components:
- States (S): All navigable grid cells (excluding obstacles)
- Actions (A): {Up, Down, Left, Right}
- Transition Model (P): Stochastic transitions with intended direction and slip probability
- Rewards (R): Goal reward, negative terminal penalty, and step cost
"""

from dataclasses import dataclass

# Action space: cardinal directions
ACTIONS = ["U", "D", "L", "R"]

# Action to grid delta mapping
A2DELTA = {
    "U": (-1, 0),  # Up: row decreases
    "D": (1, 0),   # Down: row increases
    "L": (0, -1),  # Left: column decreases
    "R": (0, 1),   # Right: column increases
}

@dataclass(frozen=True)
class GridWorldConfig:
    """
    Immutable configuration for GridWorld MDP.
    
    Attributes:
        rows: Number of rows in grid
        cols: Number of columns in grid
        start: Starting position (row, col)
        goal: Goal position with positive reward
        negative_terminal: Negative terminal position with penalty
        obstacles: Tuple of obstacle positions (impassable cells)
        step_reward: Reward for each non-terminal transition
        goal_reward: Reward for reaching goal state
        neg_reward: Penalty for reaching negative terminal
        p_intended: Probability of moving in intended direction
        p_slip: Probability of slipping to unintended directions
    """
    rows: int = 5
    cols: int = 5
    start: tuple = (4, 0)
    goal: tuple = (0, 4)
    negative_terminal: tuple = (1, 3)
    obstacles: tuple = ((1, 1), (2, 1), (3, 1))
    step_reward: float = -0.1
    goal_reward: float = 10.0
    neg_reward: float = -10.0
    p_intended: float = 0.8
    p_slip: float = 0.2

class GridWorldMDP:
    """
    GridWorld Markov Decision Process.
    
    Implements the MDP formulation with:
    - Stochastic transitions (80% intended, 20% slip)
    - Terminal states (goal and negative)
    - Obstacles (impassable states)
    - Reward function based on state transitions
    """
    
    def __init__(self, cfg: GridWorldConfig):
        """
        Initialize GridWorld MDP from configuration.
        
        Args:
            cfg: GridWorldConfig instance with MDP parameters
        """
        self.cfg = cfg
        self.rows = cfg.rows
        self.cols = cfg.cols
        self.goal = cfg.goal
        self.neg_terminal = cfg.negative_terminal
        self.obstacles = set(cfg.obstacles)
        self.terminals = {self.goal, self.neg_terminal}
        
        # Generate all valid states (excluding obstacles)
        self.states = [
            (r, c) 
            for r in range(self.rows) 
            for c in range(self.cols) 
            if (r, c) not in self.obstacles
        ]
        
        # Create bidirectional state-index mapping for efficient array access
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for s, i in self.state_to_idx.items()}

    def in_bounds(self, s):
        """Check if state is within grid boundaries."""
        r, c = s
        return 0 <= r < self.rows and 0 <= c < self.cols
    
    def is_blocked(self, s):
        """Check if state is an obstacle."""
        return s in self.obstacles
    
    def is_terminal(self, s):
        """Check if state is terminal (goal or negative)."""
        return s in self.terminals

    def reward(self, s, a, s2):
        """
        Reward function R(s, a, s').
        
        Args:
            s: Current state
            a: Action taken
            s2: Resulting state
            
        Returns:
            Immediate reward for transition
        """
        if s2 == self.goal:
            return self.cfg.goal_reward
        if s2 == self.neg_terminal:
            return self.cfg.neg_reward
        return self.cfg.step_reward

    def move(self, s, a):
        """
        Deterministic transition: where would agent end up if action succeeds?
        
        Args:
            s: Current state
            a: Action to take
            
        Returns:
            Resulting state (stays in place if blocked/out-of-bounds)
        """
        if self.is_terminal(s):
            return s
        
        dr, dc = A2DELTA[a]
        s2 = (s[0] + dr, s[1] + dc)
        
        # Stay in place if move is invalid
        if not self.in_bounds(s2) or self.is_blocked(s2):
            return s
        
        return s2

    def transitions(self, s, a):
        """
        Stochastic transition model P(s' | s, a).
        
        Implements:
        - 80% probability of intended direction
        - 20% probability distributed among other directions (slip)
        
        Args:
            s: Current state
            a: Action taken
            
        Returns:
            List of (probability, next_state, reward) tuples
        """
        if self.is_terminal(s):
            return [(1.0, s, 0.0)]
        
        probs = {}
        
        # Intended direction gets p_intended probability
        s_intended = self.move(s, a)
        probs[s_intended] = probs.get(s_intended, 0.0) + self.cfg.p_intended
        
        # Slip probability distributed among other actions
        other_actions = [x for x in ACTIONS if x != a]
        p_each = self.cfg.p_slip / len(other_actions)
        
        for a2 in other_actions:
            s2 = self.move(s, a2)
            probs[s2] = probs.get(s2, 0.0) + p_each
        
        # Return list of (probability, state, reward) tuples
        return [(p, s2, self.reward(s, a, s2)) for s2, p in probs.items()]

    def all_actions(self, s):
        """
        Available actions in state s.
        
        Returns:
            List of actions (empty for terminal states)
        """
        return [] if self.is_terminal(s) else ACTIONS