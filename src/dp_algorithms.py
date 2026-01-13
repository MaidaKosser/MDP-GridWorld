"""
Dynamic Programming Algorithms for MDPs
========================================
Implements Value Iteration and Policy Iteration algorithms.

Algorithms:
- Value Iteration: Iteratively update value function using Bellman optimality equation
- Policy Iteration: Alternate between policy evaluation and policy improvement
"""

import numpy as np
from .mdp_gridworld import ACTIONS


def one_step_lookahead(mdp, V, gamma, s):
    """
    Compute action-value function Q(s,a) for all actions in state s.
    
    Q(s,a) = Σ P(s'|s,a) [R(s,a,s') + γ V(s')]
    
    Args:
        mdp: GridWorldMDP instance
        V: Current value function (numpy array)
        gamma: Discount factor
        s: State to evaluate
        
    Returns:
        Dictionary mapping action -> Q-value
    """
    q = {}
    for a in ACTIONS:
        total = 0.0
        for p, s2, r in mdp.transitions(s, a):
            idx = mdp.state_to_idx[s2]
            total += p * (r + gamma * V[idx])
        q[a] = total
    return q


def greedy_policy_from_value(mdp, V, gamma):
    """
    Extract greedy deterministic policy from value function.
    
    π(s) = argmax_a Q(s,a)
    
    Args:
        mdp: GridWorldMDP instance
        V: Value function
        gamma: Discount factor
        
    Returns:
        Dictionary mapping state -> best action
    """
    pi = {}
    for s in mdp.states:
        if mdp.is_terminal(s):
            pi[s] = None
            continue
        
        q = one_step_lookahead(mdp, V, gamma, s)
        pi[s] = max(q, key=q.get)  # Action with highest Q-value
    
    return pi


def value_iteration(mdp, gamma=0.95, theta=1e-6, max_iters=10000):
    """
    Value Iteration Algorithm.
    
    Iteratively applies Bellman optimality equation:
    V(s) ← max_a Σ P(s'|s,a) [R(s,a,s') + γ V(s')]
    
    Converges when max change in V is below threshold θ.
    
    Args:
        mdp: GridWorldMDP instance
        gamma: Discount factor (0 ≤ γ < 1)
        theta: Convergence threshold
        max_iters: Maximum iterations
        
    Returns:
        V: Optimal value function
        pi: Optimal policy
        history: List of (V, policy) tuples for each iteration
    """
    # Initialize value function to zeros
    V = np.zeros(len(mdp.states))
    history = []

    for iteration in range(max_iters):
        delta = 0.0
        V_new = V.copy()

        # Update value for each state
        for s in mdp.states:
            if mdp.is_terminal(s):
                continue

            # Compute Q-values for all actions
            q = one_step_lookahead(mdp, V, gamma, s)
            
            # Take maximum (Bellman optimality)
            best = max(q.values())
            idx = mdp.state_to_idx[s]
            
            # Track maximum change
            delta = max(delta, abs(best - V[idx]))
            V_new[idx] = best

        V = V_new
        
        # Extract current greedy policy for visualization
        pi = greedy_policy_from_value(mdp, V, gamma)
        history.append((V.copy(), pi))

        # Check convergence
        if delta < theta:
            break

    return V, greedy_policy_from_value(mdp, V, gamma), history


def policy_evaluation(mdp, pi, gamma=0.95, theta=1e-6, max_iters=10000):
    """
    Iterative Policy Evaluation.
    
    Computes value function V^π for a given policy π:
    V^π(s) = Σ P(s'|s,π(s)) [R(s,π(s),s') + γ V^π(s')]
    
    Args:
        mdp: GridWorldMDP instance
        pi: Policy to evaluate (dict: state -> action)
        gamma: Discount factor
        theta: Convergence threshold
        max_iters: Maximum iterations
        
    Returns:
        V: Value function for policy π
    """
    V = np.zeros(len(mdp.states))

    for iteration in range(max_iters):
        delta = 0.0
        
        for s in mdp.states:
            if mdp.is_terminal(s):
                continue

            a = pi[s]
            idx = mdp.state_to_idx[s]
            v_old = V[idx]
            
            # Bellman expectation equation
            v_new = sum(
                p * (r + gamma * V[mdp.state_to_idx[s2]]) 
                for p, s2, r in mdp.transitions(s, a)
            )
            
            V[idx] = v_new
            delta = max(delta, abs(v_new - v_old))

        # Check convergence
        if delta < theta:
            break

    return V


def policy_iteration(mdp, gamma=0.95, theta=1e-6, max_eval_iters=10000, max_improve_iters=1000):
    """
    Policy Iteration Algorithm.
    
    Alternates between:
    1. Policy Evaluation: Compute V^π for current policy π
    2. Policy Improvement: Update π to be greedy w.r.t. V^π
    
    Converges when policy stops changing (policy is stable).
    
    Args:
        mdp: GridWorldMDP instance
        gamma: Discount factor
        theta: Convergence threshold for evaluation
        max_eval_iters: Max iterations for policy evaluation
        max_improve_iters: Max iterations for policy improvement
        
    Returns:
        V: Optimal value function
        pi: Optimal policy
        history: List of (V, policy) tuples for each improvement step
    """
    # Initialize policy: arbitrary (use "R" for all non-terminal states)
    pi = {s: "R" for s in mdp.states if not mdp.is_terminal(s)}
    for s in mdp.states:
        if mdp.is_terminal(s):
            pi[s] = None

    history = []

    for iteration in range(max_improve_iters):
        # 1. Policy Evaluation: Compute V^π
        V = policy_evaluation(mdp, pi, gamma, theta, max_eval_iters)

        # 2. Policy Improvement: Make policy greedy w.r.t. V
        policy_stable = True
        
        for s in mdp.states:
            if mdp.is_terminal(s):
                continue

            old_action = pi[s]
            
            # Find best action according to current V
            q = one_step_lookahead(mdp, V, gamma, s)
            pi[s] = max(q, key=q.get)

            # Check if policy changed
            if old_action != pi[s]:
                policy_stable = False

        # Store iteration history
        history.append((V.copy(), dict(pi)))

        # Check convergence: policy hasn't changed
        if policy_stable:
            break

    return V, pi, history