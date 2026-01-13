import time
import streamlit as st
import numpy as np
import pandas as pd
from src.mdp_gridworld import GridWorldConfig, GridWorldMDP
from src.dp_algorithms import value_iteration, policy_iteration
from src.render import plot_value_and_policy

# ---------------- PAGE ----------------
st.set_page_config(page_title="GridWorld MDP Visualizer", layout="wide")

# ---------------- THEME ----------------
st.markdown("""
<style>
body { background-color:#0d1f5c; color:white; }
.sidebar .sidebar-content { background-color:#081548; color:white; padding:20px;}
h1, h2, h3 { color:white; }
.stButton>button { border-radius:8px; background-color:#0d1f5c; color:white; padding:6px 12px; font-weight:bold; border:1px solid white;}
.stButton>button:hover { background-color:#081548; }
.stSlider>div>div>div>input, .stSlider>div>div>div>span { color:white; }
.stSlider>div>div>div>div { background-color:#0d1f5c; border:1px solid white;}
.stProgress>div>div { background-color:#0d1f5c; }
.stRadio>div>div>label { color:black; }
.stRadio>div>div>input { accent-color:#0d1f5c; }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("# MDP Visualizer")
st.caption("Visualization of Value Iteration & Policy Iteration")

# ---------------- MDP CONFIG ----------------
cfg = GridWorldConfig(
    rows=5, cols=5,
    start=(4,0),
    goal=(0,4),
    negative_terminal=(1,3),
    obstacles=((1,1),(2,1),(3,1)),
    step_reward=-0.1,
    goal_reward=10,
    neg_reward=-10,
    p_intended=0.8,
    p_slip=0.2
)
mdp = GridWorldMDP(cfg)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state: st.session_state.history=[]
if "i" not in st.session_state: st.session_state.i=0
if "run" not in st.session_state: st.session_state.run=False
if "theta" not in st.session_state: st.session_state.theta=1e-6
if "params" not in st.session_state: st.session_state.params=None

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## Control Panel")
    algo = st.radio("Algorithm", ["Value Iteration","Policy Iteration"])
    gamma = st.slider("Discount Factor γ",0.01,0.99,0.95,0.01)
    st.markdown("### Convergence Threshold θ")
    st.session_state.theta = st.number_input("θ value", value=st.session_state.theta, format="%.10f")
    st.divider()
    r1,r2,r3 = st.columns(3)
    start = r1.button("Start")
    stop = r2.button("Stop")
    reset = r3.button("Reset")
    r4,r5,r6 = st.columns(3)
    back = r4.button("Back")
    nextt = r5.button("Next")
    end = r6.button("End")

# ---------------- HISTORY ----------------
def compute_history():
    if algo=="Value Iteration":
        _,_,h = value_iteration(mdp,gamma=gamma,theta=st.session_state.theta)
    else:
        _,_,h = policy_iteration(mdp,gamma=gamma,theta=st.session_state.theta)
    return h

params = (algo,gamma,st.session_state.theta)
if st.session_state.params!=params or reset:
    st.session_state.history = compute_history()
    st.session_state.i = 0
    st.session_state.params = params
    st.session_state.run=False

# ---------------- BUTTON LOGIC ----------------
if start: st.session_state.run=True
if stop: st.session_state.run=False
if back: st.session_state.i = max(0, st.session_state.i-1 if st.session_state.history else 0)
if nextt: st.session_state.i = min(len(st.session_state.history)-1 if st.session_state.history else 0, st.session_state.i+1)
if end: st.session_state.i = len(st.session_state.history)-1 if st.session_state.history else 0; st.session_state.run=False

# ---------------- LAYOUT ----------------
left,center,right = st.columns([1,1,0.8])

# ---------- Original Grid ----------
left.markdown("### Initial Grid")
if st.session_state.history:
    V0,pi0 = st.session_state.history[0]
    left.pyplot(plot_value_and_policy(mdp,V0,pi0,figsize=(3,3),dpi=200),clear_figure=True)

# ---------- Dynamic Iteration Grid ----------
ph = center.empty()
if st.session_state.run:
    while st.session_state.run and st.session_state.i < len(st.session_state.history):
        V,pi = st.session_state.history[st.session_state.i]
        with ph.container():
            st.markdown(f"### Iteration {st.session_state.i+1}/{len(st.session_state.history)}")
            st.pyplot(plot_value_and_policy(mdp,V,pi,figsize=(3,3),dpi=200),clear_figure=True)
        time.sleep(0.25)
        st.session_state.i+=1
    st.session_state.run=False

if st.session_state.history:
    V,pi = st.session_state.history[st.session_state.i]
    with ph.container():
        st.markdown(f"### Iteration {st.session_state.i+1}/{len(st.session_state.history)}")
        st.pyplot(plot_value_and_policy(mdp,V,pi,figsize=(3,3),dpi=200),clear_figure=True)

# ---------- Explanation LEFT ----------
st.markdown("### Explanation & Notes")
st.markdown("""
- Values show state utilities  
- Arrows show greedy optimal policy  
- Obstacles block transitions  
- Terminal states stop episode  
- Convergence occurs when value change < θ  
- Smaller θ = more accurate but slower convergence  
""")

# Right Panel Table
right.markdown("### Summary")
summary_df = pd.DataFrame({
    "Parameter":["Algorithm","Gamma γ","Theta θ","Iterations"],
    "Value":[str(algo), str(gamma), str(st.session_state.theta), str(len(st.session_state.history))]
})
right.table(summary_df)

if st.session_state.i==len(st.session_state.history)-1 and st.session_state.history:
    right.success("Algorithm Converged Successfully")
else:
    right.info("Algorithm Running / Not Yet Converged")

right.progress((st.session_state.i+1)/max(1,len(st.session_state.history)))
