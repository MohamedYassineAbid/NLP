import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.colors import LogNorm
import time

st.set_page_config(layout="wide", page_title="Optimization Methods Comparison")

st.title("Optimization Methods Comparison")
st.write("""
This app visualizes and compares two optimization methods:
- **Gradient Descent**: First-order method using function gradient
- **Newton's Method**: Second-order method using both gradient and Hessian
""")

# Sidebar for user input
st.sidebar.header("Parameters")
x0_1 = st.sidebar.slider("Initial X coordinate", -2.0, 2.0, -1.2, 0.1)
x0_2 = st.sidebar.slider("Initial Y coordinate", -1.0, 3.0, 1.0, 0.1)
max_iter_gd = st.sidebar.slider("Max iterations (Gradient Descent)", 10, 200, 100)
max_iter_newton = st.sidebar.slider("Max iterations (Newton's Method)", 5, 100, 50)
contour_levels = st.sidebar.slider("Contour levels", 10, 40, 20)
show_animation = st.sidebar.checkbox("Show step-by-step animation", True)
animation_speed = st.sidebar.slider("Animation speed (ms)", 100, 1000, 300)

# Rosenbrock function and derivatives
def rosen(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def rosen_grad(x):
    dx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    dy = 200*(x[1] - x[0]**2)
    return np.array([dx, dy])

def rosen_hess(x):
    h11 = 2 - 400*x[1] + 1200*x[0]**2
    h12 = -400*x[0]
    return np.array([[h11, h12],
                    [h12, 200]])

# Run optimizations
@st.cache_data
def run_optimizations(x0, max_iter_gd, max_iter_newton):
    # Track paths
    paths = {'GD': [], 'Newton': []}
    def cb_gd(x): paths['GD'].append(x.copy())
    def cb_newton(x): paths['Newton'].append(x.copy())
    
    # Run Gradient Descent (as CG with exact grad)
    gd_result = minimize(rosen, x0, method='CG', jac=rosen_grad,
                callback=cb_gd, options={'maxiter': max_iter_gd, 'gtol': 1e-6})
    
    # Run Newton's Method
    newton_result = minimize(rosen, x0, method='Newton-CG',
                jac=rosen_grad, hess=rosen_hess,
                callback=cb_newton, options={'maxiter': max_iter_newton, 'xtol': 1e-8})
    
    return {
        'paths': paths,
        'gd_result': gd_result,
        'newton_result': newton_result
    }

# Run optimizations with the current parameters
x0 = np.array([x0_1, x0_2])
optimization_results = run_optimizations(x0, max_iter_gd, max_iter_newton)
paths = optimization_results['paths']
gd_path = np.array(paths['GD'])
nt_path = np.array(paths['Newton'])

# Create mesh for contour plot
x = np.linspace(-2, 2, 200)
y = np.linspace(-1, 3, 200)
X, Y = np.meshgrid(x, y)
Z = (1 - X)**2 + 100*(Y - X**2)**2

# Create two columns for the visualization
col1, col2 = st.columns([3, 2])

with col1:
    # Main plot area
    st.subheader("Optimization Paths Visualization")
    
    # Create a figure for the contour plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot contour
    contour = ax.contour(X, Y, Z, levels=np.logspace(-1, 3, contour_levels), cmap='viridis', alpha=0.7)
    fig.colorbar(contour, label='Rosenbrock function value (log scale)')
    
    # Set axis limits with some padding
    x_min, x_max = min(-2, np.min(gd_path[:, 0]), np.min(nt_path[:, 0])) - 0.2, max(2, np.max(gd_path[:, 0]), np.max(nt_path[:, 0])) + 0.2
    y_min, y_max = min(-1, np.min(gd_path[:, 1]), np.min(nt_path[:, 1])) - 0.2, max(3, np.max(gd_path[:, 1]), np.max(nt_path[:, 1])) + 0.2
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Plot initial point
    ax.plot(x0[0], x0[1], 'ko', markersize=10, label='Initial Point')
    
    # Plot the minimum point
    ax.plot(1, 1, 'g*', markersize=15, label='Global Minimum (1, 1)')
    
    # Plot full paths
    ax.plot(gd_path[:, 0], gd_path[:, 1], 'o-', color='red', label='Gradient Descent', markersize=5, alpha=0.7)
    ax.plot(nt_path[:, 0], nt_path[:, 1], 's-', color='blue', label="Newton's Method", markersize=5, alpha=0.7)
    
    # Add legend, labels, and grid
    ax.legend(loc='upper right')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    ax.set_title('Comparison of Optimization Methods')
    
    # Display the plot
    st.pyplot(fig)
    
    # Animated visualization
    if show_animation:
        st.subheader("Step-by-step Animation")
        
        # Create a placeholder for the animation
        animation_placeholder = st.empty()
        
        # Determine the maximum number of steps for animation
        max_steps = max(len(gd_path), len(nt_path))
        
        # Animation loop
        for frame in range(max_steps + 1):
            # Create a new figure for each frame
            anim_fig, anim_ax = plt.subplots(figsize=(10, 8))
            
            # Plot contour
            anim_contour = anim_ax.contour(X, Y, Z, levels=np.logspace(-1, 3, contour_levels), cmap='viridis', alpha=0.7)
            
            # Plot initial point
            anim_ax.plot(x0[0], x0[1], 'ko', markersize=10, label='Initial Point')
            
            # Plot the minimum point
            anim_ax.plot(1, 1, 'g*', markersize=15, label='Global Minimum (1, 1)')
            
            # Update GD path
            if frame > 0:
                frame_idx = min(frame, len(gd_path)) - 1
                anim_ax.plot(gd_path[:frame_idx+1, 0], gd_path[:frame_idx+1, 1], 'o-', color='red', 
                           label='Gradient Descent', markersize=5, alpha=0.7)
                if frame_idx < len(gd_path):
                    anim_ax.plot(gd_path[frame_idx, 0], gd_path[frame_idx, 1], 'o', color='red', markersize=8)
            
            # Update Newton's method path
            if frame > 0:
                frame_idx = min(frame, len(nt_path)) - 1
                anim_ax.plot(nt_path[:frame_idx+1, 0], nt_path[:frame_idx+1, 1], 's-', color='blue', 
                           label="Newton's Method", markersize=5, alpha=0.7)
                if frame_idx < len(nt_path):
                    anim_ax.plot(nt_path[frame_idx, 0], nt_path[frame_idx, 1], 's', color='blue', markersize=8)
            
            # Set title with current step
            anim_ax.set_title(f'Optimization Step: {frame}')
            
            # Set axis limits, labels, grid
            anim_ax.set_xlim(x_min, x_max)
            anim_ax.set_ylim(y_min, y_max)
            anim_ax.legend(loc='upper right')
            anim_ax.set_xlabel('x')
            anim_ax.set_ylabel('y')
            anim_ax.grid(True, alpha=0.3)
            
            # Update the placeholder with the new figure
            animation_placeholder.pyplot(anim_fig)
            plt.close(anim_fig)
            
            # Add delay for animation
            time.sleep(animation_speed / 1000)

with col2:
    # Results and statistics
    st.subheader("Optimization Results")
    
    # Create metrics for each method
    gd_steps = len(gd_path)
    newton_steps = len(nt_path)
    
    col2a, col2b = st.columns(2)
    with col2a:
        st.metric("Gradient Descent Steps", gd_steps)
        st.metric("Final GD Distance to Minimum", f"{np.linalg.norm(gd_path[-1] - np.array([1, 1])):.6f}")
        st.metric("Final GD Function Value", f"{rosen(gd_path[-1]):.6f}")
    
    with col2b:
        st.metric("Newton's Method Steps", newton_steps)
        st.metric("Final Newton Distance to Minimum", f"{np.linalg.norm(nt_path[-1] - np.array([1, 1])):.6f}")
        st.metric("Final Newton Function Value", f"{rosen(nt_path[-1]):.6f}")
    
    # Show convergence information
    st.subheader("Convergence Information")
    st.write("**Gradient Descent:**")
    st.write(f"- Success: {optimization_results['gd_result'].success}")
    st.write(f"- Status: {optimization_results['gd_result'].message}")
    st.write(f"- Final point: [{gd_path[-1, 0]:.6f}, {gd_path[-1, 1]:.6f}]")
    
    st.write("**Newton's Method:**")
    st.write(f"- Success: {optimization_results['newton_result'].success}")
    st.write(f"- Status: {optimization_results['newton_result'].message}")
    st.write(f"- Final point: [{nt_path[-1, 0]:.6f}, {nt_path[-1, 1]:.6f}]")
    
    # Information about the methods
    st.subheader("About the Methods")
    
    method_tab1, method_tab2 = st.tabs(["Gradient Descent", "Newton's Method"])
    
    with method_tab1:
        st.write("""
        **Gradient Descent** is a first-order optimization algorithm. It works by:
        - Computing the gradient (first derivative) of the function
        - Taking steps in the direction of steepest descent (negative gradient)
        - Using line search or fixed step sizes to determine how far to move
        
        Strengths:
        - Simple to implement
        - Works for a wide variety of problems
        - Less memory intensive
        
        Weaknesses:
        - Can be slow to converge, especially in ravines
        - Zigzags in narrow valleys
        - Only uses first-order information
        """)
    
    with method_tab2:
        st.write("""
        **Newton's Method** is a second-order optimization algorithm. It works by:
        - Computing both the gradient and the Hessian (second derivatives)
        - Using quadratic approximation to determine step direction
        - Often taking larger, more direct steps toward the minimum
        
        Strengths:
        - Converges much faster near the solution
        - Takes more direct paths through ravines
        - Quadratic convergence rate near the minimum
        
        Weaknesses:
        - Requires Hessian computation (or approximation)
        - More computationally expensive per iteration
        - May not work well if far from the minimum
        """)
    
    # About the Rosenbrock function
    st.subheader("About the Rosenbrock Function")
    st.write("""
    The **Rosenbrock function** is a non-convex function used as a performance test problem for optimization algorithms:
    
    $f(x,y) = (1-x)^2 + 100(y-x^2)^2$
    
    It has a global minimum at $(1,1)$ where $f(1,1) = 0$.
    
    The function is challenging because it has a narrow, curved valley where the minimum lies. 
    Simple gradient-based methods often struggle with this function, while methods that use 
    second-order information (like Newton's method) tend to perform better.
    """)

st.markdown("---")
st.caption("Interactive Optimization Methods Visualization | Created with Streamlit")
