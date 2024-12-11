import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Speed of sound (m/s)
c = 343.0

# Define microphone positions (e.g., arranged radially around origin)
# Here we place 5 microphones around the origin at different angles:
num_mics = 5
radius = 1.0  # distance of mics from origin
angles = np.linspace(0, 2*np.pi, num_mics, endpoint=False)
mic_positions = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))

# Define the source position
source_position = np.array([2.0, 2.0])

# Compute the distances from the source to each microphone
distances = np.linalg.norm(mic_positions - source_position, axis=1)
max_distance = np.max(distances)
max_distance = distances

# Compute arrival times at each microphone
arrival_times = distances / c
max_time = np.max(arrival_times) * 1.5  # a bit more than max arrival time for animation

# Create figure and axis
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title("Microphone Array based Detection")

# Plot microphones
mic_scat = ax.scatter(mic_positions[:,0], mic_positions[:,1], s=50, c='black', zorder=3, label='Microphones')

# Plot source
src_plot = ax.plot(source_position[0], source_position[1], 'r*', markersize=10, label='Source')[0]

# Initialize wavefront (circle)
wavefront, = ax.plot([], [], 'b-', lw=2, label='Wavefront')

# Text annotation for time
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)

# After all microphones are reached, we can attempt a simple triangulation visualization
triang_lines = []
triang_point = None
triangles = []
triangulation_done = False

def init():
    # Initialization function for FuncAnimation
    wavefront.set_data([], [])
    time_text.set_text('')
    return [wavefront, time_text]

def update(frame):
    # Frame corresponds to a fraction of the total time
    t = frame * (max_time / 200)  # 200 frames
    
    # Set the radius of the wavefront
    radius = c * t

    # Update wavefront
    theta = np.linspace(0, 2*np.pi, 200)
    x_wave = source_position[0] + radius * np.cos(theta)
    y_wave = source_position[1] + radius * np.sin(theta)
    wavefront.set_data(x_wave, y_wave)

    # Update microphones color when wavefront reaches them
    mic_colors = []
    for i, d in enumerate(distances):
        if radius >= d:
            mic_colors.append('red')
        else:
            mic_colors.append('black')
    mic_scat.set_color(mic_colors)

    # Update time display
    time_text.set_text(f"Time: {t:.4f} s")

    # Once all microphones are reached, show triangulation
    global triangulation_done
    triangulation_done == 0

    if all(radius >= d for d in distances) and triangulation_done==0:
        global triang_lines, triang_point, triangles
        
        # Remove old lines and triangles
        for line in triang_lines:
            line.remove()
        triang_lines.clear()
        for triangle in triangles:
            triangle.remove()
        triangles.clear()
        
        if triang_point is not None:
            triang_point.remove()

        # Draw triangles and circles for each pair of microphones
        if not triangles and not triang_lines:  # Only draw if not already drawn
            for i in range(num_mics):
                for j in range(i+1, num_mics):
                    if radius >= distances[i] + 0.1 and radius >= distances[j] + 0.1:
                        # Draw triangle between two mics and source
                        triangle = plt.Polygon([mic_positions[i], mic_positions[j], source_position], 
                                            alpha=0.2, color='yellow')
                        ax.add_patch(triangle)
                        triangles.append(triangle)
                        plt.pause(0.5)  # Add a pause to visualize each triangle formation
                        
                        # Draw circles from each microphone
                        for mic_pos, d in [(mic_positions[i], distances[i]), (mic_positions[j], distances[j])]:
                            theta = np.linspace(0, 2*np.pi, 200)
                            x_circle = mic_pos[0] + d * np.cos(theta)
                            y_circle = mic_pos[1] + d * np.sin(theta)
                            line = ax.plot(x_circle, y_circle, '--', alpha=0.3)[0]
                            triang_lines.append(line)
                            plt.pause(0.5)  # Add a pause to visualize each circle

        # Plot the triangulated point with additional delay
        if radius >= max(distances) + 1.0 and triang_point is None:  # Only plot if not already plotted
            triang_point = ax.plot(source_position[0], source_position[1], 'go', markersize=8, label='Triangulated Source')[0]
            plt.pause(0.5)  # Add a pause after plotting the triangulated point
            
        triangulation_done = 1
    
    return [wavefront, mic_scat, time_text]

ani = animation.FuncAnimation(fig, update, frames=500, init_func=init, blit=False, interval=50, repeat=False)
ax.legend()
plt.show()