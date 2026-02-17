"""
visualize_plucker.py — Understanding Plucker Coordinates Visually
=====================================================
Usage: python3 visualize_plucker.py
Output: plucker_explained.png file generated
=====================================================
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor('#1a1a2e')

# Font settings (macOS)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

colors = {
    'u': '#FF6B6B',
    'v': '#4ECDC4',
    'para': '#FFE66D',
    'text': '#E8E8E8',
    'accent': '#A8E6CF',
    'bg': '#16213E',
    'grid': '#2C3E6B',
}


# ================================================================
#  Figure 1: Dot Product vs Plucker — Basic Concept
# ================================================================

ax1 = fig.add_subplot(4, 2, 1)
ax1.set_facecolor(colors['bg'])
ax1.set_xlim(-0.5, 4.5)
ax1.set_ylim(-0.5, 3.5)
ax1.set_aspect('equal')
ax1.set_title('Dot Product = A Single Scalar', color=colors['text'], fontsize=14, fontweight='bold')

u = np.array([3, 1])
v = np.array([1, 2])

ax1.annotate('', xy=u, xytext=(0,0),
    arrowprops=dict(arrowstyle='->', color=colors['u'], lw=2.5))
ax1.annotate('', xy=v, xytext=(0,0),
    arrowprops=dict(arrowstyle='->', color=colors['v'], lw=2.5))

ax1.text(u[0]+0.1, u[1]+0.1, 'u = [3, 1]', color=colors['u'], fontsize=11, fontweight='bold')
ax1.text(v[0]+0.1, v[1]+0.1, 'v = [1, 2]', color=colors['v'], fontsize=11, fontweight='bold')

# Angle indicator
angle_u = np.arctan2(u[1], u[0])
angle_v = np.arctan2(v[1], v[0])
theta = np.linspace(angle_u, angle_v, 30)
r_arc = 0.8
ax1.plot(r_arc*np.cos(theta), r_arc*np.sin(theta), color=colors['accent'], lw=1.5, alpha=0.8)
mid_angle = (angle_u + angle_v) / 2
ax1.text(1.1*np.cos(mid_angle), 1.1*np.sin(mid_angle), 'theta',
         color=colors['accent'], fontsize=10, ha='center')

dot = np.dot(u, v)
ax1.text(2.0, 3.0, f'u . v = 3*1 + 1*2 = {dot}',
         color=colors['para'], fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#2C3E50', edgecolor=colors['para']))
ax1.text(2.0, 2.4, '"A single number containing only angle info"',
         color=colors['text'], fontsize=10, alpha=0.8)

ax1.grid(True, alpha=0.15, color=colors['grid'])
ax1.tick_params(colors=colors['text'])


# ================================================================
#  Figure 2: Plucker = Signed Area of a Parallelogram
# ================================================================

ax2 = fig.add_subplot(4, 2, 2)
ax2.set_facecolor(colors['bg'])
ax2.set_xlim(-0.5, 4.5)
ax2.set_ylim(-0.5, 3.5)
ax2.set_aspect('equal')
ax2.set_title('Plucker Coordinate = Signed Area of Parallelogram', color=colors['text'], fontsize=14, fontweight='bold')

# Parallelogram
para = plt.Polygon([
    [0, 0], u, u + v, v
], alpha=0.3, facecolor=colors['para'], edgecolor=colors['para'], lw=1.5)
ax2.add_patch(para)

ax2.annotate('', xy=u, xytext=(0,0),
    arrowprops=dict(arrowstyle='->', color=colors['u'], lw=2.5))
ax2.annotate('', xy=v, xytext=(0,0),
    arrowprops=dict(arrowstyle='->', color=colors['v'], lw=2.5))

ax2.text(u[0]+0.1, u[1]-0.25, 'u', color=colors['u'], fontsize=13, fontweight='bold')
ax2.text(v[0]-0.35, v[1]+0.1, 'v', color=colors['v'], fontsize=13, fontweight='bold')

plucker_2d = u[0]*v[1] - u[1]*v[0]
ax2.text(1.5, 1.3, f'Area = {plucker_2d}', color='#1a1a2e', fontsize=14, fontweight='bold', ha='center')

ax2.text(1.0, 3.0, f'p = u[0]*v[1] - u[1]*v[0]',
         color=colors['para'], fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#2C3E50', edgecolor=colors['para']))
ax2.text(1.0, 2.4, f'  = 3*2 - 1*1 = {plucker_2d}',
         color=colors['text'], fontsize=11)

ax2.grid(True, alpha=0.15, color=colors['grid'])
ax2.tick_params(colors=colors['text'])


# ================================================================
#  Figures 3-4: Same Dot Product, Different Plucker!
# ================================================================

# --- Pair A: xy plane ---
ax3 = fig.add_subplot(4, 2, 3, projection='3d')
ax3.set_facecolor(colors['bg'])
ax3.set_title('Pair A: Vectors in the xy plane', color=colors['text'], fontsize=13, fontweight='bold')

uA = np.array([1, 2, 0])
vA = np.array([2, 1, 0])

ax3.quiver(0, 0, 0, *uA, color=colors['u'], arrow_length_ratio=0.15, lw=2.5)
ax3.quiver(0, 0, 0, *vA, color=colors['v'], arrow_length_ratio=0.15, lw=2.5)

# Parallelogram
verts_A = np.array([[0,0,0], uA, uA+vA, vA])
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
poly_A = Poly3DCollection([verts_A], alpha=0.3, facecolor=colors['para'], edgecolor=colors['para'])
ax3.add_collection3d(poly_A)

ax3.set_xlim(0, 3); ax3.set_ylim(0, 3); ax3.set_zlim(0, 3)
ax3.set_xlabel('x', color=colors['text'])
ax3.set_ylabel('y', color=colors['text'])
ax3.set_zlabel('z', color=colors['text'])
ax3.tick_params(colors=colors['text'])
ax3.xaxis.pane.set_facecolor(colors['bg'])
ax3.yaxis.pane.set_facecolor(colors['bg'])
ax3.zaxis.pane.set_facecolor(colors['bg'])

dotA = np.dot(uA, vA)
pA = [uA[0]*vA[1]-uA[1]*vA[0], uA[0]*vA[2]-uA[2]*vA[0], uA[1]*vA[2]-uA[2]*vA[1]]
ax3.text2D(0.05, 0.92, f'u={list(uA)}  v={list(vA)}', transform=ax3.transAxes,
           color=colors['text'], fontsize=9)
ax3.text2D(0.05, 0.84, f'dot = {dotA}', transform=ax3.transAxes,
           color=colors['accent'], fontsize=11, fontweight='bold')
ax3.text2D(0.05, 0.76, f'Plucker = {pA}', transform=ax3.transAxes,
           color=colors['para'], fontsize=11, fontweight='bold')

# --- Pair B: xz plane ---
ax4 = fig.add_subplot(4, 2, 4, projection='3d')
ax4.set_facecolor(colors['bg'])
ax4.set_title('Pair B: Vectors in the xz plane', color=colors['text'], fontsize=13, fontweight='bold')

uB = np.array([1, 0, 2])
vB = np.array([2, 0, 1])

ax4.quiver(0, 0, 0, *uB, color=colors['u'], arrow_length_ratio=0.15, lw=2.5)
ax4.quiver(0, 0, 0, *vB, color=colors['v'], arrow_length_ratio=0.15, lw=2.5)

verts_B = np.array([[0,0,0], uB, uB+vB, vB])
poly_B = Poly3DCollection([verts_B], alpha=0.3, facecolor=colors['para'], edgecolor=colors['para'])
ax4.add_collection3d(poly_B)

ax4.set_xlim(0, 3); ax4.set_ylim(0, 3); ax4.set_zlim(0, 3)
ax4.set_xlabel('x', color=colors['text'])
ax4.set_ylabel('y', color=colors['text'])
ax4.set_zlabel('z', color=colors['text'])
ax4.tick_params(colors=colors['text'])
ax4.xaxis.pane.set_facecolor(colors['bg'])
ax4.yaxis.pane.set_facecolor(colors['bg'])
ax4.zaxis.pane.set_facecolor(colors['bg'])

dotB = np.dot(uB, vB)
pB = [uB[0]*vB[1]-uB[1]*vB[0], uB[0]*vB[2]-uB[2]*vB[0], uB[1]*vB[2]-uB[2]*vB[1]]
ax4.text2D(0.05, 0.92, f'u={list(uB)}  v={list(vB)}', transform=ax4.transAxes,
           color=colors['text'], fontsize=9)
ax4.text2D(0.05, 0.84, f'dot = {dotB}', transform=ax4.transAxes,
           color=colors['accent'], fontsize=11, fontweight='bold')
ax4.text2D(0.05, 0.76, f'Plucker = {pB}', transform=ax4.transAxes,
           color=colors['para'], fontsize=11, fontweight='bold')


# ================================================================
#  Figure 5: Key Insight — Same Dot Product but Different Plucker!
# ================================================================

ax5 = fig.add_subplot(4, 2, (5, 6))
ax5.set_facecolor(colors['bg'])
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 5)
ax5.axis('off')
ax5.set_title('Key Insight: Same Dot Product (4), Completely Different Plucker!', color=colors['para'], fontsize=16, fontweight='bold')

# Pair A box
box_a = patches.FancyBboxPatch((0.3, 1.0), 4.2, 3.2,
    boxstyle="round,pad=0.3", facecolor='#2C3E50', edgecolor=colors['u'], lw=2)
ax5.add_patch(box_a)
ax5.text(2.4, 3.8, 'Pair A', color=colors['u'], fontsize=14, fontweight='bold', ha='center')
ax5.text(2.4, 3.2, 'u=[1,2,0]  v=[2,1,0]', color=colors['text'], fontsize=11, ha='center')
ax5.text(2.4, 2.5, 'dot = 4', color=colors['accent'], fontsize=13, fontweight='bold', ha='center')
ax5.text(2.4, 1.8, 'Plucker = [-3, 0, 0]', color=colors['para'], fontsize=13, fontweight='bold', ha='center')
ax5.text(2.4, 1.2, 'Parallelogram in the xy plane', color=colors['text'], fontsize=10, ha='center', alpha=0.7)

# Pair B box
box_b = patches.FancyBboxPatch((5.5, 1.0), 4.2, 3.2,
    boxstyle="round,pad=0.3", facecolor='#2C3E50', edgecolor=colors['v'], lw=2)
ax5.add_patch(box_b)
ax5.text(7.6, 3.8, 'Pair B', color=colors['v'], fontsize=14, fontweight='bold', ha='center')
ax5.text(7.6, 3.2, 'u=[1,0,2]  v=[2,0,1]', color=colors['text'], fontsize=11, ha='center')
ax5.text(7.6, 2.5, 'dot = 4', color=colors['accent'], fontsize=13, fontweight='bold', ha='center')
ax5.text(7.6, 1.8, 'Plucker = [0, -3, 0]', color=colors['para'], fontsize=13, fontweight='bold', ha='center')
ax5.text(7.6, 1.2, 'Parallelogram in the xz plane', color=colors['text'], fontsize=10, ha='center', alpha=0.7)

# Center comparison
ax5.text(5.0, 2.5, 'vs', color=colors['text'], fontsize=16, fontweight='bold',
         ha='center', va='center', alpha=0.5)
ax5.text(5.0, 0.5, 'Attention (dot product) gives "4" for both -> indistinguishable!\n'
         'Grassmann (Plucker) gives [-3,0,0] vs [0,-3,0] -> completely different relationships!',
         color=colors['text'], fontsize=11, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#34495E', edgecolor=colors['accent'], lw=1.5))


# ================================================================
#  Figure 6: Plucker = "Shadow Areas" on 3 Coordinate Planes
# ================================================================

ax6 = fig.add_subplot(4, 2, (7, 8))
ax6.set_facecolor(colors['bg'])
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 5.5)
ax6.axis('off')
ax6.set_title('Intuition for Plucker Coordinates: "3 Shadow Areas"', color=colors['text'], fontsize=16, fontweight='bold')

# Central explanation
ax6.text(5.0, 5.0,
    'Project the parallelogram formed by two 3D vectors\n'
    'onto each coordinate plane (xy, xz, yz) as "shadows".\n'
    'The signed area of each shadow = each component of the Plucker coordinate.',
    color=colors['text'], fontsize=12, ha='center', va='top',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#2C3E50', edgecolor=colors['accent'], lw=1.5))

# 3 shadow boxes
shadow_data = [
    ('p_01 (xy shadow)', 'u[0]*v[1] - u[1]*v[0]', 'Area of shadow on the floor', 1.2),
    ('p_02 (xz shadow)', 'u[0]*v[2] - u[2]*v[0]', 'Area of shadow on the front', 5.0),
    ('p_12 (yz shadow)', 'u[1]*v[2] - u[2]*v[1]', 'Area of shadow on the side', 8.8),
]

shadow_colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']

for (label, formula, desc, x), sc in zip(shadow_data, shadow_colors):
    box = patches.FancyBboxPatch((x-1.3, 0.3), 2.6, 2.8,
        boxstyle="round,pad=0.2", facecolor='#2C3E50', edgecolor=sc, lw=2)
    ax6.add_patch(box)
    ax6.text(x, 2.7, label, color=sc, fontsize=11, fontweight='bold', ha='center')
    ax6.text(x, 2.1, formula, color=colors['text'], fontsize=9, ha='center', family='monospace')
    ax6.text(x, 1.2, desc, color=colors['text'], fontsize=9, ha='center', alpha=0.7)

    # Small parallelogram icon
    if 'xy' in label:
        corners = np.array([[x-0.5, 0.5], [x+0.3, 0.5], [x+0.5, 0.9], [x-0.3, 0.9]])
    elif 'xz' in label:
        corners = np.array([[x-0.5, 0.5], [x+0.3, 0.7], [x+0.5, 1.0], [x-0.3, 0.8]])
    else:
        corners = np.array([[x-0.3, 0.5], [x+0.3, 0.5], [x+0.5, 1.0], [x-0.1, 1.0]])
    para_patch = plt.Polygon(corners, alpha=0.4, facecolor=sc, edgecolor=sc)
    ax6.add_patch(para_patch)

ax6.text(5.0, 0.05,
    'Plucker = [p_01, p_02, p_12] -> These 3 numbers fully determine "which direction the plane faces"',
    color=colors['accent'], fontsize=11, ha='center', fontweight='bold')


plt.tight_layout(pad=2.0)
plt.savefig('plucker_explained.png', dpi=150, facecolor=fig.get_facecolor(),
            bbox_inches='tight', pad_inches=0.5)
plt.close()

print("plucker_explained.png saved successfully!")
print()
print("=" * 60)
print("  Plucker Coordinates — One-Line Summary")
print("=" * 60)
print()
print("  Think of the parallelogram formed by two vectors u and v.")
print()
print("  [2D] Area of the parallelogram = a single number")
print("       p = u[0]*v[1] - u[1]*v[0]")
print()
print("  [3D] Project this parallelogram onto the floor/front/side,")
print("       and you get 3 shadows.")
print("       The area of each shadow = each component of the Plucker coordinate")
print("       p = [floor area, front area, side area]")
print()
print("  [4D] There are C(4,2)=6 shadows (4D has 6 coordinate planes)")
print("       p = [6 shadow areas]")
print()
print("  The dot product only tells you the 'angle',")
print("  but Plucker also tells you 'which plane' the vectors lie in.")
print("  -> Even with the same angle, different planes yield different Plucker values!")
