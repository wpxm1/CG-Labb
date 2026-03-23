import taichi as ti
import math

# 初始化 Taichi，使用 OpenGL 后端
ti.init(arch=ti.opengl)

# 立方体顶点数
NUM_VERTICES = 8
# 立方体边数
NUM_EDGES = 12

# 声明 Taichi 的 Field 来存储顶点和转换后的屏幕坐标
vertices = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTICES)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=NUM_VERTICES)

# 边的连接关系
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # 前面
    (4, 5), (5, 6), (6, 7), (7, 4),  # 后面
    (0, 4), (1, 5), (2, 6), (3, 7)   # 连接前后
]

# 边的颜色
edge_colors = [
    0xFF0000, 0xFF0000, 0xFF0000, 0xFF0000,  # 前面 - 红色
    0x00FF00, 0x00FF00, 0x00FF00, 0x00FF00,  # 后面 - 绿色
    0x0000FF, 0x0000FF, 0x0000FF, 0x0000FF   # 连接边 - 蓝色
]

@ti.func
def get_model_matrix(angle_x, angle_y):
    """
    模型变换矩阵：绕 X 轴和 Y 轴旋转
    """
    # 绕 X 轴旋转
    rad_x = angle_x * math.pi / 180.0
    cx = ti.cos(rad_x)
    sx = ti.sin(rad_x)
    rot_x = ti.Matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cx, -sx, 0.0],
        [0.0, sx, cx, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # 绕 Y 轴旋转
    rad_y = angle_y * math.pi / 180.0
    cy = ti.cos(rad_y)
    sy = ti.sin(rad_y)
    rot_y = ti.Matrix([
        [cy, 0.0, sy, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sy, 0.0, cy, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # 组合旋转：先绕 X 轴，再绕 Y 轴
    return rot_y @ rot_x

@ti.func
def get_view_matrix(eye_pos):
    """
    视图变换矩阵：将相机移动到原点
    """
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar):
    """
    透视投影矩阵
    """
    n = -zNear
    f = -zFar
    
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r
    
    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    M_ortho_scale = ti.Matrix([
        [2.0 / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (t - b), 0.0, 0.0],
        [0.0, 0.0, 2.0 / (n - f), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    M_ortho_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    M_ortho = M_ortho_scale @ M_ortho_trans
    
    return M_ortho @ M_p2o

@ti.kernel
def compute_transform(angle_x: ti.f32, angle_y: ti.f32):
    """
    在并行架构上计算顶点的坐标变换
    """
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    model = get_model_matrix(angle_x, angle_y)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    mvp = proj @ view @ model
    
    for i in range(NUM_VERTICES):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        
        v_ndc = v_clip / v_clip[3]
        
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

def init_cube():
    """
    初始化立方体的 8 个顶点
    """
    s = 1.0  # 半边长
    vertices[0] = [-s, -s, -s]  # 左下后
    vertices[1] = [ s, -s, -s]  # 右下后
    vertices[2] = [ s,  s, -s]  # 右上后
    vertices[3] = [-s,  s, -s]  # 左上后
    vertices[4] = [-s, -s,  s]  # 左下前
    vertices[5] = [ s, -s,  s]  # 右下前
    vertices[6] = [ s,  s,  s]  # 右上前
    vertices[7] = [-s,  s,  s]  # 左上前

def main():
    # 初始化立方体顶点
    init_cube()
    
    # 创建 GUI 窗口
    gui = ti.GUI("3D Cube Rotation", res=(700, 700))
    angle_x = 0.0
    angle_y = 0.0
    
    print("控制说明：")
    print("  A/D - 绕 Y 轴旋转")
    print("  W/S - 绕 X 轴旋转")
    print("  ESC - 退出")
    
    while gui.running:
        # 处理键盘事件
        for e in gui.get_events():
            if e.type == ti.GUI.PRESS:
                if e.key == 'a':
                    angle_y += 10.0
                elif e.key == 'd':
                    angle_y -= 10.0
                elif e.key == 'w':
                    angle_x += 10.0
                elif e.key == 's':
                    angle_x -= 10.0
                elif e.key == ti.GUI.ESCAPE:
                    gui.running = False
        
        # 计算变换
        compute_transform(angle_x, angle_y)
        
        # 绘制立方体的边（使用不同颜色）
        for i, edge in enumerate(edges):
            a, b = edge
            gui.line(screen_coords[a], screen_coords[b], radius=2, color=edge_colors[i])
        
        gui.show()

if __name__ == '__main__':
    main()