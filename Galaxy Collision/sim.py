import taichi as ti
import numpy as np
import math

# --- 1. Init & Guards ---
ENABLE_PROFILER = False  # set True to profile which kernel is expensive when fps drops
try:
    ti.init(arch=ti.gpu, kernel_profiler=ENABLE_PROFILER)
except Exception:
    ti.init(arch=ti.cpu, kernel_profiler=ENABLE_PROFILER)
    print("Running on CPU, expect low FPS. Reduce N_TOTAL / GRID_RES / POISSON_ITERS if too slow.")

# --- 2. Simulation & Astrophysics Constants ---
G = 1.0
M1_CORE = 200.0
M2_CORE = 60.0
R_VIR_1 = 150.0
R_VIR_2 = 80.0
C_NFW = 5.0
SOFTENING = 0.5

BG_COLOR = ti.Vector([0.0, 0.02, 0.06])

DISK_TO_CORE_MASS_RATIO = 1.0

GRID_RES = 128
GRID_SIZE = 400.0
INV_DX = GRID_RES / GRID_SIZE
CELL_DX = GRID_SIZE / GRID_RES
CELL_VOL = CELL_DX ** 3
PI = math.pi

POISSON_ITERS = 20
GRAVITY_UPDATE_EVERY = 2

GAMMA_EOS = 5.0 / 3.0
K_EOS = 0.03
PRESSURE_MIN_RHO = 0.05

RES_X = 1920
RES_Y = 1080

N_G1_STARS = 200_000
N_G1_GAS = 200_000
N_G2_STARS = 200_000
N_G2_GAS = 200_000
N_TOTAL = N_G1_STARS + N_G1_GAS + N_G2_STARS + N_G2_GAS

TYPE_OLD_STAR = 0
TYPE_GAS = 1
TYPE_YOUNG_STAR = 2
TYPE_HOT_GAS = 3

# --- 3. Taichi Fields ---
pos = ti.Vector.field(3, dtype=ti.f32, shape=N_TOTAL)
vel = ti.Vector.field(3, dtype=ti.f32, shape=N_TOTAL)
accel = ti.Vector.field(3, dtype=ti.f32, shape=N_TOTAL)
color = ti.Vector.field(3, dtype=ti.f32, shape=N_TOTAL)
ptype = ti.field(dtype=ti.i32, shape=N_TOTAL)
pmass = ti.field(dtype=ti.f32, shape=N_TOTAL)

gas_density = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES, GRID_RES))
total_density = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES, GRID_RES))
potential = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES, GRID_RES))
potential_tmp = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES, GRID_RES))
pressure = ti.field(dtype=ti.f32, shape=(GRID_RES, GRID_RES, GRID_RES))
grid_grav = ti.Vector.field(3, dtype=ti.f32, shape=(GRID_RES, GRID_RES, GRID_RES))
grid_press = ti.Vector.field(3, dtype=ti.f32, shape=(GRID_RES, GRID_RES, GRID_RES))

# pixels_out is the ping-pong buffer for the lensing pass
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))
pixels_out = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))

core_pos = ti.Vector.field(3, dtype=ti.f32, shape=2)
core_vel = ti.Vector.field(3, dtype=ti.f32, shape=2)
core_mass = ti.field(dtype=ti.f32, shape=2)

# Only values read INSIDE kernels live here (pause flag, mouse pull target).
# Time multiplier is a plain Python var in main() to avoid a per-frame host<->device sync.
sim_state = ti.field(dtype=ti.f32, shape=4)
cam_state = ti.field(dtype=ti.f32, shape=3)

learn_mode = ti.field(dtype=ti.i32, shape=())

# --- 4. Python Setup ---
def get_nfw_mass_enclosed(r, M_vir, R_vir, c):
    Rs = R_vir / c
    x = r / Rs
    term = np.log(1.0 + x) - x / (1.0 + x)
    term_vir = np.log(1.0 + c) - c / (1.0 + c)
    return M_vir * (term / term_vir)

def build_galaxy(n_stars, n_gas, core_mass_val, R_vir, pitch_angle_deg, num_arms, is_barred):
    n_total = n_stars + n_gas
    p = np.zeros((n_total, 3), dtype=np.float32)
    v = np.zeros((n_total, 3), dtype=np.float32)
    c = np.zeros((n_total, 3), dtype=np.float32)
    t = np.zeros(n_total, dtype=np.int32)

    r_max = R_vir * 0.4
    r_bar = r_max * 0.2 if is_barred else 0.0
    pitch = np.radians(pitch_angle_deg)
    b = np.tan(pitch)

    for i in range(n_total):
        rand = np.random.rand()
        if rand < 0.15:
            r = np.abs(np.random.normal(0, r_max * 0.05))
            theta = np.random.uniform(0, 2 * np.pi)
            z = np.random.normal(0, r_max * 0.05)
        elif is_barred and rand < 0.35:
            x_bar = np.random.uniform(-r_bar, r_bar)
            y_bar = np.random.normal(0, r_max * 0.02)
            r = np.sqrt(x_bar**2 + y_bar**2)
            theta = np.arctan2(y_bar, x_bar)
            z = np.random.normal(0, 0.5)
        else:
            r = np.random.exponential(r_max * 0.3)
            r = np.clip(r, r_bar, r_max)
            arm_offset = (np.random.randint(0, num_arms) * (2 * np.pi / num_arms))
            theta_spiral = np.log(r / (r_bar + 1.0)) / (b + 1e-3)
            theta = theta_spiral + arm_offset + np.random.normal(0, 0.2)
            z = np.random.normal(0, 1.0)

        p[i] = [r * np.cos(theta), r * np.sin(theta), z]

        m_enc_halo = get_nfw_mass_enclosed(r, core_mass_val * 5.0, R_vir, C_NFW)
        v_circ = np.sqrt(G * (core_mass_val + m_enc_halo) / max(r, SOFTENING))
        v[i] = [-v_circ * np.sin(theta), v_circ * np.cos(theta), 0.0]

        if i >= n_stars:
            t[i] = TYPE_GAS
            c[i] = [0.3, 0.15, 0.1]
        else:
            t[i] = TYPE_OLD_STAR
            if rand < 0.15: c[i] = [1.0, 0.8, 0.6]
            else: c[i] = [0.7, 0.6, 0.55]

    return p, v, c, t

def initialize_simulation():
    p1, v1, c1, t1 = build_galaxy(N_G1_STARS, N_G1_GAS, M1_CORE, R_VIR_1, 18.0, 4, True)
    p2, v2, c2, t2 = build_galaxy(N_G2_STARS, N_G2_GAS, M2_CORE, R_VIR_2, 6.0, 6, False)

    p1 += np.array([-60.0, -20.0, 0.0])
    v1 += np.array([0.6, 0.2, 0.0])

    p2 += np.array([120.0, 40.0, 0.0])
    v2 += np.array([-1.5, -0.4, 0.0])

    tilt_angle = np.radians(80)
    rot_mat = np.array([[1, 0, 0], [0, np.cos(tilt_angle), -np.sin(tilt_angle)], [0, np.sin(tilt_angle), np.cos(tilt_angle)]])
    p2_centered = p2 - np.array([120.0, 40.0, 0.0])
    p2 = p2_centered.dot(rot_mat.T) + np.array([120.0, 40.0, 0.0])
    v2 = v2.dot(rot_mat.T)

    all_p = np.vstack((p1, p2)).astype(np.float32)
    all_v = np.vstack((v1, v2)).astype(np.float32)
    all_c = np.vstack((c1, c2)).astype(np.float32)
    all_t = np.concatenate((t1, t2))

    n1 = N_G1_STARS + N_G1_GAS
    n2 = N_G2_STARS + N_G2_GAS
    m1_particle = (DISK_TO_CORE_MASS_RATIO * M1_CORE) / n1
    m2_particle = (DISK_TO_CORE_MASS_RATIO * M2_CORE) / n2
    all_m = np.concatenate((
        np.full(n1, m1_particle, dtype=np.float32),
        np.full(n2, m2_particle, dtype=np.float32),
    ))

    pos.from_numpy(all_p)
    vel.from_numpy(all_v)
    color.from_numpy(all_c)
    ptype.from_numpy(all_t)
    pmass.from_numpy(all_m)
    accel.fill(0.0)

    core_pos.from_numpy(np.array([[-60.0, -20.0, 0.0], [120.0, 40.0, 0.0]], dtype=np.float32))
    core_vel.from_numpy(np.array([[0.6, 0.2, 0.0], [-1.5, -0.4, 0.0]], dtype=np.float32))
    core_mass.from_numpy(np.array([M1_CORE, M2_CORE], dtype=np.float32))

    # index 0 is unused now (time multiplier moved to a Python variable)
    sim_state.from_numpy(np.array([20.0, 0.0, 0.0, 0.0], dtype=np.float32))
    cam_state.from_numpy(np.array([0.0, 0.6, 320.0], dtype=np.float32))
    learn_mode[None] = 0

    potential.fill(0.0)
    potential_tmp.fill(0.0)

# --- 5. Physics ---
@ti.func
def nfw_acceleration(p, core_idx, r_vir):
    r_vec = core_pos[core_idx] - p
    r_safe = ti.max(r_vec.norm(), SOFTENING)
    Rs = r_vir / C_NFW
    x = r_safe / Rs
    term = ti.math.log(1.0 + x) - x / (1.0 + x)
    term_vir = ti.math.log(1.0 + C_NFW) - C_NFW / (1.0 + C_NFW)
    m_enclosed = (core_mass[core_idx] * 5.0) * (term / term_vir)
    return (r_vec / r_safe) * ((G * m_enclosed) / (r_safe * r_safe))

@ti.func
def cell_of(p):
    gx = ti.cast((p.x + GRID_SIZE * 0.5) * INV_DX, ti.i32)
    gy = ti.cast((p.y + GRID_SIZE * 0.5) * INV_DX, ti.i32)
    gz = ti.cast((p.z + GRID_SIZE * 0.5) * INV_DX, ti.i32)
    return gx, gy, gz

@ti.kernel
def deposit_density():
    for i, j, k in total_density:
        total_density[i, j, k] = 0.0
        gas_density[i, j, k] = 0.0
    for i in range(N_TOTAL):
        p = pos[i]
        gx, gy, gz = cell_of(p)
        if 0 <= gx < GRID_RES and 0 <= gy < GRID_RES and 0 <= gz < GRID_RES:
            ti.atomic_add(total_density[gx, gy, gz], pmass[i])
            if ptype[i] == TYPE_GAS or ptype[i] == TYPE_HOT_GAS:
                ti.atomic_add(gas_density[gx, gy, gz], 1.0)

@ti.kernel
def jacobi_sweep(src: ti.template(), dst: ti.template()):
    for i, j, k in dst:
        if 1 <= i < GRID_RES - 1 and 1 <= j < GRID_RES - 1 and 1 <= k < GRID_RES - 1:
            rho = total_density[i, j, k] / CELL_VOL
            neighbors = (src[i + 1, j, k] + src[i - 1, j, k] +
                         src[i, j + 1, k] + src[i, j - 1, k] +
                         src[i, j, k + 1] + src[i, j, k - 1])
            dst[i, j, k] = (neighbors - 4.0 * PI * G * rho * CELL_DX * CELL_DX) / 6.0
        else:
            dst[i, j, k] = 0.0

def solve_poisson():
    for _ in range(POISSON_ITERS // 2):
        jacobi_sweep(potential, potential_tmp)
        jacobi_sweep(potential_tmp, potential)

@ti.kernel
def compute_grid_fields():
    for i, j, k in pressure:
        pressure[i, j, k] = K_EOS * ti.math.pow(ti.max(gas_density[i, j, k], 0.0), GAMMA_EOS)
    for i, j, k in grid_grav:
        if 1 <= i < GRID_RES - 1 and 1 <= j < GRID_RES - 1 and 1 <= k < GRID_RES - 1:
            gphi_x = (potential[i + 1, j, k] - potential[i - 1, j, k]) / (2.0 * CELL_DX)
            gphi_y = (potential[i, j + 1, k] - potential[i, j - 1, k]) / (2.0 * CELL_DX)
            gphi_z = (potential[i, j, k + 1] - potential[i, j, k - 1]) / (2.0 * CELL_DX)
            grid_grav[i, j, k] = ti.Vector([-gphi_x, -gphi_y, -gphi_z])

            rho = ti.max(gas_density[i, j, k], PRESSURE_MIN_RHO)
            gp_x = (pressure[i + 1, j, k] - pressure[i - 1, j, k]) / (2.0 * CELL_DX)
            gp_y = (pressure[i, j + 1, k] - pressure[i, j - 1, k]) / (2.0 * CELL_DX)
            gp_z = (pressure[i, j, k + 1] - pressure[i, j, k - 1]) / (2.0 * CELL_DX)
            grid_press[i, j, k] = ti.Vector([-gp_x, -gp_y, -gp_z]) / rho
        else:
            grid_grav[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            grid_press[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def compute_particle_accel():
    for i in range(N_TOTAL):
        p, my_type = pos[i], ptype[i]

        acc = ti.Vector([0.0, 0.0, 0.0])
        for c in ti.static(range(2)):
            r_vec = core_pos[c] - p
            r_sq = r_vec.norm_sqr() + SOFTENING**2
            acc += (G * core_mass[c] / r_sq) * (r_vec / ti.sqrt(r_sq))
        acc += nfw_acceleration(p, 0, R_VIR_1)
        acc += nfw_acceleration(p, 1, R_VIR_2)

        gx, gy, gz = cell_of(p)
        if 0 <= gx < GRID_RES and 0 <= gy < GRID_RES and 0 <= gz < GRID_RES:
            acc += grid_grav[gx, gy, gz]

            if my_type == TYPE_GAS or my_type == TYPE_HOT_GAS:
                acc += grid_press[gx, gy, gz]

                den = gas_density[gx, gy, gz]
                if den > 5.0 and my_type == TYPE_GAS:
                    ptype[i] = TYPE_HOT_GAS
                    color[i] = ti.Vector([0.8, 0.2, 0.8])
                if den > 15.0 and ti.random() < 0.02:
                    ptype[i] = TYPE_YOUNG_STAR
                    color[i] = ti.Vector([0.3, 0.8, 1.0])

        if sim_state[2] != 0.0 or sim_state[3] != 0.0:
            com = (core_pos[0] * core_mass[0] + core_pos[1] * core_mass[1]) / (core_mass[0] + core_mass[1])
            target = ti.Vector([com.x + sim_state[2] * 150.0, com.y + sim_state[3] * 150.0, 0.0])
            pull_vec = target - p
            pull_dist = ti.max(pull_vec.norm(), 1.0)
            acc += (pull_vec / pull_dist) * 80.0 / pull_dist

        accel[i] = acc

@ti.kernel
def kick(half_dt: ti.f32):
    if sim_state[1] < 0.5:
        for i in range(N_TOTAL):
            vel[i] += accel[i] * half_dt

@ti.kernel
def drift(dt: ti.f32):
    if sim_state[1] < 0.5:
        for i in range(N_TOTAL):
            p, v, my_type = pos[i], vel[i], ptype[i]
            p += v * dt

            if my_type == TYPE_GAS or my_type == TYPE_HOT_GAS:
                v *= 0.999

            if (p - core_pos[0]).norm() > 700.0:
                p = core_pos[0] + ti.Vector([ti.random() - 0.5, ti.random() - 0.5, ti.random() - 0.5]) * 150.0
                v = core_vel[0]
                if my_type == TYPE_YOUNG_STAR or my_type == TYPE_HOT_GAS:
                    ptype[i] = TYPE_GAS
                    color[i] = ti.Vector([0.3, 0.15, 0.1])

            vel[i], pos[i] = v, p

@ti.kernel
def kick_cores(half_dt: ti.f32):
    if sim_state[1] < 0.5:
        r_vec = core_pos[1] - core_pos[0]
        r_sq = r_vec.norm_sqr() + SOFTENING**2
        f_vec = (r_vec / ti.sqrt(r_sq)) * ((G * core_mass[0] * core_mass[1]) / r_sq)
        core_vel[0] += (f_vec / core_mass[0]) * half_dt
        core_vel[1] -= (f_vec / core_mass[1]) * half_dt

@ti.kernel
def drift_cores(dt: ti.f32):
    if sim_state[1] < 0.5:
        core_pos[0] += core_vel[0] * dt
        core_pos[1] += core_vel[1] * dt

def step_simulation(dt, time_mult, frame_idx):
    """Runs the PM gravity solve every GRAVITY_UPDATE_EVERY frames (multi-stepping).
    On skipped frames, compute_particle_accel reuses the last grid_grav/grid_press."""
    t_dt = dt * time_mult
    half = t_dt * 0.5

    kick_cores(half)
    kick(half)

    drift_cores(t_dt)
    drift(t_dt)

    if frame_idx % GRAVITY_UPDATE_EVERY == 0:
        deposit_density()
        solve_poisson()
        compute_grid_fields()

    compute_particle_accel()

    kick_cores(half)
    kick(half)

# --- 6. 3D Engine & Post-Processing ---
@ti.kernel
def render_splat():
    for i, j in pixels:
        pixels[i, j] = BG_COLOR

    yaw = cam_state[0]
    pitch = cam_state[1]
    dist = cam_state[2]

    cos_y = ti.math.cos(yaw)
    sin_y = ti.math.sin(yaw)
    cos_p = ti.math.cos(pitch)
    sin_p = ti.math.sin(pitch)

    com = (core_pos[0] * core_mass[0] + core_pos[1] * core_mass[1]) / (core_mass[0] + core_mass[1])

    for i in range(N_TOTAL):
        p = pos[i] - com
        c = color[i]
        t = ptype[i]

        x1 = p.x * cos_y - p.z * sin_y
        z1 = p.x * sin_y + p.z * cos_y
        y1 = p.y

        y2 = y1 * cos_p - z1 * sin_p
        z2 = y1 * sin_p + z1 * cos_p
        x2 = x1

        z_dist = dist - z2
        if z_dist > 10.0:
            fov = 850.0
            screen_x = (x2 / z_dist) * fov + RES_X * 0.5
            screen_y = (y2 / z_dist) * fov + RES_Y * 0.5

            ix = ti.cast(screen_x, ti.i32)
            iy = ti.cast(screen_y, ti.i32)

            r = 2
            intensity = 0.05

            if t == TYPE_GAS or t == TYPE_HOT_GAS:
                intensity = 0.02
                r = 3
            elif t == TYPE_YOUNG_STAR:
                intensity = 0.12
                r = 1

            if learn_mode[None] == 1:
                # heatmap: red = high acceleration, blue = low
                acc_norm = accel[i].norm()
                heat = ti.math.min(acc_norm * 3.0, 1.0)
                c = ti.Vector([heat, 0.1, 1.0 - heat])
                intensity = 0.06

            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx = ix + dx
                    ny = iy + dy
                    if 0 <= nx < RES_X and 0 <= ny < RES_Y:
                        dist_sq = ti.cast(dx * dx + dy * dy, ti.f32)
                        max_dist_sq = ti.cast(r * r + 1, ti.f32)
                        if dist_sq < max_dist_sq:
                            weight = ti.math.exp(-dist_sq / 1.5)
                            ti.atomic_add(pixels[nx, ny], c * intensity * weight)

@ti.kernel
def apply_lensing():
    for i, j in pixels_out:
        pixels_out[i, j] = BG_COLOR

    yaw = cam_state[0]
    pitch = cam_state[1]
    dist = cam_state[2]

    cos_y = ti.math.cos(yaw)
    sin_y = ti.math.sin(yaw)
    cos_p = ti.math.cos(pitch)
    sin_p = ti.math.sin(pitch)

    com = (core_pos[0] * core_mass[0] + core_pos[1] * core_mass[1]) / (core_mass[0] + core_mass[1])

    # project both cores to screen space
    cx = ti.Vector([-1.0, -1.0])
    cy = ti.Vector([-1.0, -1.0])

    for c_idx in ti.static(range(2)):
        p = core_pos[c_idx] - com
        x1 = p.x * cos_y - p.z * sin_y
        z1 = p.x * sin_y + p.z * cos_y
        y1 = p.y
        y2 = y1 * cos_p - z1 * sin_p
        z2 = y1 * sin_p + z1 * cos_p
        x2 = x1
        z_dist = dist - z2
        if z_dist > 10.0:
            fov = 850.0
            cx[c_idx] = (x2 / z_dist) * fov + RES_X * 0.5
            cy[c_idx] = (y2 / z_dist) * fov + RES_Y * 0.5

    lensing_strength = 50.0  # distortion strength

    for i, j in pixels:
        shift_x = 0.0
        shift_y = 0.0

        for c_idx in ti.static(range(2)):
            if cx[c_idx] > 0.0:
                dx = ti.cast(i, ti.f32) - cx[c_idx]
                dy = ti.cast(j, ti.f32) - cy[c_idx]
                dist_sq = dx**2 + dy**2

                if dist_sq > 25.0:  # avoid singularity at the exact center
                    f = lensing_strength * (core_mass[c_idx] / M1_CORE) / dist_sq
                    shift_x += dx * f
                    shift_y += dy * f

        src_x = ti.cast(i + shift_x, ti.i32)
        src_y = ti.cast(j + shift_y, ti.i32)

        if 0 <= src_x < RES_X and 0 <= src_y < RES_Y:
            pixels_out[i, j] = pixels[src_x, src_y]
        else:
            pixels_out[i, j] = BG_COLOR

    for i, j in pixels:
        pixels[i, j] = pixels_out[i, j]

@ti.kernel
def tonemap():
    for i, j in pixels:
        c = pixels[i, j]
        c = 1.0 - ti.math.exp(-c * 0.9)
        pixels[i, j] = ti.math.pow(c, 1.0 / 1.2)

# --- 7. Main Loop ---
def main():
    initialize_simulation()

    deposit_density()
    solve_poisson()
    compute_grid_fields()
    compute_particle_accel()

    print("\n" + "=" * 50)
    print(" SELF-GRAVITY + LENSING SIMULATION")
    print("=" * 50)
    print(" CONTROLS:")
    print("   [LClick] : Click and drag to ORBIT the camera")
    print("   [W / S]  : Zoom In / Zoom Out")
    print("   [RClick] : Pull particles (on-screen gravity)")
    print("   [SPACE]  : Pause / Resume")
    print("   [UP/DWN] : Time Control")
    print("   [L]      : Toggle Learn Mode (acceleration heatmap)")
    print("=" * 50 + "\n")

    window = ti.ui.Window("Cosmic Encounter - Self-Gravity + Lensing", (RES_X, RES_Y), vsync=True)
    canvas = window.get_canvas()

    dt = 0.015
    last_mouse_pos = window.get_cursor_pos()

    # time_mult/paused are plain Python vars (not read from sim_state every frame)
    # to avoid a host<->device sync each frame; sim_state[1] is only WRITTEN so
    # kernels (kick/drift/etc.) still see the pause state.
    time_mult = 20.0
    paused = False
    frame_count = 0

    while window.running:
        current_mouse_pos = window.get_cursor_pos()

        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.SPACE:
                paused = not paused
                sim_state[1] = 1.0 if paused else 0.0
            elif e.key == ti.ui.UP:
                time_mult = min(time_mult + 0.5, 5.0)
            elif e.key == ti.ui.DOWN:
                time_mult = max(time_mult - 0.5, 0.1)
            elif e.key == 'r':
                initialize_simulation()
                time_mult = 20.0
                paused = False
                frame_count = 0
            elif e.key == 'l' or e.key == 'L':
                learn_mode[None] = 1 - learn_mode[None]

        if window.is_pressed('w'): cam_state[2] = max(cam_state[2] - 5.0, 50.0)
        if window.is_pressed('s'): cam_state[2] += 5.0

        if window.is_pressed(ti.ui.LMB):
            dx = current_mouse_pos[0] - last_mouse_pos[0]
            dy = current_mouse_pos[1] - last_mouse_pos[1]
            cam_state[0] += dx * 4.0
            cam_state[1] -= dy * 4.0
            cam_state[1] = max(-1.5, min(1.5, cam_state[1]))

        if window.is_pressed(ti.ui.RMB):
            sim_state[2] = (current_mouse_pos[0] - 0.5) * 2.0
            sim_state[3] = (current_mouse_pos[1] - 0.5) * 2.0
        else:
            sim_state[2], sim_state[3] = 0.0, 0.0

        last_mouse_pos = current_mouse_pos

        step_simulation(dt, time_mult, frame_count)
        render_splat()
        apply_lensing()
        tonemap()
        canvas.set_image(pixels)

        if ENABLE_PROFILER and frame_count > 0 and frame_count % 120 == 0:
            ti.profiler.print_kernel_profiler_info()
            ti.profiler.clear_kernel_profiler_info()

        gui = window.get_gui()

        ui_height = 0.40 if learn_mode[None] else 0.25
        gui.begin("Telemetry & Controls", 0.02, 0.02, 0.35, ui_height)

        gui.text("Rotate: [Left Click]")
        gui.text("Zoom: [W / S Keys]")
        gui.text(f"Time Mult: {time_mult:.1f}x")
        gui.text("Status: " + ("PAUSED" if paused else "RUNNING"))
        gui.text(f"Learn Mode [L]: {'ON' if learn_mode[None] else 'OFF'}")

        if learn_mode[None] == 1:
            gui.text("-" * 35)
            gui.text("REAL-TIME TECHNICAL DATA:")
            gui.text(f"- Poisson Grid: {GRID_RES}x{GRID_RES}x{GRID_RES}")
            gui.text(f"- Total Particles: {N_TOTAL}")
            gui.text(f"- Grid update: every {GRAVITY_UPDATE_EVERY} frames")
            gui.text("- Color Map:")
            gui.text("  > Red: High Acceleration (Core)")
            gui.text("  > Blue: Low Acceleration (Edge)")
            gui.text("- Gravitational Lensing: Active")

        gui.end()
        window.show()

        frame_count += 1

if __name__ == "__main__":
    main()