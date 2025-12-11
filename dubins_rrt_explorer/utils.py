import math
import numpy as np

# TODO: Add references

def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

class KinematicModel:
    def __init__(self, dt=0.1):
        self.dt = dt

    def get_matrices(self, theta, v_ref):
        A = np.eye(3)
        A[0, 2] = -v_ref * math.sin(theta) * self.dt
        A[1, 2] =  v_ref * math.cos(theta) * self.dt

        B = np.zeros((3, 2))
        B[0, 0] = math.cos(theta) * self.dt
        B[1, 0] = math.sin(theta) * self.dt
        B[2, 1] = self.dt

        return A, B

class DubinsSolver:
    def __init__(self, step_size=0.05):
        self.step_size = step_size 

    def get_shortest_path(self, start, end, radius):

        sx, sy, syaw = start
        gx, gy, gyaw = end

        dx = gx - sx
        dy = gy - sy
        D = math.hypot(dx, dy)
        d = D / radius

        theta = math.atan2(dy, dx) % (2 * math.pi)
        alpha = (syaw - theta) % (2 * math.pi)
        beta  = (gyaw - theta) % (2 * math.pi)

        best_cost = float('inf')
        best_mode = None
        best_params = None

        planners = [self._LSL, self._RSR, self._LSR, self._RSL, self._RLR, self._LRL]
        modes    = ["LSL", "RSR", "LSR", "RSL", "RLR", "LRL"]

        for planner, mode in zip(planners, modes):
            t, p, q = planner(alpha, beta, d)
            if t is None: continue
            
            cost = (abs(t) + abs(p) + abs(q)) * radius
            if cost < best_cost:
                best_cost = cost
                best_mode = mode
                best_params = (t, p, q)

        if best_mode:
            points = self._generate_points(start, best_mode, best_params, radius)
            return points, best_cost
        
        return None, None

    def _generate_points(self, start, mode, params, r):
        t, p, q = params
        points = []
        x, y, yaw = start
        
        operations = []
        if mode == "LSL": operations = [('L', t), ('S', p), ('L', q)]
        if mode == "RSR": operations = [('R', t), ('S', p), ('R', q)]
        if mode == "LSR": operations = [('L', t), ('S', p), ('R', q)]
        if mode == "RSL": operations = [('R', t), ('S', p), ('L', q)]
        if mode == "RLR": operations = [('R', t), ('L', p), ('R', q)]
        if mode == "LRL": operations = [('L', t), ('R', p), ('L', q)]

        points.append([x, y, yaw])

        for gear, length in operations:
            length * r if gear == 'S' else length 

            current_seg_len = 0.0
            seg_target = length 
            
            while current_seg_len < seg_target:
                 inc = min(self.step_size / r, seg_target - current_seg_len)
                 
                 x, y, yaw = self._step_dubins(x, y, yaw, gear, inc, r)
                 points.append([x, y, yaw])
                 current_seg_len += inc
                 
        return points

    def _step_dubins(self, x, y, yaw, gear, length, r):
        if gear == 'S':
            x += length * r * math.cos(yaw)
            y += length * r * math.sin(yaw)
            return x, y, yaw
        else:
            d_theta = length 
            if gear == 'R': d_theta = -length
            
            x += r * (math.sin(yaw + d_theta) - math.sin(yaw)) if gear == 'L' else r * (math.sin(yaw) - math.sin(yaw + d_theta)) # R

            if gear == 'L':
                x += r * math.sin(yaw + length) - r * math.sin(yaw)
                y += -r * math.cos(yaw + length) + r * math.cos(yaw)
                yaw += length
            elif gear == 'R':
                x += -r * math.sin(yaw - length) + r * math.sin(yaw)
                y += r * math.cos(yaw - length) - r * math.cos(yaw)
                yaw -= length
            
            return x, y, yaw

    def _mod2pi(self, theta):
        return theta % (2 * math.pi)

    def _LSL(self, alpha, beta, d):
        sa, sb = math.sin(alpha), math.sin(beta)
        ca, cb = math.cos(alpha), math.cos(beta)
        c_ab = math.cos(alpha - beta)
        tmp0 = d + sa - sb
        p_sq = 2 + d*d - 2*c_ab + 2*d*(sa - sb)
        if p_sq < 0: return None, None, None
        tmp1 = math.atan2(cb - ca, tmp0)
        t = self._mod2pi(-alpha + tmp1)
        p = math.sqrt(p_sq)
        q = self._mod2pi(beta - tmp1)
        return t, p, q

    def _RSR(self, alpha, beta, d):
        sa, sb = math.sin(alpha), math.sin(beta)
        ca, cb = math.cos(alpha), math.cos(beta)
        c_ab = math.cos(alpha - beta)
        tmp0 = d - sa + sb
        p_sq = 2 + d*d - 2*c_ab + 2*d*(sb - sa)
        if p_sq < 0: return None, None, None
        tmp1 = math.atan2(ca - cb, tmp0)
        t = self._mod2pi(alpha - tmp1)
        p = math.sqrt(p_sq)
        q = self._mod2pi(-beta + tmp1)
        return t, p, q

    def _LSR(self, alpha, beta, d):
        sa, sb = math.sin(alpha), math.sin(beta)
        ca, cb = math.cos(alpha), math.cos(beta)
        c_ab = math.cos(alpha - beta)
        p_sq = -2 + d*d + 2*c_ab + 2*d*(sa + sb)
        if p_sq < 0: return None, None, None
        p = math.sqrt(p_sq)
        tmp2 = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
        t = self._mod2pi(-alpha + tmp2)
        q = self._mod2pi(-self._mod2pi(beta) + tmp2)
        return t, p, q

    def _RSL(self, alpha, beta, d):
        sa, sb = math.sin(alpha), math.sin(beta)
        ca, cb = math.cos(alpha), math.cos(beta)
        c_ab = math.cos(alpha - beta)
        p_sq = -2 + d*d + 2*c_ab - 2*d*(sa + sb)
        if p_sq < 0: return None, None, None
        p = math.sqrt(p_sq)
        tmp2 = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
        t = self._mod2pi(alpha - tmp2)
        q = self._mod2pi(beta - tmp2)
        return t, p, q

    def _RLR(self, alpha, beta, d):
        sa, sb = math.sin(alpha), math.sin(beta)
        ca, cb = math.cos(alpha), math.cos(beta)
        c_ab = math.cos(alpha - beta)
        tmp = (6.0 - d*d + 2*c_ab + 2*d*(sa - sb)) / 8.0
        if abs(tmp) > 1.0: return None, None, None
        p = self._mod2pi(2*math.pi - math.acos(tmp))
        t = self._mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + self._mod2pi(p/2.0))
        q = self._mod2pi(alpha - beta - t + self._mod2pi(p))
        return t, p, q

    def _LRL(self, alpha, beta, d):
        sa, sb = math.sin(alpha), math.sin(beta)
        ca, cb = math.cos(alpha), math.cos(beta)
        c_ab = math.cos(alpha - beta)
        tmp = (6.0 - d*d + 2*c_ab + 2*d*(-sa + sb)) / 8.0
        if abs(tmp) > 1.0: return None, None, None
        p = self._mod2pi(2*math.pi - math.acos(tmp))
        t = self._mod2pi(-alpha + math.atan2(-ca + cb, d + sa - sb) + p/2.0)
        q = self._mod2pi(self._mod2pi(beta) - alpha - t + self._mod2pi(p))
        return t, p, q
