import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import tensorflow as tf

RAD2DEG = 180 / np.pi
DEG2RAD = np.pi / 180

def solve_rk45():
    def damped_pendulum(t, y, m, l, b, g):
        """
        감쇠 펜듈럼의 운동방정식을 정의하는 함수

        Parameters:
        - t: 시간
        - y: 상태 변수 [각도, 각 속도]
        - m: 질량
        - l: 길이
        - b: 감쇠 상수
        - g: 중력 가속도

        Returns:
        - dydt: 상태 변수의 변화율 [각 속도, 각 가속도]
        """
        theta, omega = y
        dydt = [omega, -b / m * omega - g / l * np.sin(theta)]
        return dydt

    # 초기 조건
    initial_conditions = [np.radians(10), 0.0]  # 초기 각도 10도, 초기 각 속도 0.0 rad/s

    # 물리적 매개변수
    m = 1.0  # 질량 (kg)
    l = 1.0  # 길이 (m)
    b = 0.1  # 감쇠 상수 (kg*m/s)
    g = 9.8  # 중력 가속도 (m/s^2)

    # 시뮬레이션 시간 범위
    t_span = (0, 10)

    # 운동방정식을 푸는 함수 호출
    solution = solve_ivp(damped_pendulum, t_span, initial_conditions, args=(m, l, b, g), t_eval=np.linspace(0, 10, 1000))

    # 결과 플로팅
    plt.plot(solution.t, np.degrees(solution.y[0]), label='각도 (degrees)')
    plt.plot(solution.t, np.degrees(solution.y[1]), label='각 속도 (degrees/s)')
    plt.title('감쇠 펜듈럼의 운동')
    plt.xlabel('시간 (s)')
    plt.legend()
    plt.show()

class Pendulum(object):
    def __init__(self):
        self.m = 1/(9.81)**2    #kg
        self.l = 9.81           #m
        self.g = 9.81           #m/s^2
        self.b = 1.5            #kg.m/s
        
        self.range_x = {
            'x1':{
                'min':-np.pi,
                'max':np.pi,
            },
            'x2':{
                'min':-2.,
                'max':2.,
            }
        }
        self.num_x = len(self.range_x)
        
    def solve(self, x):
        '''x : (batch, 2)
        '''
        x1_dot = x[:, 1:2]  # (batch, 1)
        x2_dot = -self.b / self.m * x[:, 1:2] - self.g / self.l * tf.math.sin(x[:, 0:1])
        # x_dot = np.concatenate((x1_dot, x2_dot), axis=-1)
        x_dot = tf.concat([x1_dot, x2_dot], axis=-1)
        
        return x_dot
        
if __name__ == "__main__":
    # pendulum = Pendulum()
    
    # print(np.sin(90*DEG2RAD))
    solve_rk45()
    
    # print(pendulum.solve(np.array([[0, 0]])))