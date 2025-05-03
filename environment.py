# Environment
import cmath

import numpy as np
import math
import random

# np.random.seed(1)
np.random.seed()


class Env():  # class 선언

    # for문으로 C_t=[[random.randint(0,9)]]를 30개 elements를 갖도록 짜주자.
    # episode를 더 갖게 하기 위해?

    def __init__(self):  # 초기 환경 설정

        self.max_episode = 300
        self.chunk_size = 299

        self.time = 0  # 시간
        self.episode = 0  # 에피소드
        self.Q_t = 0  # 큐
        self.n_t = 0  # 보낸 거

        self.LM = 0
        self.LK = 0

        # self.k_q = 1.
        self.k_q = 3
        # self.k_p = 10.
        self.k_p = 20
        self.k_l = 1

        self.bandwidth = 5  # 5e06 # bandwidth = 5MHz
        self.SNR = 1e4  # Transmit SNR = 40dB = 10*log2(10000)
        self.DistanceTtoR = 50  # distance between Tx and Rx = 50 m
        self.PathLossExponent = 2.5  # path loss exponent beta = 2.5 장애물이 있는 외부 무선환경 가정한 Loss
        self.FastFadingGain = np.random.normal(0, 1)
        self.ShadowingEffect = np.random.normal(0, 1)
        self.channelGain = cmath.sqrt(
            self.ShadowingEffect / ((self.DistanceTtoR) ** self.PathLossExponent)) * self.FastFadingGain
        self.ChannelCapacitance = self.bandwidth * math.log2(
            1 + self.SNR * abs(self.channelGain) * abs(self.channelGain))
        # channel Capacitancy의 단위는 Mbps임. 우리는chunk하나가 2초니까 단위 잘 봐야함
        self.C_t = self.ChannelCapacitance

        '''
        for i in range(self.max_episode):
            C_ep=[]
            for j in range(40):
                #self.C_t[i][j]=random.randint(0,9)
                C_ep.append(random.randint(0,9))
            self.C_t.append(C_ep)
        '''

    def reset(self):  # 환경을 reset하는 부분 초기화된 state를 return 해야함

        # self.update()
        self.episode = self.episode + 1  # reset 한다는 것은 에피소드가 끝나는 것(그래서 +1)
        self.time = 0  # 시간 reset
        self.Q_t = 0  # 큐 reset
        self.n_t = 0  # 보낸 것 reset

        return [0, 0]

    def step(self, action):  # action을 입력으로 받음. (m,k를 받음)
        # 출력은 next_state, reward, done, self.episode, self.time
        S_K = 0
        size = 0
        if action[1] == 1:
            S_K = 0.6
            size = 1.2
        elif action[1] == 2:
            S_K = 0.99
            size = 1.98
        elif action[1] == 3:
            S_K = 1.5
            size = 3
        elif action[1] == 4:
            S_K = 2.075
            size = 4.15

        print("C_t : ", self.C_t)
        action_M = 0

        # C 보다 큰 양을 보내려고 할 때
        if action[0] * size > self.C_t * 2:
            for i in range(9):
                if size * (i + 1) > self.C_t * 2:
                    self.LM = action[0] - i
                    action_M = i
                    self.LK = size
                    print("빠진 M : ", self.LM)
                    break
            if self.time == 0:
                reward = 0
            elif (self.Q_t == 0) & (self.time != 0):  # action을 action[0]==M_t, action[1]==K_t인 list로 받는다고 했을 때.
                reward = self.k_q * math.sqrt(action[0]) * S_K / 2 - self.k_p - self.k_l * max(self.LM * self.LK, 0)
            else:
                reward = self.k_q * math.sqrt(action[0]) * S_K / 2 + min(self.Q_t - self.k_p, -self.k_p / self.Q_t) - self.k_l * max(
                    self.LM * self.LK, 0)

            self.Q_t = max(self.Q_t - 1, 0) + action_M
            self.n_t = self.n_t + action_M

            # done 결정
            if self.n_t < self.chunk_size:
                done = False
            else:
                done = True
            print("DONE : ", done)

            next_state = [self.Q_t, self.n_t]  # nex_state에 update된 q, n을 넣어줌(사실상 Qt+1, Nt+1)

            self.time = self.time + 1  # environment랑 agent랑 time, episode정보를 공유하고 있지 않는 상태임 지금은. 그래서 각각 update하는 코드 넣어뒀음

            self.FastFadingGain = np.random.normal(0, 1)
            self.ShadowingEffect = np.random.normal(0, 1)
            self.channelGain = cmath.sqrt(
                self.ShadowingEffect / (self.DistanceTtoR ** self.PathLossExponent)) * self.FastFadingGain
            self.ChannelCapacitance = self.bandwidth * math.log2(
                1 + self.SNR * abs(self.channelGain) * abs(self.channelGain))
            # channel Capacitancy의 단위는 Mbps임. 우리는chunk하나가 2초니까 단위 잘 봐야함
            self.C_t = self.ChannelCapacitance

            return next_state, reward, done, self.episode, self.time, action[0], S_K / 2, self.LM, self.LK

        else:
            self.LM = 0
            self.LK = 0

            if self.time == 0:
                reward = 0
            elif (self.Q_t == 0) & (self.time != 0):  # action을 action[0]==M_t, action[1]==K_t인 list로 받는다고 했을 때.
                reward = self.k_q * math.sqrt(action[0]) * S_K / 2 - self.k_p
            else:
                reward = self.k_q * math.sqrt(action[0]) * S_K / 2 + min(self.Q_t - self.k_p, -self.k_p / self.Q_t)

            self.Q_t = max(self.Q_t - 1, 0) + action[0]
            self.n_t = self.n_t + action[0]

            # done 결정
            if self.n_t < self.chunk_size:
                done = False
            else:
                done = True
            print("DONE : ", done)

            next_state = [self.Q_t, self.n_t]  # nex_state에 update된 q, n을 넣어줌(사실상 Qt+1, Nt+1)

            self.time = self.time + 1  # environment랑 agent랑 time, episode정보를 공유하고 있지 않는 상태임 지금은. 그래서 각각 update하는 코드 넣어뒀음

            self.FastFadingGain = np.random.normal(0, 1)
            self.ShadowingEffect = np.random.normal(0, 1)
            self.channelGain = cmath.sqrt(
                self.ShadowingEffect / ((self.DistanceTtoR) ** self.PathLossExponent)) * self.FastFadingGain
            self.ChannelCapacitancy = self.bandwidth * math.log2(
                1 + self.SNR * abs(self.channelGain) * abs(self.channelGain))
            # channel Capacitancy의 단위는 bps임. 우리는chunk하나가 2초니까 단위 잘 봐야함
            self.C_t = self.ChannelCapacitancy

            return next_state, reward, done, self.episode, self.time, action[0], S_K / 2, self.LM, self.LK

    # 보상 함수
    # if문 써서 Q(t)가 0,1,2이상인 경우로 나눠서 reward값을 주기. n이 9냐 아니냐로 done에 T,F를 할당.

    # reward를 계산한 후에, next_state Q,n을 계산해준다.

    # C(t)도 리턴을 해줘야 한다. C=[상수들]로 episode처음에 할당해주면 된다.
