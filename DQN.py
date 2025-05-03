# agent
import numpy as np
# import os
# import sys
import random
import copy
# from collections import defaultdict
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform

from environment import Env

import matplotlib.pyplot as plt


class DQN(tf.keras.Model):  # 딥러닝 모델 제작
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc1 = Dense(24, activation='relu')  # input size 고려해주기
        self.fc2 = Dense(24, activation='relu')
        self.fc_out = Dense(action_size, kernel_initializer=RandomUniform(-1e-3, 1e-3))
        # https://keras.io/ko/initializers/
        # kernel_initializer, bias_initiallizer가 있는데 각각 weight, bias의 초기값을 설정하는 flag임
        # -0.001~0.001로 설정해준 이유는 reduce variance를 위해서임.
        # weight가 update되더라도, 저범위에서 시작하면 weight가 작은 값이 나올거고
        # 그러면 Q의 범위가 작아지니까.
        # M = 1~9, K=1~3으로 3*9=27 + (0,0)으로 action_size=28이다.

    def call(self, x):
        # DL에 입력 넣어주려면 최소 [[sample1],[sample2], ---] 형태로 2차원이여야함.
        # 그래서 원래 [M,K]로 했던 state를 [[M,K],[M,K],---]형태로 바꿔줘야함.
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.fc_out(x)
        return q


class DQNAgent:

    def __init__(self, state_size, action_size):
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # action은 M,K형태고 1~9,1~4 + (0,0)으로 37개[(,),(,)]
        self.action_list = []
        self.action_list.append([0, 0])

        for M in range(1, 10):
            for K in range(1, 5):
                self.action_list.append([M, K])

        # 하이퍼 파라미터 설정
        self.learning_rate = 0.001
        self.discount_factor = 0.8  # 임의로 설정한 discount_factor
        self.epsilon = 1.0  # 초기 epsilon 값
        self.epsilon_decay = 0.999  # epsilon 값 감쇄량
        self.epsilon_min = 0.01  # epsilon 최소 값
        self.batch_size = 64
        self.train_start = 100
        # self.q_table = defaultdict(float)

        # 리플레이 메모리, 최대 크기 10000
        self.memory = deque(maxlen=5000)  # 리플레이 메모리 사이즈 키워주기

        # 모델, 타깃 모델 생성
        self.behavior_model = DQN(self.action_size)  # replay 메모리 저장용
        self.target_model = DQN(self.action_size)  # 정책 학습용
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.loss = []
        # 타깃 모델 초기화
        self.update_target_model()

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        # self.target_model.set_weights(self.behavior_model.get_weights())
        weights = self.behavior_model.get_weights()
        self.target_model.set_weights(weights)

    # <s, a, r, s'> 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # (M,K)형태인 action을 입력으로 받아서 action_list에 있는 index로 mapping해주는 함수
    def action_transform(self, action):
        for i in range(len(self.action_list)):
            if (action == self.action_list[i]):
                return i

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])  # 아직 (M,K)상태
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])

        # 학습 파라미터
        model_params = self.behavior_model.trainable_variables
        with tf.GradientTape() as tape:
            # 현재 상태에 대한 모델의 큐함수
            predicts = self.behavior_model(states)
            one_hot_action = tf.one_hot(actions, self.action_size)
            # print("prediction : ", predicts)
            # print("prediction : ", tf.shape(predicts))
            # print(tf.shape(one_hot_action))
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)
            # 여기가 off policy인 이유 중 behaior가 epsilon으로 구한 action에 대한 Q값을 가져오는 거라서.

            # print("action곱 :", actions)
            # print("reduce_sum(action*predicts,axis=-1) : ", predicts)
            # 다음 상태에 대한 타깃 모델의 큐함수

            target_predicts = self.target_model(next_states)
            target_predicts = tf.stop_gradient(target_predicts)

            # target policy를 update할 때도 C가 적용되도록 짠 코드 - 여기서 시간이 오래 걸림
            # max_q = [0. for i in range(len(target_predicts))]
            # for i in range(len(targeㅠt_predicts)):
            #    for j in range(len(target_predicts[0])):
            #        if(self.action_list[j][0] * self.action_list[j][1] < Cs[i]):
            #            if(max_q[i]< target_predicts[i][j]):
            #                max_q[i] = target_predicts[i][j]

            # max_q = np.array(max_q)
            # print("max_q : ", max_q)
            # print("max_q.shape() : ", max_q.shape)
            max_q = np.amax(target_predicts, axis=-1)
            # print("max_q : ", max_q)
            # print("max_q.shape() : ", max_q.shape)

            # 여기가 Greedy policy로 off policy인 부분임
            targets = rewards + (1 - dones) * self.discount_factor * max_q
            # print("rewards in DL :", rewards)
            # print("target - predicts in DL: ", (1-dones) * self.discount_factor * max_q - predicts)
            loss = tf.reduce_mean(tf.square(targets - predicts))  # loss 함수로 mse 사용
            self.loss.append(loss)
            # 오류 함수를 줄이는 방향으로 모델 업데이트
            grads = tape.gradient(loss, model_params)
            self.optimizer.apply_gradients(zip(grads, model_params))

    # 모든 에피소드에서 에이전트가 방문한 상태의 큐 함수를 업데이트

    # 입실론 탐욕 정책에 따라서 행동을 반환 Behavior policy : epsilon greedy
    def get_action(self, state, chunk_size):
        print("current state : ", state[0])
        # print("current C_t : ", C_t)
        # for i in range(self.action_size):
        #    print("action_list : ", self.action_list[i], self.action_list[i][0], self.action_list[i][1])

        Q_t, n_t = state[0]
        if np.random.rand() <= self.epsilon:

            # for i in range(self.action_size):
            #    if ((self.action_list[i][0] + n_t < chunk_size+1) and (self.action_list[i][0] * self.action_list[i][1] <= C_t)):
            #        possible_action_list.append(self.action_list[i])
            #        #possible action list 안에는 (M,k)형태로 저장되어있음

            # print("possible_action_list : ", possible_action_list)

            # 랜덤 액션 역시 청크 사이즈에 관련없이 선택하도록 ( n 이 청크 사이즈보다 크면 에피소드 종료 _)
            action = random.choice(self.action_list)
            print('get_action의 epsilon greedy policy에 따른 random action : ', action)
            return action  # action은 (M,K) format

            """
            while (True):
                action = random.choice(self.action_list)
                if (action[0] + n_t < chunk_size + 1):
                    print('get_action의 epsilon greedy policy에 따른 random action : ', action)
                    return action  # action은 (M,K) format"""

        else:
            q_value = self.behavior_model(state)
            best_q = np.argmax(q_value)
            q_value = np.array(q_value)
            q_value = q_value.tolist()
            q_value_copy = copy.deepcopy(q_value)
            q_value_copy = np.array(q_value_copy)
            q_value_copy = q_value_copy.tolist()
            action = self.action_list[best_q]
            print('get_action의 epsilon greedy policy에 따른 greedy action : ', action)
            return action

            """for i in range(len(self.action_list)):
                
                if ((self.action_list[best_q][0] + n_t < chunk_size + 1)):  # n_t가 chunk_size+1보다 작으면
                    action = self.action_list[best_q]

                    print('get_action의 epsilon greedy policy에 따른 greedy action : ', action)
                    return action
                else:
                    # print("q_value_copy : ",q_value_copy)
                    # print("best_q : ",best_q)
                    # print("q_value : ",q_value)
                    # print("q_value[best_q] : ",q_value[0][best_q])

                    del q_value_copy[0][q_value_copy[0].index(q_value[0][best_q])]
                    best_q = q_value[0].index(q_value_copy[0][np.argmax(q_value_copy[0])])"""
    
    def get_action2(self, state, chunk_size):
        
        print("current state : ", state[0])
        Q_t, n_t = state[0]
        #weights = tf.stop_gradient(self.target_model.get_weights())        
        #self.target_model.set_weights(weights)
        q_value = self.target_model(state)
        q_value = tf.stop_gradient(q_value)
        best_q = np.argmax(q_value)
        q_value = np.array(q_value)
        q_value = q_value.tolist()
        q_value_copy = copy.deepcopy(q_value)
        q_value_copy = np.array(q_value_copy)
        q_value_copy = q_value_copy.tolist()
        action = self.action_list[best_q]
        print('get_action의 epsilon greedy policy에 따른 greedy action : ', action)
        return action

        """
        for i in range(len(self.action_list)):
            if ((self.action_list[best_q][0] + n_t < chunk_size + 1)):  # n_t가 chunk_size+1보다 작으면
                action = self.action_list[best_q]

                print('get_action의 epsilon greedy policy에 따른 greedy action : ', action)
                return action
            else:
                # print("q_value_copy : ",q_value_copy)
                # print("best_q : ",best_q)
                # print("q_value : ",q_value)
                # print("q_value[best_q] : ",q_value[0][best_q])

                del q_value_copy[0][q_value_copy[0].index(q_value[0][best_q])]
                best_q = q_value[0].index(q_value_copy[0][np.argmax(q_value_copy[0])])"""

    # 메인 함수


if __name__ == "__main__":  # 해당 .py파일이 module로 import되면 작동하지 않고, main으로 사용될 때만 동작하는 코드라는 뜻. __name__이 file이름이고 이게, main으로 되느냐임. agent에서 compile을 시작할 떄

    env = Env()  # 환경 객체 만들기(정보를 담고 있음) __init__ 받아옴

    # state_table = [[] for i in range(env.max_episode)]   # 디버깅용 밟은 state저장하는 table
    # action_table = [[] for i in range(env.max_episode)]  # 디버깅용 취한 action저장하는 table
    total_reward = [[0.] for i in range(env.max_episode)]  # 디버깅용 받은 reward저장하는 table
    total_buffering = [[0.] for i in range(env.max_episode)]  # 디버깅용 받은 buffering저장하는 table
    total_bitrate = [[0.] for i in range(env.max_episode)]  # 디버깅용 받은 buffering저장하는 table
    LossBitmean = [[0.] for i in range(env.max_episode)]  # 디버깅용 받은 buffering저장하는 table
    everyLossBit = []  # 디버깅용 받은 buffering저장하는 table
    plot_q_t = []  # 디버깅용 q값을 plot하려고 만든 table

    action_size = 37  # m(t), k(t)
    state_size = 2  # q(t), n(t), lm, lk

    counter = 0

    agent = DQNAgent(state_size, action_size)  # agent 객체 만들기 __init__ 받아옴

    for episode in range(env.max_episode):  # episode를 진행하는 반복문, max_episode 일단 1000으로 설정했음

        prev_state = [0, 0]  # state 리셋 list 형태[Q_t,n_t, lm, lk]를 state로 쓰겠다.
        prev_state = np.reshape(prev_state, [1, state_size])
        # print("state :", prev_state)
        # print("[[0,0]] : ",[[0,0]] )

        while True:  # time step을 진행하는 반복문
            plot_q_t.append(prev_state[0][0])
            print("episode, time ", episode, env.time)
            action = agent.get_action(prev_state, env.chunk_size)
            print("action : ", action)
            next_state, reward, done, episode, time, accepted_M ,accepted_bitrate, LossM, LossK = env.step(action)  # time+1 되는 시점
            everyLossBit.append(LossM*LossK)
            LossBitmean[episode][0] = LossBitmean[episode][0] + LossM*LossK

            total_bitrate[episode][0] = total_bitrate[episode][0] + accepted_M*accepted_bitrate
            
            #print("next_state :", next_state)
            next_state = np.reshape(next_state, [1, state_size])  # reshape 해주는 이유
            print("reward : ", reward)
            total_reward[episode][0] = total_reward[episode][0] + reward  # 디버깅용 reward 더하는 코드


            if (prev_state[0][0] == 0 and env.time!=0):
                total_buffering[episode][0] = total_buffering[episode][0] + 1  # 디버깅용 buffering count 더하는 코드
            # plot_q.append(agent.behavior_model(np.reshape([0, 0],[1,2]))[0])

            action = agent.action_transform(action)
            agent.append_sample(prev_state, action, reward, next_state, done)  # 여기서 sample에 저장하는 C_t는 사실

            print("\n\n\n")

            # 타임스텝마다 학습
            if len(agent.memory) >= agent.train_start:
                agent.train_model()
                # 업데이트 주기를 표기하기 위한 카운터, 매 타임 스텝 마다 +1
                counter += 1
            prev_state = next_state


            # 카운터가 특정 주기다 될 때 마다 카운터 0으로 초기화 & 타깃 모델을 업데이트.
            # 에피소드와 독립적으로 움직인다.
            if ( counter == 30 ):
                counter = 0
                agent.update_target_model()

            # 디버깅용 원하는 [[state],(action)]을 넣으면 step단위로 q_function의 변화를 볼 수 있다.

            # state_table[episode].append(next_state)  # 디버깅용 state table만드는 코드
            # action_table[episode].append(prev_action) # 디버깅용 action table만드는 코드

            if done:  # 마지막 update까지 하고 디버깅용 저장까지 하고 break해야함
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                    # 학습할 때 reward를 줄이는 방향으로 학습하지 않도록 하려면 두 가지 방법이 있음
                    # 1. target policy를 update하는 주기를 길게 잡는다.
                    # 2. epsilon decay를 0.999보다 크게 0.9999처럼 잡아서 random sample을 많이 만들어 둔다.
                LossBitmean[episode][0] /= env.time
                total_bitrate[episode][0] /= env.chunk_size
                # 에이전트 업데이트 주기 수정으로 지움
                #agent.update_target_model()
                print("reward in episode : ", total_reward[episode][0])
                # 에피소드마다 학습 결과 출력

                print("\n끝\n")
                break

        if (env.episode == env.max_episode) and (env.n_t == env.chunk_size):
            break
        env.reset()  # 환경 리셋(q,n을 가지고 있고, 이를 초기화 함)
    
    
    print("\n\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n\n")
    
    ##########################################################
    # Test Simulation
    
    env2 = Env()  # 환경 객체 만들기(정보를 담고 있음) __init__ 받아옴
    env2.max_episode = 10
    
    
    total_reward2 = [[0.] for i in range(env2.max_episode)]  # 디버깅용 받은 reward저장하는 table
    total_buffering2 = [[0.] for i in range(env2.max_episode)]  # 디버깅용 받은 buffering저장하는 table
    total_bitrate2 = [[0.] for i in range(env2.max_episode)]  # 디버깅용 받은 buffering저장하는 table
    LossBitmean2 = [[0.] for i in range(env2.max_episode)]  # 디버깅용 받은 buffering저장하는 table
    everyLossBit2 = []  # 디버깅용 받은 buffering저장하는 table
    plot_q_t2=[]

    for episode in range(env2.max_episode):  # episode를 진행하는 반복문, max_episode 일단 1000으로 설정했음

        prev_state = [0, 0]  # state 리셋 list 형태[Q_t,n_t, lm, lk]를 state로 쓰겠다.
        prev_state = np.reshape(prev_state, [1, state_size])
        # print("state :", prev_state)
        # print("[[0,0]] : ",[[0,0]] )

        while True:  # time step을 진행하는 반복문
            plot_q_t2.append(prev_state[0][0])

            print("episode, time ", episode, env2.time)
            action = agent.get_action2(prev_state, env2.chunk_size)
            print("action : ", action)
            next_state, reward, done, episode, time, accepted_M2 ,accepted_bitrate2, LossM2, LossK2   = env2.step(action)  # time+1 되는 시점
            
            everyLossBit2.append(LossM2*LossK2)            
            LossBitmean2[episode][0] = LossBitmean2[episode][0] + LossM2*LossK2
            total_bitrate2[episode][0] = total_bitrate2[episode][0] + accepted_M2*accepted_bitrate2
            
            #print("next_state :", next_state)
            next_state = np.reshape(next_state, [1, state_size])  # reshape 해주는 이유
            print("reward : ", reward)
            total_reward2[episode][0] = total_reward2[episode][0] + reward  # 디버깅용 reward 더하는 코드

            if (prev_state[0][0] == 0 and env2.time!=0):
                total_buffering2[episode][0] = total_buffering2[episode][0] + 1  # 디버깅용 buffering count 더하는 코드
            # plot_q.append(agent.behavior_model(np.reshape([0, 0],[1,2]))[0])

            action = agent.action_transform(action)
            
            print("\n\n\n")


            prev_state = next_state

            # 디버깅용 원하는 [[state],(action)]을 넣으면 step단위로 q_function의 변화를 볼 수 있다.

            # state_table[episode].append(next_state)  # 디버깅용 state table만드는 코드
            # action_table[episode].append(prev_action) # 디버깅용 action table만드는 코드

            if done:  # 마지막 update까지 하고 디버깅용 저장까지 하고 break해야함
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                    # 학습할 때 reward를 줄이는 방향으로 학습하지 않도록 하려면 두 가지 방법이 있음
                    # 1. target policy를 update하는 주기를 길게 잡는다.
                    # 2. epsilon decay를 0.999보다 크게 0.9999처럼 잡아서 random sample을 많이 만들어 둔다.
                LossBitmean2[episode][0] /= env2.time
                total_bitrate2[episode][0] /= env2.chunk_size
                print("reward in episode : ", total_reward2[episode][0])
                # 에피소드마다 학습 결과 출력

                print("\n끝\n")
                break

        if (env2.episode == env2.max_episode) and (env2.n_t == env2.chunk_size):
            break
        env2.reset()  # 환경 리셋(q,n을 가지고 있고, 이를 초기화 함)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ##########################################################
    

    # print("state_table :  ", *state_table[:],sep='\n')     #state table 기록 보여줄 때 주석 해제
    # print("action_table : ", *action_table[:],sep='\n')    #action table 기록 보여줄 때 주석 해제
    # print("q_table : ", *q_list,sep='\n')                  #최종 q_table 보여줄 때 주석 해제

    # print("plot", plot_q) # step단위 q값 변화 텍스트로 보여줄 때 주석 해제
    plt.figure()
    plt.xlabel('episode')
    plt.ylabel('total_reward')
    plt.plot(total_reward)
    plt.title("reward at each episode in Train")
    # plt.plot(total_reward,'o')  # 선 말고 dot으로 plot 해줄 때 주석 해제
    # 보통 결과 확인할 때, 매 step마다 state랑 action이 어떻게 됐는지랑 한 episode의 total reward합이 어떻게 변하는지를 보여주면 성능을 알 수 있다.
    # plt.xlabel('step')         #step단위 q값 보여줄 때 주석 해제
    # plt.ylabel('q_funtion')    #step단위 q값 보여줄 때 주석 해제
    # plt.plot(plot_q)           #step단위 q값 보여줄 때 주석 해제

#    plt.figure()
#    plt.xlabel('step')
#    plt.ylabel('loss')
#    plt.plot(agent.loss)
#    plt.title("loss at each step in Train")


    plt.figure()
    plt.xlabel('episode')
    plt.ylabel('buffering')
    plt.plot(total_buffering)
    plt.title("buffering count at each episode in Train")


    plt.figure()
    plt.xlabel('episode')
    plt.ylabel('mean of bitrate')
    plt.plot(total_bitrate)
    plt.title("mean of bitrate at each episode in Train")
    


    
    plt.figure()
    plt.xlabel('episode')
    plt.ylabel('mean of bitrate')
    plt.plot(LossBitmean)
    plt.title("mean of Loss MBit at each episode in Train")
    
    
    
    plt.figure()
    plt.xlabel('step')
    plt.ylabel('Loss Mbit')
    plt.plot(everyLossBit)
    plt.title("Loss Mbit at each step in Train")
    
    
    plt.figure()
    plt.xlabel('step')
    plt.ylabel('Q(t)')
    plt.plot(plot_q_t)
    plt.title("state Q(t) at every step in Train")
    


    #total_reward
    #total_buffering
    #total_bitrate
    #LossBitmean
    #everyLossBit
    #plot_q_t

    
    # 위에는 Train model의 plot
    ####################################################################
    # 아래는 Test model의 plot
    


    plt.figure()

    plt.xlabel('episode')
    plt.ylabel('total_reward')
    plt.plot(total_reward2)
    plt.title("reward at each episode in Test")
    
    

    plt.figure()
    plt.xlabel('episode')
    plt.ylabel('buffering')
    plt.plot(total_buffering2)
    plt.title("buffering count at each episode in Test")


    plt.figure()
    plt.xlabel('episode')
    plt.ylabel('mean of bitrate')
    plt.plot(total_bitrate2)
    plt.title("mean of bitrate at each episode in Test")
    


    
    plt.figure()
    plt.xlabel('episode')
    plt.ylabel('mean of bitrate')
    plt.plot(LossBitmean2)
    plt.title("mean of Loss MBit at each episode in Test")
    
    
    
    plt.figure()
    plt.xlabel('step')
    plt.ylabel('Loss Mbit')
    plt.plot(everyLossBit2)
    plt.title("Loss Mbit at each step in Test")
    
    
    plt.figure()
    plt.xlabel('step')
    plt.ylabel('Q(t)')
    plt.plot(plot_q_t2)
    plt.title("state Q(t) at every step in Test")
    

    plt.show()
    # 221216 2056
    # 에 start. 디버깅시간 기록하자.

    
    
    