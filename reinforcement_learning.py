# Reinforcement Learning - this is just a simple introduction and tutorial to reinforcement learning. A Jupyter Notebook
# may have been better in hindsight

# 1. What is reinforcement learning?
# To put in simply, imagine if you are a kid and you are trapped in a maze, you don't know which direction is the
# correct one, but every time you head in the right direction, you get rewarded. Eventually, the rewards will guide you
# out of the maze. If it's the wrong direction, you get stung or something I don't know.

# In RL, it usually involves an agent and an environment. In this case, the agent would be you the kiddo, and the
# environment would be the maze. This reward system would tell you if the step that you just took is good or bad,
# guiding you to take the best course of action

# 2. So do we just need an agent and environment, and reward?
# Not quite, we also need:
# a. State: current situation of agent
# b. Policy: How to map the agent's state to actions

# 3. Key concepts
# a. Markov Decision Processes: mathematical frameworks to describe an environment in reinforcement learning and almost
# all RL problems can be formalized using MDPs. An MDP consists of a set of finite environment states S, a set of
# possible actions A(s) in each state, a real valued reward function R(s) and a transition model P(s’, s | a). However,
# real world environments are more likely to lack any prior knowledge of environment dynamics. Model-free RL methods
# come handy in such cases.

# b. Q-learning: model free approach which can be used for building a self-playing PacMan agent. It revolves around
# the notion of updating Q values which denotes value of doing action a in state s. The value update rule is the core
# of the Q-learning algorithm.

# ------------------------------------- 1. Simple paddle and ball game ------------------------------------------------
# Reference code: https://github.com/shivaverma/Orbit

import turtle as t
import random
import numpy as np
from keras import Sequential
from collections import deque
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from keras.optimizers import Adam


class Paddle():
    def __init__(self):
        self.done = False
        self.reward = 0
        self.hit, self.miss = 0, 0

        # Setup Background
        self.win = t.Screen()
        self.win.title('Paddle')
        self.win.bgcolor('black')
        self.win.setup(width=600, height=600)
        self.win.tracer(0)

        # Paddle
        self.paddle = t.Turtle()
        self.paddle.speed(0)
        self.paddle.shape('square')
        self.paddle.shapesize(stretch_wid=1, stretch_len=5)
        self.paddle.color('white')
        self.paddle.penup()
        self.paddle.goto(0, -275)

        # Ball
        self.ball = t.Turtle()
        self.ball.speed(0)
        self.ball.shape('circle')
        self.ball.color('red')
        self.ball.penup()
        self.ball.goto(0, 100)
        self.ball.dx = 3
        self.ball.dy = -3

        # Keeping the scores
        self.score = t.Turtle()
        self.score.speed(0)
        self.score.color('white')
        self.score.penup()
        self.score.hideturtle()
        self.score.goto(0, 250)
        self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))

        # -------------------- Keyboard control ----------------------
        self.win.listen()
        self.win.onkey(self.paddle_right, 'Right')
        self.win.onkey(self.paddle_left, 'Left')

    # Paddle movement
    def paddle_right(self):
        x = self.paddle.xcor()
        if x < 225:
            self.paddle.setx(x+20)

    def paddle_left(self):
        x = self.paddle.xcor()
        if x > -225:
            self.paddle.setx(x-20)

    def run_frame(self):
        self.win.update()

        # Ball moving, defining the direction of the ball moving
        self.ball.setx(self.ball.xcor() + self.ball.dx)
        self.ball.sety(self.ball.ycor() + self.ball.dy)

        # Ball and Wall collision, make sure it bounces off the wall
        if self.ball.xcor() > 290:
            self.ball.setx(290)
            self.ball.dx *= -1

        if self.ball.xcor() < -290:
            self.ball.setx(-290)
            self.ball.dx *= -1

        if self.ball.ycor() > 290:
            self.ball.sety(290)
            self.ball.dy *= -1

        # Ball Ground contact, means incrementing number of misses
        if self.ball.ycor() < -290:
            self.ball.goto(0, 100)
            self.miss += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))
            self.reward -= 3
            self.done = True

        # Ball Paddle collision
        if abs(self.ball.ycor() + 250) < 2 and abs(self.paddle.xcor() - self.ball.xcor()) < 55:
            self.ball.dy *= -1
            self.hit += 1
            self.score.clear()
            self.score.write("Hit: {}   Missed: {}".format(self.hit, self.miss), align='center', font=('Courier', 24, 'normal'))
            self.reward += 3

    # ------------------------ AI control ------------------------
    # 0 move left
    # 1 do nothing
    # 2 move right

    def reset(self):
        self.paddle.goto(0, -275)
        self.ball.goto(0, 100)
        return [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx, self.ball.dy]

    def step(self, action):
        self.reward = 0
        self.done = 0

        if action == 0:
            self.paddle_left()
            self.reward -= .1

        if action == 2:
            self.paddle_right()
            self.reward -= .1

        self.run_frame()

        state = [self.paddle.xcor()*0.01, self.ball.xcor()*0.01, self.ball.ycor()*0.01, self.ball.dx, self.ball.dy]
        return self.reward, state, self.done


env = Paddle()
np.random.seed(0)

# DQN is not guaranteed to converge because of the instability caused by bootstrapping, sampling and value
# functional approximation
class DQN:
    """ Implementation of deep q learning algorithm """
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1
        self.gamma = .95
        self.batch_size = 64
        self.epsilon_min = .01
        self.epsilon_decay = .995
        self.learning_rate = 0.001
        self.memory = deque(maxlen=100000)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_shape=(self.state_space,), activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    # Replay is the act of storing and replaying game states (the state, action, reward, next_state)
    # uses Experience Replay to learn in small batches in order to avoid skewing the dataset distribution of different
    # states, actions, rewards, and next_states that the neural network will see. Importantly, the agent doesn’t need
    # to train after each step.
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(episode):
    loss = []
    action_space = 3
    state_space = 5
    max_steps = 1000

    agent = DQN(action_space, state_space)
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, state_space))
        score = 0
        for i in range(max_steps):
            action = agent.act(state)
            reward, next_state, done = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, state_space))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                print("episode: {}/{}, reward: {}".format(e, episode, score))
                break
        loss.append(score)
    return loss


if __name__ == '__main__':
    ep = 100
    loss = train_dqn(ep)
    plt.plot([i for i in range(ep)], loss)
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.show()
