from stable_baselines3 import A2C,PPO
from tetris_gym.envs.tetris_gym import TetrisGym
from tetris_gym.wrappers.observation import ExtendedObservationWrapper
from tetris_gym.utils.eval_utils import evaluate, create_videos
from sklearn.model_selection import GridSearchCV
from agent import agent
import gym

# Környezet létrehozása
env = TetrisGym(width=10, height=20)

# A megfigyelések kiterjesztése a tábla alapján számolt új jellemzők segítségével.
env = ExtendedObservationWrapper(env)
class CustomRewardWrapper(gym.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
    # Felülírjük a környezet beépített step függvényét
    def step(self, action):

        # Meghívjuk az eredeti környezet step függvényét
        obs, reward, done, info = self.env.step(action)

        # Új jutalmat számítunk, minél jobban eldől az inga vagy elmozog a
        # kocsi, annál jobban büntetjük
        line_reward = 50000
        hole_reward = 0.7
        bumpiness_reward = 0.5
        height_reward = 0.02
        lines_cleared = self.env.get_state_properties(obs['board'])[0]
        holes = self.env.get_state_properties(obs['board'])[1]
        bumpiness = self.env.get_state_properties(obs['board'])[2]
        height = self.env.get_state_properties(obs['board'])[3]
        reward = (-((hole_reward* holes) *
                   ( bumpiness_reward*bumpiness )*
                   (height_reward*height))
                  + (line_reward*lines_cleared)+100
        )
        
        return obs, reward, done, info
env = CustomRewardWrapper(env)
# Modell létrehozása
#model = A2C('MultiInputPolicy',  env, verbose=1, seed=42)
model = PPO('MultiInputPolicy',  env, verbose=1, seed=42)
model.learn(total_timesteps=100000)
#model = agent.bob(env=env)
# Model kimentése
model.save("agent/model_bob")

# Kiértékelés 10 véletlen környezetben
env = TetrisGym(width=10, height=20)

# A megfigyelések kiterjesztése a tábla alapján számolt új jellemzők segítségével.
env = ExtendedObservationWrapper(env)
score = evaluate(env, model, 10)
print("Score: {}".format(score))

# Videók készítése
#create_videos(env, model)