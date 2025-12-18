import os
import gymnasium as gym
import gym_pusht
import datetime
from gymnasium.wrappers import RecordVideo

class VideoRecorder:
    def __init__(self, individual):
        self.individual = individual
        self.video_path = "./video"
        
    def record(self):
        os.makedirs(self.video_path, exist_ok=True)
        env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
        env = RecordVideo(env, self.video_path, episode_trigger=lambda episode_id: episode_id == 0, disable_logger=True)

        fixed_state = [20.0, 250, 100.0, 200.0, 0.0]  # agent_x, agent_y, block_x, block_y, angle
        env.reset(options={"reset_to_state": fixed_state})

        individual = self.individual

        # 每兩個值是一個 action
        for i in range(len(individual) // 2):
            action = individual[2 * i: 2 * i + 2]
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset(options={"reset_to_state": fixed_state})

        env.close()

        default_video = os.path.join(self.video_path, "rl-video-episode-0.mp4")

        if os.path.exists(default_video):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_name = f"pusht_video_{timestamp}.mp4"
            new_path = os.path.join(self.video_path, new_name)
            os.rename(default_video, new_path)
            print(f"Video saved as: {new_path}")

if __name__ == "__main__":
    individual = [419.2945644084532, 98.98760695613407, 259.0923706449755, 146.40523516426043, 359.21096821541363, 335.6229731888052, 131.8661266359267, 115.7694432689721, 446.1527054685737, 51.01799049290787, 221.40025637379466, 110.76948455765324, 501.69084986867983, 287.1006104912067, 256.6301168333352, 230.95438992327973, 395.08182975977974, 107.9491712008658, 139.47890571117813, 261.6615907486071, 346.9394897242636, 385.3112142690444, 306.21559404356765, 210.30346457811316, 251.95565577311953, 154.56341827913587, 362.392895274563, 356.4263415904333, 451.74613053595994, 315.9378264323407, 133.02346347360168, 364.50087286889413, 347.20535164559766, 448.72266175070723]

    v = VideoRecorder(individual)
    v.record()