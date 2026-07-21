import argparse
import time
import os
import numpy as np
import tensorflow as tf
import gymnasium
import snake_env

def play_trained_model(model_path, env_name="snake", episodes=5, max_steps=1000, render=True):
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist.")
        return

    model = tf.keras.models.load_model(model_path)
    print(f"✓ Model loaded successfully from {model_path}")
    
    if env_name == "snake":
        render_mode = "human" if render else None
        env = snake_env.SnakeEnv(render_mode=render_mode)
    else:
        render_mode = "human" if render else None
        env = gymnasium.make('CartPole-v1', render_mode=render_mode)
        
    scores = []

    for ep in range(episodes):
        reset_res = env.reset()
        obs = reset_res[0] if isinstance(reset_res, (tuple, list)) else reset_res

        total_score = 0
        step_count = 0
        done = False
        truncated = False

        while not done and not truncated and step_count < max_steps:
            obs_input = np.expand_dims(obs, axis=0).astype(np.float32)

            q_values = model.predict(obs_input, verbose=0)[0]
            action = int(np.argmax(q_values))

            step_res = env.step(action)
            
            if isinstance(step_res, tuple) and len(step_res) == 5:
                next_obs, reward, terminated, truncated, info = step_res
                done = bool(terminated or truncated)
            elif isinstance(step_res, tuple) and len(step_res) == 4:
                next_obs, reward, done, info = step_res
                truncated = False
                done = bool(done)
            else:
                next_obs, reward, done = step_res[0], step_res[1], step_res[2]
                truncated = False

            if isinstance(next_obs, (tuple, list)):
                next_obs = next_obs[0]

            total_score += reward
            step_count += 1
            obs = next_obs

            if render:
                try:
                    env.render()
                except Exception:
                    pass

        scores.append(total_score)
        print(f"Episode {ep+1}: Score = {total_score:.1f}, Steps = {step_count}")

    try:
        env.close()
    except Exception:
        pass

    print(f"Finished {episodes} episodes. Average score: {np.mean(scores):.2f}")
    return scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained DQN Agent")
    parser.add_argument("--env", type=str, choices=["snake", "cartpole"], default="snake", help="Environment to run")
    parser.add_argument("--model", type=str, default="models/snake_model.h5", help="Path to the trained model (.h5)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    args = parser.parse_args()

    play_trained_model(
        model_path=args.model,
        env_name=args.env,
        episodes=args.episodes,
        render=not args.no_render
    )

if __name__ == "__main__":
    main()
