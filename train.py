import argparse
import time
import os
import numpy as np
import gymnasium
import snake_env
from dqn_agent import DQNAgent
from utils import plot_reward_and_variance, plot_dqn_loss

def main():
    parser = argparse.ArgumentParser(description="Train DQN Agent")
    parser.add_argument("--env", type=str, choices=["snake", "cartpole"], default="snake", help="Environment to train on")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes to train")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--update_freq", type=int, default=4, help="Network update frequency")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save the trained model")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.env == "snake":
        env = snake_env.SnakeEnv()
        model_path = os.path.join(args.save_dir, "snake_model.h5")
        solve_score = 100.0
    else:
        env = gymnasium.make("CartPole-v1")
        model_path = os.path.join(args.save_dir, "cart_pole_model.h5")
        solve_score = 100.0  # Or another appropriate score for CartPole

    reset_res = env.reset()
    state = reset_res[0] if isinstance(reset_res, (tuple, list)) else reset_res

    state_size = env.observation_space.shape
    num_actions = env.action_space.n
    print(f"Training on {args.env} | State Shape: {state_size} | Actions: {num_actions}")

    agent = DQNAgent(state_size=state_size, num_actions=num_actions)

    total_point_history = []
    total_loss_history = []
    num_p_av = 100
    epsilon = 1.0

    start_time = time.time()

    for i in range(args.episodes):
        reset_res = env.reset()
        state = reset_res[0] if isinstance(reset_res, (tuple, list)) else reset_res

        total_points = 0
        
        for t in range(args.max_steps):
            action = agent.get_action(state, epsilon)
            step_res = env.step(action)
            
            if isinstance(step_res, tuple) and len(step_res) == 5:
                next_state, reward, terminated, truncated, _ = step_res
                done = bool(terminated or truncated)
            elif isinstance(step_res, tuple) and len(step_res) == 4:
                next_state, reward, done, _ = step_res
                done = bool(done)
            else:
                next_state, reward, done = step_res[0], step_res[1], step_res[2]

            if isinstance(next_state, (tuple, list)):
                next_state = next_state[0]

            agent.store_experience(state, action, reward, next_state, done)
            
            # Update network
            if len(agent.memory_buffer) >= 1000 and t % args.update_freq == 0:
                experiences = agent.get_experiences()
                loss = agent.agent_learn(experiences)
                total_loss_history.append(loss.numpy())

            state = next_state.copy() if hasattr(next_state, 'copy') else next_state
            total_points += reward
            
            if done:
                break
                
        total_point_history.append(total_points)
        av_latest_points = np.mean(total_point_history[-num_p_av:])
        
        epsilon = agent.decay_epsilon(epsilon)

        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

        if (i+1) % num_p_av == 0:
            print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

        if av_latest_points >= solve_score:
            print(f"\n\nEnvironment solved in {i+1} episodes!")
            agent.save_model(model_path)
            print(f"Model saved to {model_path}")
            break

    tot_time = time.time() - start_time
    print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")
    
    # Only plot if not solved quickly or we want to save plots
    # We will save plots instead of showing them to prevent blocking
    plot_dqn_loss(total_loss_history, window=100, save_path=f"loss_{args.env}.png")
    plot_reward_and_variance(total_point_history, window=100, save_path=f"reward_{args.env}.png")
    print(f"Plots saved for {args.env}")

if __name__ == "__main__":
    main()
