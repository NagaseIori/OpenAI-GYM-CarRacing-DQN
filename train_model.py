import argparse
import gymnasium as gym
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from common_functions import process_state_image
from common_functions import generate_state_frame_stack_from_queue

RENDER                        = False
STARTING_EPISODE              = 1
ENDING_EPISODE                = 1000
SKIP_FRAMES                   = 3
TRAINING_BATCH_SIZE           = 128
SAVE_TRAINING_FREQUENCY       = 25
UPDATE_TARGET_MODEL_FREQUENCY = 2
CONTINUOUS                    = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a DQN agent to play CarRacing.')
    parser.add_argument('-m', '--model', help='Specify the last trained model path if you want to continue training after it.')
    parser.add_argument('-s', '--start', type=int, help='The starting episode, default to 1.')
    parser.add_argument('-e', '--end', type=int, help='The ending episode, default to 1000.')
    parser.add_argument('-p', '--epsilon', type=float, default=0.9, help='The starting epsilon of the agent, default to 0.9.')
    args = parser.parse_args()

    # Updated to use CarRacing-v2 environment
    env = gym.make('CarRacing-v2', render_mode = "human" if RENDER else None, continuous = CONTINUOUS)
    agent = CarRacingDQNAgent(epsilon=args.epsilon, action_space=env.action_space)
    if args.model:
        agent.load(args.model)
    if args.start:
        STARTING_EPISODE = args.start
    if args.end:
        ENDING_EPISODE = args.end

    for e in range(STARTING_EPISODE, ENDING_EPISODE + 1):
        # Updated reset method for compatibility with gymnasium
        init_state, _ = env.reset()
        init_state = process_state_image(init_state)

        total_reward = 0
        negative_reward_counter = 0
        state_frame_stack_queue = deque([init_state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
        time_frame_counter = 1
        done = False

        while not done:
            if RENDER:
                env.render()

            # Update the current state frame stack to match PyTorch channel order (C, H, W)
            current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            action = agent.act(current_state_frame_stack)

            reward = 0
            for __ in range(SKIP_FRAMES + 1):
                next_state, r, terminated, truncated, _ = env.step(action)
                reward += r
                done = terminated or truncated
                if done:
                    break

            # If continually getting negative reward 10 times after the tolerance steps, terminate this episode
            negative_reward_counter = negative_reward_counter + 1 if time_frame_counter > 100 and reward < 0 else 0

            total_reward += reward

            next_state = process_state_image(next_state)
            state_frame_stack_queue.append(next_state)
            next_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)

            agent.memorize(current_state_frame_stack, action, reward, next_state_frame_stack, done)

            if done:
                print('Episode: {}/{}, Scores(Time Frames): {}, Total Rewards(adjusted): {:.2f}, Epsilon: {:.2f}'.format(
                    e, ENDING_EPISODE, time_frame_counter, float(total_reward), float(agent.epsilon)))
                break
            if len(agent.memory) > TRAINING_BATCH_SIZE:
                agent.replay(TRAINING_BATCH_SIZE)
            time_frame_counter += 1

        if e % UPDATE_TARGET_MODEL_FREQUENCY == 0:
            agent.update_target_model()

        if e % SAVE_TRAINING_FREQUENCY == 0:
            agent.save('./save/trial_{}.pth'.format(e))

    env.close()