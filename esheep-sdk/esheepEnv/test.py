import time
from game_env import GameEnvironment
import random


ip = '127.0.0.1'
port = '5322'


def run():
    env = GameEnvironment(ip=ip, port=port, api_token="test")
    roomid, state = env.create_room("123")
    print("roomid:" + str(roomid))
    print("state:" + str(state))
    frame_period = env.get_frame_period()
    move, swing, fire, apply = env.get_action_space()
    for i in range(0, 400):
        frame, \
        state, \
        location, \
        immutable_element, \
        mutable_element, \
        bodies, \
        asset_ownership, \
        self_asset, \
        self_status, \
        pointer, \
        score, \
        kill, \
        health = env.get_observation_with_info()
        print("frame:"+str(frame))
        if state == 1:
            print('in_game')
            action_choose = move[random.randint(0, 3)]
            env.submit_action(frame, action_choose, None, None, None)
        elif state == 2:
            print('killed')
            rsp = env.submit_reincarnation()
            print(rsp)
        else:
            print('get state:', state)

        time.sleep(frame_period/1000)

    # rsp = env.leave_room()
    # print(rsp)


if __name__ == '__main__':
    run()