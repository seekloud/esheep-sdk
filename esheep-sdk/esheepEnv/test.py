from env import Environment
import actions_pb2 as action
import api_pb2 as api
import time
import random


ip = '127.0.0.1'
port = '5322'


def run():
    env = Environment(ip=ip,
                      port=port,
                      api_token="")

    rsp = env.create_room("123")
    print(rsp.room_id)
    actionSpace = env.get_action_space()
    print(actionSpace.move)
    # time.sleep(10000)
    actionList = [action.down, action.up, action.left, action.right]
    for i in range(0, 400):
        obs = env.get_observation()
        if obs.state == api.in_game:
            print('in_game')
            actionIndex = random.randint(0, 3)
            actionChoose = actionList[actionIndex]
            actionRsp = env.submit_action(actionChoose, action.Swing(), None, None)
            print(actionRsp)
        elif obs.state == api.killed:
            print('killed')
            rsp = env.submit_reincarnation()
            print(rsp)
        else:
            print('cannot get state:', obs.state)
        # print(obs.human_observation.pixel_length)
        # print(obs.human_observation.background.pixel_length)
        # print(obs.human_observation.things.pixel_length)
        # print(obs.human_observation.players.pixel_length)
        # print(obs.human_observation.self_player.pixel_length)
        # print(obs.human_observation.states.pixel_length)
        # print(obs.frame_index)

        time.sleep(0.15)

    # rsp = env.leave_room()
    # print(rsp)


if __name__ == '__main__':
    run()
