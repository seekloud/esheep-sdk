from grpc_client import GrpcClient
import actions_pb2 as action
import api_pb2 as api
import time
import random


ip = '127.0.0.1'
port = '5322'


def run():
    env = GrpcClient(ip=ip,
                      port=port,
                      api_token="test")

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
            actionRsp = env.submit_action(actionChoose, None, None, None)
            print(actionRsp)
        elif obs.state == api.killed:
            print('killed')
            rsp = env.submit_reincarnation()
            print(rsp)
        else:
            print('cannot get state:', obs.state)
        print("\n")
        print(str(obs.human_observation.width) + "\n")
        print(str(obs.human_observation.height) + "\n")
        print(str(obs.human_observation.pixel_length) + "\n")
        print(str(type(obs.human_observation.data)) + "\n")

        time.sleep(0.15)

    # rsp = env.leave_room()
    # print(rsp)


if __name__ == '__main__':
    run()
