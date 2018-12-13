import sys
import datetime
import grpc
import service_pb2_grpc as service
import api_pb2 as messages
import actions_pb2 as action


class Environment:
    def __init__(self, ip, port, api_token, player_id, logfile_path='./', debug=False):

        if debug:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.log_file = open(logfile_path + "log." + current_time, "w")

        self.debug = debug
        self.ip = ip
        self.port = port
        self.api_token = api_token
        self.player_id = player_id
        self.metadata = [('ip', ip)]

        channel = grpc.insecure_channel(self.ip + ':' + self.port)

        try:
            grpc.channel_ready_future(channel).result(timeout=10)
        except grpc.FutureTimeoutError:
            sys.exit('Error connecting to server')
        else:
            self.stub = service.EsheepAgentStub(channel)

    def create_room(self, password):
        response = self.stub.createRoom(
            messages.CreateRoomReq(
                credit=messages.Credit(player_id=self.player_id, api_token=self.api_token),
                password=password)
        )
        if response:
            return response
        else:
            return []

    def join_room(self, room_id, password):
        response = self.stub.joinRoom(
            messages.JoinRoomReq(
                credit=messages.Credit(player_id=self.player_id, api_token=self.api_token),
                password=password,
                room_id=room_id
            )
        )
        if response:
            return response
        else:
            return []

    def leave_room(self):
        response = self.stub.leaveRoom(
            messages.Credit(player_id=self.player_id, api_token=self.api_token)
        )
        if response:
            return response
        else:
            return []

    def get_action_space(self):
        response = self.stub.actionSpace(
            messages.Credit(player_id=self.player_id, api_token=self.api_token)
        )
        if response:
            return response
        else:
            return []

    def action(self, move, swing, fire, apply):
        response = self.stub.action(
            messages.ActionReq(
                move=action.Move(move),
                Swing=action.Swing(radian=swing[0], distance=swing[1]),
                fire=fire,
                apply=apply,
                credit=messages.Credit(player_id=self.player_id, api_token=self.api_token)
            )
        )
        if response:
            return response
        else:
            return []

    def observation(self):
        response = self.stub.observation(
            messages.Credit(player_id=self.player_id, api_token=self.api_token)
        )
        if response:
            return response
        else:
            return []

    def inform(self):
        response = self.stub.inform(
            messages.Credit(player_id=self.player_id, api_token=self.api_token)
        )
        if response:
            return response
        else:
            return []

    def reincarnation(self):
        response = self.stub.inform(
            messages.Credit(player_id=self.player_id, api_token=self.api_token)
        )
        if response:
            return response
        else:
            return []
