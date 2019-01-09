from grpc_client import GrpcClient
import threading


class GameEnvironment:
    def __init__(self, ip, port, api_token, logfile_path='./', debug=False):
        self.grpcClient = GrpcClient(ip, port, api_token, logfile_path, debug)
        self._frame_period = self.grpcClient.get_system_info().framePeriod
        self._action_space = self.grpcClient.get_action_space()
        self.frame_index = 0
        self.layered_observation = []
        self.human_observation = []

    def create_room(self, password):
        rsp = self.grpcClient.create_room(password)
        self._refresh_observation()
        return rsp.roomid, rsp.state

    def join_room(self, room_id, password):
        rsp = self.grpcClient.join_room(room_id, password)
        self._refresh_observation()
        return rsp.state

    def _refresh_observation(self):
        observation_response = self.grpcClient.get_observation()
        if observation_response.frameIndex > self.frame_index:
            self.frame_index = observation_response.frameIndex
            self.layered_observation = observation_response.layeredObservation
            self.human_observation = observation_response.humanObservation
        timer = threading.Timer(self._frame_period - 10, self._refresh_observation)
        timer.start()

    def get_action_space(self):
        return self._action_space.move, self._action_space.swing, self._action_space.fire, self._action_space.apply

    def get_inform(self):
        inform = self.grpcClient.get_inform()
        return inform.score, inform.kills, inform.health, inform.state

    def step(self, move, swing, fire, apply):
        step = self.grpcClient.submit_action(move, swing, fire, apply)
        return step.state

    def get_observation(self):
        return self.layered_observation, self.human_observation

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_space]


ACTION_MEANING = {
    0 : "UP",
    1 : "DOWN",
    2 : "LEFT",
    3 : "RIGHT",
    4 : "UPLEFT",
    5 : "UPRIGHT",
    6 : "DOWNLEFT",
    7 : "DOWNRIGHT"
}


