from grpc_client import GrpcClient
import threading
from utils import to_np_array
from rw_lock import RWLock
import api_pb2 as messages

frame_index = 0
frame_lock = RWLock()

human_observation = None
location_observation = None
immutable_element_observation = None
mutable_element_observation = None
bodies_observation = None
asset_ownership_observation = None
self_asset_observation = None
self_status_observation = None
pointer_observation = None
observation_state = None
observation_lock = RWLock()

score_inform = 0
kill_inform = 0
heath_inform = 0
inform_lock = RWLock()


class GameEnvironment:
    def __init__(self, ip, port, api_token,
                 need_human_ob=False,
                 max_containable_step=10,
                 logfile_path='./',
                 debug=False):
        self.grpc_client = GrpcClient(ip, port, api_token, logfile_path, debug)
        self.need_human_ob = need_human_ob
        self.max_containable_step = max_containable_step
        self._frame_period = self.grpc_client.get_system_info().frame_period
        self._last_action_frame = 0
        self._refresh_obs = RefreshObservation(self.grpc_client, need_human_ob, self._frame_period)
        self._action_space = None
        self._reincarnation_flag = True

    def create_room(self, password):
        rsp = self.grpc_client.create_room(password)
        if rsp.err_code == 0:
            self._refresh_obs.start()
            return rsp.room_id, rsp.state
        else:
            return None

    def join_room(self, room_id, password):
        rsp = self.grpc_client.join_room(room_id, password)
        if rsp.err_code == 0:
            self._refresh_obs.start()
            return rsp.state
        else:
            return None

    def get_action_space(self):
        if self._action_space is None:
            self._action_space = self.grpc_client.get_action_space()
        return self._action_space.move, self._action_space.swing, self._action_space.fire, self._action_space.apply

    def get_inform(self):
        inform = self.grpc_client.get_inform()
        return inform.score, inform.kills, inform.health, inform.state, inform.frame_index

    def get_frame_period(self):
        return self._frame_period

    def submit_reincarnation(self):
        rsp = self.grpc_client.submit_reincarnation()
        if rsp.err_code == 0:
            return rsp.state
        else:
            return None

    def submit_action(self, frame, move, swing, fire, apply):
        """read"""
        frame_lock.acquire_read()
        current_frame = frame_index
        frame_lock.release()
        if self._last_action_frame < frame and frame > current_frame - self.max_containable_step:
            rsp = self.grpc_client.submit_action(move, swing, fire, apply)
            if rsp.err_code == 0:
                self._last_action_frame = frame
                return rsp.state
            else:
                return None
        else:
            return None

    def get_observation_with_info(self):
        """read"""
        observation_lock.acquire_read()
        state = observation_state
        location = location_observation
        immutable_element = immutable_element_observation
        mutable_element = mutable_element_observation
        bodies = bodies_observation
        asset_ownership = asset_ownership_observation
        self_asset = self_asset_observation
        self_status = self_status_observation
        pointer = pointer_observation
        human = human_observation
        observation_lock.release()

        frame_lock.acquire_read()
        frame = frame_index
        frame_lock.release()

        inform_lock.acquire_read()
        score = score_inform
        kill = kill_inform
        heath = heath_inform
        inform_lock.release()

        if self.need_human_ob:
            return frame, \
                   state, \
                   location, \
                   immutable_element, \
                   mutable_element, \
                   bodies, \
                   asset_ownership, \
                   self_asset, \
                   self_status, \
                   pointer, \
                   human, \
                   score, \
                   kill, \
                   heath

        else:
            return frame, \
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
                   heath

    def get_move_meanings(self):
        return [MOVE_MEANING[i] for i in self._action_space.move]

    @staticmethod
    def get_state_meanings():
        return STATE_MEANING


class CheckFrame(threading.Thread):
    def __init__(self, grpc_client, need_human_ob):
        threading.Thread.__init__(self)
        self.grpc_client = grpc_client
        self.need_human_ob = need_human_ob
        self.last_frame = 0

    def run(self):
        stream = self.grpc_client.stub.currentFrame(messages.Credit(api_token=self.grpc_client.api_token))
        for response in stream:
            if response.frame > self.last_frame:
                self.last_frame = response.frame
                refresh_obs = RefreshObservation(self.grpc_client, self.need_human_ob)
                refresh_obs.start()


class RefreshObservation(threading.Thread):
    def __init__(self, grpc_client, need_human_ob, refresh_time):
        threading.Thread.__init__(self)
        self.grpc_client = grpc_client
        self.need_human_ob = need_human_ob
        self.refresh_obs_time = refresh_time

    def run(self):
        global frame_index, human_observation, location_observation, \
            immutable_element_observation, mutable_element_observation, bodies_observation, \
            asset_ownership_observation, self_asset_observation, self_status_observation,\
            pointer_observation, observation_state, score_inform, kill_inform, heath_inform

        response = self.grpc_client.get_observations_with_info()

        if response.frame_index > frame_index:
            frame_lock.acquire_write()
            frame_index = response.frame_index
            frame_lock.release()

            inform_lock.acquire_write()
            score_inform = response.score
            kill_inform = response.kills
            heath_inform = response.heath
            inform_lock.release()

            layered_observation = response.layered_observation

            """to np array"""
            location = to_np_array(layered_observation.location)
            immutable_element = to_np_array(layered_observation.immutable_element)
            mutable_element = to_np_array(layered_observation.mutable_element)
            bodies = to_np_array(layered_observation.bodies)
            asset_ownership = to_np_array(layered_observation.asset_ownership)
            self_asset = to_np_array(layered_observation.self_asset)
            self_status = to_np_array(layered_observation.self_status)
            pointer = to_np_array(layered_observation.pointer)
            human = None
            if self.need_human_ob:
                human = to_np_array(response.humanObservation)

            """write"""
            observation_lock.acquire_write()
            observation_state = response.state
            location_observation = location
            immutable_element_observation = immutable_element
            mutable_element_observation = mutable_element
            bodies_observation = bodies
            asset_ownership_observation = asset_ownership
            self_asset_observation = self_asset
            self_status_observation = self_status
            pointer_observation = pointer
            human_observation = human
            observation_lock.release()
        timer = threading.Timer((self.refresh_obs_time - 10) / 1000, self.run)
        timer.start()


MOVE_MEANING = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "UPLEFT",
    5: "UPRIGHT",
    6: "DOWNLEFT",
    7: "DOWNRIGHT"
}

STATE_MEANING = {
    0: "initGame",
    1: "inGame",
    2: "killed",
    3: "inReplay",
    4: "ended",
    15: "unknown"
}


