from grpc_client import GrpcClient
import threading
from utils import bytes_to_array
from rw_lock import RWLock
import time

frame_index = 0
frame_lock = RWLock()
observation_lock = RWLock()

human_observation = None
location_observation = None
immutable_element_observation = None
mutable_element_observation = None
bodies_observation = None
asset_ownership_observation = None
self_asset_observation = None
asset_status_observation = None
pointer_observation = None
observation_state = None


class GameEnvironment:
    def __init__(self, ip, port, api_token,
                 need_human_ob=False,
                 max_containable_step=10,
                 logfile_path='./',
                 debug=False):
        self.grpc_client = GrpcClient(ip, port, api_token, logfile_path, debug)
        self._check_frame = CheckFrame(self.grpc_client, need_human_ob)
        self.need_human_ob = need_human_ob
        self.last_action_frame = 0
        self.max_containable_step = max_containable_step
        self._action_space = None

    def create_room(self, password):
        rsp = self.grpc_client.create_room(password)
        if rsp.err_code == 0:
            self._check_frame.start()
            return rsp.room_id, rsp.state
        else:
            return None

    def join_room(self, room_id, password):
        rsp = self.grpc_client.join_room(room_id, password)
        if rsp.err_code == 0:
            self._check_frame.start()
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

    def submit_reincarnation(self):
        rsp = self.grpc_client.submit_reincarnation()
        if rsp.err_code == 0:
            return rsp.state
        else:
            return None

    def submit_action(self, frame, move, swing, fire, apply):
        """read"""
        frame_lock.acquire_read()
        current_frame = frame
        frame_lock.release()

        if self.last_action_frame < frame < current_frame - self.max_containable_step:
            rsp = self.grpc_client.submit_action(move, swing, fire, apply)
            if rsp.err_code == 0:
                self.last_action_frame = frame
                return rsp.state
            else:
                return None
        else:
            return None

    def get_observation(self):
        """read"""
        observation_lock.acquire_read()
        state = observation_state
        location = location_observation
        immutable_element = immutable_element_observation
        mutable_element = mutable_element_observation
        bodies = bodies_observation
        asset_ownership = asset_ownership_observation
        self_asset = self_asset_observation
        asset_status = asset_status_observation
        pointer = pointer_observation
        human = human_observation
        observation_lock.release()

        frame_lock.acquire_read()
        frame = frame_index
        frame_lock.release()

        if self.need_human_ob:
            return frame, \
                   state, \
                   location, \
                   immutable_element, \
                   mutable_element, \
                   bodies, \
                   asset_ownership, \
                   self_asset, \
                   asset_status, \
                   pointer, \
                   human
        else:
            return frame, \
                   state, \
                   location, \
                   immutable_element, \
                   mutable_element, \
                   bodies, \
                   asset_ownership, \
                   self_asset, \
                   asset_status, \
                   pointer

    def get_move_meanings(self):
        return [MOVE_MEANING[i] for i in self._action_space.move]

    @staticmethod
    def get_state_meanings():
        return STATE_MEANING

    def is_alive(self):
        return self._check_frame.isAlive()


class CheckFrame(threading.Thread):
    def __init__(self, grpc_client, need_human_ob):
        threading.Thread.__init__(self)
        self.grpc_client = grpc_client
        self.timer = None
        self.refresh_observation = RefreshObservation(self.grpc_client, need_human_ob)

    def run(self):
        global frame_index
        check_frame = self.grpc_client.get_frame_index().frame
        if check_frame > frame_index:
            start_time = int(round(time.time() * 1000))
            frame_lock.acquire_write()
            frame_index = check_frame
            frame_lock.release()
            print("write_frame_index time:" + str(int(round(time.time() * 1000)) - start_time))
            start_time = int(round(time.time() * 1000))
            self.refresh_observation.run()
            print("refresh_observation time:" + str(int(round(time.time() * 1000)) - start_time))
        timer = threading.Timer(0.005, self.run)
        timer.start()


class RefreshObservation(threading.Thread):
    def __init__(self, grpc_client, need_human_ob):
        threading.Thread.__init__(self)
        self.grpc_client = grpc_client
        self.need_human_ob = need_human_ob

    def run(self):
        global frame_index, human_observation, location_observation, \
            immutable_element_observation, mutable_element_observation, bodies_observation, \
            asset_ownership_observation, self_asset_observation, pointer_observation, observation_state

        observation_response = self.grpc_client.get_observations()
        frame_lock.acquire_read()
        frame_index = observation_response.frame_index
        frame_lock.release()

        layered_observation = observation_response.layered_observation

        """bytes to numpy"""
        location = bytes_to_array(layered_observation.location)
        immutable_element = bytes_to_array(layered_observation.immutable_element)
        mutable_element = bytes_to_array(layered_observation.mutable_element)
        bodies = bytes_to_array(layered_observation.bodies)
        asset_ownership = bytes_to_array(layered_observation.asset_ownership)
        self_asset = bytes_to_array(layered_observation.self_asset)
        asset_status = bytes_to_array(layered_observation.self_status)
        pointer = bytes_to_array(layered_observation.pointer)
        human = None
        if self.need_human_ob:
            human = bytes_to_array(observation_response.humanObservation)

        """write"""
        observation_lock.acquire_write()
        observation_state = observation_response.state
        location_observation = location
        immutable_element_observation = immutable_element
        mutable_element_observation = mutable_element
        bodies_observation = bodies
        asset_ownership_observation = asset_ownership
        self_asset_observation = self_asset
        asset_ownership_observation = asset_status
        pointer_observation = pointer
        human_observation = human
        observation_lock.release()


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


