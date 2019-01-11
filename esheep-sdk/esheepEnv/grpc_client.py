import sys
import datetime
import grpc
import service_pb2_grpc as service
import api_pb2 as messages
import time


class GrpcClient:
    def __init__(self, ip, port, api_token, logfile_path='./', debug=False):
        start_time = int(round(time.time() * 1000))
        if debug:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.log_file = open(logfile_path + "log." + current_time, "w")

        self.debug = debug
        self.ip = ip
        self.port = port
        self.api_token = api_token
        self.metadata = [('ip', ip)]

        channel = grpc.insecure_channel(self.ip + ':' + self.port)

        try:
            grpc.channel_ready_future(channel).result(timeout=10)
        except grpc.FutureTimeoutError:
            sys.exit('Error connecting to server')
        else:
            self.stub = service.EsheepAgentStub(channel)

        print("GrpcClient init:" + str(int(round(time.time() * 1000)) - start_time))

    def create_room(self, password):
        start_time = int(round(time.time() * 1000))
        response = self.stub.createRoom(
            messages.CreateRoomReq(
                credit=messages.Credit(api_token=self.api_token),
                password=password)
        )
        if response:
            if self.debug:
                self.log_file.write("create_room response," + "\t" +
                                    "errCode:" + str(response.err_code) + "\t" +
                                    "msg:" + str(response.msg) + "\t" +
                                    "state:" + str(response.state) + "\t" +
                                    "room_id:" + str(response.room_id) + "\n")
            print("create_room:" + str(int(round(time.time() * 1000)) - start_time))
            return response
        else:
            if self.debug:
                self.log_file.write("create_room error:can't get response." + "\n")
            sys.exit('Error when create room')

    def join_room(self, room_id, password):
        response = self.stub.joinRoom(
            messages.JoinRoomReq(
                credit=messages.Credit(api_token=self.api_token),
                password=password,
                room_id=room_id
            )
        )
        if response:
            if self.debug:
                self.log_file.write("join_room response," + "\t" +
                                    "errCode:" + str(response.err_code) + "\t" +
                                    "msg:" + str(response.msg) + "\t" +
                                    "state:" + str(response.state) + "\n")
            return response
        else:
            if self.debug:
                self.log_file.write("join_room error:can't get response." + "\n")
            sys.exit('Error when join room')

    def leave_room(self):
        response = self.stub.leaveRoom(
            messages.Credit(api_token=self.api_token)
        )
        if response:
            if self.debug:
                self.log_file.write("leave_room response," + "\t" +
                                    "errCode:" + str(response.err_code) + "\t" +
                                    "msg:" + str(response.msg) + "\t" +
                                    "state:" + str(response.state) + "\n")
            return response
        else:
            if self.debug:
                self.log_file.write("leave_room error:can't get response." + "\n")
            sys.exit('Error when leave room')

    def get_action_space(self):
        response = self.stub.actionSpace(
            messages.Credit(api_token=self.api_token)
        )
        if response:
            if self.debug:
                self.log_file.write("get_action_space response," + "\t" +
                                    "errCode:" + str(response.err_code) + "\t" +
                                    "msg:" + str(response.msg) + "\t" +
                                    "state:" + str(response.state) + "\n")
            return response
        else:
            if self.debug:
                self.log_file.write("get_action_space error:can't get response." + "\n")
            sys.exit('Error when get action space')

    def submit_action(self, move, swing, fire, apply):
        response = self.stub.action(
            messages.ActionReq(
                move=move,
                swing=swing,
                fire=fire,
                apply=apply,
                credit=messages.Credit(api_token=self.api_token)
            )
        )
        if response:
            if self.debug:
                self.log_file.write("submit_action response," + "\t" +
                                    "errCode:" + str(response.err_code) + "\t" +
                                    "msg:" + str(response.msg) + "\t" +
                                    "state:" + str(response.state) + "\t" +
                                    "frame_index" + str(response.frame_index) + "\n")
            return response
        else:
            if self.debug:
                self.log_file.write("submit_action error:can't get response." + "\n")
            sys.exit('Error when submit action')

    def get_observations(self):
        response = self.stub.observation(
            messages.Credit(api_token=self.api_token)
        )
        if response:
            if self.debug:
                self.log_file.write("get_observation response," + "\t" +
                                    "errCode:" + str(response.err_code) + "\t" +
                                    "msg:" + str(response.msg) + "\t" +
                                    "state:" + str(response.state) + "\t" +
                                    "frame_index" + str(response.frame_index) + "\n")
            return response
        else:
            if self.debug:
                self.log_file.write("get_observation error:can't get response." + "\n")
            sys.exit('Error when get observation')

    def get_inform(self):
        response = self.stub.inform(
            messages.Credit(api_token=self.api_token)
        )
        if response:
            if self.debug:
                self.log_file.write("get_inform response," + "\t" +
                                    "errCode:" + str(response.err_code) + "\t" +
                                    "msg:" + str(response.msg) + "\t" +
                                    "state:" + str(response.state) + "\t" +
                                    "score:" + str(response.score) + "\t" +
                                    "kills:" + str(response.kills) + "\t" +
                                    "heath:" + str(response.heath) + "\t" +
                                    "frame_index:" + str(response.frame_index) + "\n")
            return response
        else:
            if self.debug:
                self.log_file.write("get_inform error:can't get response." + "\n")
            sys.exit('Error when get inform')

    def submit_reincarnation(self):
        response = self.stub.reincarnation(
            messages.Credit(api_token=self.api_token)
        )
        if response:
            if self.debug:
                self.log_file.write("submit_reincarnation response," + "\t" +
                                    "errCode:" + str(response.err_code) + "\t" +
                                    "msg:" + str(response.msg) + "\t" +
                                    "state:" + str(response.state) + "\n")
            return response
        else:
            if self.debug:
                self.log_file.write("submit_reincarnation error:can't get response." + "\n")
            sys.exit('Error when submit reincarnation')

    def get_system_info(self):
        response = self.stub.systemInfo(
            messages.Credit(api_token=self.api_token)
        )
        if response:
            if self.debug:
                self.log_file.write("get_system_info response," + "\t" +
                                    "errCode:" + str(response.err_code) + "\t" +
                                    "msg:" + str(response.msg) + "\t" +
                                    "state:" + str(response.state) + "\t" +
                                    "framePeriod:" + str(response.frame_period) + "\n")
            return response
        else:
            if self.debug:
                self.log_file.write("get_system_info error:can't get response." + "\n")
            sys.exit('Error when get system info')

    def get_frame_index(self):
        response = self.stub.currentFrame(
            messages.Credit(api_token=self.api_token)
        )
        if response:
            if self.debug:
                self.log_file.write("get_frame_index response," + "\t" +
                                    "errCode:" + str(response.err_code) + "\t" +
                                    "msg:" + str(response.msg) + "\t" +
                                    "state:" + str(response.state) + "\t" +
                                    "frame:" + str(response.frame_index) + "\n")
            return response
        else:
            if self.debug:
                self.log_file.write("get_system_info error:can't get response." + "\n")
            sys.exit('Error when get frame index')
