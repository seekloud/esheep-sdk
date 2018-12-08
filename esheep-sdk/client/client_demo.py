import sys

import grpc
import service_pb2_grpc as service
import api_pb2 as messages

def run():
    channel = grpc.insecure_channel('localhost:50051')
    try:
        grpc.channel_ready_future(channel).result(timeout=10)
    except grpc.FutureTimeoutError:
        sys.exit('Error connecting to server')
    else:
        stub = service.EsheepAgentStub(channel)
        metadata = [('ip', '127.0.0.1')]
        try:
            response = stub.createRoom(
                messages.Credit(player_id='player_id', api_token="api_token"),
                metadata=metadata,
            )
        except grpc.RpcError as e:
            print('CreateUser failed with {0}: {1}'.format(e.code(), e.details()))
        else:
            print("room created:", response.room_id)


if __name__ == '__main__':
    run()