import os
import logging
from typing import Union, Type
import dataclasses

from aiohttp import web, ClientResponse

from lib.backend import Backend, LogAction
from lib.data_types import EndpointHandler
from lib.server import start_server
from .data_types import InputData


MODEL_SERVER_URL = "http://0.0.0.0:5001"

# This is the last log line that gets emitted once comfyui+extensions have been fully loaded
MODEL_SERVER_START_LOG_MSG = '"message":"Connected","target":"text_generation_router::server"'
MODEL_SERVER_ERROR_LOG_MSGS = [
    "Error: WebserverFailed", "Error: DownloadError", "Error: ShardCannotStart"
]


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


@dataclasses.dataclass
class ChatHandler(EndpointHandler[InputData]):

    @property
    def endpoint(self) -> str:
        return "/v1/chat/completions"

    @classmethod
    def payload_cls(cls) -> Type[InputData]:
        return InputData

    def make_benchmark_payload(self) -> InputData:
        return InputData.for_test()

    async def generate_client_response(
        self, client_request: web.Request, model_response: ClientResponse
    ) -> Union[web.Response, web.StreamResponse]:
        _ = client_request
        match model_response.status:
            case 200:
                log.debug("SUCCESS")
                data = await model_response.json()
                return web.json_response(data=data)
            case code:
                log.debug("SENDING RESPONSE: ERROR: unknown code")
                return web.Response(status=code)


backend = Backend(
    model_server_url=MODEL_SERVER_URL,
    model_log_file=os.environ["MODEL_LOG"],
    allow_parallel_requests=True,
    benchmark_handler=ChatHandler(benchmark_runs=3, benchmark_words=256),
    log_actions=[
        (LogAction.ModelLoaded, MODEL_SERVER_START_LOG_MSG),
        (LogAction.Info, '"message":"Download'),
        *[
            (LogAction.ModelError, error_msg)
            for error_msg in MODEL_SERVER_ERROR_LOG_MSGS
        ],
    ],
)


async def handle_ping(_):
    return web.Response(body=str(backend.metrics.asdict()))

routes = [
    web.post("/v1/chat/completions", backend.create_handler(ChatHandler())),
    web.get("/ping", handle_ping),
]

if __name__ == "__main__":
    start_server(backend, routes)
