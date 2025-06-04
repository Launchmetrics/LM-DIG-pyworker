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
    """
    Return same metrics sent to autoscaler server
    According to lib.metrics.__send_metrics_and_reset compute_autoscaler_data
    """
    #return AutoScalaerData(
    #            id=self.id,
    #            loadtime=(self.system_metrics.model_loading_time or 0.0),
    #            cur_load=(self.model_metrics.workload_processing / elapsed),
    #            max_perf=self.model_metrics.max_throughput,
    #            cur_perf=self.model_metrics.cur_perf,
    #            error_msg=self.model_metrics.error_msg or "",
    #            num_requests_working=len(self.model_metrics.requests_working),
    #            num_requests_recieved=len(self.model_metrics.requests_recieved),
    #            additional_disk_usage=self.system_metrics.additional_disk_usage,
    #            cur_capacity=0,
    #            max_capacity=0,
    #            url=self.url,
    #        )


    
    res = {
        'id': backend.metrics.id,
        'url': backend.metrics.url,
    }
    return web.json_response(res)
    # return web.Response(body=str(backend.metrics))

routes = [
    web.post("/v1/chat/completions", backend.create_handler(ChatHandler())),
    web.get("/ping", handle_ping),
]

if __name__ == "__main__":
    start_server(backend, routes)
