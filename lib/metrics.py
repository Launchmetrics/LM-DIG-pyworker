import os
import time
import logging
import json
from asyncio import sleep
from dataclasses import dataclass, asdict, field
from functools import cache

from lib.data_types import MPSScalerData, SystemMetrics, ModelMetrics
from typing import Awaitable, NoReturn, List

METRICS_UPDATE_INTERVAL = 1

log = logging.getLogger(__file__)


@cache
def get_url() -> str:
    use_ssl = os.environ.get("USE_SSL", "false") == "true"
    worker_port = os.environ['WORKER_PORT']
    public_ip = os.environ["PUBLIC_IPADDR"]
    return f"http{'s' if use_ssl else ''}://{public_ip}:{worker_port}"


@cache
def get_id() -> str:
    if 'NF_POD_ID' in os.environ.keys():
        return os.environ['NF_POD_ID']
    return os.environ['CONTAINER_ID']


@dataclass
class Metrics:
    last_metric_update: float = 0.0
    update_pending: bool = False
    id: str = field(default_factory=get_id)
    url: str = field(default_factory=get_url)
    system_metrics: SystemMetrics = field(default_factory=SystemMetrics.empty)
    model_metrics: ModelMetrics = field(default_factory=ModelMetrics.empty)
    # cache last metrics for ping pull
    last_metrics: dict = field(default_factory=lambda: {})

    def _request_start(self, num_req: int, workload: int, reqnum: int) -> None:
        """
        this function is called prior to forwarding a request to a model API.
        """
        log.debug("request start")
        self.model_metrics.requests_received += num_req
        self.model_metrics.workload_pending += workload
        self.model_metrics.batches_recieved.add(reqnum)
        self.model_metrics.requests_working.add(reqnum)

    def _request_end(self, workload: int, reqnum: int) -> None:
        """
        this function is called after handling of a request ends, regardless of the outcome
        """
        self.model_metrics.workload_pending -= workload
        self.model_metrics.requests_working.discard(reqnum)

    def _request_success(self, workload: int) -> None:
        """
        this function is called after a response from model API is received and forwarded.
        """
        self.model_metrics.workload_served += workload
        self.update_pending = True

    def _request_errored(self, workload: int) -> None:
        """
        this function is called if model API returns an error
        """
        self.model_metrics.workload_errored += workload

    def _request_canceled(self, workload: float) -> None:
        """
        this function is called if client drops connection before model API has responded
        """
        self.model_metrics.workload_cancelled += workload

    async def _send_metrics_loop(self) -> Awaitable[NoReturn]:
        while True:
            await sleep(METRICS_UPDATE_INTERVAL)
            elapsed = time.time() - self.last_metric_update
            if self.system_metrics.model_is_loaded is False and elapsed >= 10:
                log.debug(
                    f"sending loading model metrics after {int(elapsed)}s wait"
                )
                self.__send_metrics_and_reset(elapsed)
            elif self.update_pending or elapsed > 10:
                log.debug(
                    f"sending loaded model metrics after {int(elapsed)}s wait"
                )
                self.__send_metrics_and_reset(elapsed)

    def _model_loaded(self, max_throughput: float) -> None:
        self.system_metrics.model_loading_time = (
            time.time() - self.system_metrics.model_loading_start
        )
        self.system_metrics.model_is_loaded = True
        self.model_metrics.max_throughput = max_throughput

    def _model_errored(self, error_msg: str) -> None:
        self.model_metrics.set_errored(error_msg)
        self.system_metrics.model_is_loaded = True

    ####################################### Private#######################################

    def __send_metrics_and_reset(self, elapsed):

        def compute_mps_scaler_data() -> MPSScalerData:
            mps_scaler_data = MPSScalerData(
                id=self.id,
                loadtime=(self.system_metrics.model_loading_time or 0.0),
                cur_load=(self.model_metrics.workload_pending / elapsed),
                max_perf=self.model_metrics.max_throughput,
                cur_perf=self.model_metrics.cur_perf,
                error_msg=self.model_metrics.error_msg or "",
                workload_pending=self.model_metrics.workload_pending,
                num_requests_working=len(
                    self.model_metrics.requests_working
                ),
                num_requests_received=self.model_metrics.requests_received,
                num_batches_recieved=len(
                    self.model_metrics.batches_recieved
                ),
                additional_disk_usage=self.system_metrics.additional_disk_usage,
                cur_capacity=0,
                max_capacity=0,
                url=self.url,
            )
            log.debug(
                "\n".join(
                    [
                        "#" * 60,
                        f"compute_MPS_scaler_data:",
                        f"{json.dumps((asdict(mps_scaler_data)), indent=2)}",
                        "#" * 60,
                    ]
                )
            )
            # cache last metrics for ping pull
            self.last_metrics = asdict(mps_scaler_data)
            return mps_scaler_data

        ###########

        self.system_metrics.update_disk_usage()
        compute_mps_scaler_data()  # update self.last_metrics

        self.update_pending = False
        self.model_metrics.reset()
        self.system_metrics.reset()
        self.last_metric_update = time.time()
