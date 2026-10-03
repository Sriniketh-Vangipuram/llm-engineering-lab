from .batch import WorkType
from .scheduler import Scheduler


class InferenceEngine:
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self.step = 0

    def run_step(self):
        batch = self.scheduler.build_batch()

        if not batch.items:
            return None

        print(f"\n--- Step {self.step} ---")
        print(batch)

        # Execute PREFILL first.
        for item in batch.items:
            if item.work_type == WorkType.PREFILL:
                self._execute_prefill(item.request)

        # Move prefetched requests:
        #
        # WAITING -> RUNNING
        #
        waiting_requests = [
            item.request
            for item in batch.items
            if item.work_type == WorkType.PREFILL
        ]

        self.scheduler.admit_requests(waiting_requests)

        # Execute DECODE.
        for item in batch.items:
            if item.work_type == WorkType.DECODE:
                self._execute_decode(item.request)

        self.step += 1

        return batch

    def _execute_prefill(self, request):
        print(
            f"{request.request_id}: "
            f"PREFILL {request.prompt_tokens} tokens"
        )

    def _execute_decode(self, request):
        print(
            f"{request.request_id}: "
            f"DECODE token "
            f"({request.remaining_output_tokens} remaining)"
        )

        request.generated_tokens += 1

        if request.remaining_output_tokens == 0:
            self.scheduler.mark_finished(request)

            print(
                f"{request.request_id}: FINISHED"
            )
