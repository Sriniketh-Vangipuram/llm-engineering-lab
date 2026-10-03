from dataclasses import dataclass
from enum import Enum


class RequestState(Enum):
    
    WAITING="waiting"
    RUNNING="running"
    FINISHED="finished"

@dataclass
class Request:
    
    request_id:str
    prompt_tokens:int
    expected_output_tokens:int
    priority:int
    slo_seconds:float
    
    state:RequestState=RequestState.WAITING
    
    queue_wait_seconds:float=0.0
    generated_tokens:int=0
    
    @property
    def remaining_output_tokens(self)->int:
        return max(
            0,
            self.expected_output_tokens-self.generated_tokens
        )
        
    @property
    def remaining_slo_slack(self)->float:
        return max(
            0,
            self.slo_seconds-self.queue_wait_seconds
        )
        
    @property
    def total_tokens(self)->int:
        return self.prompt_tokens + self.expected_output_tokens
    
    def __repr__(self)->str:
        
        return (
            f"Request("
            f"id={self.request_id!r},"
            f"prompt={self.prompt_tokens},"
            f"remaining_output={self.remaining_output_tokens},"
            f"priority={self.priority},"
            f"slack={self.remaining_slo_slack:.2f}s"
            f")"
        )