from dataclasses import dataclass
from typing import List

from .batch import Batch, BatchItem, WorkType
from .request import (
    Request,
    RequestState,
)


@dataclass
class SchedulerConfig:
    max_batch_tokens:int=500
    
class Scheduler:
    def __init__(self,config:SchedulerConfig):
        
        self.config=config
        self.waiting_queue:List[Request]=[]
        self.running_requests:List[Request]=[]
        
        self.finished_requests:List[Request]=[]
        
        
    def add_request(self,request:Request)->None:
        request.state=RequestState.WAITING
        self.waiting_queue.append(request)
        
    def _estimated_service_time(self,request:Request)->float:
        
        """
        Very simple simulation model.
        """
        return request.remaining_output_tokens*0.01
    
    def _is_feasible(self,request:Request)->bool:
        
        estimated_time=self._estimated_service_time(request)
        
        return request.remaining_slo_slack>=estimated_time
    
    def _urgency_score(self,request:Request)->float:
        
        """
        lower slack = more urgent
        """
        
        return request.remaining_slo_slack
    
    def schedule_waiting(self)->List[Request]:
        
        if not self.waiting_queue:
            return []
        
        feasible=[
            request
            for request in self.waiting_queue
            if self._is_feasible(request)
        ]
        
        feasible.sort(
            key=lambda request:(
                self._urgency_score(request),
                -request.priority,
                request.remaining_output_tokens,
            )
        )
        
        selected=[]
        
        token_budget=0
        
        for request in feasible:
            
            request_tokens=request.prompt_tokens
                
            if token_budget+request_tokens >self.config.max_batch_tokens:
                continue
            
            selected.append(request)
            
            token_budget+=request_tokens
            
        return selected
    
    def admit_requests(self,requests:List[Request])->None:
        
        for request in requests:
            
            if request in self.waiting_queue:
                self.waiting_queue.remove(request)
                
            request.state=RequestState.RUNNING
            
            if request not in self.running_requests:
                self.running_requests.append(request)
                
    def mark_finished(self,request:Request)->None:
        
        if request in self.running_requests:
            self.running_requests.remove(request)
            
        request.state=RequestState.FINISHED
        
        if request not in self.finished_requests:
            self.finished_requests.append(request)
            
    def build_batch(self)->Batch:
        
        selected_waiting=self.schedule_waiting()
        
        items=[]
        
        for request in selected_waiting:
            items.append(
                BatchItem(
                    request=request,
                    work_type=WorkType.PREFILL,
                    tokens=request.prompt_tokens,
                )
                
            )
                            
        for request in self.running_requests:
            items.append(
                BatchItem(
                    request=request,
                    work_type=WorkType.DECODE,
                    tokens=1,
                )
            )
                
        return Batch(items)