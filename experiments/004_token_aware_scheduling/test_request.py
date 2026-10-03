from llm_engineering_lab.inference.request import Request


request = Request(
    request_id="A",
    prompt_tokens=300,
    expected_output_tokens=800,
    priority=10,
    slo_seconds=2.0,
    queue_wait_seconds=1.2,
    generated_tokens=200,
)

print(request)

print("Remaining output:", request.remaining_output_tokens)
print("Remaining SLO slack:", request.remaining_slo_slack)
print("Total tokens:", request.total_tokens)