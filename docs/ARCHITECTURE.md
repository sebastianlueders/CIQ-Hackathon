# Architecture

UI -> local controller -> inference runtime
Approved feedback -> dataset -> Fuzzball training/evaluation -> candidate
Candidate -> quality/compatibility gates -> explicit publication

Fuzzball owns workload lifecycle and placement.
The app owns request admission, skill versions and adapter publication.

Use separate serving/training nodes when available.
On one GPU, defer training during inference or use explicit training windows.

A skill contains instructions, output schema, permitted tools,
optional retrieval, optional LoRA, and an evaluation suite.

Implement one supported model/runtime first. Resolve exact model,
tokenizer, adapter and container versions before GPU execution.
