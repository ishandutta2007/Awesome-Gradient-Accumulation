# Pre-Training Web-Scale Foundational LLM Suites

## Description
Application: Scale up token ingestion throughput.

## Year First Used
2023

## Paper Link
[Llama (2023)](https://arxiv.org/abs/2302.13971)

## Diagram
```mermaid
flowchart TD
    A[8M Token Batch] --> B[Micro-Batches of 16K]
    B --> C[FSDP Accumulation]
```

[Back to Main Repository](./README.md)
