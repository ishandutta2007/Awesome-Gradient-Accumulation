# The Synchronous Pipelined Micro-Batch Era

## Description
GPipe / 1F1B, 2019–2021.

## Year First Used
2019

## Paper Link
[GPipe (2019)](https://arxiv.org/abs/1811.06965)

## Diagram
```mermaid
flowchart LR
    A[Micro-Batch 1 FWD] --> B[Micro-Batch 2 FWD]
    B --> C[Micro-Batch 1 BWD]
    C --> D[Micro-Batch 2 BWD]
```

[Back to Main Repository](./README.md)
