# The Fused Reduce-Scatter State Sharding Era

## Description
ZeRO-Stage 2, 2020–2023.

## Year First Used
2020

## Paper Link
[ZeRO (2020)](https://arxiv.org/abs/1910.02054)

## Diagram
```mermaid
flowchart TD
    A[GPU 1] --> B(Reduce-Scatter)
    C[GPU 2] --> B
    B --> D[Gradient Shard 1]
    B --> E[Gradient Shard 2]
```

[Back to Main Repository](./README.md)
