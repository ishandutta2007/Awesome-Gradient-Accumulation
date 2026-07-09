# The Precision Underflow Loss Saturation Crisis

## Description
The Problem: Underflow Errors.

## Year First Used
2017

## Paper Link
[Mixed Precision (2017)](https://arxiv.org/abs/1710.03740)

## Diagram
```mermaid
flowchart LR
    A[Loss * 65536] --> B[Backprop in FP16] --> C[Grad / 65536 in FP32]
```

[Back to Main Repository](./README.md)
