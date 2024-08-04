---
layout: distill
title: 'Flash Attention: A Brief Overview'
description: 
tags: flash attention machine learning
giscus_comments: true
date: 2024-08-02
featured: true

authors:
  - name: Rodi Düger
    affiliations:
      name: KIT

bibliography: 2024-08-02-flash-attention-brief-overview.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Introduction
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Related Work
  - name: Vanilla Attention
  - name: Flash Attention

---

# Introduction

Transformer architecture <d-cite key="aiayn"></d-cite> has been a milestone for many deep learning application areas, particularly in NLP domain as the backbone of most large language models (LLMs). Scaling up these models has been the key factor allowing them to achieve their high levels of performance and capabilities <d-cite key="scaling_laws_openai,scaling_laws_deepmind"></d-cite>. As the models grow larger, trained on more data with increased computational resources, they are able to learn more comprehensive patterns and representations, leading to improvements in understanding and generating human language as well as solving complex tasks <d-cite key="emergent_cap"></d-cite>.  

The core component of the Transformer architecture is the attention mechanism, which allows embeddings to incorporate contextual information. The standard implementation of the attention mechanism is slow due to its quadratic time and memory complexity and hence becomes a computational bottleneck, especially for long sequences. As a consequence, a primary challenge with scaling up these models is efficiency.

To address this efficiency problem, FlashAttention <d-cite key="flashattention"></d-cite> has been proposed as an exact _IO-aware_ attention algorithm. Rather than focusing on reducing
the computation of the attention algorithm, FlashAttention reduces the number of IO
operations between the GPU’s relatively slow high-bandwidth memory (HBM)
and fast on-chip SRAM and effectively utilizes the asymmetric
memory hierarchy in graphics processing units (GPUs).

# Related Work

Many approaches have been proposed to address the quadratic time and memory complexity of Transformers and to scale them to longer sequences. Some of these approaches include low-rank approximation <d-cite key="performer, performer_sublinear"></d-cite>, sparse approximation <d-cite key="reformer"></d-cite>, a combination of both <d-cite key="longformer"></d-cite>, and compression along the sequence <d-cite key="transformerx1"></d-cite>. Most of these methods offer a trade-off between efficiency and accuracy. FlashAttention distinguishes itself among these methods as an efficient and exact alternative.

# Vanilla Attention

In its simplified form without the scaling factor before applying the softmax, the vanilla attention computation can be written as 

$$
O = \text{softmax}(QK^T)V,
$$

where $$O,Q,K,V \in \mathbb{R}^{N \times d} $$.
The vanilla attention algorithm computes the output as following:

-   Load $$Q$$ and $$K$$ by blocks from HBM to SRAM \| <span
    style="color: red">IO operation !</span>

-   Compute the intermediate result $$S_0 = QK^T\in \mathbb{R}^{N \times N}$$

-   Write the intermediate result $$S_0$$ to HBM \| <span
    style="color: red">IO operation !</span>

-   Load *$$S_0$$ from HBM to SRAM \| <span style="color: red">IO
    operation !</span>

-   Apply softmax to $$S_0$$ along the second dimension, which
    results in the intermediate result
    $$S_1 = \text{softmax}(S_0) \in \mathbb{R}^{N \times N}$$

-   Write $$S_1$$ to HBM \| <span style="color: red">IO
    operation !</span>

-   Load $$S_1$$  and $$V$$ by blocks from HBM to SRAM

-   Compute the output $$O = S_1V$$

-   Write $$O$$ to HBM \| <span style="color: red">IO operation !</span>


Even in its most simplified form, attention computation requires data to move between HBM and SRAM several times due to the limited capacity of SRAM, which is e.g. approximately 20 MB in the NVIDIA A100, given it contains 108 streaming multiprocessors each equipped with 192 KB of SRAM <d-cite key="nvidiaa100"></d-cite>. Additional memory-bound operations, such as masking and dropout, also increase the IO overhead of the computation.

This demonstrates that the vanilla attention algorithm does not account for the cost of HBM reads and writes, making it _IO-unaware_. FlashAttention addresses this problem.

# Flash Attention

In contrast to the vanilla attention algorithm, FlashAttention computes exact attention with fewer HBM reads and writes. It achieves this by applying two well-established optimization techniques: tiling and recomputation. The key idea behind FlashAttention is to avoid materializing intermediate matrices and to fuse all CUDA kernels (matrix multiplication, softmax etc.) used in the vanilla attention computation into one as depicted in \cref{fig:flashattention}.

<pre id="online_softmax" class="pseudocode">
\begin{algorithm}
\caption{Online Softmax Function}
\label{alg:online_softmax}
\begin{algorithmic}[1]
\Function{OnlineSoftmax}{$x$}
    \State $m \gets [-\infty] \times (d+1)$ \Comment{Initialize $m$ with $-\infty$}
    \State $\ell \gets [0] \times (d+1)$ \Comment{Initialize $d$ with 0}

    \For{$i \gets 0$ to $d$}
        \State $m_{i + 1} \gets \max(m_i, x_i)$ \Comment{Update $m$}
        \State $\ell_{i + 1} \gets \ell_i \times e^{m_i - m_{i+1}} + e^{x_i - m_{i+1}}$ \Comment{Update $\ell$}
    \EndFor
    \State $f \gets [e^{x_0}\, \cdots \, e^{x_d}]$
    \State \Return $\frac{fe^{- m_{d+1}}}{\ell_{d+1}}$
\EndFunction
\end{algorithmic}
\end{algorithm}
</pre>