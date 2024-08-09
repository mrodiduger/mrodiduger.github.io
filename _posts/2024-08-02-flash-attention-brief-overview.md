---
layout: distill
title: "Flash Attention: A Brief Overview"
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
  - subsections:
      - name: Tiling
      - name: Recomputation
  - name: Experimental Results
---

## Introduction

Transformer architecture <d-cite key="aiayn"></d-cite> has been a milestone for many deep learning application areas, particularly in NLP domain as the backbone of most large language models (LLMs). Scaling up these models has been the key factor allowing them to achieve their high levels of performance and capabilities <d-cite key="scaling_laws_openai,scaling_laws_deepmind"></d-cite>. As the models grow larger, trained on more data with increased computational resources, they are able to learn more comprehensive patterns and representations, leading to improvements in understanding and generating human language as well as solving complex tasks <d-cite key="emergent_cap"></d-cite>.

The core component of the Transformer architecture is the attention mechanism, which allows embeddings to incorporate contextual information. The standard implementation of the attention mechanism is slow due to its quadratic time and memory complexity and hence becomes a computational bottleneck, especially for long sequences. As a consequence, a primary challenge with scaling up these models is efficiency.

To address this efficiency problem, FlashAttention <d-cite key="flashattention"></d-cite> has been proposed as an exact _IO-aware_ attention algorithm. Rather than focusing on reducing
the computation of the attention algorithm, FlashAttention reduces the number of IO
operations between the GPU’s relatively slow high-bandwidth memory (HBM)
and fast on-chip SRAM and effectively utilizes the asymmetric
memory hierarchy in graphics processing units (GPUs).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flash_attention/gpu_hierarchy.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flash_attention/attention_wallclock.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 1. <b>LEFT:</b>  GPU memory hierarchy. Relative to SRAM, HBM is slower but has more memory. <b>RIGHT:</b>  Comparison of wallclock time needed for each operation in PyTorch implementation of vanilla attention and FlashAttention. Figure taken from Tri Dao et al.<d-cite key="flashattention"></d-cite>.
</div>

## Related Work

Many approaches have been proposed to address the quadratic time and memory complexity of Transformers and to scale them to longer sequences. Some of these approaches include low-rank approximation <d-cite key="performer, performer_sublinear"></d-cite>, sparse approximation <d-cite key="reformer"></d-cite>, a combination of both <d-cite key="longformer"></d-cite>, and compression along the sequence <d-cite key="transformerx1"></d-cite>. Most of these methods offer a trade-off between efficiency and accuracy. FlashAttention distinguishes itself among these methods as an efficient and exact alternative.

## Vanilla Attention

In its simplified form without the scaling factor before applying the softmax, the vanilla attention computation can be written as

$$
O = \text{softmax}(QK^T)V,
$$

where $$O,Q,K,V \in \mathbb{R}^{N \times d} $$.
The vanilla attention algorithm computes the output as following:

- Load $$Q$$ and $$K$$ by blocks from HBM to SRAM \| <span
  style="color: red">IO operation !</span>

- Compute the intermediate result $$S_0 = QK^T\in \mathbb{R}^{N \times N}$$

- Write the intermediate result $$S_0$$ to HBM \| <span
  style="color: red">IO operation !</span>

- Load \*$$S_0$$ from HBM to SRAM \| <span style="color: red">IO
  operation !</span>

- Apply softmax to $$S_0$$ along the second dimension, which
  results in the intermediate result
  $$S_1 = \text{softmax}(S_0) \in \mathbb{R}^{N \times N}$$

- Write $$S_1$$ to HBM \| <span style="color: red">IO
  operation !</span>

- Load $$S_1$$ and $$V$$ by blocks from HBM to SRAM

- Compute the output $$O = S_1V$$

- Write $$O$$ to HBM \| <span style="color: red">IO operation !</span>

Even in its most simplified form, attention computation requires data to move between HBM and SRAM several times due to the limited capacity of SRAM, which is e.g. approximately 20 MB in the NVIDIA A100, given it contains 108 streaming multiprocessors each equipped with 192 KB of SRAM <d-cite key="nvidiaa100"></d-cite>. Additional memory-bound operations, such as masking and dropout, also increase the IO overhead of the computation.

This demonstrates that the vanilla attention algorithm does not account for the cost of HBM reads and writes, making it _IO-unaware_. FlashAttention addresses this problem.

## Flash Attention

In contrast to the vanilla attention algorithm, FlashAttention computes exact attention with fewer HBM reads and writes. It achieves this by applying two well-established optimization techniques: tiling and recomputation. The key idea behind FlashAttention is to avoid materializing intermediate matrices and to fuse all CUDA kernels (matrix multiplication, softmax etc.) used in the vanilla attention computation into one as depicted in \cref{fig:flashattention}.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flash_attention/flashattention.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
  Figure 2. Overview of FlashAttention algorithm. Figure taken from Tri Dao et. al.<d-cite key="flashattention"></d-cite>.
</div>

# Tiling

A major challenge in tiling the attention computation lies in the non-associative nature of the softmax function. Traditionally, softmax of a vector $$x \in \mathbb{R}^d$$, which can be thought as a row of the intermediate result $$S_0$$, is computed using an algorithm called "safe softmax" for numerical stability as in Algorithm 1.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flash_attention/safe_softmax.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
  Algorithm 1. Pseudocode of safe softmax algorithm.
</div>

The problem with using safe softmax while computing the attention is that it requires three iterations over the entire input vector $$x$$: one iteration to determine the maximum value $$m$$, one iteration to calculate the normalizer $$l$$ and one iteration to calculate the final output $$o$$. This, consequently leads to reads/writes from/to HBM since SRAM does not have enough capacity to materialize the entire intermediate matrix. On the other hand, online softmax <d-cite key="online_softmax"></d-cite> depicted in Algorithm 2, offers an alternative to safe softmax to calculate the maximum value $$m$$ and normalizer $$l$$ in an online manner in a single loop.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flash_attention/online_softmax.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
  Algorithm 1. Pseudocode of online softmax algorithm.
</div>

Although computing the attention matrix $$S_1$$ with the online softmax still requires two loops and hence a read/write from/to HBM, it is not necessary to materialize the attention matrix $$S_1$$ to compute the output of the atttention $$O = S_1 \cdot V$$. Thus, the output can be computed in blocks directly in a single loop with a low memory footprint that fits into the SRAM. The derivation and details of this single-loop computation are beyond the scope of this review and are left to the reader for further reading <d-cite key="from_online_softmax_to_flashattention"></d-cite>.

# Recomputation

In the context of performance optimization, recomputation refers to the concept that, in certain scenarios, recomputing data may be faster than storing intermediate results and accessing them from memory. As we discussed in [Tiling](#tiling), FlashAttention avoids materializing the intermediate matrices $$S_0$$ and $$S_1$$. As a consequence, it can also not read the intermediate matrices during the backward pass, as they are never materialized and stored. Instead, FlashAttention stores the softmax normalization statistics $$m$$ and $$\ell$$ and recomputes the $$S_0$$ and $$S_1$$ to compute the gradients of $$O$$ with respect to $$Q, K$$ and $$V$$. Although recomputation results in more FLOPs, it improves the wall clock time of the algorithm, as the slow HBM is accessed fewer times.

## Experimental Results

In this section, we analyze the experimental results of the FlashAttention.Figure 3 demonstrates the reduction in HBM memory usage compared to the vanilla attention algorithm. Since FlashAttention does not materialize the $$N \times N$$ intermediate matrices, it only requires $$O(N)$$ additional HBM memory for the output and softmax statistics as opposed to $$O(N^2)$$ memory requirement of the vanilla attention algorithm. This results in a quadratic increase in memory reduction with respect to the sequence length $$N$$.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flash_attention/flashattn_memory.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/flash_attention/flashattn_speedup.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
  <b>LEFT:</b> Figure 3. Memory reduction of FlashAttention over standard PyTorch attention implementation at different sequence lengths. <b>RIGHT:</b> Figure 4. Wallclock-time speedup of FlashAttention over standard PyTorch attention implementation at different sequence lengths on NVIDIA A100. Figures taken from Tri Dao et al.<d-cite key="flashattention"></d-cite>.
</div>

In addition to memory reduction, FlashAttention is also faster compared to the vanilla attention algorithm as depicted in Figure 4. The speedup is particularly significant when optional dropout and masking operations are applied during the attention computation. This behavior is expected, as the optimizations employed in FlashAttention aim to reduce the I/O complexity of the vanilla attention algorithm. Memory-bound operations, such as masking and dropout, are the primary sources of bottlenecks in terms of wall clock time.

## Personal Comment

"Attention is All You Need" <d-cite key="aiayn"></d-cite> in 2017 marked a pivotal moment, establishing the Transformer architecture and attention mechanism as fundamental building blocks for many groundbreaking research endeavors and widely-used products. What, I find particularly interesting about FlashAttention is how, despite several years of research in one of the most rapidly evolving scientific domains, such an elegant yet "simple" line of optimization could be overlooked for arguably one of the most crucial operations. This is especially surprising given the significant financial incentives and vast resources available to companies that would benefit from an algorithm such as FlashAttention. Of course, hindsight is 20/20. Something appears obvious and simple after it has been discovered, even though it was not initially apparent.

By the way, FlashAttention 2 <d-cite key="flashattention2"></d-cite> and FlashAttention 3 <d-cite key="flashattention3"></d-cite> are available as even more optimized attention kernels, and their adoption has been widespread across the industry. In that regard, I have become a big fan of Tri Dao's research. I understand that academic and industrial research are driven by different motivations, but I believe more academics should pay attention to real-world use cases and their computational constraints.
