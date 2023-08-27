---

layout: post
title: "Understanding RSA: Number Theory,Primes Fermat, Euler and such"
date: 2023-08-27 11:12:00-0400
description:
tags: cryptography rsa math euler fermat 
categories: cryptography
related_posts: false
toc:
  beginning: true
  
---

While searching the internet for a somehow in-depth article on the mathematics underpinning RSA encryption, I found myself somewhat dissatisfied with the available resources. This ignited a longstanding aspiration within me to start writing, so here is my debut piece !

---

# What is RSA ? 

RSA, or Rivest-Shamir-Adleman, is a public-key encryption system. It was introduced way back in 1977, making it quite old, but surprisingly, it's still widely used today (although it should not be used anymore) . Now, you might be thinking, "Why write a blog post about an ‘ancient’ encryption method?" Well, that's a fair question. The thing is, RSA isn't just a relic – it's got an educational treasure trove. It's a great example of how the difficulty of prime factorization can be turned into a one-way function. 

First, we'll take a brief look at how RSA generates keys, encrypts, and decrypts data. Then, we'll explore the math behind it, which will help us grasp why RSA was designed the way it is.

---
# Key Generation

RSA key generation involves several steps to generate a public and a private key:

- **Select two large prime numbers:** First, we pick two prime numbers, which should be large enough to ensure the security. These prime numbers are mostly denoted as **$$p$$** and **$$q$$**
- **Calculate the modulo:** The modulo is denoted as **$$n$$** and it defines the modulo algebraic structure that we operate on

$$
n = p \cdot q
$$

- **Evaluate Euler’s Phi (totient) function at _n_:** This is an important step but we will discuss the equation below later as we move forward.

$$
\phi (n) =  (p-1) \cdot (q-1)
$$

- **Choose a public exponent:** Choose a public exponent $$ e $$, which should satisfy the following condition ( $$gcd(x,y)$$ stands for greatest common divisor of $$x$$ and $$y$$ ):

$$ 
1<e<\phi(n) \; \land \; gcd(e, \phi(n)) = 1 
$$

- **Calculate the secret exponent:** Calculate the secret exponent d, such that 

$$
e \cdot d \equiv 1 \; mod \; \phi(n)
$$

Public key $$pk$$ is $$(n,e)$$ and secret key $$sk$$ is $$(n,d)$$.

---

# Encryption and Decryption

Once the public and private keys are generated, encryption and decryption is pretty simple:

$$
Enc(pk, m) := c =  m^e \; mod \; n
$$

$$
Dec(sk, c) = c^d \; mod \; n
$$

---

You might be thinking, "How does this even work? What's the trick?" Well, the magic happens behind the scenes with a bunch of number theory, and we're all set to dive right into it!

Let's introduce you to the mathematics that will help us grasp why RSA functions as it does.

---
# Euler's Totient (Phi) Function

In number theory, Euler's totient function counts the positive integers up to a given integer $$n$$ that are relatively prime to $$n$$. It is written using the Greek letter phi as $$\phi$$, and may also be called Euler's phi function. In other words, it is the number of integers $$k$$ in the range $$1 ≤ k ≤ n$$ for which the greatest common divisor $$gcd(n, k)$$ is equal to $$1$$. 

To illustrate, consider this example:

$$ \displaylines{\textrm{Let } n=6 \\ gcd(0,6) = 6 \;\; ,\;\; gcd(1,6)=1  \;\; ,\;\;  gcd(2,6) = 2 \\ gcd (3,6) = 3  \;\; ,\;\;  gcd(4,6) = 2 \;\; ,\;\; gcd(5,6) = 1} $$

Here, we see that there are only 2 integers, which are relatively prime to 6:

$$
\implies \phi(n) = \phi(6) = 2
$$

But as you can infer from the example, this isn't the most efficient way to calculate Euler's totient function for a given number $$n$$. There's actually a simpler approach to compute this function for any integer, thanks to the fact that Euler's totient function is a multiplicative function, which means:

$$
\displaylines{\textrm{Let } n= p \cdot q \textrm{  and }p \textrm{ and } q \textrm{ are coprime} \\  \phi(n)=\phi(p \cdot q) = \phi(p) \cdot \phi(q)}
$$

Multiplicative property of Euler’s totient function is actually not a trivial property. We can prove this property using Chinese Remainder Theorem (CRT). 
