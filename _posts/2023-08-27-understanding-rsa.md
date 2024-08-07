---
layout: post
title: "Understanding RSA: Number Theory,Primes, Fermat, Euler and such"
date: 2023-08-28 11:12:00-0400
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

RSA, or Rivest-Shamir-Adleman, is a public-key encryption system. It was introduced way back in 1977, making it quite old, but surprisingly, it's still widely used today (although it should not be used anymore[¹](https://blog.trailofbits.com/2019/07/08/fuck-rsa/)) . Now, you might be thinking, "Why write a blog post about an ‘ancient’ encryption method?" Well, that's a fair question. The thing is, RSA isn't just a relic – it's got an educational treasure trove. It's a great example of how the difficulty of prime factorization can be turned into a one-way function.

First, we'll take a brief look at how RSA generates keys, encrypts, and decrypts data. Then, we'll explore the math behind it, which will help us grasp why RSA was designed the way it is.

---

# Key Generation

RSA key generation involves several steps to generate a public and a private key:

- **Select two large prime numbers:** First, we pick two prime numbers **$$p$$** and **$$q$$**, which should be large enough to ensure the security. Note that $$p \neq q$$
- **Calculate the modulo:** The modulo is denoted as **$$n$$** and it defines the modulo algebraic structure that we operate on

$$
n = p \cdot q
$$

- **Evaluate Euler’s Phi (totient) function at $$n$$:** This is an important step but we will discuss the equation below later as we move forward.

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

$$ \displaylines{\textrm{Let } n=6 \\ gcd(0,6) = 6 \;\; ,\;\; gcd(1,6)=1 \;\; ,\;\; gcd(2,6) = 2 \\ gcd (3,6) = 3 \;\; ,\;\; gcd(4,6) = 2 \;\; ,\;\; gcd(5,6) = 1} $$

Here, we see that there are only 2 integers, which are relatively prime to 6:

$$
\implies \phi(n) = \phi(6) = 2
$$

But as you can infer from the example, this isn't the most efficient way to calculate Euler's totient function for a given number $$n$$. There's actually a simpler approach to compute this function for any integer, thanks to the fact that Euler's totient function is a multiplicative function, which means:

$$
\displaylines{\textrm{Let } n= p \cdot q \textrm{  and }p \textrm{ and } q \textrm{ are coprime} \\  \phi(n)=\phi(p \cdot q) = \phi(p) \cdot \phi(q)}
$$

Multiplicative property of Euler’s totient function is actually not a trivial property. We can prove this property using Chinese Remainder Theorem (CRT).

## Chinese Remainder Theorem

The Chinese Remainder Theorem (CRT) is a fundamental theorem in number theory that deals with solving systems of simultaneous modular congruences. It provides a way to find a unique solution to a set of congruences when the moduli involved are pairwise coprime (i.e., they have no common factors other than 1). The theorem is named after its historical association with ancient Chinese mathematics.

**Theorem:**

Let $$n_1​,n_2​,…,n_k$$ be pairwise coprime positive integers and let $$a_1,a_2,...,a_k$$ be any set of integers. Then the system of modular congruences

$$
\displaylines{x \equiv a_1 \;(mod \;n_1)\\x \equiv a_2 \;(mod \; n_2)\\.\\.\\x \equiv \; a_k \; (mod \; n_k) }
$$

has a **unique** solution in $$mod \; N$$, whereas $$N:= n_1 \cdot n_2 \cdot...\cdot n_k$$

Essentially, what the thorem states is that, the map

$$
x \; mod \; N \mapsto (x \; mod \; n_1,x \; mod \; n_2,...,x \; mod \; n_k )
$$

defines an isomorphism between $$\mathbb{Z}/N\mathbb{Z}$$ and $$\mathbb{Z}/n_1\mathbb{Z}\times\mathbb{Z}/n_2\mathbb{Z}\times...\times\mathbb{Z}/n_k\mathbb{Z}$$

<br/>
<br/>
<br/>

Of course, I'm not going to delve into a formal proof explaining why Euler's totient function acts as a multiplicative function under certain assumptions. However, what's essential to recognize here is that the isomorphism between these two rings leads to the pivotal observation of Euler's totient function possessing a multiplicative property.

There's another observation we can make about Euler's totient function, and it will come in handy as we proceed:

---

Let $$p$$ be a prime and $$k \geq1$$. Then the following equation holds:

$$
\phi(p^k) = p^k - p^{k-1}
$$

**Proof:** Since $$p$$ is a prime number, the only possible values of $$gcd(p^k, m)$$ are $$1, p, p^2, ..., p^k$$, and the only way to have $$gcd(p^k, m) > 1$$ is if $$m$$ is a multiple of $$p$$, that is, $$m \in {p, 2p, 3p, ..., p^{k − 1} \cdot p = p^k}$$, and there are $$p^{k − 1}$$ such multiples not greater than $$p^k$$. Therefore, the other $$p^k − p^{k − 1}$$ numbers are all relatively prime to $$p^k$$[²](https://en.wikipedia.org/wiki/Euler%27s_totient_function).

**Example:**
Let's compute $$\phi(3^2)$$:

$$
\displaylines{gcd(0,9) = 9 \;\; ,\;\; gcd(1,9)=1  \;\; ,\;\;  gcd(2,9) = 1 \\ gcd (3,9) = 3  \;\; ,\;\;  gcd(4,9) = 1 \;\; ,\;\; gcd(5,9) = 1 \\gcd(6,9) = 3 \;\; ,\;\; gcd(7,9)=1  \;\; ,\;\;  gcd(8,9) = 1\\ \implies \phi(3^2) = 6 = 3^2 - 3^1}
$$

---

Now let's take a look at the simpler approach to compute this function for any integer:

---

Let $$m$$ has the following prime factorization:

$$
m = p_1^{e_1} \cdot p_2^{e_2}\cdot...\cdot p_n^{e_n}
$$

whereas $$e_i \in \mathbb{N}$$. With a little assistance from our previous observation, we can assert the following:

$$
\phi(m) = \prod_{i=1}^n\phi(p_i^{e^i}) = \prod_{i=1}^n(p_i^{e_i} - p_i^{e_i -1})
$$

---

Now it should be more clear how we computed $$\phi(n)$$ as $$\phi(n) = \phi(p) \cdot \phi(q)$$ in the key generation section.

Prime factorization of $$n$$ is $$n = p^1 \cdot q^1$$, then:

$$
\phi(n) = \phi(p^1 \cdot q^1) = \phi(p^1) \cdot \phi(q^1) = (p^1-p^0)\cdot(q^1-q^0) = (p-1)\cdot(q-1)
$$

This insight should shed light on why we generate prime numbers to generate keys in the first place. It's crucial to emphasize that this computation becomes feasible only when we possess the prime factorization of a given integer $$n$$. Yet, for a large $$n$$, it might prove impractical to factorize it into its prime components within a reasonable timeframe. Exactly for this reason, we emphasize that **"RSA uses the complexity of prime factorization to guarantee its security,"** as this very characteristic forms the heart of the RSA cryptosystem.

If we're aware of the prime factors of a specific number $$n$$, we can efficiently calculate the Euler's totient function $$\phi(n)$$, allowing for efficient decryption. However, when we lack knowledge of the prime factors of $$n$$ efficiently computing $$\phi(n)$$ becomes a daunting task. We could resort to a brute-force approach, yet this becomes unfeasible when dealing with significantly large values of $$n$$, because of the immense time it would demand.

---

Now that we've understood some key features of Euler's totient function that are vital for key generation, let's address the question: **"Why do we even use Euler's totient function?"**

Let's explore the answer.

# Euler's Theorem

Euler's Theorem is integral to the security and functioning of RSA encryption. The use of modular exponentiation in both encryption and decryption operations leverages Euler's Theorem. Here's how:

---

**Euler's Theorem:**

Let $$a$$ and $$m$$ positive coprime integers ($$gcd(a,m)=1$$), then:

$$
a^{\phi(m)} \equiv 1 \; (mod \; m)
$$

**Example:**

Let $$m =10$$ and $$a=3$$.

Since $$ gcd(10,3) = 1 $$,  $$\;a^{\phi(m)}$$ should be equivalent to $$1 \; (mod \; m)$$. Let's check:

$$
\phi(m) = \phi(10) = \phi(5 \cdot 2) = (5-1) \cdot (2-1) = 4
$$

$$
a^{\phi(m)} = 3^4 =81 \equiv 1 \; (mod \; 10)
$$

---

Unfortunately, I'm not a mathematician. :/ So, I don't want to explain a proof, which I'm not really familiar with. Let's just accept Euler's theorem as a given truth, which, in fact, it is :D

Let's now look at a special case of Euler's Theorem, which is Fermat's Little Theorem !

---

**Fermat's Little Theorem:**

Let $$a$$ be a positive integer, $$p$$ be a prime and $$gcd(a,p) = 1$$. Then:

$$
a^p \equiv a \; (mod \; p)
$$

$$
\iff
$$

$$
a \cdot (a^{p-1} -1) \equiv 0 \; (mod \; p)
$$

$$
\iff
$$

$$
a^{p-1} \equiv 1 \; (mod \; p)
$$

**Proof**: Fermat's Little Theorem is basically a special case of Euler's theorem, if the modulo is a prime number.

Let $$a$$ be a positive integer, $$p$$ be a prime and $$gcd(a,p) = 1$$.

Since $$gcd(a,p)=1$$, using Euler's Theorem:

$$
a^{\phi(p)} \equiv 1 \; (mod \; p)
$$

We know that $$\phi(p) = p-1$$, if $$p$$ is a prime.

$$
\implies a^{p-1} \equiv 1 \; (mod \; p)
$$

---

# Correctness of RSA

Now, armed with all the necessary tools, we can demonstrate that the $$Enc(pk,m)$$ and $$Dec(sk,c)$$ operations within the RSA cryptosystem act as perfect inverses of one another.

---

**Lemma(Correctness):** A public key algorithm $$(Gen,Enc,Dec)$$ is correct if

$$
\forall m,pk,sk: (pk,sk) \leftarrow Gen(1^k) \implies Dec(sk, Enc(pk,m)) = m
$$

---

**Let's prove the correctness of RSA together, step by step !**

Let $$pk=(n,e)$$ and $$sk=(n,d)$$ be a pair of public and secret key generated by RSA key generation algorithm.

Since $$Dec(sk, Enc(pk,m)) = (m^e)^d \; mod \; n$$, what we want to show is, that

$$
(m^e)^d \equiv m \; (mod \; n)
$$

First thing to note here is, that it is enough to check that

$$
(m^e)^d \equiv m \; (mod \; p)
$$

$$
(m^e)^d \equiv m \; (mod \; q)
$$

Because by the Chinese Remainder Theorem, the map

$$
m \; mod \; n \mapsto (m \; mod \; p,m \; mod \; q)
$$

defines a ring isomorphism between $$\mathbb{Z}/n\mathbb{Z}$$ and $$\mathbb{Z}/p\mathbb{Z}\times\mathbb{Z}/q\mathbb{Z}$$.

Since $$p$$ and $$q$$ are arbitrarily chosen two prime numbers, they don't have variable specific constraints and hence it is enough to prove that

$$
(m^e)^d \equiv m \; (mod \; p)
$$

Then proof of $$(m^e)^d \equiv m \; (mod \; q)$$ would be analogue.

$$
(m^e)^d \equiv m^{e \cdot d} \; (mod \; p) \tag{1}
$$

$$
m^{e \cdot d} \equiv m^{(1 \; mod \; \phi(n))} \; (mod \; p) \tag{2}
$$

$$
m^{(1 \; mod \; \phi(n))} \equiv m^{(k \cdot \phi(n) + 1)} \; (mod \; p), \; k \in \mathbb{N}_0 \tag{3}
$$

$$
m^{(k \cdot \phi(n) + 1)} \equiv m^{k \cdot \phi(n)} \cdot m \; (mod \; p) \tag{4}
$$

$$
m^{k \cdot \phi(n)} \cdot m \equiv m^{k \cdot (p-1) \cdot (q-1)} \cdot m \; (mod \; p) \tag{5}
$$

$$
m^{k \cdot (p-1) \cdot (q-1)} \cdot m \equiv (m^{p-1})^{k \cdot (q-1)} \cdot m \; (mod \; p) \tag{6}
$$

$$
(m^{p-1})^{k \cdot (q-1)} \cdot m \equiv 1^{k \cdot (q-1)} \cdot m \; (mod \; p)\tag{7}
$$

$$
 1^{k \cdot (q-1)} \cdot m \equiv m \; (mod \; p) \tag{8}
$$

As I mentioned before, I'm not a mathematician myself, so I hope the proof provided above meets the required formality :) . While many steps are straightforward, a few might benefit from a more detailed explanation.

- Step $$(2)$$ holds true because in our key generation, we intentionally selected $$d$$ to serve as the inverse of $$e$$ in modulo $$\phi(n)$$.

- Step $$(5)$$ holds true because of properties of Euler's totient function mentioned above. If you can't understand why this step holds true, check the section about Euler's totient function and do not forget that $$n = p \cdot q$$ and $$p,q$$ are prime numbers with $$p \neq q$$

- Step $$(7)$$ holds true because of Fermat's Little Theorem

---

My goal in writing this blog post was to shed some light on the mathematical foundations of RSA encryption. I hope I have been able to fulfill this goal to some extent. Of course, it was not possible for me to prove all the theorems that have a direct use in the encryption in a formal way, but I think this blog post will be useful for many curious people who want to get away from the superficiality and take **a little** peek into the depths of cryptography.
