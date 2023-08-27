While searching the internet for a somehow in-depth article on the mathematics underpinning RSA encryption, I found myself somewhat dissatisfied with the available resources. This ignited a longstanding aspiration within me to start writing, so here is my debut piece !

# What is RSA ? 

RSA, or Rivest-Shamir-Adleman, is a public-key encryption system. It was introduced way back in 1977, making it quite old, but surprisingly, it's still widely used today (although it should not be used anymore) . Now, you might be thinking, "Why write a blog post about an ‘ancient’ encryption method?" Well, that's a fair question. The thing is, RSA isn't just a relic – it's got an educational treasure trove. It's a great example of how the difficulty of prime factorization can be turned into a one-way function. 

First, we'll take a brief look at how RSA generates keys, encrypts, and decrypts data. Then, we'll explore the math behind it, which will help us grasp why RSA was designed the way it is.

# Key Generation

RSA key generation involves several steps to generate a public and a private key:

- **Select two large prime numbers:** First, we pick two prime numbers, which should be large enough to ensure the security. These prime numbers are mostly denoted as p and q
- **Calculate the modulo:** The modulo is denoted as **_n_** and it defines the modulo algebraic structure that we operate on

$$
n = p \cdot q
$$

- **Evaluate Euler’s Phi (totient) function at _n_:** This is an important step but we will discuss the equation below later as we move forward.

$$
\phi (n) =  (p-1) \cdot (q-1)
$$

- **Choose a public exponent:** Choose a public exponent $$ e $$, which should satisfy the following conditions ( gcd(x,y) stands for greatest common divisor of x and y ):