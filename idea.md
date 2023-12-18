## Idea

Have a "Function Autoencoder":
- Have a generating function, which takes a vector of dimension n and spits out a vector of dimension m.
- Have an autoencoder which takes a vector of dimension m, and has a latent space of size l, such that m > l > n.
- See that the fractional dimension of the latent space converges on n.


Example: [x^2, 3x+2, 3-x^3] after being encoded to a 2-dimensional space should converge on a fractional dimension of n.

So far, the conclusion looks good.