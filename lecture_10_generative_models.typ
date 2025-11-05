#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "@preview/cetz:0.4.0"
#import "@preview/fletcher:0.5.8" as fletcher: node, edge
#import "common.typ": *
#import "plots.typ": *
#import "@preview/algorithmic:1.0.5"
#import algorithmic: style-algorithm, algorithm-figure, algorithm
#import "@preview/mannot:0.3.0": *

#let handout = false


// FUTURE TODO: Repeat self too much on locality/equivariance, restructure and replace with something else

#set math.vec(delim: "[")
#set math.mat(delim: "[")

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-common(handout: handout),
  config-info(
    title: [Generative Models],
    subtitle: [CISC 7026 - Introduction to Deep Learning],
    author: [Steven Morad],
    institution: [University of Macau],
    logo: image("figures/common/bolt-logo.png", width: 4cm)
  ),
  header-right: none,
  header: self => utils.display-current-heading(level: 1)
)

#title-slide()

== Outline <touying:hidden>

#components.adaptive-columns(
    outline(title: none, indent: 1em, depth: 1)
)


#set math.vec(delim: "[")
#set math.mat(delim: "[")

// Review VAE
// VAEs are a class of probabilistic generative model
// P(x | z)
// Hierarchical VAEs
// Diffusion Models
// Normalizing Flows
  // Turn base distribution into 



= Admin
==
Final project groups #pause
- If not in group must respond to message #pause
  - I will put you in a group #pause
  - If you do not respond, I will think you left the course #pause
    - No final project group, 0 on final project


= Review

= Generative Models
==

#side-by-side(align: left)[
  #cimage("figures/lecture_9/fashion-latent.png", height: 85%)
][
  Autoencoder latent space $Z in bb(R)^3$ #pause

  Images with similar semantic meaning (sneaker, sandal) cluster in Z #pause

  $Z$ forms a semantic manifold #pause
  - Local Euclidean dynamics at each datapoint #pause

  Network learns what sneakers, shirts, etc are #pause
  - Can we use this to generate *new* images?
]

==
#side-by-side(align: left)[
  #cimage("figures/lecture_9/fashion-latent.png", height: 85%)
][
  Each $bold(z)$ corresponds to $bold(x)$ in the dataset #pause

  Consider a datapoint $bold(x)_[i]$ (shirt) #pause

  We can encode $bold(x)_[i]$ 

  $ bold(z)_[i] = f (bold(x)_[i], bold(theta)_e) $ #pause

  And decode $bold(z)_[i]$ 

  $ hat(bold(x))_[i] = f^(-1) (bold(z)_[i], bold(theta)_d) $
]

==
  #side-by-side(align: left)[
    #cimage("figures/lecture_9/fashion-latent.png", height: 85%)
  ][
     $ bold(z)_[i] &= f (bold(x)_[i], bold(theta)_e) \
     hat(bold(x))_[i] &= f^(-1) (bold(z)_[i], bold(theta)_d) $ #pause
     
     Let us perturb $bold(z)$ with noise $bold(epsilon)$, then decode it #pause

     $ f^(-1) (bold(z)_[i] + #redm[$bold(epsilon)$], bold(theta)_d) $ #pause

     *Question:* What happens? #pause

     *Hint:* $Z$ is a semantic manifold (locally Euclidean)
]

==
  #side-by-side(align: left)[
    #cimage("figures/lecture_9/fashion-latent.png", height: 85%)
  ][

    $ f^(-1) (bold(z)_[i] + #redm[$bold(epsilon)$], bold(theta)_d) $ #pause

    We will create an image of a new datapoint #pause
    - Not in the dataset #pause
    - Semantically similar to $bold(x)_i$ #pause
      - But *new* and different! #pause

    $ bold(z)_[i] &-> "blue shirt" \ 
    bold(z)_[i] + bold(epsilon) &-> "red shirt" $ #pause

    Idea powers generative models
]

==
#side-by-side[

][

]
This autoencoder only works for small $d_z$ #pause

As $d_z$ grows, points spread out #pause
- Curse of dimensionality #pause
- For large $d_z$. one cluster for each datapoint #pause

Our autoencoder only works for small $d_z$ #pause
- Interesting problems are high-dimensional (large $d_z$)


= Probabilistic Generative Models

==
*Question:* How do we explain autoencoders learning our world? #pause

*Answer:* They implicitly model the *dataset distribution* #pause
- If the dataset contains our world, autoencoders learn the world #pause

We will examine learning through probability and distributions #pause
- Best way to learn generative models #pause

Let us consider a dataset of dog pictures


#slide(composer: (0.4fr, 1fr))[
  #cimage("figures/lecture_1/dog.png") 
  $ bold(x)_([i]) $
  #pause
][
  *Domain:* $X = {0 dots 255}^{3 times 28 times 28}$ #pause
  - How we represent datapoints #pause
  - $bold(x)_([i]) in X$ #pause

  *Dataset (Evidence):* $bold(X) = mat(bold(x)_([1]), dots, bold(x)_([n]))^top$ #pause
  - Finite collection of $n$ datapoints #pause

  *Dataset distribution:* $P_bold(X) (bold(x)_([i]))$ #pause
  - Probability of sampling $bold(x)_([i])$ from $bold(X)$ #pause
  - $1/n$ if $bold(x)_([i])$ in dataset, else 0 #pause

  *True data distribution:* $p_* (bold(x)_([i]))$ #pause
  - How likely is $bold(x)_([i])$ in the universe? 
  //- Superset of the dataset $bold(X)$
]

==

#side-by-side(align: left)[
  $ P_bold(X)(#cimage("figures/lecture_1/dog.png", height: 30%) ) = #pause 1 / n $ #pause
][
  $ P_bold(X)(#cimage("figures/lecture_1/muffin.png", height: 30%) ) #pause = 0 $ #pause
]

$ p_*(#cimage("figures/lecture_1/dog.png", height: 30%)) quad vec(delim: #none, =, <, >) quad p_*(#cimage("figures/lecture_1/muffin.png", height: 30%)) $

==
We want to approximate a distribution so we can generate samples #pause

#side-by-side[
  *Question:*
][
$ bold(x)_([i]) tilde P_bold(X) (bold(x); bold(theta)) $
][
or
][
$ bold(x)_([i]) tilde p_* (bold(x); bold(theta)) $
] #pause

$P_bold(X) (bold(x); bold(theta))$ samples existing datapoints, must be $p_* (bold(x); bold(theta))$ #pause

Only have $P_bold(X) (bold(x); bold(theta))$, not $p_* (bold(x); bold(theta))$ #pause
- How to approximate $p_* (bold(x); bold(theta))$? #pause

*Key idea:* Learn a continuous distribution $p (bold(x); bold(theta))$ using $P_bold(X) (bold(x))$ #pause

$ p (bold(x); bold(theta)) = P_bold(X) (bold(x)) => p (bold(x); bold(theta)) approx p_* (bold(x)) $ #pause

We call $p (bold(x); bold(theta))$ a *probabilistic generative model*

==
Understanding generative models probabilistically is very hard #pause
- Took me years, I still do not fully understand #pause
- I try and make it easier every year, but requires statistics #pause

If you understand this concept, all generative models become easy #pause
- Same objective, just different approximation methods #pause
  - GAN
  - VAE
  - Diffusion
  - Normalizing flow
  - Flow matching
  - ...

==
Let us think about the objective function #pause
- We have the dataset distribution $P_bold(X) (bold(x))$ #pause
- We have a model that outputs a distribution $p (bold(x); bold(theta))$ #pause
- Output distribution must match the dataset distribution  #pause

*Question:* What is our objective? #pause

$ argmin_(bold(theta)) space KL(P_bold(X) (bold(x)), p(bold(x); bold(theta))) $ #pause

Same idea as classification #pause
- Let us see if we can simplify the objective

==
$ argmin_bold(theta) KL(P_bold(X)(bold(x)), p(bold(x); bold(theta))) $ #pause

From the definition of $KL$

$ argmin_bold(theta) KL(P_bold(X)(bold(x)),  p(bold(x); bold(theta))) = \

argmin_bold(theta) sum_(i=1)^n P_bold(X) (bold(x)_([i])) (
  P_bold(X) (bold(x)_([i]))
) / (
  p(bold(x)_([i]); bold(theta))
)
$ #pause

However, we know the probability for all datapoints $P_bold(X) (bold(x))$ #pause

*Question:* What is it? #pause *Answer:* $1 / n$ if $bold(x)_([i])$ in dataset, else 0

==
$ argmin_bold(theta) sum_(i=1)^n P_bold(X) (bold(x)_([i])) (
  P_bold(X) (bold(x)_([i]))
) / (
  p(bold(x)_([i]); bold(theta))
)
$ #pause

With a constant $P_bold(X)$, we can simplify the expression #pause

*Question:* How do we simplify? #pause *Hint:* Same as cross entropy loss #pause

*Answer:* $P_bold(X)$ is constant $1/n$, can remove from optimization objective

$ argmin_bold(theta) sum_(i=1)^n 1 / (p(bold(x)_([i]); bold(theta))) $ #pause

==

$ argmin_bold(theta) sum_(i=1)^n 1 / (p(bold(x)_([i]); bold(theta))) $ #pause

Instead of minimizing ratio, maximize ($p$ always positive) #pause

$ argmax_bold(theta) sum_(i=1)^n p(bold(x)_([i]); bold(theta)) $ #pause

In statistics, we often optimize $log(f)$ instead of $f$ #pause
- Yields same optima, often helps with optimization #pause

$ argmax_bold(theta) sum_(i=1)^n log p(bold(x)_([i]); bold(theta)) $

==
$ argmax_bold(theta) sum_(i=1)^n log p(bold(x)_([i]); bold(theta)) $ #pause

This is known as the *log likelihood* #pause
- Used all over statistics and machine learning #pause

The log likelihood objective: 
- Learns the parameters $bold(theta)$ of a continuous distribution $p(bold(x); bold(theta))$
- That maximizes the probability of each datapoint $bold(x)_([i])$ under the model

==
#align(center, pdf_pmf)

$ p: Theta |-> Delta(X) $

==

$ argmax_bold(theta) sum_(i=1)^n log p(bold(x)_([i]); bold(theta)) $ #pause

- Log likelihood for gradient ascent #pause
- *Negative log likelihood* for gradient descent #pause

$ argmin_bold(theta) sum_(i=1)^n - log p(bold(x)_([i]); bold(theta)) $ #pause

Same thing, different names


==
To summarize, probabilistic generative models: #pause
+ Use a dataset distribution $P_bold(X)(bold(x))$ #pause
+ To approximate the true data distribution $p_* (bold(x))$ #pause
+ By learning a *continuous* distribution $p(bold(x); bold(theta))$ #pause
+ That maximizes some approximation of log likelihood objective #pause

#side-by-side[
  Generative model
][
  $ p(bold(x); bold(theta)) $
]
#side-by-side[
  Log likelihood objective
 ][
 $ argmax_bold(theta) sum_(i=1)^n log p(bold(x)_([i]); bold(theta)) $
]
  


= Bayesian Generative Models
==
Let us take a closer look at generative models #pause

#side-by-side(columns: (0.4fr, 1.0fr), align: left)[
  #cimage("figures/lecture_1/dog.png") 
  $ bold(x)_([i]) $ #pause
][
- Model $approx p(bold(x); bold(theta))$ #pause
- Objective $approx argmax_bold(theta) sum_(i=1)^n log p(bold(x)_([i]); bold(theta))$ #pause
*Question:* How to represent $p(bold(x); bold(theta))$? #pause
- $bold(x)$ is very high dimensional #pause
  - $bold(x) in [0, 1]^(3 times 28 times 28)$ #pause
- Complex relationships between pixels #pause
- Generally intractable to learn #pause
]
*Key idea:* Use low-dimensional representation of $bold(x)$ #pause
- Can reason over low-dimensional representation

==
Let $bold(y)_([i])$ be a low-dimensional label of $bold(x)_([i])$ #pause

#side-by-side(align: horizon)[
  $ bold(y)_([i]) = vec("Small", "Ugly") $ #pause
][
  $ bold(x)_([i]) = #cimage("figures/lecture_1/dog.png", height: 20%) $ #pause
]

The label tells us which $bold(x)$ to generate #pause
- Many possible dog pictures, should be specific (condition on label) #pause

$ underbrace(#cimage("figures/lecture_1/dog.png", height: 25%), bold(x)_([i])) tilde p(underbrace("Small ugly dog pictures", bold(x)) mid(|) underbrace(vec("Small", "Ugly"), bold(y)_([i])); bold(theta)) $

==
We must consider all possible $bold(y)$ to represent all possible dog pictures #pause

$ vec("Small", "Ugly"), vec("Medium", "Ugly"), vec("Big", "Not ugly"), dots $ #pause

We also need to know the label distribution $p(bold(y))$ #pause
- How common or rare are ugly dogs? Big dogs? Small dogs? #pause

$ p(bold(x); bold(theta)) = integral underbrace(p(bold(x) | bold(y); bold(theta)), "Image from label") overbrace(p(bold(y)), "Label distribution") dif bold(y) $ #pause

This is the definition of a conditional distribution

==
Bayesian generative model: #pause

$ p(bold(x); bold(theta)) = integral p(bold(x) | bold(y); bold(theta)) dot p(bold(y)) dif bold(y) $ #pause

*Question:* What was the objective? #pause

$ argmax_bold(theta) sum_(i=1)^n log p(bold(x)_([i]); bold(theta)) $ #pause

Can plug model into objective

$ argmax_bold(theta) sum_(i=1)^n log integral p(bold(x)_([i]) | bold(y); bold(theta)) dot p(bold(y)) dif bold(y) $ #pause

==
$ underbrace(argmax_bold(theta) sum_(i=1)^n log overbrace(integral p(bold(x)_([i]) | bold(y); bold(theta)) dot p(bold(y)) dif bold(y), "Model"), "Objective") $

Aftering learning $p(bold(x) | bold(y); bold(theta))$ we can generate new datapoints #pause

+ Either choose or sample label $bold(y)_([i]) tilde p(bold(y))$ #pause
+ Predict conditional distribution $p(bold(x) | bold(y)_([i]); bold(theta))$ #pause
+ Sample from conditional distribution $bold(x)_([i]) tilde p(bold(x) | bold(y)_([i]); bold(theta))$ #pause

Very easy! But this requires labels $bold(y)$ #pause
- Can we do this unsupervised (without labels/humans)?

= Variational Autoencoders
==
Consider a generative model *without labels* $bold(y)$ #pause
- We will *learn* a low-dimensional latent description $bold(z)$ instead #pause
  - $bold(z)$ encodes the structure of the dataset (and the world) #pause

$ p(bold(x); bold(theta)) = integral p(bold(x) | bold(y); bold(theta)) dot p(bold(y)) dif bold(y) => integral p(bold(x) | bold(z); bold(theta)) dot p(bold(z)) dif bold(z) $ #pause

Replace label distribution $p(bold(y))$ with latent distribution $p(bold(z))$ #pause
- We can choose this distribution, Gaussian, Bernoulli, ... #pause
- For now, make it easy $p(bold(z)) = cal(N)(bold(0), bold(I))$

==
*Question:* How can we learn structure of $bold(z)$? #pause

*Answer:* Autoencoding! #pause

$ hat(bold(x)) = f^(-1)(underbrace(f(bold(x), bold(theta)_e), bold(z)), bold(theta)_d) $ #pause

But our models are probabilistic, so it is more complex

==

#side-by-side[
$ p(bold(x); bold(theta)) = integral p(bold(x) | bold(z); bold(theta)) p(bold(z)) dif bold(z) $ #pause
][
  #vae_flow #pause
]

+ Use encoder and decoder to learn latent structure $bold(z)$ #pause
+ After learning, delete encoder (encoder only needed to learn $bold(z)$) #pause
+ Use decoder and latent distribution for generation #pause

Sound too easy? It is, making it tractable is much harder

= Practical Considerations
==
Most generative models follow a similar approach #pause

$ underbrace(argmax_bold(theta) sum_(i=1)^n log overbrace(integral p(bold(x)_([i]) | bold(z); bold(theta)) dot p(bold(z)) dif bold(z), "Model"), "Objective") $

However, this expression is  often intractable #pause

*Question:* Why? #pause *Answer:* Cannot solve integral #pause 
- No analytical solution if $p(bold(x) | bold(z); bold(theta))$ is neural network #pause
- Indefinite integration over continuous/infinite variable $bold(z)$

==
$ underbrace(argmax_bold(theta) sum_(i=1)^n log overbrace(integral p(bold(x)_([i]) | bold(z); bold(theta)) dot p(bold(z)) dif bold(z), "Model"), "Objective") $ #pause

Variational inference methods (like VAEs) approximate this objective using the *evidence lower bound* (ELBO) #pause
- Lower bounds the true log likelihood #pause

The derivation is interesting, but requires concepts you do not know #pause
- Expected values, Jensen's inequality, etc #pause
- Instead, I will show you the final result

==
The original objective

$ argmax_bold(theta) sum_(i=1)^n log integral p(bold(x)_([i]) | bold(z); bold(theta)) dot p(bold(z)) dif bold(z) $ #pause

The surrogate objective (ELBO)

$ argmax_bold(theta) sum_(i=1)^n bb(E)_(bold(z) tilde p(bold(z) | bold(x)_([i]) ; bold(theta)))[
 log p(bold(x)_([i]) | bold(z); bold(theta))
]
-KL(p(bold(z) | bold(x) ; bold(theta)), p(bold(z)))  $ #pause

Very scary looking expression #pause
- Let us try to understand it

==

#v(1em)
$ argmax_bold(theta) sum_(i=1)^n 
  markrect(bb(E)_(bold(z) tilde 
  markhl(p(bold(z) | bold(x)_([i]) ; bold(theta)),tag: #<1>, color: #orange))[
    log markhl(p(bold(x)_([i]) | bold(z); bold(theta)), tag: #<2>, color: #blue)
  ], stroke: #3pt, color: #red, radius: #5pt, tag: #<4>)
- markrect(KL(
  markhl(p(bold(z) | bold(x)_([i]) ; bold(theta)), tag: #<3>, color: #orange),
  p(bold(z))), stroke: #3pt, color: #purple, radius: #5pt, tag: #<5>) $ 

#annot((<1>), pos: top, dy: -1em, annot-text-props: (size: 1em))[Encoder]
#annot((<3>), pos: top, annot-text-props: (size: 1em))[Encoder]
#annot(<2>, pos: top, annot-text-props: (size: 1em))[Decoder]
#annot(<4>, pos: bottom,  annot-text-props: (size: 1em))[Reconstruction]
#annot(<5>, pos: bottom,  annot-text-props: (size: 1em))[Marginal matching]

#align(center, vae_flow)

= Implementation
==
If you did not understand the theory, it is ok #pause

But pay attention now, we will implement the VAE #pause
- Will use autoencoder notation to simplify when possible
==
  How do we implement encoder $f$ (i.e., $p(bold(z) | bold(x); bold(theta))$ )? #pause

  $ f : X times Theta |-> Delta Z $ #pause

  Recall we let $p(bold(z)) = cal(N)(bold(0), bold(I))$ #pause
  - Encoder should output normal distribution for analytical KL term #pause
  - Normal distribution has a mean $mu in bb(R)$ and standard deviation $sigma in bb(R)_+$ #pause
  - Our encoder should output $d_z$ means and $d_z$ standard deviations #pause

  $ f : X times Theta |-> bb(R)^(d_z) times bb(R)_+^(d_z) $

==
  ```python
  core = nn.Sequential(...)
  mu_layer = nn.Linear(d_h, d_z)
  # Neural networks output real numbers
  # But sigma must be positive
  # Output log sigma, because e^(sigma) is always positive
  log_sigma_layer = nn.Linear(d_h, d_z)

  tmp = core(x)
  mu = mu_layer(tmp)
  log_sigma = log_sigma_layer(tmp)
  distribution = (mu, exp(sigma))
  ```

==
  We covered the encoder

  $ f: X times Theta |-> Delta Z $ #pause
  
  We can use the same decoder as a standard autoencoder #pause

  $ f^(-1): Z times Theta |-> X $ #pause

  *Question:* Any issues? #pause

  *Answer:* Encoder outputs a distribution $Delta Z$ but decoder input is $Z$

==
  We can sample from the distribution 
  
  $ bold(mu), bold(sigma) &= f(bold(x), bold(theta)_e) \ 
  bold(z) & tilde cal(N)(bold(mu), bold(sigma)) $ #pause 

  But there is a problem! Sampling is not differentiable #pause

  *Question:* Why does this matter? #pause

  *Answer:* Must be differentiable for gradient descent

==
  VAE paper proposes the *reparameterization trick* #pause

  $ bold(z) & tilde cal(N)(bold(mu), bold(sigma)) $ 

  $ bold(z) = bold(mu) + bold(sigma) dot.o bold(epsilon) quad bold(epsilon) tilde cal(N)(bold(0), bold(1)) $ #pause

  Gradient can flow through $bold(mu), bold(sigma)$ #pause

  We can sample and use gradient descent

==
  Put it all together #pause

  *Step 1:* Encode the input to a normal distribution

  $ bold(mu), bold(sigma) = f(bold(x), bold(theta)_e) $ #pause

  *Step 2:* Generate a sample from distribution

  $ bold(z) = bold(mu) + bold(sigma) dot.o bold(epsilon) $ #pause

  *Step 3:* Decode the sample 

  $ hat(bold(x)) = f^(-1)(bold(z), bold(theta)_d) $

==
  Last thing, implement the loss function (ELBO) #pause


  #side-by-side[
    Decoder 
    $ p(bold(x) | bold(z); bold(theta)) $
  ][
    Encoder 
    $ p(bold(z) | bold(x); bold(theta)) $
  ][
    Marginal 
    $ p(bold(z)) = cal(N)(bold(0), bold(1)) $
  ] #pause

$ argmax_bold(theta) sum_(i=1)^n bb(E)_(bold(z) tilde p(bold(z) | bold(x)_([i]) ; bold(theta)))[
 log p(bold(x)_([i]) | bold(z); bold(theta))
]
-KL(p(bold(z) | bold(x) ; bold(theta)), p(bold(z))) $ #pause

Rewrite objective ass loss function for gradient descent #pause

$ cal(L)(bold(X), bold(theta)) = sum_(i=1)^n bb(E)_(bold(z) tilde p(bold(z) | bold(x)_([i]) ; bold(theta)))[
 -log p(bold(x)_([i]) | bold(z); bold(theta))
]
+ KL(p(bold(z) | bold(x) ; bold(theta)), p(bold(z))) $ #pause

==
  $ cal(L)(bold(X), bold(theta)) = sum_(i=1)^n bb(E)_(bold(z) tilde p(bold(z) | bold(x)_([i]) ; bold(theta)))[
  -log p(bold(x)_([i]) | bold(z); bold(theta))
  ]
  + KL(p(bold(z) | bold(x) ; bold(theta)), p(bold(z))) $ #pause

  First, rewrite KL term using our encoder $f$ #pause

  $ cal(L)(bold(X), bold(theta)) = sum_(i=1)^n bb(E)_(bold(z) tilde p(bold(z) | bold(x)_([i]) ; bold(theta)))[
  -log p(bold(x)_([i]) | bold(z); bold(theta))
  ]
  + KL(f(bold(x), bold(theta)_e), p(bold(z))) $ #pause

  //$ cal(L)(bold(x), bold(theta)) = argmin_bold(theta) [ -log P(bold(x) | bold(z)) + 1 / 2 KL(f(bold(x), bold(theta)_e), P(bold(z))) ] $ #pause

  $p(bold(z))$ and $f(bold(x), bold(theta)_e)$ are Gaussian, we can simplify KL term

  $ = sum_(i=1)^n bb(E)_(bold(z) tilde p(bold(z) | bold(x)_([i]) ; bold(theta)))[
  -log p(bold(x)_([i]) | bold(z); bold(theta))
  ]
  + 0.5 (sum_(j=1)^d_z mu^2_j + sigma^2_j - log(sigma^2) - 1) $ #pause

  //$ cal(L)(bold(x), bold(theta)) = underbrace(log P(bold(x) | bold(z)), "Reconstruction error") - (sum_(j=1)^d_z mu^2_j + sigma^2_j - log(sigma^2) - 1) $

==
  //$ cal(L)(bold(x), bold(theta)) = underbrace(log P(bold(x) | bold(z)), "Reconstruction error") - (sum_(j=1)^d_z mu^2_j + sigma^2_j - log(sigma^2) - 1) $ #pause

  $ = sum_(i=1)^n bb(E)_(bold(z) tilde p(bold(z) | bold(x)_([i]) ; bold(theta)))[
  -log p(bold(x)_([i]) | bold(z); bold(theta))
  ]
  + 0.5 sum_(j=1)^d_z mu^2_j + sigma^2_j - log(sigma^2) - 1 $ #pause

  Log probability for Gaussian is just mean square error #pause

  $ = sum_(i=1)^n sum_(j=1)^d_z (x_([i], j) - f^(-1)(f(bold(x)_([i]), bold(theta)_e), bold(theta)_d)_j )^2 - 0.5 (mu^2_j + sigma^2_j - log(sigma^2_j) - 1) $ #pause



==
  $ = sum_(i=1)^n sum_(j=1)^d_z (x_([i], j) - f^(-1)(f(bold(x)_([i]), bold(theta)_e), bold(theta)_d)_j )^2 - 0.5 (mu^2_j + sigma^2_j - log(sigma^2_j) - 1) $ #pause

  Scale of two terms can vary, we do not want one term to dominate #pause
  - Modern VAEs introduce a hyperparameter $beta$ #pause

  $ = sum_(i=1)^n sum_(j=1)^d_z (x_([i], j) - f^(-1)(f(bold(x)_([i]), bold(theta)_e), bold(theta)_d)_j )^2 - #redm[$beta$] (mu^2_j + sigma^2_j - log(sigma^2_j) - 1) $

==
  ```python
  def L(model, x, m, n, key):
    mu, sigma = model.f(x)
    epsilon = jax.random.normal(key, x.shape[0])
    z = mu + sigma * epsilon
    pred_x = model.f_inverse(z)

    recon = jnp.sum((x - pred_x) ** 2)
    kl = jnp.sum(mu ** 2 + sigma ** 2 - jnp.log(sigma ** 2) - 1)

    return recon + beta * kl
  ```

= Hierarchical VAEs
==
VAEs are just too easy #pause

Some crazy people decided to make them more complex #pause
- *Hierarchical VAEs* are many VAEs stacked together #pause
  - Provide better results than VAEs
==

  #align(center, scale(80%, reflow: true, vae_flow)) #pause

  #align(center, scale(80%, reflow: true, hvae_flow))

==

$ cal(L)_"VAE" (bold(X), bold(theta)) = sum_(i=1)^n underbrace(bb(E)_(bold(z) tilde p(bold(z) | bold(x)_([i]) ; bold(theta)))[
 -log p(bold(x)_([i]) | bold(z); bold(theta))
], "Reconstruction")
+ underbrace(KL(p(bold(z) | bold(x) ; bold(theta)), p(bold(z))), "Marginal match") $ #pause

$ cal(L)_H = sum_(i=1)^n underbrace(bb(E)_(bold(z)_1 tilde p(bold(z)_1 | bold(x)_([i]) ; bold(theta)_(e 1)))[
 -log p(bold(x)_([i]) | bold(z)_1; bold(theta)_(d 1))
], "Reconstruction")
+ underbrace(KL(p(bold(z) | bold(x) ; bold(theta)_(e 1)), p(bold(z))), "Mariginal match") \
+ sum_(t=2)^T underbrace(KL(
  p (bold(z)_t | bold(z)_(t-1) ; bold(theta)_(e t)),  
  p (bold(z)_t | bold(z)_(t+1) ; bold(theta)_(d t))
), "Consistency") $ 

==

$ sum_(t=2)^T underbrace(KL(
  p (bold(z)_t | bold(z)_(t-1) ; bold(theta)_(e t)),  
  p (bold(z)_t | bold(z)_(t+1) ; bold(theta)_(d t))
), "Consistency") $ 
#hvae_flow


= Diffusion Models
==
Diffusion models are simplified HVAEs #pause
- Instead of learning encoder, use fixed encoder #pause
  - Operates in input space $X$ instead of latent space $Z$ #pause
- Share parameters for all decoders #pause
==

  #align(center, scale(80%, reflow: true, hvae_flow)) #pause
  #align(center, scale(80%, reflow: true, diffusion_flow))

==

$ cal(L)_"H" = sum_(i=1)^n bb(E)_(bold(z)_1 tilde p(bold(z)_1 | bold(x)_([i]) ; bold(theta)_(e 1)))[
 -log p(bold(x)_([i]) | bold(z)_1; bold(theta)_(d 1))
]
+ KL(p(bold(z) | bold(x) ; bold(theta)_(e 1)), p(bold(z))) \
+ sum_(t=2)^T KL(
  p (bold(z)_t | bold(z)_(t-1) ; bold(theta)_(e t)),  
  p (bold(z)_t | bold(z)_(t+1) ; bold(theta)_(d t))
) $ #pause

$ cal(L)_"Dif" = sum_(i=1)^n underbrace(bb(E)_(bold(z)_1 tilde p(bold(x)_2 | bold(x)_(1, [i])))[
 -log p(bold(x)_(1, [i]) | bold(x)_2; bold(theta))
], "Reconstruction")
 \
+ sum_(t=2)^T underbrace(KL(
  p (bold(x)_t | bold(x)_(t-1)),  
  p (bold(x)_t | bold(x)_(t+1) ; bold(theta))
), "Consistency (Denoising)") $ 

==
What is encoder $p (bold(x)_t | bold(x)_(t-1))$?

Gaussian noise #pause

$ p (bold(x)_t | bold(x)_(t-1)) = bold(x)_(t-1) + cal(N)(bold(0), bold(I)) $

It is actually slightly more complicated, but same idea

$ p (bold(x)_t | bold(x)_(t-1)) = cal(N)(sqrt(alpha_t) dot bold(x)_(t-1), (1 - alpha_t) dot bold(I) ) $ #pause

*Question:* How is this different than VAE encoder? #pause
- VAE *learns* to output $cal(N)(bold(0), bold(I))$, diffusion model outputs fixed $cal(N)(bold(0), bold(I))$ #pause
  - VAE noise in latent space, diffusion in input space #pause
- Hierarchical VAEs strictly more powerful than diffusion models 

==
How should I choose VAE/HVAE/diffusion model?

Choose VAE for: #pause
- Low-dimensional representation $bold(z)$ #pause
- Simpler problem #pause

Choose HVAE for: #pause
- Low-dimensional representation $bold(z)$ #pause
- Harder problem #pause

Choose diffusion model if: #pause
- Do not care about $bold(z)$ #pause
- Have 16 GPUs

= Coding
==
Let's code a VAE

https://colab.research.google.com/drive/1UyR_W6NDIujaJXYlHZh6O3NfaCAMscpH#scrollTo=NLTDp6ipBtJU