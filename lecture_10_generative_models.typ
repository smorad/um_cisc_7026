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
/*
#side-by-side[
  $ p(bold(x); bold(theta)) = integral underbrace(p(bold(x) | bold(y); bold(theta)), "Learn it") dot p(bold(y)) dif bold(y) $ #pause
][
  $ argmin_bold(theta) KL(P_bold(X)(bold(x)), p(bold(x); bold(theta))) $ #pause
]*/
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
  #fletcher-diagram(
    node-stroke: .15em,
    node-fill: blue.lighten(50%),
    edge-stroke: .1em,
    
    node((0,0), $ bold(x) $, radius: 2em, name: <A>),
    node((1,0), $ bold(z) $, radius: 2em, fill: gray.lighten(50%), name: <B>),
    // Define edges
    edge(<A>, <B>, "-|>", bend: 45deg, $ "Encoder" p(bold(z) | bold(x); bold(theta)_e) $),
    edge(<A>, <B>, "<|-", bend: -45deg, $ "Decoder" p(bold(x) | bold(z); bold(theta)_d) $),
  ) #pause
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
  markhl(p(bold(z) | bold(x) ; bold(theta)), tag: #<3>, color: #orange),
  p(bold(z))), stroke: #3pt, color: #purple, radius: #5pt, tag: #<5>) $ 

#annot((<1>), pos: top, dy: -1em, annot-text-props: (size: 1em))[Encoder]
#annot((<3>), pos: top, dy: -1em, annot-text-props: (size: 1em))[Encoder]
#annot(<2>, pos: top, dy: -1em, annot-text-props: (size: 1em))[Decoder]
#annot(<4>, pos: bottom,  annot-text-props: (size: 1em))[Reconstruction]
#annot(<5>, pos: bottom,  annot-text-props: (size: 1em))[Marginal matching]


#align(center, fletcher-diagram(
  node-stroke: .15em,
  node-fill: blue.lighten(50%),
  edge-stroke: .1em,
  
  node((0,0), $ bold(x) $, radius: 2em, name: <A>),
  node((1,0), $ bold(z) $, radius: 2em, fill: gray.lighten(50%), name: <B>),
  // Define edges
  edge(<A>, <B>, "-|>", bend: 45deg, $ "Encoder" p(bold(z) | bold(x); bold(theta)_e) $),
  edge(<A>, <B>, "<|-", bend: -45deg, $ "Decoder" p(bold(x) | bold(z); bold(theta)_d) $),
))

==


VAEs approximate the model with *variational inference* #pause
- Assume the latent space $p(bold(z))$ follows a tractable distribution #pause

$ p(bold(x); bold(theta)) = integral p(bold(x) | bold(y); bold(theta)) dot p(bold(y)) dif bold(y) $


TODO: Should this go after HVAE and Diffusion?
==
#side-by-side[
  $ p(bold(x); bold(theta)_d) = integral underbrace(p(bold(x) | bold(z); bold(theta)_d), "Decoder") p(bold(z)) dif bold(z) $
][
  $ underbrace(p(bold(z) | bold(x); bold(theta)_e), "Encoder") $
]
  $ argmin_bold(theta) KL(P_bold(X)(bold(x)), p(bold(x); bold(theta))) $
+ Encoder must map to marginal distribution $p(bold(z))$
+ Integral is intractable
+ Objective (KL divergence) is intractable



= Hierarchical VAEs
==
  #fletcher-diagram(
    node-stroke: .15em,
    node-fill: blue.lighten(50%),
    edge-stroke: .1em,
    
    node((0,0), $ bold(x) $, radius: 2em, name: <A>),
    node((1,0), $ bold(z)_1 $, fill: gray.lighten(50%), radius: 2em, name: <B>),
    node((2,0), $ bold(z)_2 $, fill: gray.lighten(50%), radius: 2em, name: <C>),
    node((2.75,0), $ dots $, fill: none, stroke: none, radius: 2em, name: <D>),
    node((3.75,0), $ bold(z)_T $, fill: gray.lighten(50%), radius: 2em, name: <E>),
    // Define edges
    edge(<A>, <B>, "-|>", bend: 45deg, $ p(bold(z)_1 | bold(x); bold(theta)_(e 1)) $),
    edge(<B>, <C>, "-|>", bend: 45deg, $ p(bold(z)_2 | bold(z)_1; bold(theta)_(e 2)) $),
    edge(<C>, <D>, "-", bend: 45deg),
    edge(<D>, <E>, "-|>", bend: 45deg, $ p(bold(z)_T | bold(z)_(T-1); bold(theta)_(e T)) $),

    edge(<A>, <B>, "<|-", bend: -45deg, $ p(bold(x) | bold(z)_1; bold(theta)_(d 1)) $),
    edge(<B>, <C>, "<|-", bend: -45deg, $ p(bold(z)_1 | bold(z)_2; bold(theta)_(d 2)) $),
    edge(<C>, <D>, "-", bend: -45deg),
    edge(<D>, <E>, "<|-", bend: -45deg, $ p(bold(z)_(T-1) | bold(z)_T; bold(theta)_(d T)) $),
    )


= Diffusion Models
==
  #fletcher-diagram(
    node-stroke: .15em,
    node-fill: blue.lighten(50%),
    edge-stroke: .1em,
    
    node((0,0), $ bold(x) $, radius: 2em, name: <A>),
    node((1,0), $ bold(z)_1 $, fill: gray.lighten(50%), radius: 2em, name: <B>),
    node((2,0), $ bold(z)_2 $, fill: gray.lighten(50%), radius: 2em, name: <C>),
    node((2.75,0), $ dots $, fill: none, stroke: none, radius: 2em, name: <D>),
    node((3.75,0), $ bold(z)_T $, fill: gray.lighten(50%), radius: 2em, name: <E>),
    // Define edges
    edge(<A>, <B>, "-|>", bend: 45deg, $ p(bold(z)_1 | bold(x)) $),
    edge(<B>, <C>, "-|>", bend: 45deg, $ p(bold(z)_2 | bold(z)_1) $),
    edge(<C>, <D>, "-", bend: 45deg),
    edge(<D>, <E>, "-|>", bend: 45deg, $ p(bold(z)_T | bold(z)_(T-1)) $),

    edge(<A>, <B>, "<|-", bend: -45deg, $ p(bold(x) | bold(z)_1; bold(theta)) $),
    edge(<B>, <C>, "<|-", bend: -45deg, $ p(bold(z)_1 | bold(z)_2; bold(theta)) $),
    edge(<C>, <D>, "-", bend: -45deg),
    edge(<D>, <E>, "<|-", bend: -45deg, $ p(bold(z)_(T-1) | bold(z)_T; bold(theta)) $),
    )


= Backup
= Variational Modeling
==
  Autoencoders are useful for compression and denoising #pause

  But we can also use them as *generative models* #pause

  A generative model learns the structure of data #pause

  Using this structure, it generates *new* data #pause

  - Train on face dataset, generate *new* pictures #pause
  - Train on book dataset, write a *new* book #pause
  - Train on protein dataset, create *new* proteins #pause

  How does this work?

==
  Latent space $Z$ after training on the clothes dataset with $d_z = 3$

  #cimage("figures/lecture_9/fashion-latent.png", height: 85%)

  // #cimage("figures/lecture_9/latent_space.png") #pause

==
  What happens if we decode a new point?

  #cimage("figures/lecture_9/fashion-latent.png", height: 85%)

==
  #side-by-side[
    #cimage("figures/lecture_9/fashion-latent.png")
  ][
  Autoencoder generative model: #pause
  
  Encode $ vec(bold(x)_[1], dots.v, bold(x)_[n])$ into $vec(bold(z)_[1], dots.v, bold(z)_[n]) $ #pause

  Pick a point $bold(z)_[k]$ #pause

  Add some noise $bold(z)_"new" = bold(z)_[k] + bold(epsilon)$ #pause

  Decode $bold(z)_"new"$ into $bold(x)_"new"$
  ]

/*
==
  #cimage("figures/lecture_9/vae_gen_faces.png", height: 70%) #pause

  These pictures were created by a *variational* autoencoder #pause

  But these people do not exist!
*/
==
  #cimage("figures/lecture_9/vae_gen_faces.png", height: 70%) #pause

  $ f^(-1)(bold(z)_k + bold(epsilon), bold(theta)_d) $

==
  #side-by-side(align: left)[
  
  But there is a problem, the *curse of dimensionality* #pause

  As $d_z$ increases, points move further and further apart #pause

  ][
    #cimage("figures/lecture_9/curse.png") #pause
  ]

  $f^(-1)(bold(z) + epsilon)$ will produce either garbage, or $bold(z)$

==
  *Question:* What can we do? #pause

  *Answer:* Force the points to be close together! #pause

  We will use a *variational autoencoder* (VAE)

==
  VAE discovered by Diederik Kingma (also adam optimizer) #pause

  #cimage("figures/lecture_9/durk.jpg", height: 80%) 

==
  Variational autoencoders (VAEs) do three things: #pause
  + Make it easy to sample random $bold(z)$ #pause
  + Keep all $bold(z)_[1], dots bold(z)_[n]$ close together in a small region #pause
  + Ensure that $bold(z) + bold(epsilon)$ is always meaningful #pause

  How? #pause

  Make $bold(z)_[1], dots, bold(z)_[n]$ normally distributed #pause

  $ bold(z) tilde cal(N)(mu, sigma), quad mu = 0, sigma = 1 $

==
  //#cimage("figures/lecture_2/normal_dist.png")
  #align(center, normal)

==
  If $bold(z)_[1], dots, bold(z)_[n]$ are distributed following $cal(N)(0, 1)$: #pause

  + 99.7% of $bold(z)_[1], dots, bold(z)_[n]$ lie within $3 sigma = [-3, 3]$ #pause

  + Make it easy to generate new $bold(z)$, just sample $bold(z) tilde cal(N)(0, 1)$

==
  So how do we ensure that $bold(z)_[1], dots, bold(z)_[n]$ are normally distributed? #pause

  We have to remember conditional probabilities #pause

  $ P("rain" | "cloud") = "Probability of rain, given that it is cloudy" $ #pause

  //First, let us assume we already have some latent variable $bold(z)$, and focus on the decoder #pause

==
  *Key idea 1:* We want to model the distribution over the dataset $X$

  $ P(bold(x); bold(theta)), quad bold(x) tilde bold(X) $ #pause

  We want to learn $bold(theta)$ that best models the distribution of possible faces #pause

  #side-by-side[
    Large $P(bold(x); bold(theta))$
  ][
    $P(bold(x); bold(theta)) approx 0$
  ]
  #side-by-side[
    #cimage("figures/lecture_9/vae_gen_faces.png", height: 40%)
  ][
    #cimage("figures/lecture_1/muffin.png", height: 40%)
  ]

==
  *Key idea 2:* There is some latent variable $bold(z)$ which generates data $bold(x)$ #pause

  $ x: #cimage("figures/lecture_9/vae_slider.png") $

  $ z: mat("woman", "brown hair", ("frown" | "smile")) $

==
  #cimage("figures/lecture_9/vae_slider.png")

  Network can only see $bold(x)$, it cannot directly observe $bold(z)$ #pause

  Given $bold(x)$, find the probability that the person is smiling $P(bold(z) | bold(x); bold(theta))$

==



TODO

==
  We cast the autoencoding task as a *variational inference* problem #pause

  #align(center, varinf)

  #side-by-side[
    Decoder 
    $ P(bold(x) | bold(z); bold(theta)) $
  ][
    Encoder 
    $ P(bold(z) | bold(x); bold(theta)) $
  ] #pause

  We want to learn both the encoder and decoder: $P(bold(z), bold(x); bold(theta))$

==
  To generate data, we only need the *marginal* and *decoder* #pause

  #align(center, varinf)

  $ P(bold(x); bold(theta)) = integral_(Z) P(bold(x) | bold(z); bold(theta)) P(bold(z)) $

  But the encoder $P(bold(z) | bold(x); bold(theta))$ is necessary to learn meaningful latent $p(bold(z))$

==
  $ P(bold(z), bold(x); bold(theta)) = P(bold(x) | bold(z); bold(theta)) space P(bold(z)) $ #pause

  We can choose any distribution for $P(bold(z))$ #pause

  $ P(bold(z)) = cal(N)(bold(0), bold(1)) $ #pause

  We can generate all possible $bold(x)$ by sampling $bold(z) tilde cal(N)(bold(0), bold(1))$ #pause

  We can randomly generate $bold(z)$, which we can decode into new $bold(x)$!

==
  Now, all we must do is find $bold(theta)$ that best explains the dataset distribution #pause

  Learned distribution $P(bold(x); bold(theta))$ to be close to dataset $P(bold(x)), quad bold(x) tilde X$ #pause

  We need some error function between $P(bold(x); bold(theta))$ and $P(bold(x))$ #pause
  
  *Question:* How do we measure error between probability distributions?  #pause

  *Answer:* KL divergence

==

  #cimage("figures/lecture_5/forwardkl.png", height: 50%)
  
  $ KL(P, Q) = sum_i P(i) log P(i) / Q(i) $

==
  Learn the parameters for our model #pause

  $ argmin_bold(theta) KL(P(bold(x)), P(bold(x); bold(theta))) $ #pause

  Unfortunately, this objective is intractable to optimize #pause

  *Question:* Why? #pause

  $ P(bold(x))= integral_(Z) $

  The support of $P(bold(x))$ is infinitely large: $bb(R)_(d_z)$

==

  The paper provides surrogate objective

  $ argmin_bold(theta) [ -log P(bold(x) | bold(z); bold(theta)) + 1 / 2 KL(P(bold(z) | bold(x); bold(theta)), P(bold(z)))] $ #pause

  We call this the *Evidence Lower Bound Objective* (ELBO) 

==
  $ argmin_bold(theta)  [- log P(bold(x) | bold(z); bold(theta)) + 1/2 KL(P(bold(z) | bold(x); bold(theta)), P(bold(z))) ] $ #pause

  How is this ELBO helpful? #pause

  #side-by-side[
    Decoder 
    $ P(bold(x) | bold(z); bold(theta)) $
  ][
    Encoder 
    $ P(bold(z) | bold(x); bold(theta)) $
  ][
    Prior
    $ P(bold(z)) = cal(N)(bold(0), bold(1)) $
  ] #pause

  $  argmin_bold(theta) [ underbrace(-log P(bold(x) | bold(z); bold(theta)), "Reconstruction error") + 1 / 2 underbrace(KL(P(bold(z) | bold(x); bold(theta)), P(bold(z))), "Constrain latent") ] $ #pause

  Now we know how to train our autoencoder!

= Implementation
==
  How do we implement $f$ (i.e., $P(bold(z) | bold(x); bold(theta))$ )? #pause

  $ f : X times Theta |-> Delta Z $ #pause

  Normal distribution has a mean $mu in bb(R)$ and standard deviation $sigma in bb(R)_+$ #pause

  Our encoder should output $d_z$ means and $d_z$ standard deviations #pause

  $ f : X times Theta |-> bb(R)^(d_z) times bb(R)_+^(d_z) $

==
  ```python
  core = nn.Sequential(...)
  mu_layer = nn.Linear(d_h, d_z)
  # Neural networks output real numbers
  # But sigma must be positive
  # Output log sigma, because e^(sigma) is always positive
  log_sigma_layer = nn.Linear(d_h, d_z)
  # Alternatively, one sigma for all data
  log_sigma = jnp.ones((d_z,))

  tmp = core(x)
  mu = mu_layer(tmp)
  log_sigma = log_sigma_layer(tmp)
  distribution = (mu, exp(sigma))
  ```

==
  We covered the encoder

  $ f: X times Theta |-> Delta Z $
  
  We can use the same decoder as a standard autoencoder #pause

  $ f^(-1): Z times Theta |-> X $

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

  We can sample and use gradient descent #pause

  This trick only works with certain distributions

==
  Put it all together #pause

  *Step 1:* Encode the input to a normal distribution

  $ bold(mu), bold(sigma) = f(bold(x), bold(theta)_e) $ #pause

  *Step 2:* Generate a sample from distribution

  $ bold(z) = bold(mu) + bold(sigma) dot.o bold(epsilon) $ #pause

  *Step 3:* Decode the sample 

  $ bold(x) = f^(-1)(bold(z), bold(theta)_d) $

==
  One last thing, implement the loss function #pause


  #side-by-side[
    Decoder 
    $ P(bold(x) | bold(z); bold(theta)) $
  ][
    Encoder 
    $ P(bold(z) | bold(x); bold(theta)) $
  ][
    Prior
    $ P(bold(z)) = cal(N)(bold(0), bold(1)) $
  ] #pause

  $ cal(L)(bold(x), bold(theta)) = argmin_bold(theta) [ underbrace(-log P(bold(x) | bold(z); bold(theta)), "Reconstruction error") + underbrace( 1 / 2 KL(P(bold(z) | bold(x); bold(theta)), P(bold(z))), "Constrain latent") ] $ #pause

  Start with the KL term first

==
  $ cal(L)(bold(x), bold(theta)) = argmin_bold(theta) [ -log P(bold(x) | bold(z); bold(theta)) + 1 / 2 KL(P(bold(z) | bold(x); bold(theta)), P(bold(z))) ] $ #pause

  First, rewrite KL term using our encoder $f$ #pause

  $ cal(L)(bold(x), bold(theta)) = argmin_bold(theta) [ -log P(bold(x) | bold(z)) + 1 / 2 KL(f(bold(x), bold(theta)_e), P(bold(z))) ] $ #pause

  $P(bold(z))$ and $f(bold(x), bold(theta)_e)$ are Gaussian, we can simplify KL term

  $ cal(L)(bold(x), bold(theta)) = underbrace(log P(bold(x) | bold(z)), "Reconstruction error") - (sum_(j=1)^d_z mu^2_j + sigma^2_j - log(sigma^2) - 1) $

==
  $ cal(L)(bold(x), bold(theta)) = underbrace(log P(bold(x) | bold(z)), "Reconstruction error") - (sum_(j=1)^d_z mu^2_j + sigma^2_j - log(sigma^2) - 1) $ #pause

  Next, plug in square error for reconstruction error #pause

  $ = sum_(j=1)^d_z (x_j - f^(-1)(f(bold(x), bold(theta)_e), bold(theta)_d)_j )^2 - (sum_(j=1)^d_z mu^2_j + sigma^2_j - log(sigma^2_j) - 1) $ #pause

  $ cal(L)(bold(x), bold(theta)) = sum_(j=1)^d_z (x_j - f^(-1)(f(bold(x), bold(theta)_e), bold(theta)_d)_j )^2 - (sum_(j=1)^d_z mu^2_j + sigma^2_j - log(sigma^2_j) - 1) $

==
  $ cal(L)(bold(x), bold(theta)) = sum_(j=1)^d_z (x_j - f^(-1)(f(bold(x), bold(theta)_e), bold(theta)_d)_j )^2 - (sum_(j=1)^d_z mu^2_j + sigma^2_j - log(sigma^2_j) - 1) $ #pause

  Finally, define over the entire dataset

  $ cal(L)(bold(X), bold(theta)) &= sum_(i=1)^n sum_(j=1)^d_z (x_([i],j) - f^(-1)(f(bold(x)_[i], bold(theta)_e), bold(theta)_d)_(j) )^2 - \ 
  &(sum_(i=1)^n sum_(j=1)^d_z mu^2_([i],j) + sigma^2_([i],j) - log(sigma_([i], j)^2) - 1) $

==
  $ cal(L)(bold(X), bold(theta)) &= sum_(i=1)^n sum_(j=1)^d_z (x_([i],j) - f^(-1)(f(bold(x)_[i], bold(theta)_e), bold(theta)_d)_(j) )^2 - \ 
   &(sum_(i=1)^n sum_(j=1)^d_z mu^2_([i],j) + sigma^2_([i],j) - log(sigma_([i], j)^2) - 1) $

  Scale of two terms can vary, we do not want one term to dominate

==
  Paper suggests using minibatch size $m$ and dataset size $n$ #pause

  $ cal(L)(bold(X), bold(theta)) &= #redm[$m / n$] sum_(i=1)^n sum_(j=1)^d_z (x_([i],j) - f^(-1)(f(bold(x)_[i], bold(theta)_e), bold(theta)_d)_(j) )^2 - \ 
   &(sum_(i=1)^n sum_(j=1)^d_z mu^2_([i],j) + sigma^2_([i],j) - log(sigma_([i], j)^2) - 1) $

==
  Another paper finds hyperparameter $beta$ also helps #pause

  $ cal(L)(bold(X), bold(theta)) &= #redm[$m / n$] sum_(i=1)^n sum_(j=1)^d_z (x_([i],j) - f^(-1)(f(bold(x)_[i], bold(theta)_e), bold(theta)_d)_(j) )^2 - \ 
   & #redm[$beta$] (sum_(i=1)^n sum_(j=1)^d_z mu^2_([i],j) + sigma^2_([i],j) - log(sigma_([i], j)^2) - 1) $

==
  ```python
  def L(model, x, m, n, key):
    mu, sigma = model.f(x)
    epsilon = jax.random.normal(key, x.shape[0])
    z = mu + sigma * epsilon
    pred_x = model.f_inverse(z)

    recon = jnp.sum((x - pred_x) ** 2)
    kl = jnp.sum(mu ** 2 + sigma ** 2 - jnp.log(sigma ** 2) - 1)

    return m / n * recon + kl
  ```