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

= Review

= Generative Models
==

#side-by-side(align: left)[
  #cimage("figures/lecture_9/fashion-latent.png", height: 85%)
][
  Autoencoder latent space $Z in bb(R)^3$ 

  Images with similar semantic meaning (sneaker, sandal) cluster in Z

  $Z$ forms a semantic manifold
  - Euclidean space near each point

  Network learns what sneakers, shirts, etc are
  - Can we use this to generate *new* images?
]

==
#side-by-side(align: left)[
  #cimage("figures/lecture_9/fashion-latent.png", height: 85%)
][
  Each $bold(z)$ corresponds to $bold(x)$ in the dataset

  Consider a datapoint $bold(x)_[i]$ (shirt)

  We can encode $bold(x)_[i]$

  $ bold(z)_[i] = f (bold(x)_[i], bold(theta)_e) $

  And decode $bold(z)_[i]$

  $ hat(bold(x))_[i] = f^(-1) (bold(z)_[i], bold(theta)_d) $
]

==
  #side-by-side(align: left)[
    #cimage("figures/lecture_9/fashion-latent.png", height: 85%)
  ][
     $ bold(z)_[i] &= f (bold(x)_[i], bold(theta)_e) \
     hat(bold(x))_[i] &= f^(-1) (bold(z)_[i], bold(theta)_d) $
     
     Let us we perturb $bold(z)$ with noise $bold(epsilon)$, then decode it

     $ f^(-1) (bold(z)_[i] + #redm[$bold(epsilon)$], bold(theta)_d) $

     *Question:* What happens?

     *Hint:* $Z$ is a semantic manifold (locally Euclidean)
]

==
  #side-by-side(align: left)[
    #cimage("figures/lecture_9/fashion-latent.png", height: 85%)
  ][

    $ f^(-1) (bold(z)_[i] + #redm[$bold(epsilon)$], bold(theta)_d) $

    We will create an image of a new datapoint
    - Not in the dataset
    - Semantically similar to $bold(x)_i$
      - But different!

    $ bold(z)_[i] &-> "blue shirt" \ 
    bold(z)_[i] + bold(epsilon) &-> "red shirt" $

    Idea powers generative models
]

==
#side-by-side[

][

]
This idea only works for small $d_z$

As $d_z$ grows, points spread out
- Curse of dimensionality
- For large $d_z$. one cluster for each datapoint

Our generative autoencoder only works for small $d_z$
- Interesting problems are high-dimensional (large $d_z$)


= Probabilistic Generative Models

==
*Question:* How do we explain autoencoders learning our world?

*Answer:* They implicitly model the *dataset distribution*
- If the dataset contains our world, autoencoders learn the world

We will examine learning through probability and distributions
- Best way to learn generative models

Let us consider a dataset of dog pictures


#slide(composer: (0.4fr, 1fr))[
  #cimage("figures/lecture_1/dog.png") 
  $ bold(x)_([i]) $
  #pause
][
  *Domain:* $X = {0 dots 255}^{3 times 28 times 28}$ 
  - How we represent datapoints
  - $bold(x) in X$

  *Dataset (Evidence):* $bold(X) = mat(bold(x)_([1]), dots, bold(x)_([n]))^top$
  - Finite collection of datapoints 

  *Dataset distribution:* $P_bold(X) (bold(x)_([i]))$
  - Probability of sampling $bold(x)_([i])$ from $bold(X)$
  - $1/n$ if $bold(x)_([i])$ in dataset, else 0

  *True data distribution:* $p_* (bold(x)_([i]))$
  - How likely is $bold(x)_([i])$ in the universe?
  - Superset of the dataset $bold(X)$
]

==

#side-by-side(align: left)[
  $ P_bold(X)(#cimage("figures/lecture_1/dog.png", height: 30%) ) = 1 / n $
][
  $ P_bold(X)(#cimage("figures/lecture_1/muffin.png", height: 30%) ) = 0 $
]

$ p_*(#cimage("figures/lecture_1/dog.png", height: 30%)) quad vec(delim: #none, =, <, >) quad p_*(#cimage("figures/lecture_1/muffin.png", height: 30%)) $

==
We want to approximate distribution so we can generate samples

#side-by-side[
  *Question:*
][
$ bold(x)_([i]) tilde P_bold(X) (bold(x); bold(theta)) $
][
or
][
$ bold(x)_([i]) tilde p_* (bold(x); bold(theta)) $
]

$P_bold(X) (bold(x); bold(theta))$ samples existing datapoints, must be $p_* (bold(x); bold(theta))$

Only have $P_bold(X) (bold(x); bold(theta))$, not $p_* (bold(x); bold(theta))$
- How to approximate $p_* (bold(x); bold(theta))$?

*Key idea:* Learn a continuous distribution $p (bold(x); bold(theta))$ using $P_bold(X) (bold(x))$

$ p (bold(x); bold(theta)) = P_bold(X) (bold(x)) => p (bold(x); bold(theta)) approx p_* (bold(x)) $

We call $p (bold(x); bold(theta))$ a *probabilistic generative model*

==
Understanding generative models probabilistically is very hard 
- Requires statistics background most CS students lack
- Students always struggle with this lecture
- I change it every year to try to improve it

If you understand this concept, all generative models become easy
- Same objective, just different approximation methods
  - GAN
  - VAE
  - Diffusion
  - Normalizing flow
  - Flow matching
  - ...

==
Let us think about the objective function
- We have the dataset distribution $P_bold(X) (bold(x))$
- We have a model that outputs a distribution $p (bold(x); bold(theta))$
- Output distribution must match the dataset distribution 

*Question:* What is our objective?

$ argmax_(bold(theta)) space KL(P_bold(X) (bold(x)), p(bold(x); bold(theta))) $

Same idea as classification
- Almost all generative models use this objective
  - They represent it in different ways

==
To summarize, probabilistic generative models:
+ Use a dataset distribution $P_bold(X)(bold(x))$
+ To approximate the true data distribution $p_* (bold(x))$
+ By learning a *continuous* distribution $p(bold(x); bold(theta))$
+ That minimizes some approximation of $KL(p(bold(x); bold(theta)), P_bold(X)(bold(x)))$

总结来说，概率生成模型：
+ 使用数据集的经验分布 $P_bold(X)(bold(x))$
+ 来近似真实的数据分布 $p_*(bold(x))$
+ 其方法是学习一个 *连续* 的参数化分布 $p(bold(x); bold(theta))$
+ 该分布通过最小化 $KL(p(bold(x); bold(theta)), P_bold(X)(bold(x)))$ 来进行近似。


= Bayesian Inference
==

#side-by-side(columns: (0.4fr, 1.0fr), align: left)[
  #cimage("figures/lecture_1/dog.png") 
  $ bold(x) $
][
- Model $=p(bold(x); bold(theta))$
- Objective $approx KL(p(bold(x); bold(theta)), P_bold(X)(bold(x)))$
*Question:* Represent $p(bold(x); bold(theta))$?
- $bold(x)$ is very high dimensional
  - $bold(x) in {0 dots 255}^(3 times 28 times 28)$
- Complex relationships between pixels
- Generally intractable
]
*Key idea:* Use low-dimensional representation of $bold(x)$ 
- Can reason over low-dimensional representation

/*
*Question:* How?

*Answer:* Autoencoders!
- Learns semantic, low-dimensional representation $bold(z)$
*/

==
Let $bold(y)_([i])$ be a low-dimensional label of $bold(x)_([i])$

#side-by-side(align: horizon)[
  $ bold(y)_([i]) = vec("Small", "Ugly") $
][
  $ bold(x)_([i]) = #cimage("figures/lecture_1/dog.png", height: 25%) $
]

Probabilistic model should output distribution

$ underbrace(#cimage("figures/lecture_1/dog.png", height: 25%), bold(x)_([i])) tilde p(underbrace("Small ugly dog pictures", bold(x)) mid(|) underbrace(vec("Small", "Ugly"), bold(y)_([i])); bold(theta)) $

==
We must consider all possible $bold(y)$ to represent all possible dog pictures

$ vec("Small", "Ugly"), vec("Medium", "Ugly"), vec("Big", "Not ugly"), dots $

We also need to know the label distribution $bold(y)$
- How common or rare are ugly dogs? Big dogs? Small dogs?

$ p(bold(x); bold(theta)) = integral underbrace(p(bold(x) | bold(y); bold(theta)), "Image from label") overbrace(p(bold(y)), "Label distribution") dif bold(y) $

This is the definition of a conditional distribution

==

/*
*Question:* Do we know $bold(z)$ in unsupervised learning?

*Answer:* No! Only have $bold(x)$, not $bold(z)$

*Question:* Can we learn $p(bold(x), bold(z); bold(theta))$ with only $bold(x)$? How? *Hint:* Marginal


$ p(bold(x); bold(theta)) = integral p(bold(x), bold(z); bold(theta)) dif bold(z) $

==
$ p(bold(x); bold(theta)) = integral p(bold(x), bold(z); bold(theta)) dif bold(z) $

Recall the original problem is $bold(x)$ being continuous and high-dimensional

*Question:* What is the problem with this integral?

*Answer:* $p(bold(x), bold(z); bold(theta))$ is even higher dimensional than $p(bold(x); bold(theta))$
- Made the problem even harder, impossible to learn this distribution
- Factorize problem to make it easier

From the definition of conditional probability
$ p(bold(x), bold(z); bold(theta)) = p(bold(x) | bold(z); bold(theta)) dot p(bold(z); bold(theta)) $

*/
==
Bayesian inference model 

$ p(bold(x); bold(theta)) = integral p(bold(x) | bold(y); bold(theta)) dot p(bold(y)) dif bold(y) $

*Question:* What was the objective?

$ argmax_bold(theta) KL(P_bold(X)(bold(x)), p(bold(x); bold(theta))) $

This is Bayesian inference! 

==
#side-by-side[
  $ p(bold(x); bold(theta)) = integral underbrace(p(bold(x) | bold(y); bold(theta)), "Learn it") dot p(bold(y)) dif bold(y) $
][
  $ argmax_bold(theta) KL(P_bold(X)(bold(x)), p(bold(x); bold(theta))) $
]

Aftering learning $p(bold(x) | bold(y); bold(theta))$ we can generate new datapoints

+ Either choose or sample label $bold(y)_([i]) tilde p(bold(y))$
+ Predict conditional distribution $p(bold(x) | bold(y)_([i]); bold(theta))$
+ Sample from conditional distribution $bold(x)_([i]) tilde p(bold(x) | bold(y)_([i]); bold(theta))$

Very easy! But this requires labels $bold(y)$
- Can we do this unsupervised (without labels)?
- Then, model can learn labels $bold(y)$ without human help

= Variational Autoencoders
==
Consider Bayesian inference *without labels* $bold(y)$
- We will *learn* a low-dimensional latent description $bold(z)$ instead
  - $bold(z)$ encodes the structure of the dataset (and the world)

$ p(bold(x); bold(theta)) = integral p(bold(x) | bold(y); bold(theta)) dot p(bold(y)) dif bold(y) => integral p(bold(x) | bold(z); bold(theta)) dot p(bold(z)) dif bold(z) $

We no longer have a label distribution $p(bold(y))$
- Replace with a latent distribution $p(bold(z))$
- We can choose this distribution! Gaussian, Bernoulli, ...
- For now, make it easy $p(bold(z)) = cal(N)(bold(0), bold(I))$

==
*Question:* How can we learn $bold(z)$?

*Answer:* Autoencoding! 

$ hat(bold(x)) = f^(-1)(underbrace(f(bold(x), bold(theta)_e), bold(z)), bold(theta)_d) $

But our models are probabilistic, so it is more complex

==

/*
#side-by-side[
  #fletcher-diagram(
    node-stroke: .15em,
    node-fill: blue.lighten(50%),
    edge-stroke: .1em,
    
    node((0,0), $ bold(x) $, radius: 2em, name: <A>),
    node((1,0), $ bold(z) $, fill: gray.lighten(50%), radius: 2em, name: "B"),
    // Define edges
    edge(<A>, <B>, "-|>"),
    )

    Encoder $p(bold(z) | bold(x); bold(theta)_e)$

][
  #fletcher-diagram(
  node-stroke: .15em,
  node-fill: blue.lighten(50%),
  edge-stroke: .1em,
  
  node((0,0), $ bold(x) $, radius: 2em, name: <A>),
  node((1,0), $ bold(z) $, fill: gray.lighten(50%), radius: 2em, name: "B"),
  // Define edges
  edge(<A>, <B>, "<|-"),
  )

  Decoder $p(bold(x) | bold(z); bold(theta)_d)$
]
*/

#side-by-side[
  #fletcher-diagram(
    node-stroke: .15em,
    node-fill: blue.lighten(50%),
    edge-stroke: .1em,
    
    node((0,0), $ bold(x) $, radius: 2em, name: <A>),
    node((1,0), $ bold(z) $, radius: 2em, fill: gray.lighten(50%), name: <B>),
    // Define edges
    edge(<A>, <B>, "-|>", bend: 45deg, $ "Encoder" p(bold(z) | bold(x); bold(theta)_e) $),
    edge(<A>, <B>, "<|-", bend: -45deg, $ "Decoder" p(bold(x) | bold(z); bold(theta)_d) $),
  )
][
$ p(bold(x)) = integral p(bold(x) | bold(z); bold(theta)) p(bold(z)) dif bold(z) $
]

+ Use encoder and decoder to learn latent structure $bold(z)$ 
+ After learning, delete encoder (encoder only needed to learn $bold(z)$)
+ Do Bayesian inference with decoder and latent distribution

Sound too easy? Implementation is much harder

= Implementation Details
==
TODO: Should this go after HVAE and Diffusion?
==
#side-by-side[
  $ p(bold(x); bold(theta)_d) = integral underbrace(p(bold(x) | bold(z); bold(theta)_d), "Decoder") p(bold(z)) dif bold(z) $
][
  $ underbrace(p(bold(z) | bold(x); bold(theta)_e), "Encoder") $
]
  $ argmax_bold(theta) KL(P_bold(X)(bold(x)), p(bold(x); bold(theta))) $
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