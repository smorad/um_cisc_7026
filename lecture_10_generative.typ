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

// Review VAE
// VAEs are a class of probabilistic generative model
// P(x | z)
// Hierarchical VAEs
// Diffusion Models
// Normalizing Flows
  // Turn base distribution into 



= Admin
==
How was the last homework? #pause
- Finished with homework! #pause

What's left: #pause
- Final exam (for some students) #pause
- Group project (for everyone)
==
Final project groups #pause
- Merged group G (2 people) and group R (3 people) #pause
- Added last student to group E (randomly selected) #pause
- All groups should be 4-5 students now #pause

Groups G, E have extension for project plan #pause
- Submit by 11 Nov for full credit #pause

*If you are not in a group of 4-5, you cannot submit final project* #pause
- You must be in a group #pause
- Come see me immediately

= Review

==
#side-by-side(align: left)[#cimage("figures/lecture_9/kungfu.jpg", height: 100%) #pause][
  *Question:* You watch a film. How do you communicate information about the film with a friend? #pause

  *Answer:* Fat panda becomes kung fu master to defend his village from an evil snow leopard. #pause

  *Question:* What is missing? #pause

  *Answer:* Goose father, tortoise teacher, tiger friend, magic scroll, etc
]

==
  When you discuss films with friends, you summarize them #pause

  This is a form of *compression* #pause

  $ f(vec(bold(x)_1, dots.v, bold(x)_n)) = "Fat panda saves village" $ 

==
  Encoders and decoders for images, videos, and music are functions #pause

  Neural networks can represent any continuous function #pause

  We can use neural networks to represent encoders and decoders #pause

  $ f: X times Theta |-> Z $ #pause

  $ f^(-1): Z times Theta |-> X $ #pause

  We call this an *autoencoder* #pause

  Notice no labels $Y$ this time 

==
  In supervised learning, humans provide the model with *inputs* $bold(X)$ and corresponding *outputs* $bold(Y)$ #pause

  $ bold(X) = mat(bold(x)_[1], bold(x)_[2], dots, bold(x)_[n])^top quad bold(Y) = mat(bold(y)_[1], bold(y)_[2], dots, bold(y)_[n])^top $ #pause

  In unsupervised learning, humans only provide *input* #pause

  $ bold(X) = mat(bold(x)_[1], bold(x)_[2], dots, bold(x)_[n])^top $ #pause

  The training algorithm will learn *unsupervised* (only from $bold(X)$) #pause
  - Humans do not need to provide labels!
==
  $f, f^(-1)$ may be any neural network #pause

  $ bold(x) = f^(-1)(f(bold(x), bold(theta)_e), bold(theta)_d) $ #pause

  Turn this into a loss function using the square error #pause

  $ cal(L)(bold(x), bold(theta)) = sum_(j=1)^(d_x) (x_j - f^(-1)(f(bold(x), bold(theta)_e), bold(theta)_d)_j)^2 $ #pause

  Forces the networks to compress and reconstruct $bold(x)$

==
Can make many types of autoencoders #pause
- Convolutional autoencoder #pause
- Recurrent autoencoder #pause
- Graph neural network autoencoders #pause
- Transformer autoencoders #pause

The reconstruction objective is useful for compression 

$ cal(L)(bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f(bold(x)_[i], bold(theta)_e), bold(theta)_d)_j)^2 $ #pause

Could we use autoencoders for other tasks?

==
*Task:* Fix blurry and noisy image #pause

$ cal(L)(bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f( #redm[blur];(bold(x)_[i] + #bluem[$bold(epsilon)$]), bold(theta)_e), bold(theta)_d)_j)^2 $ #pause

#side-by-side[
  #cimage("figures/lecture_9/enhance0.jpg")
][
  #cimage("figures/lecture_9/enhance1.jpg")
]

==

*Task:* Fix image with missing pixels #pause


$ "Reconstruction loss" quad cal(L)(bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f(bold(x)_[i], bold(theta)_e), bold(theta)_d)_j)^2 $ #pause


#side-by-side[Sample Bernoulli noise][$ bold(b) tilde cal(B)(0.2) $] #pause

Masked reconstruction loss

$ cal(L)(bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f(bold(x)_[i] #redm[$dot.o bold(b)$] , bold(theta)_e), bold(theta)_d)_j)^2 $ 

==
#side-by-side(align: left)[
  #cimage("figures/lecture_9/masked.png") #pause
][
  Understanding emerges from reconstruction objectives #pause

  *Without labels*, autoencoders learn what a bird is, what a bear is #pause

  They learn the structure and distribution of the input data $X$ #pause

  If the input data $X$ is from our world, then they begin to understand our world
]

= Generative Models
==

#side-by-side(align: left)[
  #cimage("figures/lecture_9/fashion-latent.png", height: 85%)
][
  Autoencoder latent space $Z in bb(R)^3$ #pause

  Images with similar semantic meaning (sneaker, sandal) cluster in Z #pause

  $Z$ forms a semantic manifold #pause
  - Each datapoint locally Euclidean #pause

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
    - Image not in the dataset #pause
    - Semantically similar to $bold(x)_i$ #pause
      - But *new* and different! #pause

    $ bold(z)_[i] &-> "blue shirt" \ 
    bold(z)_[i] + bold(epsilon) &-> "red shirt" $ #pause

    Idea powers generative models
]

==
Generating new clothing pictures is not so exciting #pause

But we can apply generative models to many exciting problems #pause
- Train on books, generate new stories #pause
- Train on music, generate new music #pause
- Train on medicine, generate new medicine #pause

Is it that easy? Train autoencoder on medicine to create new medicines? #pause

Unfortunately, no. We can do it, but not with a simple autoencoder 

==
This autoencoder only works for small $d_z$ #pause
- More interesting problems are high-dimensional (large $d_z$) #pause

#cimage("figures/lecture_9/curse.png") #pause

As $d_z$ grows, points spread out (curse of dimensionality) #pause
- For large $d_z$, one cluster for each datapoint

==
#side-by-side[
  #cimage("figures/lecture_9/curse.png") #pause
][
$ f^(-1) (bold(z) + bold(epsilon)); d_z >> 0 $ #pause
]

Adding noise and decoding does not work for large $d_z$ #pause

#cimage("figures/lecture_9/fail_recon.png", height: 40%)

==
Autoencoder latent space has learned structure #pause
- We must enforce some latent structure for generative models #pause
- We use a probabilistic framework for this #pause
- Very active area of research 


= Probabilistic Generative Models

==
*Question:* How do we explain autoencoders learning our world? #pause

*Answer:* They implicitly model the *dataset distribution* #pause
- If the dataset contains our world, autoencoders learn the world #pause

Must rethink machine learning to understand generative models #pause
- Most lectures I try to skip/ignore probability theory #pause
- Understanding probability is necessary for generative models #pause
  - Please try your best to pay attention! #pause

Let us consider a dataset of dog pictures


#slide(composer: (0.4fr, 1fr))[
  #cimage("figures/lecture_1/dog.png") 
  $ bold(x)_([i]) $
  #pause
][
  *Domain:* $X = [0, 1]^{3 times 28 times 28}$ #pause
  - How we represent datapoints #pause
  - $bold(x)_([i]) in X$ #pause

  *Dataset (Evidence):* $bold(X) = mat(bold(x)_([1]), dots, bold(x)_([n]))^top$ #pause
  - Finite collection of $n$ datapoints #pause

  *Dataset distribution:* $P_bold(X) (bold(x)_([i]))$ #pause
  - Probability of uniformly sampling $bold(x)_([i])$ from $bold(X)$ #pause
  - $1/n$ if $bold(x)_([i])$ in dataset, else 0 #pause

  *True data distribution:* $p_* (bold(x)_([i]))$ #pause
  - How likely is $bold(x)_([i])$ to exist in the universe? 
  //- Superset of the dataset $bold(X)$
]

==

#side-by-side(align: left)[
  $ P_bold(X)(#cimage("figures/lecture_1/dog.png", height: 30%) ) = #pause 1 / n $ #pause
][
  $ P_bold(X)(#cimage("figures/lecture_1/muffin.png", height: 30%) ) #pause = 0 $ #pause
]

$ p_*(#cimage("figures/lecture_1/dog.png", height: 30%)) quad vec(delim: #none, =, <, >) quad p_*(#cimage("figures/lecture_1/muffin.png", height: 30%)) $ #pause

The probability of a point in a continuous distribution is always 0 #pause
- For continuous distributions, we consider density not probability #pause

==
#cimage("figures/lecture_9/prob-vs-density.png")

We let small $p$ represent likelihood

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

$P_bold(X) (bold(x); bold(theta))$ samples finite, existing datapoints, must be $p_* (bold(x); bold(theta))$ #pause

Only have $P_bold(X) (bold(x); bold(theta))$, not $p_* (bold(x); bold(theta))$ #pause
- How to approximate $p_* (bold(x); bold(theta))$? #pause

*Key idea:* Learn a continuous distribution $p (bold(x); bold(theta))$ using $P_bold(X) (bold(x))$ #pause

$ p (bold(x); bold(theta)) = P_bold(X) (bold(x)) => p (bold(x); bold(theta)) approx p_* (bold(x)) $ #pause

We call $p (bold(x); bold(theta))$ a *probabilistic generative model*

==
#align(center, pdf_pmf)

==
Truly understanding generative models is very hard #pause
- Took me years, I still do not fully understand #pause

If you understand this concept, all generative models become easy #pause
- Same objective, just different approximation methods #pause
  - GAN
  - VAE
  - Diffusion
  - Normalizing flow
  - Flow matching
  - ...

==
Let us think about the objective function for a generative model #pause
- We have the dataset distribution $P_bold(X) (bold(x))$ #pause
- We have a model that outputs a distribution $p (bold(x); bold(theta))$ #pause
- Output distribution must match the dataset distribution  #pause

*Question:* What is our objective? #pause *Hint:* Two distributions #pause

$ argmin_(bold(theta)) space KL(P_bold(X) (bold(x)), p(bold(x); bold(theta))) $ #pause

Same idea as classification #pause
- Let us see if we can simplify the objective

==
$ argmin_bold(theta) KL(P_bold(X)(bold(x)), p(bold(x); bold(theta))) $ #pause

From the definition of $KL$

$ argmin_bold(theta) KL(P_bold(X)(bold(x)),  p(bold(x); bold(theta))) = \

argmin_bold(theta) sum_(i=1)^n P_bold(X) (bold(x)_([i])) log (
  P_bold(X) (bold(x)_([i]))
) / (
  p(bold(x)_([i]); bold(theta))
)
$ #pause

However, we know the probability for all datapoints $P_bold(X) (bold(x))$ #pause

*Question:* What is it? #pause *Answer:* $1 / n$ if $bold(x)_([i])$ in dataset, else 0

==
$ argmin_bold(theta) sum_(i=1)^n P_bold(X) (bold(x)_([i])) log (
  P_bold(X) (bold(x)_([i]))
) / (
  p(bold(x)_([i]); bold(theta))
)
$ #pause

//Since $P_bold(X)$ is constant, let us try and factor it out #pause
//- Then we can remove it from the objective #pause

$ argmin_bold(theta) sum_(i=1)^n P_bold(X) (bold(x)_([i])) [ log 
  P_bold(X) (bold(x)_([i]))
 - log p(bold(x)_([i]); bold(theta))
 ]
$ #pause

$ argmin_bold(theta) sum_(i=1)^n P_bold(X) (bold(x)_([i]))  log 
  P_bold(X) (bold(x)_([i]))
 - P_bold(X) (bold(x)_([i])) log p(bold(x)_([i]); bold(theta))
$ 

$ argmin_bold(theta) (sum_(i=1)^n P_bold(X) (bold(x)_([i])) log 
  P_bold(X) (bold(x)_([i]))) 
 - (sum_(i=1)^n P_bold(X) (bold(x)_([i])) log p(bold(x)_([i]); bold(theta)))
$ #pause


==

$ argmin_bold(theta) (sum_(i=1)^n P_bold(X) (bold(x)_([i])) log 
  P_bold(X) (bold(x)_([i]))) 
 - (sum_(i=1)^n P_bold(X) (bold(x)_([i])) log p(bold(x)_([i]); bold(theta)))
$ #pause

First term is constant and does not depend on $bold(theta)$, can delete it!

$ argmin_bold(theta) sum_(i=1)^n - P_bold(X) (bold(x)_([i])) log p(bold(x)_([i]); bold(theta))
$ 

Very similar to classification loss derivation so far 

==
$ argmin_bold(theta) - sum_(i=1)^n P_bold(X) (bold(x)_([i])) log p(bold(x)_([i]); bold(theta)) $ #pause

Let us think about $P_bold(X) (bold(x)_([i]))$, can we rewrite it? #pause
- *Hint:* The sum is over the dataset #pause
- *Hint:* What is $P_bold(X)$ for a datapoint in the dataset? #pause

$ P_bold(X) (bold(x)_([i])) = 1 / n $ #pause

$ argmin_bold(theta) - sum_(i=1)^n 1 / n log p(bold(x)_([i]); bold(theta)) $ #pause

==

$ argmin_bold(theta) - sum_(i=1)^n 1 / n log p(bold(x)_([i]); bold(theta)) $ #pause

$1 / n$ is a constant, does not affect minima #pause

$ argmin_bold(theta) - sum_(i=1)^n log p(bold(x)_([i]); bold(theta)) $ #pause

==
$ argmin_bold(theta) - sum_(i=1)^n log p(bold(x)_([i]); bold(theta)) $ #pause

$ argmin_bold(theta) sum_(i=1)^n - log p(bold(x)_([i]); bold(theta)) $ #pause

This is the *negative log likelihood* #pause
- Used all over statistics and machine learning (e.g. LLM) #pause

The log likelihood objective: #pause
- Learns the parameters $bold(theta)$ of a continuous distribution $p(bold(x); bold(theta))$ #pause
- That maximizes the likelihood of each datapoint $bold(x)_([i])$ under the model
==

$ argmin_bold(theta) sum_(i=1)^n - log p(bold(x)_([i]); bold(theta)) $ #pause

It may not seem intuitive #pause
- More intuitive to think about equivalent maximization objective #pause
  - $min => max$, $- => +$ #pause

$ argmax_bold(theta) sum_(i=1)^n log p(bold(x)_([i]); bold(theta)) $ #pause

Find $bold(theta)$ so likelihood $p$ is maximized at datapoints $bold(x)_([i])$

==
#align(center, pdf_pmf)

$ p: Theta |-> Delta(X) $

==
To summarize, probabilistic generative models: #pause
+ Use a finite dataset distribution $P_bold(X)(bold(x))$ #pause
+ To approximate the true data distribution $p_* (bold(x))$ #pause
+ By learning a *continuous* distribution $p(bold(x); bold(theta))$ #pause
+ That minimizes some approximation of negative log likelihood #pause

#side-by-side[
  Generative model
][
  $ p(bold(x); bold(theta)) $
]
#side-by-side[
  Negative log likelihood objective
 ][
 $ argmin_bold(theta) sum_(i=1)^n - log p(bold(x)_([i]); bold(theta)) $
]
  


= Bayesian Generative Models
==
Let us start to implement generative models #pause

#side-by-side(columns: (0.4fr, 1.0fr), align: left)[
  #cimage("figures/lecture_1/dog.png") 
  $ bold(x)_([i]) $ #pause
][
- Model $approx p(bold(x); bold(theta))$ #pause
- Objective $approx argmin_bold(theta) sum_(i=1)^n - log p(bold(x)_([i]); bold(theta))$ #pause
*Question:* How to represent $p(bold(x); bold(theta))$? #pause
- $bold(x)$ is very high dimensional #pause
  - $bold(x) in [0, 1]^(3 times 28 times 28)$ #pause
- Complex relationships between pixels #pause
  - Generally intractable to approximate $p_* (bold(x))$ #pause
]
*Key idea:* Approximate low-dimensional representation of $bold(x)$ #pause
- Can reason over low-dimensional representation

==
Let $bold(y)_([i])$ be a low-dimensional label of $bold(x)_([i])$ #pause

#side-by-side(align: horizon)[
  $ bold(x)_([i]) = #cimage("figures/lecture_1/dog.png", height: 20%) $ #pause
][
  $ bold(y)_([i]) = vec("Small", "Ugly") $ #pause
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

$ argmin_bold(theta) sum_(i=1)^n - log p(bold(x)_([i]); bold(theta)) $ #pause

Can plug model into objective

$ argmin_bold(theta) sum_(i=1)^n - log integral p(bold(x)_([i]);  | bold(y); bold(theta)) dot p(bold(y)) dif bold(y) $ #pause

==
$ underbrace(argmin_bold(theta) sum_(i=1)^n - log overbrace(integral p(bold(x)_([i]) | bold(y); bold(theta)) dot p(bold(y)) dif bold(y), "Model"), "Objective") $

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
  - Replace label distribution $p(bold(y))$ with latent distribution $p(bold(z))$ #pause

$ p(bold(x); bold(theta)) = integral p(bold(x) | bold(y); bold(theta)) dot p(bold(y)) dif bold(y) $ #pause

$ p(bold(x); bold(theta)) = integral p(bold(x) | bold(z); bold(theta)) dot p(bold(z)) dif bold(z) $ #pause

We can choose this distribution, Gaussian, Bernoulli, ... #pause
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

$ underbrace(argmin_bold(theta) sum_(i=1)^n - log overbrace(integral p(bold(x)_([i]) | bold(z); bold(theta)) dot p(bold(z)) dif bold(z), "Model"), "Objective") $ #pause

However, this expression is  often intractable #pause

*Question:* Why? #pause *Answer:* Cannot solve integral #pause 
- No analytical solution if $p(bold(x) | bold(z); bold(theta))$ is neural network #pause
- Indefinite integration over continuous/infinite variable $bold(z)$

==
$ underbrace(argmin_bold(theta) sum_(i=1)^n - log overbrace(integral p(bold(x)_([i]) | bold(z); bold(theta)) dot p(bold(z)) dif bold(z), "Model"), "Objective") $

Variational inference methods (like VAEs) approximate this objective using the *evidence lower bound* (ELBO) #pause
- Lower bounds the true log likelihood #pause

The derivation is interesting, but requires concepts you do not know #pause
- Expected values, Jensen's inequality, etc #pause
- Instead, I will show you the final result

==
The original objective

$ underbrace(argmin_bold(theta) sum_(i=1)^n - log overbrace(integral p(bold(x)_([i]) | bold(z); bold(theta)) dot p(bold(z)) dif bold(z), "Model"), "Objective") $

The surrogate objective (ELBO)

$ argmin_bold(theta) sum_(i=1)^n bb(E)_(bold(z) tilde p(bold(z) | bold(x)_([i]) ; bold(theta)_e))[
 -log p(bold(x)_([i]) | bold(z); bold(theta)_d)
]
+KL(p(bold(z) | bold(x) ; bold(theta)_e), p(bold(z)))  $ #pause

This is a *variational autoencoder* (VAE)

==

#v(1em)
$ argmin_bold(theta) sum_(i=1)^n 
  markrect(bb(E)_(bold(z) tilde 
  markhl(p(bold(z) | bold(x)_([i]) ; bold(theta)_e),tag: #<1>, color: #orange))[
    - log markhl(p(bold(x)_([i]) | bold(z); bold(theta)_d), tag: #<2>, color: #blue)
  ], stroke: #3pt, color: #red, radius: #5pt, tag: #<4>)
+ markrect(KL(
  markhl(p(bold(z) | bold(x)_([i]) ; bold(theta)_e), tag: #<3>, color: #orange),
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
  - Encoder should output normal distribution so KL term is analytical #pause
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
    $ p(bold(x) | bold(z); bold(theta)_d) $
  ][
    Encoder 
    $ p(bold(z) | bold(x); bold(theta)_e) $
  ][
    Marginal 
    $ p(bold(z)) = cal(N)(bold(0), bold(1)) $
  ] #pause

$ argmin_bold(theta) sum_(i=1)^n bb(E)_(bold(z) tilde p(bold(z) | bold(x)_([i]) ; bold(theta)_e))[
 - log p(bold(x)_([i]) | bold(z); bold(theta)_d)
]
+ KL(p(bold(z) | bold(x) ; bold(theta)_e), p(bold(z))) $ #pause

Rewrite objective ass loss function for gradient descent #pause

$ cal(L)(bold(X), bold(theta)) = sum_(i=1)^n bb(E)_(bold(z) tilde p(bold(z) | bold(x)_([i]) ; bold(theta)_e))[
 - log p(bold(x)_([i]) | bold(z); bold(theta)_d)
]
+ KL(p(bold(z) | bold(x) ; bold(theta)_e), p(bold(z))) $ 

==
$ cal(L)(bold(X), bold(theta)) = sum_(i=1)^n bb(E)_(bold(z) tilde p(bold(z) | bold(x)_([i]) ; bold(theta)_e))[
 - log p(bold(x)_([i]) | bold(z); bold(theta)_d)
]
+ KL(p(bold(z) | bold(x) ; bold(theta)_e), p(bold(z))) $ #pause

  First, rewrite KL term using our encoder $f$ #pause

  $ cal(L)(bold(X), bold(theta)) = sum_(i=1)^n bb(E)_(bold(z) tilde p(bold(z) | bold(x)_([i]) ; bold(theta)_e))[
  -log p(bold(x)_([i]) | bold(z); bold(theta)_d)
  ]
  + KL(f(bold(x), bold(theta)_e), p(bold(z))) $ #pause

  $p(bold(z))$ and $f(bold(x), bold(theta)_e) = bold(mu), bold(sigma)$ are Gaussian, we can simplify KL term

  $ = sum_(i=1)^n bb(E)_(bold(z) tilde p(bold(z) | bold(x)_([i]) ; bold(theta)_e))[
  -log p(bold(x)_([i]) | bold(z); bold(theta)_d)
  ]
  + 0.5 (sum_(j=1)^d_z mu^2_j + sigma^2_j - log(sigma^2) - 1) $ 


==
  $ = sum_(i=1)^n bb(E)_(bold(z) tilde p(bold(z) | bold(x)_([i]) ; bold(theta)_e))[
  -log p(bold(x)_([i]) | bold(z); bold(theta)_d)
  ]
  + 0.5 (sum_(j=1)^d_z mu^2_j + sigma^2_j - log(sigma^2) - 1) $  #pause

  Log probability for unit Gaussian is just mean square error #pause

  $ log p prop log exp((x - mu)^2) = (x - mu)^2  $ #pause

  $ sum_(i=1)^n (sum_(j=1)^d_x (x_([i], j) - f^(-1)(f(bold(x)_([i]), bold(theta)_e), bold(theta)_d)_j )^2 - 0.5 sum_(j=1)^d_z (mu^2_j + sigma^2_j - log(sigma^2_j) - 1)) $ 

==
  $ sum_(i=1)^n (sum_(j=1)^d_x (x_([i], j) - f^(-1)(f(bold(x)_([i]), bold(theta)_e), bold(theta)_d)_j )^2 - 0.5 sum_(j=1)^d_z (mu^2_j + sigma^2_j - log(sigma^2_j) - 1)) $ #pause

  Scale of two terms can vary, we do not want one term to dominate #pause
  - Modern VAEs introduce a hyperparameter $beta$ #pause

  $ sum_(i=1)^n (sum_(j=1)^d_x (x_([i], j) - f^(-1)(f(bold(x)_([i]), bold(theta)_e), bold(theta)_d)_j )^2 - 0.5 #redm[$beta$] sum_(j=1)^d_z (mu^2_j + sigma^2_j - log(sigma^2_j) - 1)) $ #pause

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

$ cal(L)_"VAE" (bold(X), bold(theta)) = sum_(i=1)^n underbrace(bb(E)_(bold(z) tilde p(bold(z) | bold(x)_([i]) ; bold(theta)_e))[
 -log p(bold(x)_([i]) | bold(z); bold(theta)_d)
], "Reconstruction")
+ underbrace(KL(p(bold(z) | bold(x) ; bold(theta)_e), p(bold(z))), "Marginal match") $ #pause

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