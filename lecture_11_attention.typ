#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "@preview/cetz:0.4.0"
#import "@preview/fletcher:0.5.8" as fletcher: node, edge
#import "common.typ": *
#import "plots.typ": *
#import "@preview/algorithmic:1.0.5"
#import algorithmic: style-algorithm, algorithm-figure, algorithm
#import "@preview/mannot:0.3.0": *

#let handout = true

#set math.vec(delim: "[")
#set math.mat(delim: "[")

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-common(handout: handout),
  config-info(
    title: [Attention],
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

= Admin
==
Last exam next week #pause

Preliminary questions (still making exam, may change) #pause
- 1 Question perceptron autoencoder and objective
- 1 Question rewriting function as recurrent function 
- 1 Question evaluating scans
- 1 Question log likelihood derivation
- 1 Question self attention ($bold(Q) bold(K)^top$, $softmax$)

// TODO: Cleanup attention shapes

= Review
== 
Finish VAE coding

https://colab.research.google.com/drive/1UyR_W6NDIujaJXYlHZh6O3NfaCAMscpH#scrollTo=NLTDp6ipBtJU


/*
#sslide[

  We can use autoencoders as *generative models* #pause

  A generative model learns the structure of data #pause

  Using this structure, it generates *new* data #pause

  - Train on face dataset, generate *new* faces #pause
  - Train on book dataset, write a *new* book #pause
  - Train on human knowledge, create *new* knowledge #pause

  Generative models learn the dataset *distribution* $P(bold(x)); quad bold(x) in X$ 
]

#sslide[
  Generative models learn the dataset *distribution* $P(bold(x)); quad bold(x) in X$ 

  Virtually *all* generative methods learn the dataset distribution #pause
  - Variational autoencoders #pause
  - Diffusion models #pause
  - Generative adversarial networks #pause
  - Normalizing flows #pause

  If you like generative models, you should study Bayesian statistics #pause

  Back to the variational autoencoder
]

#sslide[
  Latent space $Z$ for autoencoder on the clothes dataset with $d_z = 3$

  #cimage("figures/lecture_9/fashion-latent.png", height: 85%)

  // #cimage("figures/lecture_9/latent_space.png") #pause
]

#sslide[
  What happens if we decode a new point?

  #cimage("figures/lecture_9/fashion-latent.png", height: 85%)
]

#sslide[
  #side-by-side[
    #cimage("figures/lecture_9/fashion-latent.png")
  ][
    #set align(left)
  Autoencoder generative model: #pause
  
  1. Encode $ vec(bold(x)_[1], dots.v, bold(x)_[n])$ into $vec(bold(z)_[1], dots.v, bold(z)_[n]) $ #pause

  2. Pick a point $bold(z)_[k]$ #pause

  3. Add some noise $bold(z)_"new" = bold(z)_[k] + bold(epsilon)$ #pause

  4. Decode $bold(z)_"new"$ into $bold(x)_"new"$
  ]
]

#sslide[
  #cimage("figures/lecture_9/vae_gen_faces.png", height: 70%) #pause

  $ f^(-1)(bold(z)_k + bold(epsilon), bold(theta)_d) $
]

#sslide[
  #side-by-side[
  
  But there is a problem, the *curse of dimensionality* #pause

  As $d_z$ increases, points move further and further apart #pause

  ][
    #cimage("figures/lecture_9/curse.png") #pause
  ]

  $f^(-1)(bold(z) + epsilon)$ will produce either garbage, or $bold(z)$
]

#sslide[
  Variational autoencoders (VAEs) make $bold(z)_[1], dots bold(z)_[n]$ normally distributed #pause

  This keeps the point close together #pause

  We can sample new points $bold(z)_"new" tilde N(bold(0), bold(1))$
]

#sslide[
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
]

#sslide[
    Given a prior distribution
    
    $ P(bold(z)) = cal(N)(bold(0), bold(1)) $ #pause

    We want to approximate the dataset distribution

    $ P(bold(x)) $ #pause

    By approximating the marginal likelihood (from Bayes rule)

    $ P(bold(x); bold(theta)) = integral_bold(z) P(bold(x) | bold(z); bold(theta)) P(bold(z)) dif bold(z) $
]


#sslide[
  *Key idea 2:* There is some latent variable $bold(z)$ which generates data $bold(x)$ #pause

  $ x: #cimage("figures/lecture_9/vae_slider.png") $

  $ z: mat("woman", "brown hair", ("frown" | "smile")) $
]

/*
#sslide[
  #cimage("figures/lecture_9/vae_slider.png") #pause

  Network can only see $bold(x)$, it cannot directly observe $bold(z)$ #pause

  Given $bold(x)$, find the probability that the person is smiling $P(bold(z) | bold(x); bold(theta))$
]
*/

// TODO: Our generative model needs to approximate P(x) = P(x | z) P (z)

#sslide[
  We cast the autoencoding task as a *variational inference* problem #pause

  #align(center, varinf)

  #side-by-side[
    Decoder 
    $ P(bold(x) | bold(z); bold(theta)) $
    Generate new $bold(x)$
  ][
    Encoder 
    $ P(bold(z) | bold(x); bold(theta)) $
    Learn meaningful $bold(z)$
  ] 

  //We want to learn both the encoder and decoder: $P(bold(z), bold(x); bold(theta))$
]

#sslide[
  How do we implement encoder $f$ (i.e., $P(bold(z) | bold(x); bold(theta))$ )? #pause

  $ f : X times Theta |-> Delta Z $ #pause

  Normal distribution has a mean $mu in bb(R)$ and standard deviation $sigma in bb(R)_+$ #pause

  Our encoder should output $d_z$ means and $d_z$ standard deviations #pause

  $ f : X times Theta |-> bb(R)^(d_z) times bb(R)_+^(d_z) $
]

#sslide[
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
  distribution = (mu, exp(log_sigma))
  ```
]

#sslide[
  We covered the encoder

  $ f: X times Theta |-> Delta Z $
  
  We can use the same decoder as a standard autoencoder #pause

  $ f^(-1): Z times Theta |-> X $ #pause

  Encoder outputs a distribution $Delta Z$ but decoder input is $Z$ #pause

  *Solution:* Sample a vector $bold(z)$ from the distribution $Delta Z$ 
]

#sslide[
  Put it all together #pause

  *Step 1:* Encode the input to a normal distribution

  $ bold(mu), bold(sigma) = f(bold(x), bold(theta)_e) $ #pause

  *Step 2:* Generate a sample from distribution

  $ bold(z) = bold(mu) + bold(sigma) dot.o bold(epsilon) $ #pause

  *Step 3:* Decode the sample 

  $ bold(x) = f^(-1)(bold(z), bold(theta)_d) $
]

#sslide[
    ```python
    # Create normal distribution from input
    mu, sigma = model.f(x)

    # Randomly sample a z vector from our distribution
    epsilon = jax.random.normal(key, x.shape[0])
    z = mu + sigma * epsilon

    # Decode/reconstruct z back into x
    pred_x = model.f_inverse(z)
    ```
]

#sslide[
  Now, all we must do is find $bold(theta)$ that best explains the dataset distribution #pause

  Learned distribution $P(bold(x); bold(theta))$ to be close to dataset $P(bold(x)), quad bold(x) tilde X$ #pause

  We started with the KL divergence

  $ argmin_bold(theta) KL(P(bold(x)), P(bold(x); bold(theta))) $ #pause
]

#sslide[
  From the KL divergence, we derived the *ELBO* loss for the VAE 

    $ cal(L)(bold(X), bold(theta)) &= underbrace(m / n sum_(i=1)^n sum_(j=1)^d_z (x_([i],j) - f^(-1)(f(bold(x)_[i], bold(theta)_e), bold(theta)_d)_(j) )^2, "Reconstruct" bold(x)) - \ 
    & underbrace(beta (sum_(i=1)^n sum_(j=1)^d_z mu^2_([i],j) + sigma^2_([i],j) - log(sigma_([i], j)^2) - 1), "Make" bold(z) "normally distributed") $
]


#sslide[
  ```python
  def L(model, x, m, n, beta, key):
    mu, sigma = model.f(x) # Encode input into distribution 
    # Sample from distribution
    z = mu + sigma * jax.random.normal(key, x.shape[0])
    # Reconstruct input
    pred_x = model.f_inverse(z) 
    # Compute reconstruction and kl loss terms
    recon = jnp.sum((x - pred_x) ** 2)
    kl = jnp.sum(mu ** 2 + sigma ** 2 - jnp.log(sigma ** 2) - 1)
    # Loss function contains reconstruction and kl terms
    return m / n * recon + beta * kl
  ```
]

#sslide[
  https://colab.research.google.com/drive/1UyR_W6NDIujaJXYlHZh6O3NfaCAMscpH#scrollTo=nmyQ8aE2pSbb

  https://colab.research.google.com/drive/1fwbkU46kvRWoisgIIZ2CFcynMReJsDUA

  Show my results

  https://openai.com/index/glow/
]
*/

= Attention
// Discussed seq models and conv
// We find various issues
// Locality not present
// seq models have vanishing gradient
// return to composite memory
    // faces at party
    // show this does not have vanishing gradient
    // we treat all faces as equal or forget based on time 
    // maybe only pay attention to the pretty ones
    // how would we model this?
    // sigmoid attention
        // function of current input or prior inputs?
    // but sigmoid thinks everyone is pretty
    // limited memory, make sure we choose just a few faces
    // softmax sum to one
    // maybe instead of talking

    // what if instead of pretty, we care about remembering important people

// attention paper over rnn states
// gated convolution
// attention is all you need
// function over sets

// implement transformer
// general purpose architecture
// ViT?
// other applications of attention


==
    Attention and transformers are the "hottest" topic in deep learning #pause

    People use them for almost every task (even if they shouldn't!) #pause

    Let's review some projects based on attention

==
    AlphaFold (Nobel prize)

    #cimage("figures/lecture_11/alphafold.png", height: 85%)

==
    ChatGPT, Qwen, LLaMA, Mistral, Doubou, Ernie chatbots

    #cimage("figures/lecture_11/llm.jpg", height: 85%)

==
    MusicTransformer, MuLan

    #cimage("figures/lecture_11/mulan.jpg", height: 85%)

==
    Google Translate, Baidu Translate, Apple Translate

    #cimage("figures/lecture_11/translate.jpeg", height: 85%)

==
    ViT, DinoV2

    #cimage("figures/lecture_9/dino.png", height: 85%)

==
    All these models are *transformers* #pause

    At the core of each transformer is *attention* #pause

    We can derive attention from composite memory

==

    #side-by-side[
        Francis Galton (1822-1911) \ photo composite memory

        #cimage("figures/lecture_8/galton.jpg", height: 70%) #pause
    ][
        Composite photo of members of a party 

        #cimage("figures/lecture_8/composite_memory.jpg", height: 70%)
    ]

==
    *Task:* Find a mathematical model of how our mind represents memories #pause

    $ X: bb(R)^(h times w) quad "People you see at the party" $ #pause

    $ H: bb(R)^(h times w) quad "The image in your mind" $ #pause

    $ f: X^T times Theta |-> H $ #pause

    Composite photography/memory uses a weighted sum #pause

    $ f(bold(x), bold(theta)) = sum_(i=1)^T bold(theta)^top overline(bold(x))_i $

==
    Limited space, cannot remember everything #pause

    Introduced forgetting term $gamma in [0, 1]$ #pause

    #side-by-side(columns: (0.4fr, 0.05fr, 0.6fr))[$ f(bold(x), bold(theta)) = sum_(i=1)^T gamma^(T - i) dot bold(theta)^top overline(bold(x))_i $ #pause][][
        #align(center)[#forgetting]
    ] #pause

    *Question:* Does this accurately model what *you* remember?

==
    *Example:* We attend a party in 1850s #pause

    We talk with many people at this party #pause

    #cimage("figures/lecture_11/composite0.svg") #pause

    #side-by-side[
    10 PM #pause
    ][
    11 PM #pause
    ][
    12 AM #pause
    ][
    1 AM
    ]

==
    According to forgetting, the memories should fade with time #pause

    #cimage("figures/lecture_11/composite_fade.png") #pause

    #side-by-side[
    $ gamma^3 bold(theta)^top overline(bold(x))_1 $ #pause
    ][
    $ gamma^2 bold(theta)^top overline(bold(x))_2 $ #pause
    ][
    $ gamma^1 bold(theta)^top overline(bold(x))_3 $ #pause
    ][
    $ gamma^0 bold(theta)^top overline(bold(x))_4 $
    ]

==
    Consider another party, with one more guest #pause

    #cimage("figures/lecture_11/composite_swift.png") #pause


    #side-by-side[
    $ gamma^4 bold(theta)^top overline(bold(x))_1 $
    ][
    $ gamma^3 bold(theta)^top overline(bold(x))_2 $
    ][
    $ gamma^2 bold(theta)^top overline(bold(x))_3 $
    ][
    $ gamma^1 bold(theta)^top overline(bold(x))_4 $
    ][
    $ gamma^0 bold(theta)^top overline(bold(x))_5 $
    ] #pause

    *Question:* What will happen to Taylor Swift?

==
    #cimage("figures/lecture_11/composite_swift_fade.png")

    #side-by-side[
    $ gamma^4 bold(theta)^top overline(bold(x))_1 $
    ][
    $ gamma^3 bold(theta)^top overline(bold(x))_2 $
    ][
    $ gamma^2 bold(theta)^top overline(bold(x))_3 $
    ][
    $ gamma^1 bold(theta)^top overline(bold(x))_4 $
    ][
    $ gamma^0 bold(theta)^top overline(bold(x))_5 $
    ] #pause

    We will forget meeting her! #pause

    *Question:* Would you forget meeting Taylor Swift?

==
    Our model of memory is incomplete #pause

    Memories are not created equal, some are more important than others #pause

    Important memories persist longer than unimportant memories #pause

    We will pay more *attention* to certain memories #pause
    - And we will forget other memories

==
    What does human memory actually look like?

    #cimage("figures/lecture_11/composite_softmax.png") #pause

    #side-by-side[
    $ 1.0 dot bold(theta)^top overline(bold(x))_1 $ #pause
    ][
    $ 0.1 dot bold(theta)^top overline(bold(x))_2 $ #pause
    ][
    $ 0.1 dot bold(theta)^top overline(bold(x))_3 $ #pause
    ][
    $ 0.5 dot bold(theta)^top overline(bold(x))_4 $ #pause
    ][
    $ 0.1 dot bold(theta)^top overline(bold(x))_5 $
    ] #pause

    *Question:* How can we achieve this forgetting?

==
    In our composite model, forgetting is a function of time #pause

    *Question:* Any forgetting mechanism that is not a function of time? #pause

    *Answer:* Forgetting in recurrent neural network is function of input! #pause

    $ f_"forget" (bold(x), bold(theta)) = sigma(bold(theta)^top_lambda overline(bold(x))) $ #pause

    $ f(bold(h), bold(x), bold(theta)) = f_"forget" (bold(x), bold(theta)) dot.o bold(h) +  bold(theta)_x^top overline(bold(x)) $ 

==
    First, write our forgetting function with slightly different notation #pause
    $
        lambda(bold(x), bold(theta)_lambda) = sigma(bold(theta)^top_lambda overline(bold(x))); quad bold(theta)_lambda in bb(R)^((d_x + 1) times 1)
    $ #pause

    #side-by-side[*Question:* Shape of $lambda(bold(x), bold(theta)_lambda)$? #pause][*Answer:* Scalar! #pause]

    Then, write our composite memory model with forgetting #pause

    $ f(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = sum_(i=1)^T bold(theta)^top bold(x)_i dot lambda(bold(x)_i, bold(theta)_lambda)
    $ #pause

    Only pay attention to important inputs 

==
    We can use this simple form of attention to remember Taylor Swift #pause

    #cimage("figures/lecture_11/composite_swift.png")

== 
    #side-by-side[
    $ lambda(bold(x)_1, bold(theta)_lambda) \ dot bold(theta)^top overline(bold(x))_1 $
    ][
    $ lambda(bold(x)_2, bold(theta)_lambda) \ dot bold(theta)^top overline(bold(x))_2 $
    ][
    $ lambda(bold(x)_3, bold(theta)_lambda) \ dot bold(theta)^top overline(bold(x))_3 $
    ][
    $ lambda(bold(x)_4, bold(theta)_lambda) \ dot bold(theta)^top overline(bold(x))_4 $
    ][
    $ lambda(bold(x)_5, bold(theta)_lambda) \ dot bold(theta)^top overline(bold(x))_5 $
    ] #pause

    *Question:* What do the images look like now? #pause

    #cimage("figures/lecture_11/composite_swift.png")

==
    This form of attention will learn to pay attention to everyone! #pause

    #side-by-side[
    $ 1.0 dot bold(theta)^top overline(bold(x))_1 $
    ][
    $ 1.0 dot bold(theta)^top overline(bold(x))_2 $
    ][
    $ 1.0 dot bold(theta)^top overline(bold(x))_3 $
    ][
    $ 1.0 dot bold(theta)^top overline(bold(x))_4 $
    ][
    $ 1.0 dot bold(theta)^top overline(bold(x))_5 $
    ] #pause
    
    #cimage("figures/lecture_11/composite_swift.png") 

==
    #side-by-side[
    $ 1.0 dot bold(theta)^top overline(bold(x))_1 $
    ][
    $ 1.0 dot bold(theta)^top overline(bold(x))_2 $
    ][
    $ 1.0 dot bold(theta)^top overline(bold(x))_3 $
    ][
    $ 1.0 dot bold(theta)^top overline(bold(x))_4 $
    ][
    $ 1.0 dot bold(theta)^top overline(bold(x))_5 $
    ] #pause

    $ f(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = sum_(i=1)^T bold(theta)^top bold(x)_i dot lambda(bold(x)_i, bold(theta)_lambda)
    $ #pause

    Why does this model of attention pay attention to everyone? #pause
    - $lambda$ is not aware of other party members $bold(x)_j, j!=i$ #pause
        - Can assign all members maximum attention #pause
        - Humans have limited attention span, we should model this

==
    We should normalize $lambda(bold(x), bold(theta)_lambda)$ to model finite (human) attention span #pause

    For example, normalize attention to sum to one

    $ sum_(i=1)^T lambda(bold(x)_i, bold(theta)_lambda) = 1 $ #pause

    Now the model must choose who to remember! #pause
    - Normalization makes $lambda$ a function of all inputs $bold(x)_1 dots bold(x)_t$ #pause
        - Big $lambda(bold(x)_i, bold(theta))$ makes other $lambda(bold(x)_j, bold(theta))$ smaller #pause

    *Question:* How can we ensure that the attention sums to one? #pause

    *Answer:* Softmax!

==
  The softmax function maps real numbers to the simplex (probabilities) #pause

  $ "softmax": bb(R)^k |-> Delta^(k - 1) $ #pause

  $ "softmax"(vec(x_1, dots.v, x_k)) = (exp(bold(x))) / (sum_(i=1)^k exp(x_i)) = vec(
    exp(x_1) / (exp(x_1) + exp(x_2) + dots exp(x_k)),
    exp(x_2) / (exp(x_1) + exp(x_2) + dots exp(x_k)),
    dots.v,
    exp(x_k) / (exp(x_1) + exp(x_2) + dots exp(x_k)),
  ) $

==
    //$ lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda) = exp(bold(theta)^top_lambda overline(bold(x))) / (sum_(j=1)^k exp(bold(theta)^top_lambda overline(bold(x))_j)) $ #pause

    Let us rewrite attention using softmax #pause

    The attention we pay to person $i$ is

    $ lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda)_i 
    = softmax(vec(
        bold(theta)_lambda^top overline(bold(x))_1,
        dots.v,
        bold(theta)_lambda^top overline(bold(x))_T,
    ))_i 
    = exp(bold(theta)^top_lambda overline(bold(x))_i) 
        / (sum_(j=1)^T exp(bold(theta)^top_lambda overline(bold(x))_j)) 
    $ #pause

    Notice $lambda$ now a function of all inputs, $lambda(dot)_i$ is attention for $bold(x)_i$

==
    #cimage("figures/lecture_11/composite_swift.png") #pause
    
    #side-by-side[
    $ lambda(vec(bold(x)_1, dots.v, bold(x)_5), bold(theta)_lambda)_1 \ dot bold(theta)^top overline(bold(x))_1 $ #pause
    ][
    $ lambda(vec(bold(x)_1, dots.v, bold(x)_5), bold(theta)_lambda)_2 \ dot bold(theta)^top overline(bold(x))_2 $ #pause
    ][
    $ lambda(vec(bold(x)_1, dots.v, bold(x)_5), bold(theta)_lambda)_3 \ dot bold(theta)^top overline(bold(x))_3 $ #pause
    ][
    $ lambda(vec(bold(x)_1, dots.v, bold(x)_5), bold(theta)_lambda)_4 \ dot bold(theta)^top overline(bold(x))_4 $ #pause
    ][
    $ lambda(vec(bold(x)_1, dots.v, bold(x)_5), bold(theta)_lambda)_5 \ dot bold(theta)^top overline(bold(x))_5 $
    ] 

==
    #cimage("figures/lecture_11/composite_softmax.png") #pause

    #side-by-side[
    $ 0.70 dot bold(theta)^top overline(bold(x))_1 $ #pause
    ][
    $ 0.04 dot bold(theta)^top overline(bold(x))_2 $ #pause
    ][
    $ 0.03 dot bold(theta)^top overline(bold(x))_3 $ #pause
    ][
    $ 0.20 dot bold(theta)^top overline(bold(x))_4 $ #pause
    ][
    $ 0.03 dot bold(theta)^top overline(bold(x))_5 $
    ] #pause

    $ 0.70 + 0.04 + 0.03 + 0.20 + 0.03 = 1.0 $

==
    $ lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda)_i = exp(bold(theta)^top_lambda overline(bold(x)_i)) / (sum_(j=1)^T exp(bold(theta)^top_lambda overline(bold(x))_j)) $ #pause

    Compute attention for all inputs at once #pause

    $ lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda) = vec(
        exp(bold(theta)^top_lambda overline(bold(x))_#redm[$1$]) / (sum_(j=1)^T exp(bold(theta)^top_lambda overline(bold(x))_j)), 
        dots.v,
        exp(bold(theta)^top_lambda overline(bold(x))_#redm[$T$]) / (sum_(j=1)^T exp(bold(theta)^top_lambda overline(bold(x))_j))
    ) $

==
    This is a simple form of attention #pause
    - Look at all party guests, choose who to remember and forget #pause

    Next, we will investigate the attention used in transformers

= Keys and Queries
==
    The modern form of attention behaves like a database #pause

    We label each person at the party with a *key* #pause

    The key describes the content of each $bold(x)$ #pause

    #cimage("figures/lecture_11/composite_swift_einstein.svg") #pause

    #side-by-side[Musician #pause][Lawyer #pause][Shopkeeper #pause][Chef #pause][Scientist ]

==
    We can search through our keys using a *query* #pause

    *Query:* Which person will help me on my exam? #pause

    #only((3,4))[
        #cimage("figures/lecture_11/composite_swift_einstein.svg")
    ] #pause

    #side-by-side[Musician][Lawyer][Shopkeeper][Chef][Scientist] #pause

    #only(5)[#cimage("figures/lecture_11/composite_swift_einstein_attn_einstein.png")]

==
    *Query:* I want to have fun #pause

    #only((2,3))[
        #cimage("figures/lecture_11/composite_swift_einstein.svg")
    ] #pause

    #side-by-side[Musician][Lawyer][Shopkeeper][Chef][Scientist] #pause

    #only((4,5))[#cimage("figures/lecture_11/composite_swift_einstein_attn_chef.png")] #pause

    How do we represent keys and queries mathematically?

==
    For each input, we create a key $bold(k)$ #pause

    $ bold(k)_j = bold(theta)^top_K bold(x)_j, quad bold(theta)_K in bb(R)^(d_x times d_h), quad bold(k)_j in bb(R)^(d_h) $ #pause

    The key $bold(k)_j$ is a latent description of $bold(x)_j$

==
    Now, create a query from some $bold(x)_q$ #pause

    $ bold(q) = bold(theta)^top_Q bold(x)_q,  quad bold(theta)_Q in bb(R)^(d_x times d_h), quad bold(q) in bb(R)^(d_h) $ #pause

    To determine if a key and query match, we will take the dot product #pause

    $ bold(q)^top bold(k)_i = (bold(theta)_Q^top bold(x)_q)^top (bold(theta)_K^top bold(x)_i) $ #pause

    *Question:* What is the shape of $bold(q)^top bold(k)_i$? #pause

    *Answer:* $(1, d_h) times (d_h, 1) = 1$, the output is a scalar #pause

    Large dot product $=>$ match! Small dot product $=>$ no match.

==
    *Example:* #pause

    #side-by-side[
        $bold(k)_i = bold(theta)_K^top #image("figures/lecture_11/swift.jpg", height: 20%)$ #pause
    ][
        $bold(q) = bold(theta)_Q^top "Musician"$ #pause
    ]

    $ bold(q)^top bold(k)_i = (bold(theta)_Q^top "Musician")^top (bold(theta)_K^top #image("figures/lecture_11/swift.jpg", height: 20%)) = 100 $ #pause

    Large attention!

==
    *Example:*

    #side-by-side[
        $bold(k)_i = bold(theta)_K^top #image("figures/lecture_11/swift.jpg", height: 20%)$ #pause
    ][
        $bold(q) = bold(theta)_Q^top "Mathematician"$ #pause
    ]

    $ bold(q)^top bold(k)_i = (bold(theta)_Q^top "Mathematician")^top (bold(theta)_K^top #image("figures/lecture_11/swift.jpg", height: 20%)) = -50 $ #pause

    Small attention!

==
    *Example:*

    #side-by-side[
        $bold(k)_i = bold(theta)_K^top #image("figures/lecture_11/einstein.jpg", height: 20%)$ #pause
    ][
        $bold(q) = bold(theta)_Q^top "Mathematician"$ #pause
    ]

    $ bold(q)^top bold(k)_i = (bold(theta)_Q^top "Mathematician")^top (bold(theta)_K^top #image("figures/lecture_11/einstein.jpg", height: 20%)) = 90 $ #pause

    Large attention!

==
So far, we only consider single operations $bold(q)^top bold(k)_i$ #pause

But we have many $bold(k)$, can compute attention over all keys at once? #pause

First, let us construct a vector of keys (matrix of scalars) #pause

$ bold(K) 
    = vec(bold(k)_1, bold(k)_2, dots.v, bold(k)_T) 
    = vec(bold(theta)_K^top bold(x)_1, bold(theta)_K^top bold(x)_2, dots.v, bold(theta)_K^top bold(x)_T), 
    quad bold(K) in bb(R)^(T times d_h) 
$ 

==
Consider the shapes of our tensors #pause

$ bold(q) in bb(R)^(d_h times 1); bold(K) in bb(R)^(T times d_h) $ #pause

$ bold(K) bold(q) in bb(R)^(T times 1) $ #pause

$ bold(K) bold(q) = vec(bold(k)_1, dots.v, bold(k)_T) bold(q) #pause = vec(bold(k)_1^top bold(q), dots.v, bold(k)^top_T bold(q)) #pause = vec(bold(q)^top bold(k)_1, dots.v, bold(q)^top bold(k)_T) $ #pause

Can also write transpose (output row instead of column) #pause

$ bold(q)^top bold(K)^top = bold(q)^top mat(bold(k)_1, dots, bold(k)_T) = mat(bold(q)^top bold(k)_1, dots, bold(q)^top bold(k)_T) $

==
$ bold(K) bold(q) = vec(bold(k)_1, dots.v, bold(k)_T) bold(q) = vec(bold(k)_1^top bold(q), dots.v, bold(k)^top_T bold(q)) = vec(bold(q)^top bold(k)_1, dots.v, bold(q)^top bold(k)_T) $ #pause

$ bold(q)^top bold(K)^top = bold(q)^top mat(bold(k)_1, dots, bold(k)_T) = mat(bold(q)^top bold(k)_1, dots, bold(q)^top bold(k)_T) $ #pause

*Question:* Are we missing anything? #pause

$ softmax(bold(K) bold(q)) = softmax(vec(bold(k)_1^top bold(q), dots.v, bold(k)^top_T bold(q))) $ #pause

$ softmax(bold(q)^top bold(K)^top) = softmax(mat(bold(q)^top bold(k)_1, dots, bold(q)^top bold(k)_T)) $

==
$ softmax(bold(K) bold(q)) = softmax(vec(bold(k)_1^top bold(q), dots.v, bold(k)^top_T bold(q))) $

$ softmax(bold(q)^top bold(K)^top) = softmax(mat(bold(q)^top bold(k)_1, dots, bold(q)^top bold(k)_T)) $

    We call this *dot-product attention*

==
    *Query:* Which person will help me on my exam? #pause

    #only((2,3,4))[
        #cimage("figures/lecture_11/composite_swift_einstein.svg", width: 70%)
    ] 

    #only(5)[#cimage("figures/lecture_11/composite_swift_einstein_attn_einstein.png", width: 70%)]

    #side-by-side[][
        $bold(q)^top bold(k)_1$
    ][
        $bold(q)^top bold(k)_2$
    ][
        $bold(q)^top bold(k)_3$
    ][
        $bold(q)^top bold(k)_4$
    ][
        $bold(q)^top bold(k)_5$
    ][] #pause

    #side-by-side[][
        $-1.71$
    ][
        $0.60$
    ][
        $-1.01$
    ][
        $-0.61$
    ][
        $2.73$
    ][] 

    $ softmax $ #pause

    #side-by-side[][
        $0.01$
    ][
        $0.10$
    ][
        $0.02$
    ][
        $0.03$
    ][
        $0.84$
    ][] 

==
    Put dot product attention into our composite model #pause

    $
        lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda)_i = softmax(bold(q)^top bold(K)^top)_i = 
        softmax(bold(q)^top mat(bold(k)_1, bold(k)_2, dots, bold(k)_T))_i 
    
    $ #pause

    Then, write our composite memory model with dot product attention #pause

    $ f(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = sum_(i=1)^T bold(theta)^top bold(x)_i dot lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda)_i
    $ 


==
    $ f(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = sum_(i=1)^T bold(theta)^top bold(x)_i dot lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda)_i
    $ #pause

    We will relabel $bold(theta)$ to $bold(theta)_V$ #pause

    $ f(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = sum_(i=1)^T bold(theta)_#redm[$V$]^top bold(x)_i dot lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda)_i $ #pause

    In dot-product attention, we call $bold(theta)_V^top bold(x)_i$ the *value*



= Self Attention
==

    Previously, we chose our own query #pause $bold(x)_q = "Musician"$ #pause

    *Self attention* creates create queries from the inputs #pause

    $ bold(Q) = vec(bold(q)_1, bold(q)_2, dots.v, bold(q)_T) = vec(bold(theta)_Q^top bold(x)_1, bold(theta)_Q^top bold(x)_2, dots.v, bold(theta)_Q^top bold(x)_T), quad bold(Q) in bb(R)^(T times d_h) $ #pause

    Call it self-attention because keys and queries from the same inputs #pause

    Let us see how to combine $bold(Q)$ and $bold(K)$

==

#side-by-side[
    $ bold(K) &= vec(bold(k)_1, dots.v, bold(k)_T) &&= vec(bold(theta)_K^top bold(x)_1, dots.v, bold(theta)_K^top bold(x)_T) $ #pause

][
$ bold(Q) &= vec(bold(q)_1, dots.v, bold(q)_T) &&= vec(bold(theta)_Q^top bold(x)_1, dots.v, bold(theta)_Q^top bold(x)_T) $ #pause
]

$ bold(q)^top bold(k) => bold(q)^top bold(K)^top => dots $ #pause

$ bold(Q) bold(K)^top = vec(bold(q)_1, dots.v, bold(q)_T) mat(bold(k)_1, dots, bold(k)_T) #pause = mat(
bold(q)_1 bold(k)_1, dots, bold(q)_1 bold(k)_T;
bold(q)_2 bold(k)_1, dots, bold(q)_2 bold(k)_T;
dots.v, dots.down, dots.v;
bold(q)_T bold(k)_1,  dots, bold(q)_T bold(k)_T;
) $ #pause

By stacking $bold(q)$ row-wise, we do not need a transpose

==

$ underbrace(bold(Q),bb(R)^(T times d_h))  
  overbrace(bold(K)^top, bb(R)^(d_h times T)) = vec(bold(q)_1, dots.v, bold(q)_T) mat(bold(k)_1, dots, bold(k)_T) = underbrace(mat(
bold(q)_1 bold(k)_1, dots, bold(q)_1 bold(k)_T;
bold(q)_2 bold(k)_1, dots, bold(q)_2 bold(k)_T;
dots.v, dots.down, dots.v;
bold(q)_T bold(k)_1,  dots, bold(q)_T bold(k)_T;
), bb(R)^(d_h times d_h)) $ #pause

Do not forget softmax! #pause

$ softmax(bold(Q) bold(K)^top) = softmax(mat(
bold(q)_1 bold(k)_1, dots, bold(q)_1 bold(k)_T;
bold(q)_2 bold(k)_1, dots, bold(q)_2 bold(k)_T;
dots.v, dots.down, dots.v;
bold(q)_T bold(k)_1,  dots, bold(q)_T bold(k)_T;
)) $

==

$ softmax(bold(Q) bold(K)^top) = softmax(mat(
bold(q)_1 bold(k)_1, dots, bold(q)_1 bold(k)_T;
bold(q)_2 bold(k)_1, dots, bold(q)_2 bold(k)_T;
dots.v, dots.down, dots.v;
bold(q)_T bold(k)_1,  dots, bold(q)_T bold(k)_T;
)) $

$ lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda)_i = softmax(bold(Q) bold(K)^top)_i $ #pause

$ f(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = sum_(i=1)^T bold(theta)_V^top bold(x)_i dot lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda)_i $ 

==
$ f(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = sum_(i=1)^T bold(theta)_V^top bold(x)_i dot lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda)_i $ #pause

This is ugly and confusing #pause

Try and delete the sum and indexes using a *value* matrix #pause

$ bold(V) &= vec(bold(v)_1, dots.v, bold(v)_T) &&= vec(bold(theta)_V^top bold(x)_1, dots.v, bold(theta)_V^top bold(x)_T) $ 

==
$ bold(V) &= vec(bold(v)_1, dots.v, bold(v)_T) &&= vec(bold(theta)_V^top bold(x)_1, dots.v, bold(theta)_V^top bold(x)_T) $ #pause

$
f(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) #pause = sum_(i=1)^T bold(theta)_V^top bold(x)_i dot lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda)_i #pause = 
sum_(i=1)^T bold(v)_i dot softmax(bold(Q) bold(K)^top)_i #pause \
= softmax(mat(
bold(q)_1 bold(k)_1, dots, bold(q)_1 bold(k)_T;
bold(q)_2 bold(k)_1, dots, bold(q)_2 bold(k)_T;
dots.v, dots.down, dots.v;
bold(q)_T bold(k)_1, dots, bold(q)_T bold(k)_T;
)) vec(bold(v)_1, dots.v, bold(v)_T) #pause = softmax(bold(Q) bold(K)^top) bold(V)
$ 

==
We can forget all the sums, and write self attention in matrix form #pause

$ 
f(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = softmax(bold(Q) bold(K)^top) bold(V)
$ #pause

To get the result for the $i$ element, just index the result #pause

$ 
f(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta))_i = [softmax(bold(Q) bold(K)^top) bold(V)]_i
$ 

==

$ softmax(bold(Q) bold(K)^top) bold(V) = softmax(mat(
    bold(q)_1 bold(k)_1, dots, bold(q)_1 bold(k)_T;
    dots.v, dots.down, dots.v;
    bold(q)_T bold(k)_1,  dots, bold(q)_T bold(k)_T;
    ))  vec(bold(v)_1, dots.v, bold(v)_T) $  #pause

Be very careful with softmax dimension #pause
- Defined over vectors, not matrices #pause
- Writing attention different ways uses different softmax dimensions #pause

*Question:* Softmax rows or columns? #pause *Answer:* Rows #pause

#side-by-side[
    Many keys, one query
  ][
    $ softmax(mat(bold(q)_i bold(k)_1, dots, bold(q)_i bold(k)_T)) vec(bold(v)_1, dots.v, bold(v)_T) $ 
  ]  

==
    Attention paper suggests normalizing constant for faster learning #pause

    $ f(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = softmax( (bold(Q) bold(K)^top) / sqrt(d_h)) bold(V) $ #pause

    This operation powers today's biggest models #pause
    - Two matrix multiplies and softmax

==
    ```python

    class Attention(nn.Module):
        def __init__(self):
            self.theta_K = nn.Linear(d_x, d_h, bias=False)
            self.theta_Q = nn.Linear(d_x, d_h, bias=False)
            self.theta_V = nn.Linear(d_x, d_h, bias=False)

        def forward(self, x):
            # Shape:  (column, row), be very careful with axis!
            A = softmax(
                self.theta_Q(x) @ self.theta_K(x).T / d_h, axis=1
            )
            V = self.theta_V(x)
            return A @ V 
    ```

= Short Intro to Graph Neural Networks
==
    #side-by-side[
        #cimage("figures/lecture_11/graph.png") #pause
    ][
        A *node* is a vector of information #pause

        An *edge* connects two nodes  #pause
    ] 
    
    If we connect nodes $i$ and $j$ with edge $(i, j)$, then $i$ and $j$ are *neighbors* #pause

    The *neighborhood* $bold(N)(i)$ contains all neighbors of node $i$

==
    Let us think of graphs as *signals* #pause

    *Question:* Where did we see signals before? #pause

    *Answer:* Convolution #pause

    Rather than time $t$ or space $u, v$, graphs are a function of neighborhood #pause

    $ "Node" i quad bold(x)(i) in bb(R)^(d_x); quad i in 1, dots, T $ #pause

    $ "Neighborhood of" i quad bold(N)(i) = vec(i, j, k, dots.v); quad bold(N)(i) in cal(P)(i); quad i in 1, dots, T $

==
    Consider a *graph convolution layer* #pause

    For a node $i$, the graph convolution layer is: #pause

    $ f(bold(x), bold(N), bold(theta))(i) = sigma(sum_(j in bold(N)(i)) bold(theta)^top overline(bold(x))(j)) $ #pause

    Combine information from the neighbors of $bold(x)(i)$ #pause

    This is just one node, we use this graph layer for all nodes in the graph

==
    Apply graph convolution over all nodes in the graph #pause

    $ f(bold(x), bold(N), bold(theta)) = vec(
        f(bold(x), bold(N), bold(theta))(1),
        f(bold(x), bold(N), bold(theta))(2),
        dots.v,
        f(bold(x), bold(N), bold(theta))(T),
    ) = vec(
        sigma(sum_(j in bold(N)(1)) bold(theta)^top overline(bold(x))(j)),
        sigma(sum_(j in bold(N)(2)) bold(theta)^top overline(bold(x))(j)),
        dots.v,
        sigma(sum_(j in bold(N)(T)) bold(theta)^top overline(bold(x))(j))
    )
    $ #pause

    How does this compare to regular convolution (images, sound, etc)?

==
    #side-by-side[
    Standard 1D convolution 

    $ vec(
        sigma(sum_(j=1)^(k) bold(theta)^top overline(bold(x))(j)),
        sigma(sum_(j=2)^(k+1) bold(theta)^top overline(bold(x))(j)),
        dots.v,
        sigma(sum_(j=T-k)^(T) bold(theta)^top overline(bold(x))(j)),
    )
    $ #pause

    ][
    Graph convolution

    $ vec(
        sigma(sum_(j in bold(N)(1)) bold(theta)^top overline(bold(x))(j)),
        sigma(sum_(j in bold(N)(2)) bold(theta)^top overline(bold(x))(j)),
        dots.v,
        sigma(sum_(j in bold(N)(T)) bold(theta)^top overline(bold(x))(j))
    )
    $ #pause
    ]

    *Question:* What is the output size of standard convolution? #pause

    *Answer:* $(T - k - 1) times d_h$

==
    #side-by-side[
    Standard 1D convolution 

    $ vec(
        sigma(sum_(j=1)^(k) bold(theta)^top overline(bold(x))(j)),
        sigma(sum_(j=2)^(k+1) bold(theta)^top overline(bold(x))(j)),
        dots.v,
        sigma(sum_(j=T-k)^(T) bold(theta)^top overline(bold(x))(j)),
    )
    $ 

    ][
    Graph convolution

    $ vec(
        sigma(sum_(j in bold(N)(1)) bold(theta)^top overline(bold(x))(j)),
        sigma(sum_(j in bold(N)(2)) bold(theta)^top overline(bold(x))(j)),
        dots.v,
        sigma(sum_(j in bold(N)(T)) bold(theta)^top overline(bold(x))(j))
    )
    $ #pause
    ]

    *Question:* What is the output size of graph convolution? #pause

    *Answer:* $T times d_h$

==
    We can use pooling with graph convolutions too

    $ "SumPool"(vec(
        sigma(sum_(j in bold(N)(1)) bold(theta)^top overline(bold(x))(j)),
        sigma(sum_(j in bold(N)(2)) bold(theta)^top overline(bold(x))(j)),
        dots.v,
        sigma(sum_(j in bold(N)(T)) bold(theta)^top overline(bold(x))(j))
    )) = \ 
    sigma(sum_(j in bold(N)(1)) bold(theta)^top overline(bold(x))(j)) +  sigma(sum_(j in bold(N)(2)) bold(theta)^top overline(bold(x))(j)) + dots + sigma(sum_(j in bold(N)(T)) bold(theta)^top overline(bold(x))(j))
    $ 

==
We can write attention and transformers as a graph neural network #pause
    - Make the neighborhood the entire graph $bold(N)(i) = bold(X)$ #pause
    - Learnable continuous edge weights (key $times$ query) #pause

#side-by-side(columns: (0.4fr, 0.6fr))[
    Edge weights
][
    $ bold(A)(i) = softmax(((bold(theta)_Q^top bold(x)_i) (bold(theta)_K^top bold(X))) / sqrt(d_z)) $
] #pause

#side-by-side(columns: (0.4fr, 0.6fr))[
    Graph convolution
][
    $ f(bold(X), i) = sum_(j in bold(N)(i)) bold(theta)^top_V bold(x)_j dot bold(A)(i)_j $
] #pause

Many ways to teach attention!

= Guest Lecture - Dr. Matteo Bettini
==
#side-by-side(columns: (0.4fr, 0.7fr))[
   #cimage("figures/lecture_11/matteo.jpg") 
][
    Dr. Matteo Bettini is a good friend of mine #pause

    We did our PhDs together
]
==
Dr. Matteo Bettini 
- Incoming Research Scientist at Meta's SuperIntelligence Lab #pause
- Focus on agentic LLMs #pause
- Previously worked at PyTorch #pause
- PhD in Computer Science at Cambridge #pause
- MS in Computer Science at Cambridge #pause
- BS in Computer Engineering at Milan Polytechnic