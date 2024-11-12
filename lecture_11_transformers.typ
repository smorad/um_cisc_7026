#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/cetz:0.2.2": canvas, draw, plot
#import "common.typ": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm
#import "@preview/fletcher:0.5.2" as fletcher: diagram, node, edge

#set math.vec(delim: "[")
#set math.mat(delim: "[")

#let varinf = diagram(
  node-stroke: .1em,
  spacing: 4em,
  node((0,0), $bold(z)$, radius: 2em),
  edge($P(bold(x) | bold(z); bold(theta))$, "-|>"),
  node((2,0), $bold(x)$, radius: 2em),
  edge((0,0), (2,0), $P(bold(z) | bold(x); bold(theta))$, "<|-", bend: -40deg),
)

#let forgetting = { 
    set text(size: 22pt)
    canvas(length: 1cm, {
  plot.plot(size: (8, 6),
    x-tick-step: 10,
    y-tick-step: 0.5,
    y-min: 0,
    y-max: 1,
    x-label: [Time],
    y-label: [],
    {
      plot.add(
        domain: (0, 40), 
        label: [Memory Strength],
        style: (stroke: (thickness: 5pt, paint: red)),
        t => calc.pow(0.9, t)
      )
    })
})}


#let ag = (
  [GNN Review],
  [VAE Review and Coding],
  [Attention],
  [Positional Encoding],
  [Transformers],
  [Coding]
)

#show: university-theme.with(
  aspect-ratio: "16-9",
  short-title: "CISC 7026: Introduction to Deep Learning",
  short-author: "Steven Morad",
  short-date: "Lecture 10: Attention"
)

#title-slide(
  title: [Attention and Transformers],
  subtitle: "CISC 7026: Introduction to Deep Learning",
  institution-name: "University of Macau",
)

#aslide(ag, none)
#aslide(ag, 0)

#sslide[
    Graph is a structure of nodes (vertices) and edges

    #side-by-side[
        $ G = (bold(X), bold(E)) $
    ][
        $ bold(X) in bb(R)^(T times d_x) $
    ][
        $ bold(E) in cal(P)(bb(Z)_T times bb(Z)_T) $
    ] #pause

    #side-by-side[
        #cimage("figures/lecture_11/graph.png") #pause
    ][
        A *node* is a vector of information #pause

        An *edge* connects two nodes 
    ]

    //#A *graph neural network* (GNN) is a model for learning on graph data 
]

#sslide[
    #side-by-side[
        #cimage("figures/lecture_11/graph.png")
    ][
        A *node* is a vector of information 

        An *edge* connects two nodes 

    ]
    If we connect nodes $i$ and $j$ with edge $(i, j)$, then $i$ and $j$ are *neighbors* #pause

    The *neighborhood* $N(j)$ contains all neighbors of node $j$
]

#sslide[
    Prof. Li introduced the *graph convolution layer* #pause


    For a node $j$, the graph convolution layer is: #pause

    $ f(bold(X), bold(E), bold(theta))_j = sigma(bold(theta)_1^top overline(bold(x))_j + sum_(i in N(j)) bold(theta)_2^top bold(x)_i) $ #pause

    Combine information from current node $bold(x)_j$ with neighbors $bold(x)_i$ #pause

    This is just one node, we use this graph layer for all nodes in the graph
]

#sslide[
    Graph convolution over all nodes in the graph #pause

    $ f(bold(X), bold(E), bold(theta)) = vec(
        f(bold(X), bold(E), bold(theta))_1,
        f(bold(X), bold(E), bold(theta))_2,
        dots.v,
        f(bold(X), bold(E), bold(theta))_T,
    ) = vec(
        sigma(bold(theta)_1^top overline(bold(x))_1 + sum_(i in N(1)) bold(theta)_2^top bold(x)_i),
        sigma(bold(theta)_1^top overline(bold(x))_2 + sum_(i in N(2)) bold(theta)_2^top bold(x)_i),
        dots.v,
        sigma(bold(theta)_1^top overline(bold(x))_T + sum_(i in N(T)) bold(theta)_2^top bold(x)_i),
    )
    $ #pause

    How does this compare to regular convolution (images, sound, etc)?
]

#sslide[
    #side-by-side[
    Standard convolution 

    $ vec(
        sigma(bold(theta)_1^top overline(bold(x))_1 + sum_(i=1)^(k) bold(theta)_2^top bold(x)_i),
        sigma(bold(theta)_1^top overline(bold(x))_2 + sum_(i=2)^(k+1) bold(theta)_2^top bold(x)_i),
        dots.v,
        sigma(bold(theta)_1^top overline(bold(x))_T + sum_(i=T-k)^(T) bold(theta)_2^top bold(x)_i),
    )
    $ #pause

    ][
    Graph convolution

    $ vec(
        sigma(bold(theta)_1^top overline(bold(x))_1 + sum_(i in N(1)) bold(theta)_2^top bold(x)_i),
        sigma(bold(theta)_1^top overline(bold(x))_2 + sum_(i in N(2)) bold(theta)_2^top bold(x)_i),
        dots.v,
        sigma(bold(theta)_1^top overline(bold(x))_T + sum_(i in N(T)) bold(theta)_2^top bold(x)_i),
    )
    $ #pause
    ]

    *Question:* What is the output size of standard convolution? #pause

    *Answer:* $T - k - 1 times d_h$
]



#sslide[
    #side-by-side[
    Standard convolution 

    $ vec(
        sigma(bold(theta)_1^top overline(bold(x))_1 + sum_(i=1)^(k) bold(theta)_2^top bold(x)_i),
        sigma(bold(theta)_1^top overline(bold(x))_2 + sum_(i=2)^(k+1) bold(theta)_2^top bold(x)_i),
        dots.v,
        sigma(bold(theta)_1^top overline(bold(x))_T + sum_(i=T-k)^(T) bold(theta)_2^top bold(x)_i),
    )
    $ 

    ][
    Graph convolution

    $ vec(
        sigma(bold(theta)_1^top overline(bold(x))_1 + sum_(i in N(1)) bold(theta)_2^top bold(x)_i),
        sigma(bold(theta)_1^top overline(bold(x))_2 + sum_(i in N(2)) bold(theta)_2^top bold(x)_i),
        dots.v,
        sigma(bold(theta)_1^top overline(bold(x))_T + sum_(i in N(T)) bold(theta)_2^top bold(x)_i),
    )
    $ 
    ]

    *Question:* What is the output size of graph convolution? #pause

    *Answer:* $T times d_h$
]

#sslide[
    We can use pooling with graph convolutions too

    $ "SumPool"( vec(
        sigma(bold(theta)_1^top overline(bold(x))_1 + sum_(i in N(1)) bold(theta)_2^top bold(x)_i),
        sigma(bold(theta)_1^top overline(bold(x))_2 + sum_(i in N(2)) bold(theta)_2^top bold(x)_i),
        dots.v,
        sigma(bold(theta)_1^top overline(bold(x))_T + sum_(i in N(T)) bold(theta)_2^top bold(x)_i),
    )) = \ 
    sigma(bold(theta)_1^top overline(bold(x))_1 + sum_(i in N(1)) bold(theta)_2^top bold(x)_i) + sigma(bold(theta)_1^top overline(bold(x))_2 + sum_(i in N(2)) bold(theta)_2^top bold(x)_i) 
    + dots
    $ 
]
#sslide[
    Standard convolution has an ordering, graph convolution is *permutation invariant* #pause

    $ "SumPool"(f(bold(X), bold(E), bold(theta))) &= "SumPool"(vec(
        f(bold(X), bold(E), bold(theta))_1,
        f(bold(X), bold(E), bold(theta))_2,
        dots.v,
        f(bold(X), bold(E), bold(theta))_T,
    )) \ &= "SumPool"(vec(
        f(bold(X), bold(E), bold(theta))_2,
        f(bold(X), bold(E), bold(theta))_T,
        dots.v,
        f(bold(X), bold(E), bold(theta))_1,
    ))
    $ #pause
]

#aslide(ag, 0)
#aslide(ag, 1)

#sslide[
  Autoencoders are useful for compression and denoising #pause

  But we can also use them as *generative models* #pause

  A generative model learns the structure of data #pause

  Using this structure, it generates *new* data #pause

  - Train on face dataset, generate *new* pictures #pause
  - Train on book dataset, write a *new* book #pause
  - Train on protein dataset, create *new* proteins #pause

  How does this work?
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
  Autoencoder generative model: #pause
  
  Encode $ vec(bold(x)_[1], dots.v, bold(x)_[n])$ into $vec(bold(z)_[1], dots.v, bold(z)_[n]) $ #pause

  Pick a point $bold(z)_[k]$ #pause

  Add some noise $bold(z)_"new" = bold(z)_[k] + bold(epsilon)$ #pause

  Decode $bold(z)_"new"$ into $bold(x)_"new"$
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
  Variational autoencoders (VAEs) do three things: #pause
  + Make it easy to sample random $bold(z)$ #pause
  + Keep all $bold(z)_[1], dots bold(z)_[n]$ close together in a small region #pause
  + Ensure that $bold(z) + bold(epsilon)$ is always meaningful #pause

  How? #pause

  Make $bold(z)_[1], dots, bold(z)_[n]$ normally distributed #pause

  $ bold(z) tilde cal(N)(mu, sigma), quad mu = 0, sigma = 1 $
]

//#sslide[
  //#cimage("figures/lecture_2/normal_dist.png")
//  #align(center, normal)
//]

#sslide[
  If $bold(z)_[1], dots, bold(z)_[n]$ are distributed following $cal(N)(0, 1)$: #pause

  + 99.7% of $bold(z)_[1], dots, bold(z)_[n]$ lie within $3 sigma = [-3, 3]$ #pause

  + Make it easy to generate new $bold(z)$, just sample $bold(z) tilde cal(N)(0, 1)$
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
  *Key idea 2:* There is some latent variable $bold(z)$ which generates data $bold(x)$ #pause

  $ x: #cimage("figures/lecture_9/vae_slider.png") $

  $ z: mat("woman", "brown hair", ("frown" | "smile")) $
]

#sslide[
  #cimage("figures/lecture_9/vae_slider.png") #pause

  Network can only see $bold(x)$, it cannot directly observe $bold(z)$ #pause

  Given $bold(x)$, find the probability that the person is smiling $P(bold(z) | bold(x); bold(theta))$
]

#sslide[
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
]

#sslide[
  How do we implement $f$ (i.e., $P(bold(z) | bold(x); bold(theta))$ )? #pause

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
  distribution = (mu, exp(sigma))
  ```
]

#sslide[
  We covered the encoder

  $ f: X times Theta |-> Delta Z $
  
  We can use the same decoder as a standard autoencoder #pause

  $ f^(-1): Z times Theta |-> X $

  Encoder outputs a distribution $Delta Z$ but decoder input is $Z$ #pause

  *Solution:* Sample a vector $bold(z)$ from the distribution $Delta Z$ 
]

#sslide[
  Put it all together #pause

  *Step 1:* Encode the input to a normal distribution

  $ bold(mu), bold(sigma) = f(bold(x), bold(theta)_e) $ #pause

  *Step 2:* Generate a sample from distribution

  $ bold(z) = bold(mu) + bold(sigma) dot.circle bold(epsilon) $ #pause

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
  def L(model, x, m, n, key):
    mu, sigma = model.f(x) # Encode input into distribution 
    # Sample from distribution
    z = mu + sigma * jax.random.normal(key, x.shape[0])
    # Reconstruct input
    pred_x = model.f_inverse(z) 
    # Compute reconstruction and kl loss terms
    recon = jnp.sum((x - pred_x) ** 2)
    kl = jnp.sum(mu ** 2 + sigma ** 2 - jnp.log(sigma ** 2) - 1)
    # Loss function contains reconstruction and kl terms
    return m / n * recon + kl
  ```
]

#sslide[
  https://colab.research.google.com/drive/1UyR_W6NDIujaJXYlHZh6O3NfaCAMscpH#scrollTo=nmyQ8aE2pSbb
]

#aslide(ag, 1)
#aslide(ag, 2)

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


#sslide[
    Today, we will investigate attention and transformers #pause

    Attention and transformers are the "hottest" topic in deep learning #pause

    People use them for almost every task (even if they shouldn't!) #pause

    Let's review some products based on attention
]

#sslide[
    AlphaFold (Nobel prize)

    #cimage("figures/lecture_11/alphafold.png", height: 85%)
]

#sslide[
    ChatGPT, Qwen, LLaMA, Mistral, Doubou, Ernie chatbots

    #cimage("figures/lecture_11/llm.jpg", height: 85%)
]

#sslide[
    MusicTransformer, MuLan

    #cimage("figures/lecture_11/mulan.jpg", height: 85%)
]

#sslide[
    Google Translate, Baidu Translate, Apple Translate

    #cimage("figures/lecture_11/translate.jpeg", height: 85%)
]

#sslide[
    ViT, DinoV2

    #cimage("figures/lecture_9/dino.png", height: 85%)
]

#sslide[
    All these models are *transformers* #pause

    At the core of each transformer is *attention* #pause

    We can derive attention from composite memory
]

#sslide[
    #side-by-side[
        Francis Galton (1822-1911) \ photo composite memory

        #cimage("figures/lecture_8/galton.jpg", height: 70%) #pause
    ][
        Composite photo of members of a party 

        #cimage("figures/lecture_8/composite_memory.jpg", height: 70%)
    ]
]

#sslide[
    *Task:* Find a mathematical model of how our mind represents memories #pause

    $ X: bb(R)^(h times w) quad "People you see at the party" $ #pause

    $ H: bb(R)^(h times w) quad "The image in your mind" $ #pause

    $ f: X^T times Theta |-> H $ #pause

    Composite photography/memory uses a weighted sum #pause

    $ f(bold(x), bold(theta)) = sum_(i=1)^T bold(theta)^top overline(bold(x))_i $
]

#sslide[
    Limited space, cannot fit everything #pause

    Introduced forgetting

    #side-by-side[$ sum_(i=1)^n gamma^(n - i) bold(theta)^top bold(x)_i $ #pause][
        #align(center)[#forgetting]
    ] #pause

    *Question:* Does this accurately model what you remember?
]

#sslide[
    You go to a party and meet these people in order

    #cimage("figures/lecture_11/composite0.svg")
]

#sslide[
    According to forgetting, the memories should fade with time

    #cimage("figures/lecture_11/composite_fade.png")

    #side-by-side[
    $ gamma^3 bold(theta)^top overline(bold(x))_1 $ #pause
    ][
    $ gamma^2 bold(theta)^top overline(bold(x))_2 $ #pause
    ][
    $ gamma^1 bold(theta)^top overline(bold(x))_3 $ #pause
    ][
    $ gamma^0 bold(theta)^top overline(bold(x))_4 $
    ]
]

#sslide[
    Consider the following case #pause

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
]

#sslide[
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

    Would you forget Taylor Swift at a party?
]

#sslide[
    Our model of memory is incomplete #pause

    Memories are not created equal, some are more important than others
]

#slide[
    My memory might actually be

    #cimage("figures/lecture_11/composite_softmax.png")

    #side-by-side[
    $ 1.0 dot bold(theta)^top overline(bold(x))_1 $
    ][
    $ 0.1 dot bold(theta)^top overline(bold(x))_2 $
    ][
    $ 0.1 dot bold(theta)^top overline(bold(x))_3 $
    ][
    $ 0.5 dot bold(theta)^top overline(bold(x))_4 $
    ][
    $ 0.1 dot bold(theta)^top overline(bold(x))_5 $
    ] #pause

    *Question:* How can we achieve this forgetting?
]

#sslide[
    *Question:* How did we achieve forgetting in our recurrent neural network? #pause

    $ f_"forget" (bold(x), bold(theta)) = sigma(bold(theta)^top_lambda overline(bold(x))) $ #pause

    $ f(bold(h), bold(x), bold(theta)) = f_"forget" (bold(x), bold(theta)) dot.circle bold(h) +  bold(theta)_x^top overline(bold(x)) $ #pause

    Let us do something similar #pause

    However, we will write it slightly differently (without recurrence)
]
#sslide[
    First, write our forgetting function #pause
    $
        lambda(bold(x), bold(theta)_lambda) = sigma(bold(theta)^top_lambda overline(bold(x))); quad bold(theta)_lambda in bb(R)^((d_x + 1) times 1)
    $ #pause

    Then, write our composite memory model with forgetting #pause

    $ f(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = sum_(i=1)^T bold(theta)^top bold(x)_i dot lambda(bold(x)_i, bold(theta)_lambda)
    $ #pause

    This is one form of *attention* #pause

    We only pay attention to specific inputs
]

#sslide[
    We can use this simple form of attention to pay attention to Taylor Swift #pause

    #cimage("figures/lecture_11/composite_swift.png")
]

#sslide[
    
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
]

#sslide[
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
    
    #cimage("figures/lecture_11/composite_swift.png") #pause

    Not a good model of attention!
]

#sslide[
    We should normalize $lambda(bold(x), bold(theta)_lambda)$ to model a finite attention span #pause

    For example, we can make

    $ sum_(i=1)^T lambda(bold(x), bold(theta)_lambda) = 1 $ #pause

    This will limit the total amount of attenion that we have #pause

    *Question:* Do we know of any functions with this property? #pause

    *Answer:* Softmax!
]

#sslide[
  The softmax function maps real numbers to the simplex (probabilities)

  $ "softmax": bb(R)^k |-> Delta^(k - 1) $ #pause

  $ "softmax"(vec(x_1, dots.v, x_k)) = (exp(bold(x))) / (sum_(i=1)^k exp(x_i)) = vec(
    exp(x_1) / (exp(x_1) + exp(x_2) + dots exp(x_k)),
    exp(x_2) / (exp(x_1) + e^(x_2) + dots exp(x_k)),
    dots.v,
    exp(x_k) / (exp(x_1) + exp(x_2) + dots exp(x_k)),
  ) $
]

#sslide[
    //$ lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda) = exp(bold(theta)^top_lambda overline(bold(x))) / (sum_(j=1)^k exp(bold(theta)^top_lambda overline(bold(x))_j)) $ #pause

    Let us rewrite attention using softmax #pause

    The attention we pay to person $i$ is

    $ lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda)_i = exp(bold(theta)^top_lambda overline(bold(x))_i) / (sum_(j=1)^T exp(bold(theta)^top_lambda overline(bold(x))_j)) $
]

#sslide[
    
    #side-by-side[
    $ lambda(vec(bold(x)_1, dots.v, bold(x)_5), bold(theta)_lambda)_1 \ dot bold(theta)^top overline(bold(x))_1 $
    ][
    $ lambda(vec(bold(x)_1, dots.v, bold(x)_5), bold(theta)_lambda)_2 \ dot bold(theta)^top overline(bold(x))_2 $
    ][
    $ lambda(vec(bold(x)_1, dots.v, bold(x)_5), bold(theta)_lambda)_3 \ dot bold(theta)^top overline(bold(x))_3 $
    ][
    $ lambda(vec(bold(x)_1, dots.v, bold(x)_5), bold(theta)_lambda)_4 \ dot bold(theta)^top overline(bold(x))_4 $
    ][
    $ lambda(vec(bold(x)_1, dots.v, bold(x)_5), bold(theta)_lambda)_5 \ dot bold(theta)^top overline(bold(x))_5 $
    ] #pause

    #cimage("figures/lecture_11/composite_softmax.png")
]


#sslide[
    
    #side-by-side[
    $ 0.70 dot bold(theta)^top overline(bold(x))_1 $
    ][
    $ 0.04 dot bold(theta)^top overline(bold(x))_2 $
    ][
    $ 0.03 dot bold(theta)^top overline(bold(x))_3 $
    ][
    $ 0.20 dot bold(theta)^top overline(bold(x))_4 $
    ][
    $ 0.03 dot bold(theta)^top overline(bold(x))_5 $
    ] #pause

    #cimage("figures/lecture_11/composite_softmax.png") #pause

    $ 0.70 + 0.04 + 0.03 + 0.20 + 0.03 = 1.0 $
]

#sslide[
    $ lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda)_i = exp(bold(theta)^top_lambda overline(bold(x))) / (sum_(j=1)^T exp(bold(theta)^top_lambda overline(bold(x))_j)) $

    $ lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda) = vec(
        exp(bold(theta)^top_lambda overline(bold(x))_#redm[$1$]) / (sum_(j=1)^T exp(bold(theta)^top_lambda overline(bold(x))_j)), 
        dots.v,
        exp(bold(theta)^top_lambda overline(bold(x))_#redm[$T$]) / (sum_(j=1)^T exp(bold(theta)^top_lambda overline(bold(x))_j))
    ) $
]

#sslide[
    Keys and Queries
]

#sslide[
    The modern form of attention behaves like a database #pause

    We label each person at the party with a *key* #pause

    The key describes the content of each $bold(x)$ #pause

    #cimage("figures/lecture_11/composite_swift_einstein.svg") #pause

    #side-by-side[Musician #pause][Lawyer #pause][Shopkeeper #pause][Chef #pause][Scientist #pause]
]

#sslide[
]

#sslide[
    Then we compute a *query* that corresponds to the key

    *Query:* Which person will help me on my exam? #pause

    #only(3)[
        #cimage("figures/lecture_11/composite_swift_einstein.svg")
    ] 
    #side-by-side[Musician][Lawyer][Shopkeeper][Chef][Scientist] #pause

    #only(4)[#cimage("figures/lecture_11/composite_swift_einstein_attn_einstein.png")]
]

#sslide[
    *Query:* I want to have fun #pause

    #only(3)[
        #cimage("figures/lecture_11/composite_swift_einstein.svg")
    ] 
    #side-by-side[Musician][Lawyer][Shopkeeper][Chef][Scientist] #pause

    #only(4)[#cimage("figures/lecture_11/composite_swift_einstein_attn_chef.png")] #pause

    How do we represent this mathematically?
]

#sslide[
    For each input, we create a key $bold(k)$ #pause

    $ bold(k)_j = bold(theta)^top_K bold(x)_j, quad bold(k)_j in bb(R)^(d_h) $ #pause

    Do this for all inputs #pause

    $ bold(K) = vec(bold(k)_1, bold(k)_2, dots.v, bold(k)_T) = vec(bold(theta)_K^top bold(x)_1, bold(theta)_K^top bold(x)_2, dots.v, bold(theta)_K^top bold(x)_T), quad bold(K) in bb(R)^(T times d_h) $
]

#sslide[
    Now, create a query from some $bold(x)_q$ #pause

    $ bold(q) = bold(theta)^top_Q bold(x)_q, quad bold(q) in bb(R)^(d_h) $ #pause

    We determine how well a key matches the query with the dot product #pause

    $ bold(q)^top bold(k)_i = (bold(theta)_Q^top bold(x)_q)^top (bold(theta)_K^top bold(x)_i) $ #pause

    *Question:* What is the shape of $bold(q)^top bold(k)_i$? #pause

    *Answer:* $(1, d_h) times (d_h, 1) = 1$, the output is a scalar #pause
]

#sslide[
    *Example:*

    #side-by-side[
        $bold(k)_i = bold(theta)_K^top #image("figures/lecture_11/swift.jpg", height: 20%)$ #pause
    ][
        $bold(q) = bold(theta)_Q^top "Musician"$ #pause
    ]

    $ bold(q)^top bold(k)_i = (bold(theta)_Q^top "Musician")^top (bold(theta)_K^top #image("figures/lecture_11/swift.jpg", height: 20%)) = 5.6 $ #pause

    Large attention!
]

#sslide[
    *Example:*

    #side-by-side[
        $bold(k)_i = bold(theta)_K^top #image("figures/lecture_11/swift.jpg", height: 20%)$ #pause
    ][
        $bold(q) = bold(theta)_Q^top "Mathematician"$ #pause
    ]

    $ bold(q)^top bold(k)_i = (bold(theta)_Q^top "Mathematician")^top (bold(theta)_K^top #image("figures/lecture_11/swift.jpg", height: 20%)) = -4.5 $ #pause

    Small attention!
]

#sslide[
    We compute the similarity between keys and queries using the dot product #pause

    $ bold(q)^top bold(K) = bold(q)^top vec(bold(k)_1, bold(k)_2, dots.v, bold(k)_T) = vec( 
        (bold(theta)_Q^top bold(x)_q)^top (bold(theta)_K^top bold(x)_1),
        (bold(theta)_Q^top bold(x)_q)^top (bold(theta)_K^top bold(x)_2),
        dots.v,
        (bold(theta)_Q^top bold(x)_q)^top (bold(theta)_K^top bold(x)_T),
    )
    $
]

#sslide[
    Another guest shows up to the party

    #cimage("figures/lecture_11/composite_swift_einstein.svg")

    Who do we pay attention to, Taylor Swift or Einstein?

    It depends if you like music or science more
]

#sslide[
    $ lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda) = exp(bold(theta)^top_lambda overline(bold(x))) / (sum_(j=1)^k exp(bold(theta)^top_lambda overline(bold(x))_j)) $
    
    Let us rewrite $f$ as a single function

    $ f_i (vec(bold(x)_1, dots.v, bold(x)_T), bold(x)_i, bold(theta)) = ( exp(bold(theta)^top_lambda overline(bold(x))_i) / (sum_(j=1)^T exp(bold(theta)^top_lambda overline(bold(x))_j))) bold(theta)^top overline(bold(x))_i $

    Parameters notation is a bit messy, let us clarify

    $ f_i (vec(bold(x)_1, dots.v, bold(x)_T), bold(x)_i, bold(theta)) = ( exp(bold(theta)^top_lambda overline(bold(x))_i) / (sum_(j=1)^T exp(bold(theta)^top_lambda overline(bold(x))_j))) bold(theta)^top_V overline(bold(x))_i $
]

#sslide[
    $ f (vec(bold(x)_1, dots.v, bold(x)_T), vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = sum_(i=1)^T ( exp(bold(theta)^top_lambda overline(bold(x))_i) / (sum_(j=1)^T exp(bold(theta)^top_lambda overline(bold(x))_j))) bold(theta)^top overline(bold(x))_i $

]

#sslide[
    TODO: Add query with science/einstein, music/taylor

    TODO: Remove xbar because attention does not use bias?
]
