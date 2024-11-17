#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/cetz:0.2.2": canvas, draw, plot
#import "common.typ": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm
#import "@preview/fletcher:0.5.2" as fletcher: diagram, node, edge

#set math.vec(delim: "[")
#set math.mat(delim: "[")

// TODO: Make conv and graph conv notation consistent

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
  [Keys and Queries],
  [Transformer],
  [Positional Encoding],
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
    Graph is a structure of nodes (vertices) and edges #pause

    #side-by-side[
        #cimage("figures/lecture_11/graph.png") #pause
    ][
        A *node* is a vector of information #pause

        An *edge* connects two nodes 
    ] #pause

    /*
    #side-by-side[
        $ G = (bold(X), bold(E)) $
    ][
        $ bold(X) in bb(R)^(T times d_x) $
    ][
        $ bold(E) in cal(P)(bb(Z)_T times bb(Z)_T) $
    ] 
    */


    //#A *graph neural network* (GNN) is a model for learning on graph data 
]

#sslide[
    #side-by-side[
        #cimage("figures/lecture_11/graph.png")
    ][
        A *node* is a vector of information 

        An *edge* connects two nodes 

    ] #pause
    If we connect nodes $i$ and $j$ with edge $(i, j)$, then $i$ and $j$ are *neighbors* #pause

    The *neighborhood* $bold(N)(i)$ contains all neighbors of node $i$
]

#sslide[
    Let us think of graphs as *signals* #pause

    *Question:* Where did we see signals before? #pause

    Rather than time $t$ or space $u, v$, graphs are a function of index $i$ #pause

    $ "Node" i quad bold(x)(i) in bb(R)^(d_x); quad i in 1, dots, T $ #pause

    $ "Neighborhood of" i quad bold(N)(i) = vec(i, j, k, dots.v); quad bold(N)(i) in cal(P)(i); quad i in 1, dots, T $
]


#sslide[
    Prof. Li introduced the *graph convolution layer* #pause

    For a node $i$, the graph convolution layer is: #pause

    $ f(bold(x), bold(N), bold(theta))(i) = sigma(sum_(j in bold(N)(i)) bold(theta)^top overline(bold(x))(j)) $ #pause

    Combine information from the neighbors of $bold(x)(i)$ #pause

    This is just one node, we use this graph layer for all nodes in the graph
]

#sslide[
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
]

#sslide[
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
]



#sslide[
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
]

#sslide[
    We can use pooling with graph convolutions too

    $ "SumPool"(vec(
        sigma(sum_(j in bold(N)(1)) bold(theta)^top overline(bold(x))(j)),
        sigma(sum_(j in bold(N)(2)) bold(theta)^top overline(bold(x))(j)),
        dots.v,
        sigma(sum_(j in bold(N)(T)) bold(theta)^top overline(bold(x))(j))
    )) = \ 
    sigma(sum_(j in bold(N)(1)) bold(theta)^top overline(bold(x))(j)) +  sigma(sum_(j in bold(N)(2)) bold(theta)^top overline(bold(x))(j)) + dots + sigma(sum_(j in bold(N)(T)) bold(theta)^top overline(bold(x))(j))
    $ 
]
/*
#sslide[
    Standard convolution is *translation equivariant*, graph convolution is *permutation equivariant* #pause

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
*/

#aslide(ag, 0)
#aslide(ag, 1)

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
  ] #pause

  //We want to learn both the encoder and decoder: $P(bold(z), bold(x); bold(theta))$
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
  https://colab.research.google.com/drive/1UyR_W6NDIujaJXYlHZh6O3NfaCAMscpH#scrollTo=nmyQ8aE2pSbb #pause

  https://openai.com/index/glow/
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
    Attention and transformers are the "hottest" topic in deep learning #pause

    People use them for almost every task (even if they shouldn't!) #pause

    Let's review some projects based on attention
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
    Limited space, cannot remember everything #pause

    Introduced forgetting term $gamma in [0, 1]$ #pause

    #side-by-side[$ f(bold(x), bold(theta)) = sum_(i=1)^T gamma^(T - i) dot bold(theta)^top overline(bold(x))_i $ #pause][
        #align(center)[#forgetting]
    ] #pause

    *Question:* Does this accurately model what *you* remember?
]

#sslide[
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
]

#sslide[
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
    ] #pause
]

#sslide[
    Any questions before moving on?
]

#sslide[
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

    *Question:* Would you forget meeting Taylor Swift?
]

#sslide[
    Our model of memory is incomplete #pause

    Memories are not created equal, some are more important than others #pause

    Important memories persist longer than unimportant memories #pause

    We will *pay more attention* to certain memories
]

#slide[
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
]

#sslide[
    In our composite model, forgetting is a function of time #pause

    *Question:* Any forgetting mechanism that is not a function of time? #pause

    *Answer:* Forgetting in recurrent neural network is function of input! #pause

    $ f_"forget" (bold(x), bold(theta)) = sigma(bold(theta)^top_lambda overline(bold(x))) $ #pause

    $ f(bold(h), bold(x), bold(theta)) = f_"forget" (bold(x), bold(theta)) dot.circle bold(h) +  bold(theta)_x^top overline(bold(x)) $ 
]

/*
#sslide[
    *Question:* How did we achieve forgetting in our recurrent neural network? #pause

    $ f_"forget" (bold(x), bold(theta)) = sigma(bold(theta)^top_lambda overline(bold(x))) $ #pause

    $ f(bold(h), bold(x), bold(theta)) = f_"forget" (bold(x), bold(theta)) dot.circle bold(h) +  bold(theta)_x^top overline(bold(x)) $ #pause

    Let us do something similar #pause

    However, we will write it slightly differently (without recurrence)
]
*/
#sslide[
    First, write our forgetting function with slightly different notation #pause
    $
        lambda(bold(x), bold(theta)_lambda) = sigma(bold(theta)^top_lambda overline(bold(x))); quad bold(theta)_lambda in bb(R)^((d_x + 1) times 1)
    $ #pause

    #side-by-side[*Question:* Shape of $lambda(bold(x), bold(theta)_lambda)$? #pause][*Answer:* Scalar! #pause]

    Then, write our composite memory model with forgetting #pause

    $ f(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = sum_(i=1)^T bold(theta)^top bold(x)_i dot lambda(bold(x)_i, bold(theta)_lambda)
    $ #pause

    Only pay attention to important inputs #pause
]

#sslide[
    We can use this simple form of attention to remember Taylor Swift #pause

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
    We should normalize $lambda(bold(x), bold(theta)_lambda)$ to model finite (human) attention span #pause

    For example, normalize attention to sum to one

    $ sum_(i=1)^T lambda(bold(x)_i, bold(theta)_lambda) = 1 $ #pause

    Now the model must choose who to remember! #pause

    *Question:* How can we ensure that the attention sums to one? #pause

    *Answer:* Softmax!
]

#sslide[
  The softmax function maps real numbers to the simplex (probabilities) #pause

  $ "softmax": bb(R)^k |-> Delta^(k - 1) $ #pause

  $ "softmax"(vec(x_1, dots.v, x_k)) = (exp(bold(x))) / (sum_(i=1)^k exp(x_i)) = vec(
    exp(x_1) / (exp(x_1) + exp(x_2) + dots exp(x_k)),
    exp(x_2) / (exp(x_1) + exp(x_2) + dots exp(x_k)),
    dots.v,
    exp(x_k) / (exp(x_1) + exp(x_2) + dots exp(x_k)),
  ) $
]

#sslide[
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
    $
]

#sslide[
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
    ] #pause

]


#sslide[
    
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
]

#sslide[
    $ lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda)_i = exp(bold(theta)^top_lambda overline(bold(x)_i)) / (sum_(j=1)^T exp(bold(theta)^top_lambda overline(bold(x))_j)) $ #pause

    Compute attention for all inputs at once #pause

    $ lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda) = vec(
        exp(bold(theta)^top_lambda overline(bold(x))_#redm[$1$]) / (sum_(j=1)^T exp(bold(theta)^top_lambda overline(bold(x))_j)), 
        dots.v,
        exp(bold(theta)^top_lambda overline(bold(x))_#redm[$T$]) / (sum_(j=1)^T exp(bold(theta)^top_lambda overline(bold(x))_j))
    ) $
]

#sslide[
    This is a simple form of attention #pause

    Next, we will investigate the attention used in transformers
]

#aslide(ag, 2)
#aslide(ag, 3)

#sslide[
    The modern form of attention behaves like a database #pause

    We label each person at the party with a *key* #pause

    The key describes the content of each $bold(x)$ #pause

    #cimage("figures/lecture_11/composite_swift_einstein.svg") #pause

    #side-by-side[Musician #pause][Lawyer #pause][Shopkeeper #pause][Chef #pause][Scientist ]
]

#sslide[
    We can search through our keys using a *query* #pause

    *Query:* Which person will help me on my exam? #pause

    #only((3,4))[
        #cimage("figures/lecture_11/composite_swift_einstein.svg")
    ] #pause

    #side-by-side[Musician][Lawyer][Shopkeeper][Chef][Scientist] #pause

    #only(5)[#cimage("figures/lecture_11/composite_swift_einstein_attn_einstein.png")]
]

#sslide[
    *Query:* I want to have fun #pause

    #only((2,3))[
        #cimage("figures/lecture_11/composite_swift_einstein.svg")
    ] #pause

    #side-by-side[Musician][Lawyer][Shopkeeper][Chef][Scientist] #pause

    #only((4,5))[#cimage("figures/lecture_11/composite_swift_einstein_attn_chef.png")] #pause

    How do we represent keys and queries mathematically?
]

#sslide[
    For each input, we create a key $bold(k)$ #pause

    $ bold(k)_j = bold(theta)^top_K bold(x)_j, quad bold(theta)_K in bb(R)^(d_x times d_h), quad bold(k)_j in bb(R)^(d_h) $ #pause

    Create keys for all inputs #pause

    $ bold(K) = vec(bold(k)_1, bold(k)_2, dots.v, bold(k)_T) = vec(bold(theta)_K^top bold(x)_1, bold(theta)_K^top bold(x)_2, dots.v, bold(theta)_K^top bold(x)_T), quad bold(K) in bb(R)^(T times d_h) $
]

#sslide[
    Now, create a query from some $bold(x)_q$ #pause

    $ bold(q) = bold(theta)^top_Q bold(x)_q,  quad bold(theta)_Q in bb(R)^(d_x times d_h), quad bold(q) in bb(R)^(d_h) $ #pause

    To determine if a key and query match, we will take the dot product #pause

    $ bold(q)^top bold(k)_i = (bold(theta)_Q^top bold(x)_q)^top (bold(theta)_K^top bold(x)_i) $ #pause

    *Question:* What is the shape of $bold(q)^top bold(k)_i$? #pause

    *Answer:* $(1, d_h) times (d_h, 1) = 1$, the output is a scalar #pause

    Large dot product $=>$ match! Small dot product $=>$ no match.
]

#sslide[
    *Example:* #pause

    #side-by-side[
        $bold(k)_i = bold(theta)_K^top #image("figures/lecture_11/swift.jpg", height: 20%)$ #pause
    ][
        $bold(q) = bold(theta)_Q^top "Musician"$ #pause
    ]

    $ bold(q)^top bold(k)_i = (bold(theta)_Q^top "Musician")^top (bold(theta)_K^top #image("figures/lecture_11/swift.jpg", height: 20%)) = 100 $ #pause

    Large attention!
]

#sslide[
    *Example:*

    #side-by-side[
        $bold(k)_i = bold(theta)_K^top #image("figures/lecture_11/swift.jpg", height: 20%)$ #pause
    ][
        $bold(q) = bold(theta)_Q^top "Mathematician"$ #pause
    ]

    $ bold(q)^top bold(k)_i = (bold(theta)_Q^top "Mathematician")^top (bold(theta)_K^top #image("figures/lecture_11/swift.jpg", height: 20%)) = -50 $ #pause

    Small attention!
]

#sslide[
    *Example:*

    #side-by-side[
        $bold(k)_i = bold(theta)_K^top #image("figures/lecture_11/einstein.jpg", height: 20%)$ #pause
    ][
        $bold(q) = bold(theta)_Q^top "Mathematician"$ #pause
    ]

    $ bold(q)^top bold(k)_i = (bold(theta)_Q^top "Mathematician")^top (bold(theta)_K^top #image("figures/lecture_11/einstein.jpg", height: 20%)) = 90 $ #pause

    Large attention! #pause

    Remember, there are multiple inputs to pay attention to
]

#sslide[
    We compute attention for each input #pause

    $ bold(q)^top bold(K) = bold(q)^top vec(bold(k)_1, bold(k)_2, dots.v, bold(k)_T) = vec( 
        (bold(theta)_Q^top bold(x)_q)^top (bold(theta)_K^top bold(x)_1),
        (bold(theta)_Q^top bold(x)_q)^top (bold(theta)_K^top bold(x)_2),
        dots.v,
        (bold(theta)_Q^top bold(x)_q)^top (bold(theta)_K^top bold(x)_T),
    )
    $ #pause

    *Question:* Anything missing from before? #pause

    *Answer:* Normalize attention to sum to one!
]


#sslide[
    Normalize, only pay attention to important matches #pause

    $ softmax(bold(q)^top bold(K)) = softmax(bold(q)^top vec(bold(k)_1, bold(k)_2, dots.v, bold(k)_T)) = softmax(vec( 
        (bold(theta)_Q^top bold(x)_q)^top (bold(theta)_K^top bold(x)_1),
        (bold(theta)_Q^top bold(x)_q)^top (bold(theta)_K^top bold(x)_2),
        dots.v,
        (bold(theta)_Q^top bold(x)_q)^top (bold(theta)_K^top bold(x)_T),
    ))
    $ #pause

    We call this *dot-product attention*
]

#sslide[
    *Query:* Which person will help me on my exam? #pause

    #only((2,3,4,5))[
        #cimage("figures/lecture_11/composite_swift_einstein.svg", width: 70%)
    ] 

    #only(6)[#cimage("figures/lecture_11/composite_swift_einstein_attn_einstein.png", width: 70%)]

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
]

#sslide[
    Put dot product attention into our composite model #pause

    $
        lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda)_i = 
        softmax(bold(q)^top vec(bold(k)_1, bold(k)_2, dots.v, bold(k)_T))_i 
    
    $ #pause

    Then, write our composite memory model with forgetting #pause

    $ f(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = sum_(i=1)^T bold(theta)^top bold(x)_i dot lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda)_i
    $ #pause

    //We call $bold(theta)_V^top bold(x)_i$ the *value*
]

#sslide[
    $ f(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = sum_(i=1)^T bold(theta)^top bold(x)_i dot lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda)_i
    $ #pause

    $ f(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = sum_(i=1)^T bold(theta)_#redm[$V$]^top bold(x)_i dot lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda)_i $ #pause

    In dot-product attention, we call $bold(theta)_V^top bold(x)_i$ the *value*
]


#sslide[
    #cimage("figures/lecture_11/composite_swift_einstein_attn_einstein.png", width: 70%)
        #side-by-side[][
        $bold(q)^top bold(k)_1 \
        dot bold(theta)_V^top bold(x)_1$
    ][
        $bold(q)^top bold(k)_2 \ 
        dot bold(theta)_V^top bold(x)_2$
    ][
        $bold(q)^top bold(k)_3 \ 
        dot bold(theta)_V^top bold(x)_3$
    ][
        $bold(q)^top bold(k)_4 \ 
        dot bold(theta)_V^top bold(x)_4$
    ][
        $bold(q)^top bold(k)_5 \ 
        dot bold(theta)_V^top bold(x)_5$
    ][] #pause
]

#aslide(ag, 3)
#aslide(ag, 4)

#sslide[
    Previously, we chose our own $bold(x)_q = "Musician"$ #pause

    We can also create queries from all the inputs #pause

    $ bold(Q) = vec(bold(q)_1, bold(q)_2, dots.v, bold(q)_T) = vec(bold(theta)_Q^top bold(x)_1, bold(theta)_Q^top bold(x)_2, dots.v, bold(theta)_Q^top bold(x)_T), quad bold(Q) in bb(R)^(T times d_h) $ #pause

    We call this dot-product *self* attention 
]

#sslide[
    Writing dot-product self attention in matrix form is easier #pause

    $
    bold(Q) = vec(bold(q)_1, bold(q)_2, dots.v, bold(q)_T) = vec(bold(theta)_Q^top bold(x)_1, bold(theta)_Q^top bold(x)_2, dots.v, bold(theta)_Q^top bold(x)_T) quad

    bold(K) = vec(bold(k)_1, bold(k)_2, dots.v, bold(k)_T) = vec(bold(theta)_K^top bold(x)_1, bold(theta)_K^top bold(x)_2, dots.v, bold(theta)_K^top bold(x)_T) quad

    bold(V) = vec(bold(v)_1, bold(v)_2, dots.v, bold(v)_T) = vec(bold(theta)_V^top bold(x)_1, bold(theta)_V^top bold(x)_2, dots.v, bold(theta)_V^top bold(x)_T) quad

    $ #pause

    $ "attn"(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = softmax( (bold(Q) bold(K)^top) / sqrt(d_h)) bold(V) $ #pause

    This operation powers today's biggest models
]

#sslide[
    $ bold(Q) in bb(R)^(T times d_h) quad bold(K) in bb(R)^(T times d_h) quad bold(V) in bb(R)^(T times d_h) $

    $ underbrace("attn"(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)), bb(R)^(T times d_h)) = softmax overbrace( (( bold(Q) bold(K)^top, ) / sqrt(d_h)), bb(R)^(T times T) ) underbrace(bold(V), bb(R)^(T times d_h)) $ #pause
]

#sslide[
    ```python

    class Attention(nn.Module):
        def __init__(self):
            self.theta_K = nn.Linear(d_x, d_h, bias=False)
            self.theta_Q = nn.Linear(d_x, d_h, bias=False)
            self.theta_V = nn.Linear(d_x, d_h, bias=False)

        def forward(self, x):
            A = softmax(self.theta_Q(x) @ self.theta_K(x) / d_h)
            return A @ self.theta_V(x)
    ```
]

#sslide[Transformers]

// Todo embeddings, positional encoding, output selection

#sslide[
    If you understand attention, transformers are very simple #pause

    Each transformer consists of many "transformer blocks" #pause

    A transformer block is attention and an MLP
]

#sslide[
    ```python
    class TransformerBlock(nn.Module):
        def __init__(self):
            self.attn = Attention()
            self.mlp = Sequential(
                Linear(d_h, d_h), LeakyReLU(), Linear(d_h, d_h))
            self.norm1 = nn.LayerNorm(d_h)
            self.norm2 = nn.LayerNorm(d_h)

        def forward(self, x):
            # Residual connection
            x = self.norm1(self.attn(x) + x)
            x = self.norm2(self.mlp(x) + x)
            return x
    ```
]

#sslide[
    ```python
    class Transformer(nn.Module):
        def __init__(self):
            self.block1 = TransformerBlock()
            self.block2 = TransformerBlock()
            ...
        
        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            ...
            return x
    ```
]

#sslide[
    ```python
    ```
]

#aslide(ag, 5)
#aslide(ag, 6)

// Transformer operates ons equences
// Set operation
// Permutation invariant
// Equation
// Do not understand time
//
// Transformers are an operation on *sets*

#sslide[
    *Question:* Do we care about the order of inputs $bold(x)_1, bold(x)_2, dots$ in attention? #pause

    Let us see! #pause

    Define a permutation matrix $bold(P) in {0, 1}^(T times T)$ #pause

    *Example:*

    $ P = mat(
        1, 0, 0;
        0, 1, 0;
        0, 0, 1;
    ); quad a = vec(3, 4, 5) $ #pause

    $ P a = vec(3, 4, 5) $
]

#sslide[
    *Example:*

    $ P = mat(
        0, 1, 0;
        1, 0, 0;
        0, 0, 1;
    ); quad a = vec(3, 4, 5) $

    $ P a = vec(4, 3, 5) $
]

#sslide[
    Recall attention

    $
    bold(Q) = vec(bold(q)_1, bold(q)_2, dots.v, bold(q)_T) = vec(bold(theta)_Q^top bold(x)_1, bold(theta)_Q^top bold(x)_2, dots.v, bold(theta)_Q^top bold(x)_T) quad

    bold(K) = vec(bold(k)_1, bold(k)_2, dots.v, bold(k)_T) = vec(bold(theta)_K^top bold(x)_1, bold(theta)_K^top bold(x)_2, dots.v, bold(theta)_K^top bold(x)_T) quad

    bold(V) = vec(bold(v)_1, bold(v)_2, dots.v, bold(v)_T) = vec(bold(theta)_V^top bold(x)_1, bold(theta)_V^top bold(x)_2, dots.v, bold(theta)_V^top bold(x)_T) quad

    $ #pause

    $ "attn"(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = softmax( (bold(Q) bold(K)^top) / sqrt(d_h)) bold(V) $
]

#sslide[
    What if we permute $mat(bold(x)_1, dots, bold(x)_T)^top$ by $bold(P)$? #pause

    $ "attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = softmax( (bold(P) bold(Q) bold(P) bold(K)^top) / sqrt(d_h)) bold(P) bold(V) $

    $
    bold(P) bold(Q) = bold(P) vec(bold(q)_1, bold(q)_2, dots.v, bold(q)_T) quad

    bold(P) bold(K) = bold(P) vec(bold(k)_1, bold(k)_2, dots.v, bold(k)_T)  quad

    bold(P) bold(V) = bold(P) vec(bold(v)_1, bold(v)_2, dots.v, bold(v)_T)  quad

    $ #pause

]

#sslide[
    *Example:* #pause

    $ "attn"(vec(bold(x)_T, bold(x)_1, dots.v, bold(x)_2), bold(theta)) = softmax( (bold(Q)_P bold(K)_P^top) / sqrt(d_h)) bold(V)_P $ #pause

    $
    bold(Q)_P = vec(bold(q)_T, bold(q)_1, dots.v, bold(q)_2) quad

    bold(K)_P = vec(bold(k)_T, bold(k)_1, dots.v, bold(k)_2)  quad

    bold(V)_P = vec(bold(v)_T, bold(v)_1, dots.v, bold(v)_2)  quad

    $

]

#sslide[
    $ bold(P) "attn"(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = "attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) $

    Attention is *permutation equivariant* #pause

    Order of the inputs does not matter
]

#sslide[
    This makes sense, in our party example with attention we never consider the order #pause

    #cimage("figures/lecture_11/composite_swift_einstein_attn_einstein.png") #pause

    *Question:* Any situations where input order matters?
]

#sslide[
    What about language? #pause

    #side-by-side[
        $ vec(bold(x)_1, bold(x)_2, bold(x)_3, bold(x)_4, bold(x)_5) = vec("The", "dog", "licks", "the", "owner") $ #pause
    ][
        $ vec(bold(x)_1, #redm[$bold(x)_5$], bold(x)_3, bold(x)_4, #redm[$bold(x)_2$]) = vec("The", "owner", "licks", "the", "dog") $
    ] #pause

    *Question:* Do these have the same meaning? #pause

    To attention, these have the same meaning! #pause

    We want to *break* the permutation equivariance for certain tasks
]

#sslide[
    *Question:* What are some ways we can introduce ordering? #pause

    *Answer 1:* We can introduce forgetting #pause

    *ALiBi:* _Press, Ofir, Noah Smith, and Mike Lewis. "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." International Conference on Learning Representations._ #pause

    *RoPE:* _Su, Jianlin, et al. "Roformer: Enhanced transformer with rotary position embedding." Neurocomputing._ #pause

    *Answer 2:* We can modify the inputs based on their ordering #pause

    We will focus on answer 2 because it was used first
]

#sslide[
    *Approach 2:* Modify inputs based on ordering #pause

    #side-by-side[
        $ "attn"(vec(bold(x)_1, bold(x)_2, dots.v, bold(x)_T), bold(theta)) $ #pause
    ][
        $ "attn"(vec(bold(x)_1, bold(x)_2, dots.v, bold(x)_T) + vec(f_"pos" (1), f_"pos" (2), dots.v, f_"pos" (3)), bold(theta)) $ #pause
    ]

    $
    bold(Q) = vec(bold(theta)_Q^top bold(x)_1, bold(theta)_Q^top bold(x)_2, dots.v, bold(theta)_Q^top bold(x)_T) quad

    bold(K) = vec(bold(theta)_K^top bold(x)_1, bold(theta)_K^top bold(x)_2, dots.v, bold(theta)_K^top bold(x)_T) quad

    bold(V) = vec(bold(theta)_V^top bold(x)_1, bold(theta)_V^top bold(x)_2, dots.v, bold(theta)_V^top bold(x)_T) quad

    $ #pause

]
#sslide[
    Now, keys and values 

    Now what happens if we permute the inputs? #pause

    $ "attn"(vec(bold(x)_1, bold(x)_2, dots.v, bold(x)_T) + vec(f_"pos" (1), f_"pos" (2), dots.v, f_"pos" (3)), bold(theta)) $ #pause

    
]

#sslide[
    ```python
    class PositionalTransformer(nn.Module):
        def __init__(self):
            self.f_pos = nn.Embedding(1024, d_x)
            self.block1 = TransformerBlock()
            self.block2 = TransformerBlock()
        
        def forward(self, x):
            timesteps = torch.arange(x.shape[0])
            x = x + self.embedding(timesteps)
            x = self.block1(x)
            x = self.block2(x)
            return x
    ```
]

#sslide[

    Homework
    
    https://colab.research.google.com/drive/18VBb7sz0u8ul5vsFEJnQaepn0pQy4cUa?usp=sharing
]