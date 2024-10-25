#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/cetz:0.2.2": canvas, draw, plot
#import "common.typ": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm

#set math.vec(delim: "[")
#set math.mat(delim: "[")


// TODO: Why is encoder tractable but decoder is not?

#let ag = (
  [Review],
  [Compression],
  [Autoencoders],
  [Applications],
  [Variational Models],
  [Coding]
)

#show: university-theme.with(
  aspect-ratio: "16-9",
  short-title: "CISC 7026: Introduction to Deep Learning",
  short-author: "Steven Morad",
  short-date: "Lecture 9: Autoencoders"
)

#title-slide(
  title: [Autoencoders and Generative Models],
  subtitle: "CISC 7026: Introduction to Deep Learning",
  institution-name: "University of Macau",
)


#aslide(ag, none)
#aslide(ag, 0)

// TODO Review

#sslide[

]

#aslide(ag, 0)
#aslide(ag, 1)

#sslide[
  // See a movie (shrek), how do you tell a friend?
  // You cannot draw each frame from memory
  // Summarization (capture key elements)
  // This is a lossy form of Compression
  // We call this unsupervised learning because I am not telling you how to communicate the film
  // Can we do the same thing with models?
  // Linear autoencoder
  // Problem setup, mnist autoencoder
  // Encoder, bottleneck, decoder
  // Problem, denoising autoencoder
  // Problem setup, future prediction model 
  // Combine memory w/ Encoder
  // VAE
  // Beta VAE
  // VQ-VAE
  
  // Q: Why do we need a small bottleneck?
  // Q: Why do we want to do this?
]

#sslide[
  #side-by-side[#cimage("figures/lecture_9/shrek.jpg") #pause][
    *Question:* You watch a film. How do you communicate information about the film with a friend? #pause

    *Answer:* An ogre and donkey rescue a princess, discovering friendship and love along the way. #pause

    *Question:* What is missing? #pause

    *Answer:* Shrek lives in a swamp, Lord Farquaad, dragons, etc
  ]
]

#sslide[
  When you discuss concepts with friends (paintings, music, films, etc), you summarize them #pause

  This is a form of *compression* #pause

  $ f(vec(bold(x)_1, dots.v, bold(x)_n)) = "Green ogre and donkey save princess" $ #pause

  In compression, we take some data and reduce its size by removing unnecessary information #pause

  Let us examine a more principled form of video compression

]

#sslide[
  Shrek in 4k UHD: $ X in bb(Z)_(255)^(3 times 3840 times 2160), X^(90 times 60 times 24) $ #pause

  *Question:* How many GB? #pause

  *Answer:* 3000 GB #pause

  Fortunately, very talented engineers created the H.264 video codec #pause

  H.264 MPEG-AVC *encoder* transforms videos into a more compact representation 

]


// Encoder and decoder of H264
// Why do we need to encode and decode
#sslide[
  H.264 encoder selects $16 times 16$ pixel blocks, estimates shift between frames, applies cosine transform, ... #pause

  The result is an `.mp4` file $Z$ #pause

  #side-by-side[$ f: X^t |-> Z $][$ Z in {0, 1}^n $] #pause

  *Question:* What is the size of $Z$ in GB? #pause

  *Answer:* 60 GB, original size was 3000 GB #pause

  We achieve a compression ratio of $3000 "GB /" 60 "GB" = 50$ 
]

#sslide[

  #side-by-side[We download $Z$ from the internet #pause][$ Z in {0, 1}^n $] #pause

  Information is no longer pixels, it is a string of bits #pause

  *Question:* How do we watch the video? #pause

  *Answer:* Transform or *decode* $Z$ back into pixels #pause

  We need to undo (invert) the encoding function $f$

  $ f^(-1): Z |-> X^t $ #pause

  You CPU has a H.264 decoder built in to make this fast
]

#sslide[
  To summarize: #pause

  We encode pixels into a bit string to save space #pause

  $ f: X^t |-> Z $ #pause

  We store the film as a bit string on websites or your computer #pause

  When you want to watch, we decode the string back into pixels #pause

  $ f^(-1): Z |-> X^t $
]

#sslide[
  #side-by-side[
    Compression may be *lossy* #pause
  ][
    or *lossless*
  ] #pause
  #cimage("figures/lecture_9/lossy.jpg", height: 70%)

  *Question:* Which is H.264?
]

#aslide(ag, 1)
#aslide(ag, 2)

#sslide[
  The encoders and decoders for images, videos, etc are very complex #pause

  Neural networks can represent any continuous function #pause

  Let us learn neural network encoders and decoders #pause

  $ f: X times Theta |-> Z $ #pause

  $ f^(-1): Z times Theta |-> X $ #pause

  We call this an *autoencoder*

  Notice there is no $Y$ this time #pause

  Training autoencoders is different than what we have seen before
]

#sslide[
    #cimage("figures/lecture_8/supervised_unsupervised.png")
]

#sslide[
    In supervised learning, humans provide the model with *inputs* $bold(X)$ and corresponding *outputs* $bold(Y)$ #pause

    $ bold(X) = mat(x_[1], x_[2], dots, x_[n])^top quad bold(Y) = mat(y_[1], y_[2], dots, y_[n])^top $ #pause

    In unsupervised learning, humans only provide *input* #pause

    $ bold(X) = mat(x_[1], x_[2], dots, x_[n])^top $ #pause

    The training algorithm may generate labels
]

/*
#sslide[
    Unsupervised learning is not an accurate term, because there is some supervision #pause

    "I now call it *self-supervised learning*, because *unsupervised* is both a loaded and confusing term. … Self-supervised learning uses way more supervisory signals than supervised learning, and enormously more than reinforcement learning. That’s why calling it “unsupervised” is totally misleading." - Yann LeCun, Godfather of Deep Learning 

    We will use the term *self-supervised* learning, although many textbooks still call it unsupervised learning #pause
]
*/


#sslide[
  //TODO: Benefits of data-specific encoding vs general

  *Task:* Compress images for your clothing website to save on costs #pause

  #cimage("figures/lecture_5/classify_input.svg", width: 80%)

  #side-by-side[$ d_x: 28 times 28 $][$ d_z: 4 $]

  #side-by-side[$ X: [0, 1]^(d_x) $][$ Z: bb(R)^(d_z) $] #pause

  #side-by-side[$ f(bold(x), bold(theta)) = bold(z) $][
    $ f^(-1)(bold(z), bold(theta)) = bold(x) $
  ]

  #side-by-side[What is the structure of $f, f^(-1)$? #pause][How do we find $bold(theta)$?] 
]

#sslide[
  Let us find $f$, then find the inverse #pause

  Start with a perceptron #pause

  $ f(bold(x), bold(theta)) = sigma(bold(theta)^top overline(bold(x))); quad bold(theta) in bb(R)^(d_x, d_z) $ #pause

  $ bold(z) = sigma(bold(theta)^top overline(bold(x))) $

  Solve for $bold(x)$ to find the inverse

  $ sigma^(-1)(bold(z)) = sigma^(-1)(sigma(bold(theta)^top overline(bold(x)))) $

  $ sigma^(-1)(bold(z)) = bold(theta)^top overline(bold(x)) $
]

#sslide[
  $ sigma^(-1)(bold(z)) = bold(theta)^top overline(bold(x)) $

  $ (bold(theta)^top)^(-1) sigma^(-1)(bold(z)) = (bold(theta)^top)^(-1) bold(theta)^top overline(bold(x)) $

  $ (bold(theta)^top)^(-1) sigma^(-1)(bold(z)) = bold(I) overline(bold(x)) $

  $ (bold(theta)^top)^(-1) sigma^(-1)(bold(z)) = overline(bold(x)) $


  $ f^(-1)(bold(z), bold(theta)) =  (bold(theta)^top)^(-1) sigma^(-1)(bold(z)) $
]

#sslide[
  #side-by-side[
  $ f(bold(x), bold(theta)) = sigma(bold(theta)^top overline(bold(x))) $
  ][
    $ f^(-1)(bold(z), bold(theta)) = (bold(theta)^top)^(-1) sigma^(-1)(bold(z)) $
  ][
    $ bold(theta) in bb(R)^(d_x, d_z) $
  ] #pause

  *Question:* Any issues? #h(5em) *Hint:* What if $d_x != d_z$? #pause

  *Answer:* Can only invert square matrices, $bold(theta)^top$ only invertible if $d_z = d_x$ #pause

  *Question:* What kind of compression can we achieve if $d_z = d_x$ ? #pause

  *Answer:* None! We need $d_z < d_x$ for compression #pause

  Look for another solution
  
  // TODO: Lossy compression required for invertability? Means we cannot use bottleneck
  // Cannot be invertible for all useful information, but can be invertible over a subset (the dataset)
]

#sslide[
  Let us try another way

  $ bold(z) = f(bold(x), bold(theta)_e) = sigma(bold(theta)_e^top bold(overline(x))) $

  $ bold(x) = f^(-1)(bold(z), bold(theta)_d) = sigma(bold(theta)_d^top bold(overline(z))) $

  What if we plug $bold(z)$ into the second equation?
]
#sslide[
  Let us try another way

  $ bold(z) = #redm[$f(bold(x), bold(theta)_e)$] = #redm[$sigma(bold(theta)_e^top bold(overline(x)))$] $

  $ bold(x) = f^(-1)(bold(z), bold(theta)_d) = sigma(bold(theta)_d^top bold(overline(z))) $

  What if we plug $bold(z)$ into the second equation?

  $ bold(x) = f^(-1)(#redm[$f(bold(x), bold(theta)_e)$], bold(theta)_d) = sigma(bold(theta)_d^top #redm[$bold(sigma(bold(theta)_e^top bold(overline(x))))$]) $

  //$ f^(-1)(f(bold(x), bold(theta)_e), bold(theta)_d) $
]

#sslide[
  $ bold(x) = f^(-1)((bold(x), bold(theta)_e), bold(theta)_d) = sigma(bold(theta)_d^top bold(sigma(bold(theta)_e^top bold(overline(x))))) $

  More generally, $f, f^(-1)$ may be any neural network #pause

  $ bold(x) = f^(-1)(f(bold(x), bold(theta)_e), bold(theta)_d) $ #pause

  Turn this into a loss function using the square error #pause

  $ cal(L)(bold(x), bold(theta)) = sum_(j=1)^(d_x) (x_j - f^(-1)(f(bold(x), bold(theta)_e), bold(theta)_d)_j)^2 $
]

#sslide[
  $ cal(L)(bold(x), bold(theta)) = sum_(j=1)^(d_x) (x_j - f^(-1)(f(bold(x), bold(theta)_e), bold(theta)_d)_j)^2 $ #pause

  Define over the entire dataset

  $ cal(L)(bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f(bold(x)_[i], bold(theta)_e), bold(theta)_d)_j)^2 $ #pause

  We call this the *reconstruction loss* #pause

  It is an unsupervised loss because we only provide $bold(X)$ and not $bold(Y)$!
]

#sslide[
  First coding exercise

  https://colab.research.google.com/drive/1UyR_W6NDIujaJXYlHZh6O3NfaCAMscpH#scrollTo=nmyQ8aE2pSbb

  https://www.youtube.com/watch?v=UZDiGooFs54
]

// Denoising autoencoder
#aslide(ag, 2)
#aslide(ag, 3)

#sslide[
  We can use autoencoders for other tasks too #pause

  We can make *denoising autoencoders* that remove noise #pause

  #cimage("figures/lecture_9/denoise.jpg", height: 70%)

]

#sslide[
  #side-by-side[Generate some noise][
    $ bold(epsilon) tilde cal(N)(bold(mu), bold(Sigma)) $
  ]

  #side-by-side[Add noise to the image][
    $ bold(x) + bold(epsilon) $
  ]


  $ "Original loss" quad cal(L)(bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f(bold(x)_[i], bold(theta)_e), bold(theta)_d)_j)^2 $ #pause

  $ "Denoising loss" quad cal(L)(bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f(bold(x)_[i] #redm[$+ bold(epsilon)$], bold(theta)_e), bold(theta)_d)_j)^2 $ 
]

#sslide[
  We can add camera blur too #pause

  #cimage("figures/lecture_9/blur.jpg", height: 80%)
]

#sslide[
  $ "blur"(bold(x) + bold(epsilon)) $ #pause

  Denoising deblur loss

  $ cal(L)(bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f("blur"(bold(x)_[i] + bold(epsilon)), bold(theta)_e), bold(theta)_d)_j)^2 $ 
]

#sslide[
  Now we can "enhance" images like in crime tv shows #pause

  #side-by-side[
    #cimage("figures/lecture_9/enhance0.jpg")
  ][
    #cimage("figures/lecture_9/enhance1.jpg")
  ]
]

#sslide[
  We can deblur faces from security cameras

  #cimage("figures/lecture_9/deblur.gif", height: 80%)
]

#sslide[
  We can even hide parts of the image #pause

  A *masked autoencoder* will reconstruct the missing data #pause

  #cimage("figures/lecture_9/masked.png", height: 70%)
]

#sslide[
  What is happening here? How can these models do this? #pause

  Autoencoders learn the structure of the dataset

]

#sslide[
  #side-by-side[
    #cimage("figures/lecture_9/lung.jpg", height: 100%) #pause
  ][
    $X:$ Pictures of human lungs #pause

    $Z in bb(R)^2$ #pause

    Learns the structure of lungs from images #pause

    Differentiates sick and healthy lungs without being told

  ]
]

#sslide[
  If the dataset is pictures from our world, then the autoencoders learn the structure of the world #pause

  Nobody tells them what a dog or cat is #pause

  They learn that on their own
]

#sslide[
  #cimage("figures/lecture_9/clustering.png", height: 100%)
]

#sslide[
  #cimage("figures/lecture_9/dino.png", height: 100%)
]

#sslide[
  Some say "neural networks do not understand", they just learn patterns #pause

  Humans are also pattern recognition machines #pause

  Clearly, these networks can understand our world
]

#aslide(ag, 3)
#aslide(ag, 4)

#sslide[
  #cimage("figures/lecture_9/vae_gen_faces.png", height: 70%) #pause

  These pictures were created by an autoencoder #pause

  But these people do not exist!
]

#sslide[
  Autoencoders are useful for compression and denoising #pause

  But we can also use them as *generative models* #pause

  A generative model is a "creative" model that creates new data #pause
  - Generate pictures of people that do not exist #pause
  - Write a new book #pause
  - Create new proteins #pause

  How does this work?
]

#sslide[
  Latent space $Z$ after training on the clothes dataset with $d_z = 2$

  #cimage("figures/lecture_9/latent_space.png") #pause
]

#sslide[
  What happens if we decode a new point?

  #cimage("figures/lecture_9/latent_space.png")
]

#sslide[
  #side-by-side[
  #cimage("figures/lecture_9/latent_space.png")
  ][
  Autoencoder generative model: #pause
  
  Encode $ vec(bold(x)_[1], dots.v, bold(x)_[n])$ into $vec(bold(z)_[1], dots.v, bold(z)_[n]) $ #pause

  Pick a point $bold(z)_[k]$ #pause

  Add some noise $bold(z)_"new" = bold(z)_[k] + bold(epsilon)$ #pause

  Decode $bold(z)_"new"$ into a new $bold(x)$
  ]
]

#sslide[
  #cimage("figures/lecture_9/vae_gen_faces.png", height: 70%) #pause

  $ f^(-1)(bold(z)_k + bold(epsilon), bold(theta)_d) $
]

#sslide[
  #side-by-side[
  But there is a problem #pause

  As $d_z$ increases, it becomes difficult to find useful parts of latent space #pause

  For large $d_z$, similar inputs map to $bold(z)$ that are very far from each other #pause

  Points $bold(z) + bold(epsilon)$ decode into garbage
  ][
  #cimage("figures/lecture_9/3d_latent_space.png", height: 100%)
  ]
]

#sslide[
  *Question:* What can we do? #pause

  *Answer:* Force the points to be close together! #pause

  We will use a *variational autoencoder* (VAE)
]

#sslide[
  VAE discovered by Diederik Kingma (also adam optimizer) #pause

  #cimage("figures/lecture_9/durk.jpg", height: 80%) 
]

#sslide[
  Variational autoencoders (VAEs) do three things: #pause
  + Make it easy to sample random $bold(z)$ #pause
  + Keep all $bold(z)_[1], dots bold(z)_[n]$ close together in a small region #pause
  + Ensure that $bold(z) + bold(epsilon)$ is always meaningful #pause

  How? #pause

  Make $bold(z)_[i], dots, bold(z)_[n]$ normally distributed #pause

  $ bold(z) tilde cal(N)(mu, sigma), quad mu = 0, sigma = 1 $
]

#sslide[
  If $bold(z)_[i], dots, bold(z)_[n]$ are distributed following $cal(N)(0, 1)$: #pause

  + 99.7% of $bold(z)_[1], dots, bold(z)_[n]$ lie within $3 sigma = [-3, 3]$ #pause

  + Make it easy to generate new $bold(z)$, just sample $bold(z) tilde cal(N)(0, 1)$
]


#sslide[
  So how do we ensure that $bold(z)_[i], dots, bold(z)_[n]$ are normally distributed? #pause

  We have to remember conditional probabilities

  $ P("rain" | "cloud") = "Probability of rain, given that it is cloudy" $ #pause

  //First, let us assume we already have some latent variable $bold(z)$, and focus on the decoder #pause
]

#sslide[
  Introduce one more Bayesian concept called *marginalization* #pause

  Assume we have the probability of two events 

  $ P(A sect B) = P(A, B) $ #pause

  From Bayes rule 

  $ P(A sect B) = P(A | B) P(B) $ #pause

  We can find $P(A)$ by marginalizing out $B$ #pause

  $ P(A) = integral_B P(A | B) P(B) dif B $
]

#sslide[
  *Warning:* Derivation of variational autoencoders is difficult #pause

  I have done my best to simplify it, but it is still complex #pause

  There might be small mistakes!
]

#sslide[
  *Key idea:*

  *Step 1:* Encoder produces a distribution $P(bold(z))$ from an input $bold(x)$

  $ P(bold(z) | bold(x)) $ #pause

  *Step 2:* Decoder produces $bold(x)_"new"$ sampled from a distribution 

  $ bold(x)_"new" tilde P(bold(x) | bold(z)) P(bold(z)) $ #pause

  VAE is generative because we sample from a distribution #pause

  We are sampling new $bold(z)$ that were not in the dataset!
]

#sslide[
  Let us start with the encoder #pause

  We want to turn an input $bold(x)$ into a latent normal distribution over $Z$ #pause

  Start with the joint probability $P(bold(z), bold(x))$

  $ P(bold(z), bold(x)) = underbrace(P(bold(z) | bold(x)), "Likelihood") underbrace(P(bold(x)), "Prior") $

  #side-by-side[
    $ P(bold(x)) $
  ][Probability of a specific input (image) existing in the world] #pause

  #side-by-side[
    $ P(bold(z) | bold(x)) $
  ][
    Latent distribution that summarizes $bold(x)$
  ]
]

#sslide[
  We can marginalize to get the latent distribution $P(bold(z))$

  $ P(bold(z)) = integral_bold(x) P(bold(z) | bold(x)) P(bold(x)) dif x $ 

  We approximate this function using the encoder

  $ f(bold(x), bold(theta)_e) = P(bold(z)) $

  *Question:* What distribution should $P(bold(z))$ be? #pause

  *Answer:* A normal distribution $cal(N)(bold(mu), bold(sigma))$
]


#sslide[
  We represent a normal distribution with a *mean* $mu in M$ and a *standard deviation* $sigma in Sigma$ #pause

  $ f : X times Theta |-> M times Sigma  $ #pause

  So our encoder should output two values

  $ M in bb(R)^(d_z), Sigma in bb(R)_+^(d_z) $
]

#sslide[
  ```python
  core = nn.Sequential(...)
  mu_layer = nn.Linear(d_h, d_z)
  # Neural networks output real numbers
  # But sigma must be positive
  # So we output log sigma, because e^(sigma) is always positive
  log_sigma_layer = nn.Linear(d_h, d_z)

  tmp = core(x)
  mu = mu_layer(tmp)
  log_sigma = log_sigma_layer(tmp)
  dist = (mu, exp(sigma))
  ```
]

#sslide[
  Decoder maps the distribution $P(bold(z))$ to $P(bold(x))$

  $ P(bold(x)) = integral_z P(bold(x) | bold(z)) P(bold(z)) $

  $ bold(x) tilde P(bold(x)) $

  #side-by-side[
    $ P(bold(z)) $
  ][Normal dist. given by encoder] #pause

  #side-by-side[
    $ P(bold(x) | bold(z)) $
  ][
    Output distribution 
  ]
  #side-by-side[
    $ bold(x) $
  ][
    Output (e.g., an image)
  ]
]

#sslide[
  $ P(bold(x)) = integral_z P(bold(x) | bold(z)) P(bold(z)) $

  $ bold(x) tilde P(bold(x)) $

  Integral is intractable, so the decoder inputs and outputs a *sample*

  $ f^(-1): Z times Theta |-> X $
]

#sslide[
  ```python
  decoder = nn.Sequential(
    nn.Linear(d_z, d_h),
    ...
    nn.Linear(d_h, d_x)
  )
  ```

  Decoder is the same as a normal autoencoder
]

#sslide[
  So far, we have the encoder

  $ f : X times Theta |-> M times Sigma  $ #pause

  And the decoder

  $ f^(-1): Z times Theta |-> X $ #pause
  
  *Question:* Any problems? #pause

  *Answer:* The encoder output is a distribution $M times Sigma$, but the decoder input is a sample $Z$
]

#sslide[
  To convert from a distribution over $Z$ to a $bold(z)$, we must sample #pause

  $ bold(z) tilde P(bold(z)) $ #pause

  But sampling from a distribution is not differentiable #pause

  The authors discover the *reparameterization trick* #pause

  $ bold(z) = bold(mu) + bold(sigma) dot.circle bold(epsilon) quad bold(epsilon) tilde cal(U)(0, 1) $ #pause

  Gradient can flow through $bold(mu), bold(sigma)$ #pause

  We can draw samples but still use gradient descent
]

#sslide[
  Put it all together #pause

  *Step 1:* Encode the input to a latent distribution

  $ bold(mu), bold(sigma) = f(bold(x), bold(theta)_e) $

  *Step 2:* Generate a sample from distribution

  $ bold(z) = bold(mu) + bold(sigma) dot.circle bold(epsilon) $

  *Step 3:* Decode the sample 

  $ bold(x) = f^(-1)(bold(z), bold(theta)_d) $
]

#sslide[

]

#sslide[
  Todo loss
]


#sslide[
  We want to constrain the distribution 
  
  $ P(bold(z) | bold(x)) approx cal(N)(0, 1) $ #pause

  We will implement this constraint in the loss function #pause

  *Question:* How to measure distance between two distributions?
]

#sslide[
  *Answer:* KL divergence

  #cimage("figures/lecture_5/forwardkl.png", height: 50%)
  
  $ KL(P, Q) = sum_i P(i) log P(i) / Q(i) $
]

#sslide[
  $ cal(L)_"KL" (bold(x), bold(theta)) = sum_(j=1)^d_z KL(P(z_j | bold(x)), cal(N)(bold(0), bold(1))) $

  $ cal(L)_"KL" (bold(x), bold(theta)) = sum_(j=1)^d_z KL(f(bold(x), bold(theta)_e)_j, cal(N)(bold(0), bold(1))) $

  This ensures that the latent representations $bold(z)$ are easy to sample 
]

#sslide[
  $ cal(L)_"KL" (bold(x), bold(theta)) = sum_(j=1)^d_z KL(P(z_j | bold(x)), cal(N)(0, 1)) $

  Remember, our neural network outputs $P(bold(z))$ as $bold(mu), bold(sigma)$

  $ f(bold(x), bold(theta)) = bold(mu), bold(sigma) $ #pause

  The KL divergence between two normal distributions simplifies to

  $ cal(L)_"KL" (bold(x), bold(theta)) = (sum_(j=1)^d_z 1 + sigma_j^2 + mu_j^2 - exp(sigma_j^2)) $ #pause

]

#sslide[
  $ cal(L)_"KL" (bold(x), bold(theta)) = (sum_(j=1)^d_z 1 + sigma_j^2 + mu_j^2 - exp(sigma_j^2)) $ #pause

  Over the whole dataset

  $ cal(L)_"KL" (bold(X), bold(theta)) = sum_(i=1)^n (sum_(j=1)^d_z 1 + sigma_([i],j)^2 + mu_([i],j)^2 - exp(sigma_([i], j)^2)) $ 
]

#sslide[
  $ cal(L)_"KL" (bold(X), bold(theta)) = sum_(i=1)^n (sum_(j=1)^d_z 1 + sigma_([i],j)^2 + mu_([i],j)^2 - exp(sigma_([i], j)^2)) $  #pause

  The KL loss ensures that $bold(z) tilde cal(N)(bold(0), bold(1))$ #pause

  We also want to make sure we reconstruct the input!

  $ 
   cal(L)_"recon" (bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f(bold(x)_[i], bold(theta)_e), bold(theta)_d)_j)^2 
  $ #pause

  cal(L)(bold(X), bold(theta))
  
]

#sslide[
  The loss for a VAE contains the reconstruction error and the KL loss
]
