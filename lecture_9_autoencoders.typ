#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/cetz:0.2.2": canvas, draw, plot
#import "common.typ": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm
#import "@preview/fletcher:0.5.2" as fletcher: diagram, node, edge

#let varinf = diagram(
  node-stroke: .1em,
  spacing: 4em,
  node((0,0), $bold(z)$, radius: 2em),
  edge($P(bold(x) | bold(z); bold(theta))$, "-|>"),
  node((2,0), $bold(x)$, radius: 2em),
  edge((0,0), (2,0), $P(bold(z) | bold(x); bold(theta))$, "<|-", bend: -40deg),
)

#let normal = { 
    canvas(length: 1cm, {
  plot.plot(size: (16, 10),
    x-tick-step: 2,
    y-tick-step: 0.5,
    y-min: 0,
    y-max: 1,
    x-label: [$ bold(z) $],
    y-label: [$ P(bold(z)) $],
    {
      plot.add(
        domain: (-4, 4), 
        style: (stroke: (thickness: 5pt, paint: red)),
        x => calc.pow(calc.e, -(0.5 * calc.pow(x, 2)))
      )
    })
})}

#set math.vec(delim: "[")
#set math.mat(delim: "[")


// TODO: Why is encoder tractable but decoder is not?

#let ag = (
  [Review],
  [Compression],
  [Autoencoders],
  [Applications],
  [Variational Modeling],
  [VAE Implementation],
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

#sslide[
    Convolution works over inputs of any variables (time, space, etc) #pause

    Recurrent neural networks only work with time #pause

    Convolution makes use of locality and translation equivariance properties #pause

    Recurrent models do not assume locality or equivariance #pause

    Equivariance and locality make learning more efficient, but not all problems have this structure
]

#sslide[
    How do humans process temporal data? #pause

    Humans only perceive the present #pause

    Humans process temporal data by storing and recalling memories
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
    $ f(bold(x), bold(theta)) = sum_(i=1)^T bold(theta)^top overline(bold(x))_i $ #pause

    #side-by-side[What if we see a new face? #pause][
        $ f(bold(x), bold(theta)) = (sum_(i=1)^T bold(theta)^top overline(bold(x))_i) + bold(theta)^top overline(bold(x))_"new" $ #pause
    ]

    We repeat the same process for each new face #pause

    We can rewrite $f$ as a *recurrent function*

]
#sslide[
    Let us rewrite composite memory as a recurrent function #pause

    $ f(bold(x), bold(theta)) = underbrace((sum_(i=1)^T bold(theta)^top overline(bold(x))_i), bold(h)) + bold(theta)^top overline(bold(x))_"new" $ #pause

    $ f(bold(h), bold(x), bold(theta)) = bold(h) + bold(theta)^top overline(bold(x)) $ #pause

    $ bold(x) in bb(R)^(d_x), quad bold(h) in bb(R)^(d_h) $
]

#sslide[
    #side-by-side[$  bold(x) in bb(R)^(d_x), quad bold(h) in bb(R)^(d_h) $][
    $ f(bold(h), bold(x), bold(theta)) = bold(h) + bold(theta)^top overline(bold(x)) $] #pause

    $ #redm[$bold(h)_1$] = f(bold(0), bold(x)_1, bold(theta)) = bold(0) + bold(theta)^top overline(bold(x))_1 $ #pause

    $ #greenm[$bold(h)_2$] = f(#redm[$bold(h)_1$], bold(x)_2, bold(theta)) = bold(h)_1 + bold(theta)^top overline(bold(x))_2 $ #pause

    $ bold(h)_3 = f(#greenm[$bold(h)_2$], bold(x)_3, bold(theta)) = bold(h)_2 + bold(theta)^top overline(bold(x))_3 $ #pause

    $ dots.v $

    $ bold(h)_T = f(bold(h)_(T-1), bold(x)_T, bold(theta)) = bold(h)_(T-1) + bold(theta)^top overline(bold(x))_T $ #pause

    //We *scan* through the inputs $bold(x)_1, bold(x)_2, dots, bold(x)_T$
]

#sslide[
    Right now, our model remembers everything #pause

    But $bold(h)$ is a fixed size, what if $T$ is very large? #pause

    If we keep adding and adding $bold(x)$ into $bold(h)$, we will run out of space #pause

    Humans forget old information
]

#sslide[
    #side-by-side[Murdock (1982) #cimage("figures/lecture_8/murdock.jpg", height: 80%) #pause ][
        
        $ f(bold(h), bold(x), bold(theta)) = #redm[$gamma$] bold(h) + bold(theta)^top overline(bold(x)); quad 0 < gamma < 1 $ #pause

        *Key Idea:* $lim_(T -> oo) gamma^T = 0$ #pause 

        Let $gamma = 0.9$ #pause

        $ 0.9 dot 0.9 dot bold(h) = 0.81 bold(h) $ #pause
        $ 0.9 dot 0.9 dot 0.9 dot bold(h) = 0.72 bold(h) $ #pause
        $ 0.9 dot 0.9 dot 0.9 dot dots dot bold(h) = 0 $

        ] 
]

#sslide[
    $ bold(h)_T = gamma^3 bold(h)_(T - 3) + gamma^2 bold(theta)^top overline(bold(x))_(T - 2) + gamma bold(theta)^top overline(bold(x))_(T - 1) + bold(theta)^top overline(bold(x))_T $ #pause

    $ bold(h)_T = gamma^T bold(theta)^top overline(bold(x))_1 + gamma^(T - 1) bold(theta)^top overline(bold(x))_2 + dots + gamma^2 bold(theta)^top overline(bold(x))_(T - 2) + gamma bold(theta)^top overline(bold(x))_(T - 1) + bold(theta)^top overline(bold(x))_T $
]
#sslide[
    Our function $f$ is just defined for a single $X$ #pause

    $ f: H times X times Theta |-> H $ #pause

    To extend $f$ to sequences, we scan $f$ over the inputs

    $ scan(f): H times X^T times Theta |-> H^T $ #pause

    $ scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(h_1, h_2, dots.v, h_T) $ 
]

#sslide[
    $ f: H times X times Theta |-> H, quad scan(f): H times X^T times Theta |-> H^T $ #pause

    $ scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(h_1, h_2, dots.v, h_T) $ #pause
    
    $ scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(f(h_0, x_1, bold(theta)), f(h_1, x_2, bold(theta)), dots.v, f(h_(T-1), x_T, bold(theta))) = vec(f(h_0, x_1), f(f(h_0, x_1), x_2), dots.v, f( dots f(h_0, x_1) dots, x_T)) $
]

#sslide[
    Let $g$ define our memory recall function #pause 

    $ g: H times X times Theta |-> Y $ #pause

    $g$ searches your memories $h$ using the input $x$, to produce output $y$ #pause

    $bold(x):$ "What is your favorite ice cream flavor?" #pause 

    $bold(h):$ Everything you remember (hometown, birthday, etc) #pause

    $g:$ Searches your memories for ice cream information, and responds "chocolate"
]

#sslide[
    *Step 1:* Perform scan to find recurrent states #pause

    $ vec(bold(h)_1, dots.v, bold(h)_T) = scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_f) $ #pause

    *Step 2:* Perform recall on recurrent states #pause

    $ vec(bold(y)_1, dots.v, bold(y)_T) = vec(
        g(bold(h)_1, bold(x)_1, bold(theta)_g),
        dots.v,
        g(bold(h)_T, bold(x)_T, bold(theta)_g),
    ) $
]

#sslide[
    The simplest recurrent neural network is the *Elman Network* #pause

    $ f(bold(h), bold(x), bold(theta)) = sigma(bold(theta)^top_1 bold(h) + bold(theta)^top_2 overline(bold(x))) $ #pause

    $ g(bold(h), bold(x), bold(theta)) = 
        bold(theta)^top_3 bold(h)
    $
]

#sslide[
    Add forgetting 

    $ 
    f (bold(h), bold(x), bold(theta)) = 
    sigma(
        bold(theta)_1^top bold(h) #redm[$dot.circle f_"forget" (bold(h), bold(x), bold(theta))$] + bold(theta)_2^top overline(bold(x)) 
    )
    $ #pause


    $ 
    f_"forget" (bold(h), bold(x), bold(theta)) = sigma(
        bold(theta)_1^top overline(bold(x)) +  bold(theta)_2^top bold(h)
    )
    $ #pause

    When $f_"forget" < 1$, we forget some of our memories! #pause

    Through gradient descent, neural network learns which memories to forget
]

#sslide[
    *Minimal gated unit* (MGU) is a modern RNN #pause
    
    MGU defines two helper functions #pause

    $ 
    f_"forget" (bold(h), bold(x), bold(theta)) = sigma(
        bold(theta)_1^top overline(bold(x)) +  bold(theta)_2^top bold(h)
    ) 
    $ #pause

    $ 
    f_2(bold(h), bold(x), bold(theta)) = sigma(
        bold(theta)_3^top overline(bold(x)) + bold(theta)_4^top 
            f_"forget" (bold(h), bold(x), bold(theta)) dot.circle bold(h)
    ) 
    $ #pause

    $
    f(bold(h), bold(x), bold(theta)) = 
        f_"forget" (bold(h), bold(x), bold(theta)) dot.circle bold(h) + (1 - f_"forget" (bold(h), bold(x), bold(theta))) dot.circle f_2(bold(h), bold(x), bold(theta))
    $ #pause

    Left term forgets old, right term replaces forgotten memories
]

#sslide[
    Jax RNN https://colab.research.google.com/drive/147z7FNGyERV8oQ_4gZmxDVdeoNt0hKta#scrollTo=TUMonlJ1u8Va
]

#aslide(ag, 0)
#aslide(ag, 1)

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

#sslide[
  #side-by-side[#cimage("figures/lecture_9/shrek.jpg") #pause][
    *Question:* You watch a film. How do you communicate information about the film with a friend? #pause

    *Answer:* An ogre and donkey rescue a princess, discovering friendship and love along the way. #pause

    *Question:* What is missing? #pause

    *Answer:* Shrek lives in a swamp, Lord Farquaad, dragons, etc
  ]
]

#sslide[
  When you discuss films with friends, you summarize them #pause

  This is a form of *compression* #pause

  $ f(vec(bold(x)_1, dots.v, bold(x)_n)) = "Green ogre and donkey save princess" $ #pause

  In compression, we reduce the size of data by removing information #pause

  Let us examine a more principled form of video compression
]

#sslide[
  Shrek in 4k UHD: $ X in bb(Z)_(255)^(3 times 3840 times 2160), X^(90 times 60 times 24) $ #pause

  *Question:* How many GB? #pause

  *Answer:* 3000 GB #pause

  But if you download films, you know they are smaller than 3 TB #pause

  Today, we use the H.264 video *encoder* to transform videos into a more compact representation
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
  What happens on your computer when you watch Shrek? #pause

  #side-by-side[Download $bold(z) in Z$ from the internet #pause][$ Z in {0, 1}^n $] #pause

  Information is no longer pixels, it is a string of bits #pause

  We must *decode* $bold(z)$ back into pixels #pause

  We need to undo (invert) the encoder $f$

  $ f: X^t |-> Z $

  $ f^(-1): Z |-> X^t $ #pause

  You CPU has a H.264 decoder built in to make this fast
]

/*
#sslide[
  To summarize: #pause

  We encode pixels into a bit string to save space #pause

  $ f: X^t |-> Z $ #pause

  We store the film as a bit string on websites or your computer #pause

  When you want to watch, we decode the string back into pixels #pause

  $ f^(-1): Z |-> X^t $
]
*/

#sslide[
  #side-by-side[
    Compression may be *lossy* #pause
  ][
    or *lossless*
  ] #pause
  #cimage("figures/lecture_9/lossy.jpg", height: 70%) #pause

  *Question:* Which is H.264?
]

#aslide(ag, 1)
#aslide(ag, 2)



#sslide[
  Encoders and decoders for images, videos, and music are functions #pause

  Neural networks can represent any continuous function #pause

  Let us learn neural network encoders and decoders #pause

  $ f: X times Theta |-> Z $ #pause

  $ f^(-1): Z times Theta |-> X $ #pause

  We call this an *autoencoder* #pause

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

    The training algorithm will learn *unsupervised* (only from $bold(X)$)
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

  #cimage("figures/lecture_5/classify_input.svg", width: 80%) #pause

  #side-by-side[$ X in [0, 1]^(d_x) $][$ Z in bb(R)^(d_z) $] #pause

  #side-by-side[$ d_x: 28 times 28 $][$ d_z: 4 $] #pause

  #side-by-side[$ f: X times Theta |-> Z $][
    $ f^(-1): Z times Theta |-> X $
  ] #pause

  #side-by-side[What is the structure of $f, f^(-1)$? #pause][How do we find $bold(theta)$?] 
]

#sslide[
  Let us find $f$, then find the inverse $f^(-1)$ #pause

  Start with a perceptron #pause

  $ f(bold(x), bold(theta)) = sigma(bold(theta)^top overline(bold(x))); quad bold(theta) in bb(R)^(d_x times d_z) $ #pause

  $ bold(z) = sigma(bold(theta)^top overline(bold(x))) $ #pause

  Solve for $bold(x)$ to find the inverse #pause

  $ sigma^(-1)(bold(z)) = sigma^(-1)(sigma(bold(theta)^top overline(bold(x)))) $ #pause

  $ sigma^(-1)(bold(z)) = bold(theta)^top overline(bold(x)) $
]

#sslide[
  $ sigma^(-1)(bold(z)) = bold(theta)^top overline(bold(x)) $ #pause

  $ (bold(theta)^top)^(-1) sigma^(-1)(bold(z)) = (bold(theta)^top)^(-1) bold(theta)^top overline(bold(x)) $ #pause

  $ (bold(theta)^top)^(-1) sigma^(-1)(bold(z)) = bold(I) overline(bold(x)) $ #pause

  $ (bold(theta)^top)^(-1) sigma^(-1)(bold(z)) = overline(bold(x)) $ #pause

  $ f^(-1)(bold(z), bold(theta)) =  (bold(theta)^top)^(-1) sigma^(-1)(bold(z)) $
]

#sslide[
  #side-by-side[
  $ f(bold(x), bold(theta)) = sigma(bold(theta)^top overline(bold(x))) $
  ][
    $ f^(-1)(bold(z), bold(theta)) = (bold(theta)^top)^(-1) sigma^(-1)(bold(z)) $
  ][
    $ bold(theta) in bb(R)^(d_x times d_z) $
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
  Let us try another way #pause

  $ bold(z) = f(bold(x), bold(theta)_e) = sigma(bold(theta)_e^top bold(overline(x))) $

  $ bold(x) = f^(-1)(bold(z), bold(theta)_d) = sigma(bold(theta)_d^top bold(overline(z))) $ #pause

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
  $ bold(x) = f^(-1)(f(bold(x), bold(theta)_e), bold(theta)_d) = sigma(bold(theta)_d^top bold(sigma(bold(theta)_e^top bold(overline(x))))) $ #pause

  More generally, $f, f^(-1)$ may be any neural network #pause

  $ bold(x) = f^(-1)(f(bold(x), bold(theta)_e), bold(theta)_d) $ #pause

  Turn this into a loss function using the square error #pause

  $ cal(L)(bold(x), bold(theta)) = sum_(j=1)^(d_x) (x_j - f^(-1)(f(bold(x), bold(theta)_e), bold(theta)_d)_j)^2 $ #pause

  Forces the networks to compress and reconstruct $bold(x)$
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
  We can use autoencoders for more than compression #pause

  We can make *denoising autoencoders* that remove noise #pause

  #cimage("figures/lecture_9/denoise.jpg", height: 70%)

]

#sslide[
  #side-by-side[Generate some noise][
    $ bold(epsilon) tilde cal(N)(bold(mu), bold(sigma)) $
  ] #pause

  #side-by-side[Add noise to the image][
    $ bold(x) + bold(epsilon) $
  ] #pause

  $ "Original loss" quad cal(L)(bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f(bold(x)_[i], bold(theta)_e), bold(theta)_d)_j)^2 $ #pause

  $ "Denoising loss" quad cal(L)(bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f(bold(x)_[i] #redm[$+ bold(epsilon)$], bold(theta)_e), bold(theta)_d)_j)^2 $ #pause

  Autoencoder will learn to remove noise when reconstructing image
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
  If the dataset is lung images, the model learns the structure of lungs #pause

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

  #cimage("figures/lecture_9/face.jpg", height: 60%) #pause

  These networks can understand our world
]

#aslide(ag, 3)
#aslide(ag, 4)

#sslide[
  #cimage("figures/lecture_9/vae_gen_faces.png", height: 70%) #pause

  These pictures were created by a *variational* autoencoder #pause

  But these people do not exist!
]

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
  Latent space $Z$ after training on the clothes dataset with $d_z = 3$

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

  Make $bold(z)_[1], dots, bold(z)_[n]$ normally distributed #pause

  $ bold(z) tilde cal(N)(mu, sigma), quad mu = 0, sigma = 1 $
]

#sslide[
  //#cimage("figures/lecture_2/normal_dist.png")
  #align(center, normal)
]

#sslide[
  If $bold(z)_[1], dots, bold(z)_[n]$ are distributed following $cal(N)(0, 1)$: #pause

  + 99.7% of $bold(z)_[1], dots, bold(z)_[n]$ lie within $3 sigma = [-3, 3]$ #pause

  + Make it easy to generate new $bold(z)$, just sample $bold(z) tilde cal(N)(0, 1)$
]


#sslide[
  So how do we ensure that $bold(z)_[1], dots, bold(z)_[n]$ are normally distributed? #pause

  We have to remember conditional probabilities #pause

  $ P("rain" | "cloud") = "Probability of rain, given that it is cloudy" $ #pause

  //First, let us assume we already have some latent variable $bold(z)$, and focus on the decoder #pause
]

/*
#sslide[
  Introduce one more Bayesian concept called *marginalization* #pause

  Assume we have the probability of two events 

  $ P(A sect B) = P(A, B) $ #pause

  We can find $P(A)$ by marginalizing out $B$ #pause

  $ P(A) = integral P(A, B) dif B $
]
*/

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
  $ P(bold(z), bold(x); bold(theta)) = P(bold(x) | bold(z); bold(theta)) space P(bold(z); bold(theta))  $ #pause

  We can choose any distribution for $P(bold(z))$ #pause

  $ P(bold(z)) = cal(N)(bold(0), bold(1)) $ #pause

  We can generate all possible $bold(x)$ by sampling $bold(z) tilde cal(N)(bold(0), bold(1))$ #pause

  We can randomly generate $bold(z)$, which we can decode into new $bold(x)$!
]

#sslide[
  Now, all we must do is find $bold(theta)$ that best explains the dataset distribution #pause

  Learned distribution $P(bold(x); bold(theta))$ to be close to dataset $P(bold(x)), quad bold(x) tilde X$ #pause

  We need some error function between $P(bold(x); bold(theta))$ and $P(bold(x))$ #pause
  
  *Question:* How do we measure the distance between probability distributions? 
]

#sslide[
  *Answer:* KL divergence #pause

  #cimage("figures/lecture_5/forwardkl.png", height: 50%)
  
  $ KL(P, Q) = sum_i P(i) log P(i) / Q(i) $
]

#sslide[
  Learn the parameters for our model #pause

  $ argmin_bold(theta) KL(P(bold(x)), P(bold(x); bold(theta))) $ #pause

  Unfortunately, this objective is intractable to optimize #pause

  The paper provides surrogate objective

  $ argmin_bold(theta) [ -log P(bold(x) | bold(z); bold(theta)) + 1 / 2 KL(P(bold(z) | bold(x); bold(theta)), P(bold(z)))] $ #pause

  We call this the *Evidence Lower Bound Objective* (ELBO) 
]

#sslide[
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
]

#aslide(ag, 4)
#aslide(ag, 5)


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

  *Question:* Any issues? #pause

  *Answer:* Encoder outputs a distribution $Delta Z$ but decoder input is $Z$
]

#sslide[
  We can sample from the distribution 
  
  $ bold(mu), bold(sigma) &= f(bold(x), bold(theta)_e) \ 
  bold(z) & tilde cal(N)(bold(mu), bold(sigma)) $ #pause 

  But there is a problem! Sampling is not differentiable #pause

  *Question:* Why does this matter? #pause

  *Answer:* Must be differentiable for gradient descent
]

#sslide[
  VAE paper proposes the *reparameterization trick* #pause

  $ bold(z) & tilde cal(N)(bold(mu), bold(sigma)) $ 

  $ bold(z) = bold(mu) + bold(sigma) dot.circle bold(epsilon) quad bold(epsilon) tilde cal(N)(bold(0), bold(1)) $ #pause

  Gradient can flow through $bold(mu), bold(sigma)$ #pause

  We can sample and use gradient descent #pause

  This trick only works with certain distributions
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
]

#sslide[
  $ cal(L)(bold(x), bold(theta)) = argmin_bold(theta) [ -log P(bold(x) | bold(z); bold(theta)) + 1 / 2 KL(P(bold(z) | bold(x); bold(theta)), P(bold(z))) ] $ #pause

  First, rewrite KL term using our encoder $f$ #pause

  $ cal(L)(bold(x), bold(theta)) = argmin_bold(theta) [ -log P(bold(x) | bold(z)) + 1 / 2 KL(f(bold(x), bold(theta)_e), P(bold(z))) ] $ #pause

  $P(bold(z))$ and $f(bold(x), bold(theta)_e)$ are Gaussian, we can simplify KL term

  $ cal(L)(bold(x), bold(theta)) = underbrace(log P(bold(x) | bold(z)), "Reconstruction error") - (sum_(j=1)^d_z mu^2_j + sigma^2_j - log(sigma^2) - 1) $
]

#sslide[
  $ cal(L)(bold(x), bold(theta)) = underbrace(log P(bold(x) | bold(z)), "Reconstruction error") - (sum_(j=1)^d_z mu^2_j + sigma^2_j - log(sigma^2) - 1) $ #pause

  Next, plug in square error for reconstruction error #pause

  $ = sum_(j=1)^d_z (x_j - f^(-1)(f(bold(x), bold(theta)_e), bold(theta)_d)_j )^2 - (sum_(j=1)^d_z mu^2_j + sigma^2_j - log(sigma^2_j) - 1) $ #pause

  $ cal(L)(bold(x), bold(theta)) = sum_(j=1)^d_z (x_j - f^(-1)(f(bold(x), bold(theta)_e), bold(theta)_d)_j )^2 - (sum_(j=1)^d_z mu^2_j + sigma^2_j - log(sigma^2_j) - 1) $
]

#sslide[
  $ cal(L)(bold(x), bold(theta)) = sum_(j=1)^d_z (x_j - f^(-1)(f(bold(x), bold(theta)_e), bold(theta)_d)_j )^2 - (sum_(j=1)^d_z mu^2_j + sigma^2_j - log(sigma^2_j) - 1) $ #pause

  Finally, define over the entire dataset

  $ cal(L)(bold(X), bold(theta)) &= sum_(i=1)^n sum_(j=1)^d_z (x_([i],j) - f^(-1)(f(bold(x)_[i], bold(theta)_e), bold(theta)_d)_(j) )^2 - \ 
  &(sum_(i=1)^n sum_(j=1)^d_z mu^2_([i],j) + sigma^2_([i],j) - log(sigma_([i], j)^2) - 1) $
]

#sslide[
  $ cal(L)(bold(X), bold(theta)) &= sum_(i=1)^n sum_(j=1)^d_z (x_([i],j) - f^(-1)(f(bold(x)_[i], bold(theta)_e), bold(theta)_d)_(j) )^2 - \ 
   &(sum_(i=1)^n sum_(j=1)^d_z mu^2_([i],j) + sigma^2_([i],j) - log(sigma_([i], j)^2) - 1) $

  Scale of two terms can vary, we do not want one term to dominate
]

#sslide[
  Paper suggests using minibatch size $m$ and dataset size $n$ #pause

  $ cal(L)(bold(X), bold(theta)) &= #redm[$m / n$] sum_(i=1)^n sum_(j=1)^d_z (x_([i],j) - f^(-1)(f(bold(x)_[i], bold(theta)_e), bold(theta)_d)_(j) )^2 - \ 
   &(sum_(i=1)^n sum_(j=1)^d_z mu^2_([i],j) + sigma^2_([i],j) - log(sigma_([i], j)^2) - 1) $
]

#sslide[
  Another paper finds hyperparameter $beta$ also helps #pause

  $ cal(L)(bold(X), bold(theta)) &= #redm[$m / n$] sum_(i=1)^n sum_(j=1)^d_z (x_([i],j) - f^(-1)(f(bold(x)_[i], bold(theta)_e), bold(theta)_d)_(j) )^2 - \ 
   & #redm[$beta$] (sum_(i=1)^n sum_(j=1)^d_z mu^2_([i],j) + sigma^2_([i],j) - log(sigma_([i], j)^2) - 1) $
]

#sslide[
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
]

#sslide[
  https://colab.research.google.com/drive/1UyR_W6NDIujaJXYlHZh6O3NfaCAMscpH#scrollTo=nmyQ8aE2pSbb

  https://www.youtube.com/watch?v=UZDiGooFs54


  4 Nov - Compensatory rest day (holiday), no lecture

  11 Nov - Quiz on autoencoders (not VAE) and Prof. Li on GNNs

  18 Nov - Attention and transformers

  25 Nov - Reinforcement learning

  1 Dec - Foundation models (virtual, optional)
]