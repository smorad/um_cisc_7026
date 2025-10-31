#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "@preview/cetz:0.4.0"
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
    title: [Autoencoders],
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


// TODO: Why is encoder tractable but decoder is not?

= Admin
==
Exam 2 grades released #pause
- Mean score: 78.3% #pause
  - 10% higher than exam 1! #pause
- Extra credit added to exam score #pause
  - Students with over 100% have "Exam 2 Extra Credit" #pause
    - Moodle does not allow more than 100%

==
Some students got 100% or more on both exams! #pause

Great job! For you, there is no need to take exam 3 #pause
- Agustin Yan
- Chen Zhonghong
- Cui Shuhang
- Lai Yongchao
- Gao Yuchen
- Sheng Junwei
- Wu Chengqin #pause

Some students are very close (Exam 1: 95%, Exam 2: 100%) #pause
  - Can also skip exam 3 if you do not care about 5%

==
Homework 3 fully graded #pause
- Mean score 97% #pause

Last assignment is RNN! 

==
Please find groups for final project! #pause

Project plan must contain 2 projects #pause
- I will check projects to make sure they are possible #pause
- If project 1 too hard, can do project 2 instead #pause

Things that will make you lose many points #pause
- Heavy use of LLM/CoPilot #pause
- Copy dataset and model from Kaggle/GitHub/etc

==
The final project has two purposes: #pause
1. Show that you understand the course material enough to apply it #pause
2. Give you freedom to work on something you like #pause
  - No supervisor telling you what to work on #pause

Code everything yourself and invest sufficient time and effort #pause
- You will get good marks #pause

Some projects from last year with full marks: #pause
- Detecting depression in brain scans #pause
- Generating MIDI music using RNN #pause
- Comparing graph neural network architectures


= Review
==
    Convolution works over inputs of any variables (time, space, etc) #pause

    Recurrent neural networks only work with time #pause

    Convolution makes use of locality and translation equivariance properties #pause

    Recurrent models do not assume locality or equivariance #pause

    Equivariance and locality make learning more efficient, but not all problems have this structure

==
    How do humans process temporal data? #pause

    Humans only perceive the present #pause

    Humans process temporal data by storing and recalling memories

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
    $ f(bold(x), bold(theta)) = sum_(i=1)^T bold(theta)^top overline(bold(x))_i $ #pause

    #side-by-side[What if we see a new face? #pause][
        $ f(bold(x), bold(theta)) = (sum_(i=1)^T bold(theta)^top overline(bold(x))_i) + bold(theta)^top overline(bold(x))_(T+1) $ #pause
    ]

    We repeat the same process for each new face #pause

    We can rewrite $f$ as a *recurrent function*

==
    Rewrote composite memory as a recurrent function #pause

    $ f(bold(x), bold(theta)) = underbrace((sum_(i=1)^T bold(theta)^top overline(bold(x))_i), bold(h)) + bold(theta)^top overline(bold(x))_(T+1) $ #pause

    $ f(bold(h), bold(x), bold(theta)) = bold(h) + bold(theta)^top overline(bold(x)) $ #pause

    $ bold(x) in bb(R)^(d_x), quad bold(h) in bb(R)^(d_h) $

==
    #side-by-side[$  bold(x) in bb(R)^(d_x), quad bold(h) in bb(R)^(d_h) $][
    $ f(bold(h), bold(x), bold(theta)) = bold(h) + bold(theta)^top overline(bold(x)) $] #pause

    $ #redm[$bold(h)_1$] = f(bold(0), bold(x)_1, bold(theta)) = bold(0) + bold(theta)^top overline(bold(x))_1 $ #pause

    $ #greenm[$bold(h)_2$] = f(#redm[$bold(h)_1$], bold(x)_2, bold(theta)) = bold(h)_1 + bold(theta)^top overline(bold(x))_2 $ #pause

    $ bold(h)_3 = f(#greenm[$bold(h)_2$], bold(x)_3, bold(theta)) = bold(h)_2 + bold(theta)^top overline(bold(x))_3 $ #pause

    $ dots.v $

    $ bold(h)_T = f(bold(h)_(T-1), bold(x)_T, bold(theta)) = bold(h)_(T-1) + bold(theta)^top overline(bold(x))_T $ #pause

    //We *scan* through the inputs $bold(x)_1, bold(x)_2, dots, bold(x)_T$

==
    This model remembers everything #pause

    But $bold(h)$ is a fixed size, what if $T$ is very large? #pause

    If we keep adding and adding $bold(x)$ into $bold(h)$, we will run out of space #pause

    Humans forget old information

==
    #side-by-side[Murdock (1982) #cimage("figures/lecture_8/murdock.jpg", height: 80%) #pause ][
        
        $ f(bold(h), bold(x), bold(theta)) = #redm[$gamma$] bold(h) + bold(theta)^top overline(bold(x)); quad 0 < gamma < 1 $ #pause

        *Key Idea:* $lim_(T -> oo) gamma^T = 0$ #pause 

        Let $gamma = 0.9$ #pause

        $ 0.9 dot 0.9 dot bold(h) = 0.81 bold(h) $ #pause
        $ 0.9 dot 0.9 dot 0.9 dot bold(h) = 0.72 bold(h) $ #pause
        $ 0.9 dot 0.9 dot 0.9 dot dots dot bold(h) = 0 $

        ] 

==
    $ bold(h)_T = gamma^3 bold(h)_(T - 3) + gamma^2 bold(theta)^top overline(bold(x))_(T - 2) + gamma bold(theta)^top overline(bold(x))_(T - 1) + bold(theta)^top overline(bold(x))_T $ #pause

    $ bold(h)_T = gamma^T bold(theta)^top overline(bold(x))_1 + gamma^(T - 1) bold(theta)^top overline(bold(x))_2 + dots + gamma^2 bold(theta)^top overline(bold(x))_(T - 2) + gamma bold(theta)^top overline(bold(x))_(T - 1) + bold(theta)^top overline(bold(x))_T $
==
    Our function $f$ is just defined for a single $X$ #pause

    $ f: H times X times Theta |-> H $ #pause

    To extend $f$ to sequences, we scan $f$ over the inputs

    $ scan(f): H times X^T times Theta |-> H^T $ #pause

    $ scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(h_1, h_2, dots.v, h_T) $ 
==
    $ f: H times X times Theta |-> H, quad scan(f): H times X^T times Theta |-> H^T $ #pause

    $ scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(bold(h)_1, bold(h)_2, dots.v, bold(h)_T) $ #pause

    $ scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = vec(f(bold(h)_0, bold(x)_1, bold(theta)), f(bold(h)_1, bold(x)_2, bold(theta)), dots.v, f(bold(h)_(T-1), bold(x)_T, bold(theta))) = vec(f(bold(h)_0, x_1), f(f(bold(h)_0, bold(x)_1), bold(x)_2), dots.v, f( dots f(bold(h)_0, bold(x)_1) dots, bold(x)_T)) $
==
    Let $g$ define the memory recall function #pause 

    $ g: H times X times Theta |-> Y $ #pause

    $g$ searches your memories $h$ using the input $x$, to produce output $y$ #pause

    $bold(x):$ "What is your favorite ice cream flavor?" #pause 

    $bold(h):$ Everything you remember (hometown, birthday, etc) #pause

    $g:$ Searches your memories for ice cream information, and responds "chocolate"

==
    *Step 1:* Perform scan to find recurrent states #pause

    $ vec(bold(h)_1, dots.v, bold(h)_T) = scan(f)(bold(h)_0, vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_f) $ #pause

    *Step 2:* Perform recall on recurrent states #pause

    $ vec(hat(bold(y))_1, dots.v, hat(bold(y))_T) = vec(
        g(bold(h)_1, bold(x)_1, bold(theta)_g),
        dots.v,
        g(bold(h)_T, bold(x)_T, bold(theta)_g),
    ) $ 
==
    The simplest recurrent neural network is the *Elman Network* #pause

    $ f(bold(h), bold(x), bold(theta)) = sigma(bold(theta)^top_1 bold(h) + bold(theta)^top_2 overline(bold(x))) $ #pause

    $ g(bold(h), bold(x), bold(theta)) = 
        bold(theta)^top_3 bold(h)
    $

==
    Add forgetting 

    $ 
    f (bold(h), bold(x), bold(theta)) = 
    sigma(
        bold(theta)_1^top bold(h) #redm[$dot.o f_"forget" (bold(h), bold(x), bold(theta))$] + bold(theta)_2^top overline(bold(x)) 
    )
    $ #pause


    $ 
    f_"forget" (bold(h), bold(x), bold(theta)) = sigma(
        bold(theta)_1^top overline(bold(x)) +  bold(theta)_2^top bold(h)
    )
    $ #pause

    When $f_"forget" < 1$, we forget some of our memories! #pause

    Through gradient descent, neural network learns which memories to forget

==
    *Minimal gated unit* (MGU) is a modern RNN #pause
    
    MGU defines two helper functions #pause

    $ 
    f_"forget" (bold(h), bold(x), bold(theta)) = sigma(
        bold(theta)_1^top overline(bold(x)) +  bold(theta)_2^top bold(h)
    ) 
    $ #pause

    $ 
    f_2(bold(h), bold(x), bold(theta)) = underbrace(sigma(
        bold(theta)_3^top overline(bold(x)) + bold(theta)_4^top 
            f_"forget" (bold(h), bold(x), bold(theta)) dot.o bold(h)
    ), "Make new memories") 
    $ #pause

    $
    f(bold(h), bold(x), bold(theta)) = 
        underbrace(
          f_"forget" (bold(h), bold(x), bold(theta)) dot.o bold(h)
        , "Forget old memories") + 
        underbrace((1 - f_"forget" (bold(h), bold(x), bold(theta))), "Replace") dot.o underbrace(f_2(bold(h), bold(x), bold(theta)), "New memories")
    $ #pause

    Only forget when we have new information to remember

= Compression
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

  $ f(vec(bold(x)_1, dots.v, bold(x)_n)) = "Fat panda saves village" $ #pause

  In compression, we reduce the size of data by removing information #pause

  Let us examine a more principled form of video compression

==
  Kung Fu Panda in 4k UHD: $ X in {0, dots, 255}^(3 times 3840 times 2160), X^(90 times 60 times 24) $ #pause

  *Question:* How many GB? #pause

  *Answer:* 3000 GB #pause

  But if you download films, you know they are smaller than 3 TB #pause

  Usually you download an `mp4` or `mpeg4` file #pause

  `mp4` uses `H.264` video *encoder* to compress information 


// Encoder and decoder of H264
// Why do we need to encode and decode
==
  `H.264` encoder selects $16 times 16$ pixel blocks, estimates shift between frames, applies cosine transform, ... #pause

  The result is an `.mp4` file $Z$ #pause

  #side-by-side[$ f: X^t |-> Z $][$ Z in {0, 1}^k $] #pause

  *Question:* What is the size of $Z$ in GB? #pause

  *Answer:* 60 GB, original size was 3000 GB #pause

  We achieve a compression ratio of $3000 "GB /" 60 "GB" = 50$ 

==
  What happens on your computer when you watch Kung Fu Panda? #pause

  #side-by-side[Download $bold(z) in Z$ from the internet #pause][$ Z in {0, 1}^n $] #pause

  Information is no longer pixels, it is a string of bits #pause

  We must *decode* $bold(z)$ back into pixels #pause

  We need to undo (invert) the encoder $f$ #pause

  $ f: X^t |-> Z $

  $ f^(-1): Z |-> X^t $ #pause

  You CPU has an `H.264` hardware decoder to make this fast

==
  #side-by-side[
    Compression may be *lossy* #pause
  ][
    or *lossless*
  ] #pause
  #cimage("figures/lecture_9/lossy.jpg", height: 70%) #pause

  *Question:* Which is `H.264`?

= Autoencoders
==
  Encoders and decoders for images, videos, and music are functions #pause

  Neural networks can represent any continuous function #pause

  We can use neural networks to represent encoders and decoders #pause

  $ f: X times Theta |-> Z $ #pause

  $ f^(-1): Z times Theta |-> X $ #pause

  We call this an *autoencoder* #pause

  Notice there is no $Y$ this time #pause

  Training autoencoders is different than what we have seen before

==
    #cimage("figures/lecture_8/supervised_unsupervised.png")

==
    In supervised learning, humans provide the model with *inputs* $bold(X)$ and corresponding *outputs* $bold(Y)$ #pause

    $ bold(X) = mat(x_[1], x_[2], dots, x_[n])^top quad bold(Y) = mat(y_[1], y_[2], dots, y_[n])^top $ #pause

    In unsupervised learning, humans only provide *input* #pause

    $ bold(X) = mat(x_[1], x_[2], dots, x_[n])^top $ #pause

    The training algorithm will learn *unsupervised* (only from $bold(X)$) #pause
    - Humans do not need to provide labels!

==
  //TODO: Benefits of data-specific encoding vs general

  *Task:* Compress images for your clothing website to save on costs #pause

  #cimage("figures/lecture_5/classify_input.svg", width: 80%) #pause

  #side-by-side[$ X in [0, 1]^(d_x) $][$ Z in bb(R)^(d_z) $] #pause

  #side-by-side[$ d_x: 28 times 28 $][$ d_z: 3 $] #pause

  #side-by-side[$ f: X times Theta |-> Z $][
    $ f^(-1): Z times Theta |-> X $
  ] #pause

  #side-by-side[What is the structure of $f, f^(-1)$? #pause][How do we find $bold(theta)$?] 

==
  Let us find $f$, then find the inverse $f^(-1)$ #pause

  Start with a perceptron #pause

  $ f(bold(x), bold(theta)) &= sigma(bold(theta)^top overline(bold(x))); quad bold(theta) in bb(R)^((d_x + 1) times d_z) #pause \

  bold(z) &= sigma(bold(theta)^top overline(bold(x))) $ #pause

  Solve for $bold(x)$ to find the inverse #pause

  $ sigma^(-1)(bold(z)) &= sigma^(-1)(sigma(bold(theta)^top overline(bold(x)))) #pause \

  sigma^(-1)(bold(z)) &= bold(theta)^top overline(bold(x)) $

==
  $ sigma^(-1)(bold(z)) &= bold(theta)^top overline(bold(x)) #pause \

  (bold(theta)^top)^(-1) sigma^(-1)(bold(z)) &= (bold(theta)^top)^(-1) bold(theta)^top overline(bold(x)) #pause \

  (bold(theta)^top)^(-1) sigma^(-1)(bold(z)) &= bold(I) overline(bold(x)) #pause \

  (bold(theta)^top)^(-1) sigma^(-1)(bold(z)) &= overline(bold(x)) #pause \

  f^(-1)(bold(z), bold(theta)) &=  (bold(theta)^top)^(-1) sigma^(-1)(bold(z)) $

==
  #side-by-side[
  $ f(bold(x), bold(theta)) = sigma(bold(theta)^top overline(bold(x))) $
  ][
    $ f^(-1)(bold(z), bold(theta)) = (bold(theta)^top)^(-1) sigma^(-1)(bold(z)) $
  ][
    $ bold(theta) in bb(R)^((d_x + 1) times d_z) $
  ] #pause

  *Question:* Any issues? #pause #h(5em) *Hint:* What if $d_x != d_z$? #pause

  *Answer:* Can only invert square matrices, $bold(theta)^top$ only invertible if $d_z = d_x$ #pause

  *Question:* What kind of compression can we achieve if $d_z = d_x$ ? #pause

  *Answer:* None! We need $d_z < d_x$ for compression #pause

  Look for another solution
  
  // TODO: Lossy compression required for invertability? Means we cannot use bottleneck
  // Cannot be invertible for all useful information, but can be invertible over a subset (the dataset)

==
  Let us try another way #pause

  $ bold(z) &= f(bold(x), bold(theta)_e) & = sigma(bold(theta)_e^top bold(overline(x))) \ #pause

  bold(x) &= f^(-1)(bold(z), bold(theta)_d) & = sigma(bold(theta)_d^top bold(overline(z))) $ #pause

  What if we plug $bold(z)$ into the second equation?

==
  Let us try another way

  $ bold(z) = #redm[$f(bold(x), bold(theta)_e)$] = #redm[$sigma(bold(theta)_e^top bold(overline(x)))$] $

  $ bold(x) = f^(-1)(bold(z), bold(theta)_d) = sigma(bold(theta)_d^top bold(overline(z))) $

  What if we plug $bold(z)$ into the second equation?

  $ bold(x) = f^(-1)(#redm[$f(bold(x), bold(theta)_e)$], bold(theta)_d) = sigma(bold(theta)_d^top #redm[$bold(sigma(bold(theta)_e^top bold(overline(x))))$]) $

  //$ f^(-1)(f(bold(x), bold(theta)_e), bold(theta)_d) $

==
  $ bold(x) = f^(-1)(f(bold(x), bold(theta)_e), bold(theta)_d) = sigma(bold(theta)_d^top sigma(bold(theta)_e^top bold(overline(x)))) $ #pause

  More generally, $f, f^(-1)$ may be any neural network #pause

  $ bold(x) = f^(-1)(f(bold(x), bold(theta)_e), bold(theta)_d) $ #pause

  Turn this into a loss function using the square error #pause

  $ cal(L)(bold(x), bold(theta)) = sum_(j=1)^(d_x) (x_j - f^(-1)(f(bold(x), bold(theta)_e), bold(theta)_d)_j)^2 $ #pause

  Forces the networks to compress and reconstruct $bold(x)$

==
  $ cal(L)(bold(x), bold(theta)) = sum_(j=1)^(d_x) (x_j - f^(-1)(f(bold(x), bold(theta)_e), bold(theta)_d)_j)^2 $ #pause

  Define over the entire dataset

  $ cal(L)(bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f(bold(x)_[i], bold(theta)_e), bold(theta)_d)_j)^2 $ #pause

  We call this the *reconstruction loss* #pause

==
  $ cal(L)(bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f(bold(x)_[i], bold(theta)_e), bold(theta)_d)_j)^2 $ #pause

  The reconstruction loss is an *unsupervised loss* #pause
  - No traditional labels $Y$! #pause
  - Learns from unlabeled data $X$

==
So far, we consider only a perceptron and mean square error #pause

There are many types of autoencoder architectures and objectives 

= Convolutional Autoencoders
==
#side-by-side(align: left)[#cimage("figures/lecture_9/kungfu.jpg", height: 80%) #cimage("figures/lecture_9/kungfu.jpg", height: 10%) #pause][
*Task:* Compress an image #pause

$ 1"MB" -> 10"kb" $ #pause

For Fashion MNIST we used perceptron #pause

*Question:* Do we have better networks for images? #pause

*Question:* Use convolutional network. Why? #pause

*Answer:* More efficient than perceptrons (data/parameters)
]
==
With autoencoders we must contract then expand #pause

$ f: &bb(R)^(d_x) times Theta &&|-> bb(R)^(d_z) quad d_z < d_x \
f^(-1): &bb(R)^(d_z) times Theta &&|-> bb(R)^(d_x) $ #pause

*Question:* Do convolution and pooling contract or expand? #pause

*Answer:* Usually contract, reduce the size of the signal #pause

To create a convolutional decoder, we must expand the signal #pause

To expand the signal, we must "invert" convolution and pooling

==
Let us first try and "invert" pooling #pause

Consider mean pooling (adaptive $2 times 2$) #pause

  #let c = cetz-canvas({
    import cetz.draw: *

    content((2, 2), image("figures/lecture_7/ghost_dog_bw.svg", width: 4cm))
    
    draw_filter(
      0, 0,
      (
        (1, 1, 1, 1),
        (0, 0, 1, 0),
        (1, 0, 0, 1),
        (1, 0, 1, 1),
      )
    )

  content((2, -1), text(size: 40pt)[$sum$])
  content((0, -3.15), text(size: 40pt)[$1/4 dot$])

  draw_filter(
    1, -4,
    (
      (" ", " "),
      (" ", " ")
    )
  ) 

  })
  
  #let d = cetz-canvas({
    import cetz.draw: *

    content((2, 2), image("figures/lecture_7/ghost_dog_bw.svg", width: 4cm))
    
    draw_filter(
      0, 0,
      (
        (" ", " ", " ", " "),
        (" ", " ", " ", " "),
        (" ", " ", " ", " "),
        (" ", " ", " ", " "),
      )
    )

  content((2, -1), text(size: 40pt)[$sum$])

  content((0, -3.15), text(size: 40pt)[$1/4 dot$])

  draw_filter(
    1, -4,
    (
      (2, 3),
      (2, 3)
    )
  ) 
})

  #side-by-side[#c #pause][#d #pause]

  This "inverse" pooling has two names: *unpooling* and *upsampling*

==
Upsampling does not use parameters! #pause
  - Only upscaling $bold(z)$ will not provide a good reconstruction #pause
  - Need to introduce learned parameters #pause
    - Combine with convolution layers or "inverse" convolution layers

==

  Let us try and "invert" convolution #pause

  #let c = cetz-canvas({
    import cetz.draw: *

    content((2, 2), image("figures/lecture_7/ghost_dog_bw.svg", width: 4cm))
    
    draw_filter(
      0, 0,
      (
        (1, 1, 1, 1),
        (0, 0, 1, 0),
        (1, 0, 0, 1),
        (1, 0, 0, 1),
      )
    )

  content((2, -1), text(size: 40pt)[$*$])

  draw_filter(
    1, -4,
    (
      (1, 0),
      (0, 1)
    )
  ) 

  content((6, 0), text(size: 40pt)[$=$])
  
  draw_filter(
    7, -1,
    (
      (" ", " "),
      (" ", " ")
    )
  )
  })
  
  #let d = cetz-canvas({
  import cetz.draw: *

  content((2, 2), image("figures/lecture_7/ghost_dog_bw.svg", width: 4cm))
  
  draw_filter(
    0, 0,
    (
      (" ", " ", " ", " "),
      (" ", " ", " ", " "),
      (" ", " ", " ", " "),
      (" ", " ", " ", " "),
    )
  )

content((2, -1), text(size: 40pt)[$*$])

draw_filter(
  1, -4,
  (
    (1, 0),
    (0, 1)
  )
) 

content((6, 0), text(size: 40pt)[$=$])


draw_filter(
  7, -1,
  (
    (1, 0),
    (1, 1)
  )
)
})

  #side-by-side[#c #pause][#d #pause]

  We call this *deconvolution* or *transposed convolution*

==
Transposed convolution is similar to upsampling #pause
- Unlike upsampling, we learn the filters (parameters) #pause

New "inverse" operations let us construct a convolutional decoder #pause
- Many different ways to do this

==
Symmetric convolution-only autoencoder

#side-by-side(align: top)[
  ```python
  encoder = Sequential(

    Conv2d(3, 16, 3) 
    LeakyReLU()

    Conv2d(16, 32, 3)
    LeakyReLU() 
    ...
    Conv2d(32, 32, 3)
    Flatten()
  )
  ```
][
  ```python
  decoder = Sequential(
    Unflatten()
    ConvTranspose2d(32, 32, 3)
    LeakyReLU()

    ConvTranspose2d(32, 16, 3)
    LeakyReLU()
    ...
    ConvTranspose2d(16, 3, 3)

  )
  ```
]

==
Other architectures use asymmetric `conv -> upsample` blocks

#side-by-side(align: top)[
  ```python
  encoder = Sequential(

    # Block
    Conv2d() 
    MaxPool()
    LeakyReLU()

    ...
    Conv2d()
    Flatten()
  )
  ```
][
  ```python
  decoder = Sequential(
    Unflatten()
    # Block
    Upsample()
    Conv2d()
    LeakyReLU()

    ...
    Conv2d()
  )
  ```
]

==
I find that asymmetric upsample autoencoders tend to work better #pause
- `Upsample -> Conv -> Activation` #pause
- Finding filter size, stride, etc that produce the correct shape is painful #pause

I use symmetric autoencoders because they are easier to implement #pause
- `ConvTranspose -> Activation` #pause
- Usually just reorder encoder arguments #pause
  - `Conv2d(A, B, C) -> ConvTranspose2d(B, A, C)`

= Recurrent Autoencoders
==
#side-by-side(align: left)[
    #cimage("figures/lecture_9/kfp1.jpg")

    #cimage("figures/lecture_9/kfp2.jpg")

    $ dots.v $
][
  *Task:* Compress entire film (sequence of images) #pause

  *Step 1:* Compress image with CNN $ -> bold(x)$ #pause

  *Step 2:* #pause Compress $bold(x)_1, dots bold(x)_T -> bold(h)$ #pause

  *Question:* What models process a sequence of inputs? #pause

  *Answer:* Recurrent neural networks!
]
==
Recall the recurrent neural network type signatures #pause

$ f&: H times X times Theta &&|-> H \ #pause
scan(f)&: H times X^T times Theta &&|-> H^T \ #pause
$

*Question:* This is the encoder, what about decoder? #pause

$ g: H times X times Theta |-> X $ #pause

*Question:* Any problems? #pause *Answer:* Need input to reconstruct input #pause

$ f^(-1): H times Theta |-> X $ #pause

$ scan(f^(-1)): H times Theta |-> H^T times X^T $ #pause

==

$ f(bold(h)_0, 
    underbrace(#image("figures/lecture_9/kfp1.jpg", height: 20%), bold(x)_1),
bold(theta)_e) = bold(h)_1 $ #pause

$ dots.v $ #pause

$ f(bold(h)_(T - 1), 
    underbrace(#image("figures/lecture_9/kfp2.jpg", height: 20%), bold(x)_T),
bold(theta)_e) = bold(h)_T $ #pause

==
$ f(bold(h)_T, bold(theta)_d) = hat(bold(h))_(T - 1), underbrace(#image("figures/lecture_9/kfp2.jpg", height: 20%), bold(x)_T) $ #pause

$ dots.v $ #pause

$ f(bold(h)_1, bold(theta)_d) = hat(bold(h))_(0), underbrace(#image("figures/lecture_9/kfp1.jpg", height: 20%), bold(x)_T) $ 

==
```python
# Compress images to latent sequence
zs = vmap(cnn_enc)(frames)
h0 = zeros(d_h) # Initial state
# Compress sequence to latent vector
hT, hs = scan(rnn_enc, init=h0, xs=zs)
# Final state should hold all film info
# Decompress final state into latent sequence
recon_zT, recon_zs = scan(rnn_dec, init=hT, length=T)
# Decompress latent sequence into images 
recon_frames = vmap(cnn_dec)(recon_zs)
# Compute loss and update params
grads = grad(lambda x, xhat: mean((x - xhat) ** 2))(
  frames, recon_frames)
model = update((cnn_enc, rnn_enc, rnn_dec, cnn_dec), grads)
```

==
Older language models used this approach #pause
- Encode sequence of characters into meaning #pause
- ELMo was a precursor to GPT 

= Applications and Objectives
==
Can make many types of autoencoders #pause
- Graph neural network autoencoders #pause
- Transformer autoencoders #pause

So far, we considered the reconstruction objective #pause

$ cal(L)(bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f(bold(x)_[i], bold(theta)_e), bold(theta)_d)_j)^2 $ #pause

This objective is useful for compression #pause

Could we use autoencoders for other tasks?

==
*Task:* Remove noise from an image #pause

#cimage("figures/lecture_9/denoise.jpg", height: 70%) #pause

Can we do this with autoencoders?

==

  $ "Reconstruction loss" quad cal(L)(bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f(bold(x)_[i], bold(theta)_e), bold(theta)_d)_j)^2 $ #pause

  #side-by-side[Generate some noise][
    $ bold(epsilon) tilde cal(N)(bold(mu), bold(sigma)) $
  ] #pause

  $ "Denoising loss" quad cal(L)(bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f(bold(x)_[i] #redm[$+ bold(epsilon)$], bold(theta)_e), bold(theta)_d)_j)^2 $ #pause

  Add noise to input $bold(x) + bold(epsilon)$, reconstruct $bold(x)$ without noise #pause

  Autoencoder will learn to remove noise when reconstructing image

==
  *Task:* Remove blurring from an image #pause

  #cimage("figures/lecture_9/blur.jpg", height: 80%)

==
  $ "Reconstruction loss" quad cal(L)(bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f(bold(x)_[i], bold(theta)_e), bold(theta)_d)_j)^2 $ #pause
  
  #side-by-side[Create blurry image][$ "blur"(bold(x)) $] #pause

  Denoising deblur loss

  $ cal(L)(bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f( #redm[blur];(bold(x)_[i]), bold(theta)_e), bold(theta)_d)_j)^2 $ 

==
  #cimage("figures/lecture_9/deblur.gif", height: 80%) #pause

  No more privacy in 2025

==
*Task:* Fix blurry and noisy image #pause

$ cal(L)(bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f( #redm[blur];(bold(x)_[i] + #bluem[$bold(epsilon)$]), bold(theta)_e), bold(theta)_d)_j)^2 $ #pause

#side-by-side[
  #cimage("figures/lecture_9/enhance0.jpg")
][
  #cimage("figures/lecture_9/enhance1.jpg")
]


= Emergent Intelligence

==
Something very interesting happens with autoencoders #pause

This becomes clear when we consider one last application

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

  //To compress, denoise, deblur, or demask inputs, autoencoders must understand our world #pause

  //Reconstruction losses are a surrogate objective for world understanding 
]

==
  #side-by-side(align: left)[
    #cimage("figures/lecture_9/lung.jpg", height: 100%) #pause
  ][
    $X:$ Pictures of human lungs #pause

    $Z in bb(R)^2$ #pause

    Learns the structure of lungs from images #pause

    Separates sick from healthy lungs without labels #pause
    - "Sick" or "healthy" information improves compression

  ]

==
  If the dataset is lung images, the model learns the structure of lungs #pause

  If the dataset is pictures from our world, then the autoencoders learn the structure of the world #pause

  Nobody tells an autoencoder what a dog or cat is #pause

  They learn this on their own

==
  #cimage("figures/lecture_9/clustering.png", height: 100%)

==
  #cimage("figures/lecture_9/dino.png", height: 100%)

==
  Some say "neural networks do not understand", they just learn patterns #pause

  Humans are also pattern recognition machines #pause

  #cimage("figures/lecture_9/face.jpg", height: 60%) #pause

  *Opinion:* These networks understand our world as well as humans

==
Let us demonstrate this 
  
https://colab.research.google.com/drive/1UyR_W6NDIujaJXYlHZh6O3NfaCAMscpH#scrollTo=nmyQ8aE2pSbb

https://www.youtube.com/watch?v=UZDiGooFs54

= Final Project Groups
==
You must stay 10 minutes to find a group of 4-5 people #pause

Groups with 4 members, the lonely students are often smart #pause 
- Please consider inviting them to your group #pause

Following groups have 5 members and can leave now
- H, O, A, L, S, P, M, AA, X



/*
= Applications
==
  We can use autoencoders for more than compression #pause

  We can make *denoising autoencoders* that remove noise #pause

  #cimage("figures/lecture_9/denoise.jpg", height: 70%)

==
  #side-by-side[Generate some noise][
    $ bold(epsilon) tilde cal(N)(bold(mu), bold(sigma)) $
  ] #pause

  #side-by-side[Add noise to the image][
    $ bold(x) + bold(epsilon) $
  ] #pause

  $ "Original loss" quad cal(L)(bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f(bold(x)_[i], bold(theta)_e), bold(theta)_d)_j)^2 $ #pause

  $ "Denoising loss" quad cal(L)(bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f(bold(x)_[i] #redm[$+ bold(epsilon)$], bold(theta)_e), bold(theta)_d)_j)^2 $ #pause

  Autoencoder will learn to remove noise when reconstructing image

==
  We can add camera blur too #pause

  #cimage("figures/lecture_9/blur.jpg", height: 80%)

==
  $ "blur"(bold(x) + bold(epsilon)) $ #pause

  Denoising deblur loss

  $ cal(L)(bold(X), bold(theta)) = sum_(i=1)^n sum_(j=1)^(d_x) (x_([i],j) - f^(-1)(f("blur"(bold(x)_[i] + bold(epsilon)), bold(theta)_e), bold(theta)_d)_j)^2 $ 

==
  Now we can "enhance" images like in crime tv shows #pause

  #side-by-side[
    #cimage("figures/lecture_9/enhance0.jpg")
  ][
    #cimage("figures/lecture_9/enhance1.jpg")
  ]

==
  We can deblur faces from security cameras

  #cimage("figures/lecture_9/deblur.gif", height: 80%)

==
  We can even hide parts of the image #pause

  A *masked autoencoder* will reconstruct the missing data #pause

  #cimage("figures/lecture_9/masked.png", height: 70%)

==
  What is happening here? How can these models do this? #pause

  Autoencoders learn the structure of the dataset

==
  #side-by-side[
    #cimage("figures/lecture_9/lung.jpg", height: 100%) #pause
  ][
    $X:$ Pictures of human lungs #pause

    $Z in bb(R)^2$ #pause

    Learns the structure of lungs from images #pause

    Differentiates sick and healthy lungs without being told

  ]

==
  If the dataset is lung images, the model learns the structure of lungs #pause

  If the dataset is pictures from our world, then the autoencoders learn the structure of the world #pause

  Nobody tells them what a dog or cat is #pause

  They learn that on their own

==
  #cimage("figures/lecture_9/clustering.png", height: 100%)

==
  #cimage("figures/lecture_9/dino.png", height: 100%)

==
  Some say "neural networks do not understand", they just learn patterns #pause

  Humans are also pattern recognition machines #pause

  #cimage("figures/lecture_9/face.jpg", height: 60%) #pause

  These networks can understand our world

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

==
  https://colab.research.google.com/drive/1UyR_W6NDIujaJXYlHZh6O3NfaCAMscpH#scrollTo=nmyQ8aE2pSbb

  https://www.youtube.com/watch?v=UZDiGooFs54


  4 Nov - Compensatory rest day (holiday), no lecture

  11 Nov - Quiz on autoencoders (not VAE) and Prof. Li on GNNs

  18 Nov - Attention and transformers

  25 Nov - Reinforcement learning

  1 Dec - Foundation models (virtual, optional)

*/