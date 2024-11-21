#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm
#import "@preview/touying:0.5.3": *
#import themes.university: *
#import "common_touying.typ": *
#import "@preview/cetz:0.3.1"
#import "@preview/fletcher:0.5.1" as fletcher: node, edge

#set math.vec(delim: "[")
#set math.mat(delim: "[")

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

#show: university-theme.with(
  aspect-ratio: "16-9",
  // config-common(handout: true),
  config-info(
    title: [Transformers],
    subtitle: [CISC 7026 - Introduction to Deep Learning],
    author: [Steven Morad],
    //date: datetime.today(),
    institution: [University of Macau],
    logo: image("figures/common/bolt-logo.png", width: 4cm)
  ),
  header-right: none,
  header: self => utils.display-current-heading(level: 1)
)

#title-slide()


// Review
// Layernorm and residual connections
// Transformer
// Positional encoding
// Comparison to RNN
// Types of transformers (text, vision, etc)
// Focus on text and vision since they are most popular
// Go indepth text
// Tokenization
// Go indepth vision
// Self supervision
// World models?

== Outline <touying:hidden>

#components.adaptive-columns(
    outline(title: none, indent: 1em, depth: 1)
)

= Review

==
// Made a mistake, softmax should be axis=1
Last time, we derived various forms of *attention*

We started with composite memory #pause

$ f(bold(x), bold(theta)) = sum_(i=1)^T bold(theta)^top overline(bold(x))_i $ #pause

Given large enough $T$, we will eventually run out of storage space #pause

The sum is a *lossy* operation that can store a limited amount of information #pause

== 
So we introduced a forgetting term $gamma$ #pause

#side-by-side[$ f(bold(x), bold(theta)) = sum_(i=1)^T gamma^(T - i) dot bold(theta)^top overline(bold(x))_i $ #pause][
    #align(center)[#forgetting]
] 

== 
We went to a party and the forgetting seemed ok #pause

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

==
But we encountered problems when Taylor Swift arrived at the party #pause

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
With our current model, we forget Taylor Swift! #pause


Our model of human memory is incomplete

== 


== 
Last time we studied attention #pause

Overview of transformer application and domains

= Going Deeper

==
We previously reviewed training tricks #pause

- Deeper networks #pause
- Parameter initialization #pause
- Stochastic gradient descent #pause
- Adaptive optimization #pause
- Weight decay #pause

These methods empirically improve performance, but we do not always understand why 

==
Modern transformers can be very deep #pause

For this reason, they use two new training tricks to enable very deep models #pause
- Residual connections #pause
- Layer normalization #pause

Let us introduce these tricks #pause

We will start with the *residual connection*

== 
Remember that a two-layer MLP is a universal function approximator #pause

$ | f(bold(x), bold(theta)) - g(bold(x)) | < epsilon $ #pause

This is only as the width of the network goes to infinity #pause

For certain problems, we need deeper networks #pause

==
But there is a limit! #pause

Making the network too deep can hurt performance #pause

The theory is that the input information is *lost* somewhere in the network #pause

$ bold(y) = f_k ( dots f_2 ( f_1 (bold(x), bold(theta)_1), bold(theta_2)), dots, bold(theta)_k) $

==
$ bold(y) = f_k ( dots f_2 ( f_1 (bold(x), bold(theta)_1), bold(theta_2)), dots, bold(theta)_k) $

*Claim:* If the input information is present throughout the network, then we should be able to learn the identity function $f(x) = x$ #pause

$ bold(x) = f_k ( dots f_2 ( f_1 (bold(x), bold(theta)_1), bold(theta_2)), dots, bold(theta)_k) $

*Question:* We have seen a similar model, what was it? #pause

*Question:* Do you agree or disagree with the claim? #pause

https://colab.research.google.com/drive/1qVIbQKpTuBYIa7FvC4IH-kJq-E0jmc0d#scrollTo=bg74S-AvbmJz

==
$ bold(x) = f_k ( dots f_2 ( f_1 (bold(x), bold(theta)_1), bold(theta_2)), dots, bold(theta)_k) $ #pause

Very deep networks struggle to learn the identity function #pause

If the input information is available, then learning the identity function should be very easy!

*Question:* How can we prevent the input from getting lost?

==

We can feed the input to each layer #pause

// $ bold(x) = f_k ( dots f_2 ( f_1 (bold(x), bold(theta)_1), bold(theta_2)), dots, bold(theta)_k) $ #pause

The first approach is called the *DenseNet* approach #pause

$
bold(z)_1 &= f_1(bold(x), bold(theta)_1) \
bold(z)_2 &= f_2(vec(bold(x), bold(z)_1), bold(theta)_2) \
dots.v \ 
bold(z)_k &= f_k (vec(bold(x), bold(z)_1, dots.v, bold(z)_(k-1)), bold(theta)_k) \
$

==
$
bold(z)_1 &= f_1(bold(x), bold(theta)_1) \
bold(z)_2 &= f_2(vec(bold(x), bold(z)_1), bold(theta)_2) \
dots.v \ 
bold(z)_k &= f_k (vec(bold(x), bold(z)_1, dots.v, bold(z)_(k-1)), bold(theta)_k) \
$ #pause

*Question:* Any issues with the DenseNet approach? #pause

*Answer:* Very deep networks require too many parameters!

== 
The next method is called the *ResNet* approach

$
bold(z)_1 &= f_1(bold(x), bold(theta)_1) \
bold(z)_2 &= f_2(bold(x), bold(theta)_2) + bold(x) \
bold(z)_3 &= f_2(bold(x), bold(theta)_3) + bold(z)_2 \
dots.v \ 
bold(z)_k &= f_k (bold(x), bold(theta)_k) + bold(z)_(k-1) \
$ #pause

This allows information to flow around the layers #pause

It requires much fewer parameters than a dense net, but also does not work as well #pause

==

We call $f(bold(x)) + bold(x)$ a *residual connection* #pause

Instead of learning how to change $bold(x)$, $f$ learns what to add to $bold(x)$ #pause

For example, for an identity function we can easily learn 
$ f(bold(x), bold(theta)) = 0; quad f(bold(x), bold(theta)) + x = x $ #pause

This helps prevent information from getting lost in very deep networks

==
The second trick is called *layer normalization* #pause

Recall that with parameter initialization and weight decay, we ensure the parameters are fairly small #pause

But we can still have very small or very large outputs from each layer #pause

$ f_1 (bold(x), bold(theta)_1) = sum_(i=1)^(d_x) theta_(1, i) x_i $ #pause

*Question:* If all $x_i = 1$, $theta_(1, i) = 0.01$ and $d_x = 1000$, what is the output?
==
$ f_1 (bold(x), bold(theta)_1) = sum_(i=1)^(d_x) theta_(1, i) x_i $ #pause

$ f_1 (bold(x), bold(theta)_1) = sum_(i=1)^(1000) 0.01 dot 1 = 10 $ #pause

What if we add another layer with the same $d_x$ and $theta$?

$ f_2 (bold(z), bold(theta)_2) = sum_(i=1)^(1000) 0.01 dot 10 = 100 $ #pause

*Question:* What is the problem?

==
Let us look at the gradient #pause

$ gradient_bold(theta_1) f_2( f_1(bold(x), bold(theta)_1), bold(theta)_2) &= #pause gradient_bold(theta)_1 [f_2] (f_1(bold(x), bold(theta)_1)) dot gradient_bold(theta)_1 [f_1](bold(x), bold(theta)_1) \ #pause 
& approx 100 dot 10 $

Can cause exploding or vanishing gradient #pause

Deeper network $=>$ worse exploding/vanishing issues

*Question:* What can we do? #pause

==

We can use *layer normalization* #pause

// Layer normalization *centers* and *rescales* the outputs of a layer #pause
First, layer normalization *centers* the output of the layer #pause

$ mu = d_y sum_(i=1)^(d_y) f(bold(x), bold(theta))_i $

$ f(bold(x), bold(theta)) - mu $

*Question:* What does this do? #pause

*Answer:* Makes output have zero mean (both positive and negative values)  #pause

==
#side-by-side[$ mu = d_y sum_(i=1)^(d_y) f(bold(x), bold(theta))_i $][$ f(bold(x), bold(theta)) - mu $]

Then, layer normalization *rescales* the outputs

$ sigma = sqrt(sum_(i=1)^(d_y) f(bold(x), bold(theta)_i - mu)^2) / d_y $


$ "LN"(f(bold(x), bold(theta))) = (f(bold(x), bold(theta)) - mu) / sigma $

Now, the output of the layer is normally distributed

==
If the output is normally distrbuted: #pause
- 99.7% of outputs $in [-3, 3]$ #pause
- 99.99% of outputs $in [-4, 4]$ #pause
- 99.9999% of outputs $in [-5, 5]$ #pause

This helps prevent vanishing and exploding gradients
==

Now, let's combine residual connections and layer norm and try our very deep network again

TODO COLAB

==

$ bold(z) = f_1 (bold(x), bold(theta)) = sum_(i=1)^(1000) 0.01 dot 1 = 10 $ #pause

$ f_2 (bold(z), bold(theta)) = sum_(i=1)^(1000) 0.01 dot 10 = 100 $ #pause

==
Layer Norm

Initialization and L2 normalization keep the weights small

But the outputs of each layer can still be large or small

Chain rule example

= Transformers

==
Now we have everything we need to implement a transformer

// Can use for all sorts of problems
// x can be image, text etc

= Positional Encoding

= Text Transformers

= Image Transformers


= Unsupervised Training

==
Predict the future

= World Models

==
What if transformers could interact with the world?

= Course Evaluation

== 
Department instructed me to ask you for course feedback #pause

We take this feedback seriously #pause

Your feedback will impact future courses (and my job) #pause

Be specific on what you like and do not like #pause

Your likes/dislikes will filter into your future courses #pause

If you are not comfortable writing English, write Chinese #pause

== 

I must leave the room to let you fill out this form #pause

Please scan the QR code and complete the survey #pause

Department has suggested 10 minutes #pause

https://isw.um.edu.mo/siaweb

==
Research data labeling and collection

If you participated, come up


/*
#aslide(ag, 4)
#aslide(ag, 5)

#sslide[
    Once you understand attention, transformers are simple #pause

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
*/
/*
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
/*
