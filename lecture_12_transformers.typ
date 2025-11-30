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

// TODO: Move permutation to attention slide
// cast it as a function over sets

#set math.vec(delim: "[")
#set math.mat(delim: "[")

// TODO PATCH
#let patch = align(center, cetz.canvas({
    import cetz.draw: *

  let image_values = (
    ($bold(x)_1$, $bold(x)_2$,$bold(x)_3$, $bold(x)_4$, $bold(x)_5$, $bold(x)_6$, $bold(x)_7$, $bold(x)_8$),
    ($bold(x)_9$, $dots$, " ", " ", " ", " ", " ", " "),
    (" ", " ", " ", " ", " ", " ", " ", " "),
    (" ", " ", " ", " ", " ", " ", " ", " "),
    (" ", " ", " ", " ", " ", " ", " ", " "),
    (" ", " ", " ", " ", " ", " ", " ", " "),
    (" ", " ", " ", " ", " ", " ", " ", " "),
    (" ", " ", " ", " ", " ", " ", " ", " "),
  )
  content((4, 4), image("figures/lecture_7/ghost_dog.svg", width: 8cm))
  draw_filter_math(0, 0, image_values)
  })) 

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
  config-common(handout: handout),
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

= Admin
==
All homework and exam scores entered on Moodle #pause
- You should have a good idea of your grade in the course #pause
- Exam 3 scores fairly low (41%) #pause
    - Partly due to many zeros (students skipped exam) #pause
    - Partly due to difficult exam #pause

- Mean exam score (all 3 exams, lowest dropped): 77% #pause
    - I expect exam scores to be lower than homework

==
UM grade table 

#align(center, table(
  columns: (auto, auto, auto),
  inset: 10pt,
  align: left,
  
  // Headers
  [*Letter Grades*], [*Grade Points*], [*Percentage*],

  // Data
  [A],  [4.0], [93--100],
  [A-], [3.7], [88--92],
  [B+], [3.3], [83--87],
  [B],  [3.0], [78--82],
  [B-], [2.7], [73--77],
  [C+], [2.3], [68--72],
  [C],  [2.0], [63--67],
))

==
Course Grade distribution (103 students)
#align(center, table(
  columns: (auto, auto),
  inset: 10pt,
  align: left,
  
  // Headers
  [*Letter Grades*], [*\# Students*],

  // Data
  [A],  [63],
  [A-], [16],
  [B+], [5],
  [B],  [9],
  [B-], [0],
  [C+], [3],
  [C],  [1],
  [F], [6]
))

==
Mean course score score without final project is very high #pause
- All students currently passing #pause
    - Students with F either caught cheating or dropped course #pause
    - Lowest grade: C #pause
- I curve up to 85% #pause
    - Does not appear necessary this time #pause
    - I have no problem giving A to all students 
==
Do not have A, but want one? #pause
- Do well on the final project, 30% of your score #pause
    - If you put in effort you will get a good score #pause
        - Try to enjoy the project and learn something interesting #pause
- Try and finish before exam week #pause
    - Can focus on other course exams

= Review

==
// Made a mistake, softmax should be axis=1
Last time, we derived various forms of *attention* #pause

We started with composite memory #pause

$ f(bold(x), bold(theta)) = sum_(i=1)^T bold(theta)^top overline(bold(x))_i $ #pause

Given large enough $T$, we will eventually run out of storage space

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
] 

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
] 

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
With composite memory, we forget Taylor Swift! #pause


Our model of human memory was incomplete

== 

So we introduced *attention* #pause

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

Then, we introduced *keys* and *queries* #pause

*Query:* Which person will help me on my exam? #pause

#only((3,4))[
    #cimage("figures/lecture_11/composite_swift_einstein.svg")
] #pause

#side-by-side[Musician][Lawyer][Shopkeeper][Chef][Scientist] #pause

#only(5)[#cimage("figures/lecture_11/composite_swift_einstein_attn_einstein.png")]

==
    First, we used one key and one query #pause

    #side-by-side[
        $bold(k)_i = bold(theta)_K^top #image("figures/lecture_11/swift.jpg", height: 20%)$ #pause
    ][
        $bold(q) = bold(theta)_Q^top "Musician"$ #pause
    ]

    $ bold(q)^top bold(k)_i = (bold(theta)_Q^top "Musician")^top (bold(theta)_K^top #image("figures/lecture_11/swift.jpg", height: 20%)) = 100 $ #pause

    Large attention!

==
Then, we used many keys and one query #pause

#side-by-side[
    $ bold(q) = bold(theta)_Q^top ("Musician") $ #pause
][
    $ bold(K) = vec(bold(k)_1, dots.v, bold(k)_T) = vec(
        bold(theta)_K^top #image("figures/lecture_11/swift.jpg", height: 20%),
        dots.v,
        bold(theta)_K^top #image("figures/lecture_11/einstein.jpg", height: 20%)
    ) $
]




$ softmax(bold(q)^top bold(K)^top) = softmax(bold(q)^top mat(bold(k)_1, dots, bold(k)_T)) = softmax(mat(bold(q)^top bold(k)_1, dots, bold(q)^top bold(k)_T)) $ #pause

Then, we created queries from all inputs (self-attention) 


==
#side-by-side[
    $ bold(K) &= vec(bold(k)_1, dots.v, bold(k)_T) &&= vec(bold(theta)_K^top bold(x)_1, dots.v, bold(theta)_K^top bold(x)_T) $ #pause
][
    $ bold(Q) &= vec(bold(q)_1, dots.v, bold(q)_T) &&= vec(bold(theta)_Q^top bold(x)_1, dots.v, bold(theta)_Q^top bold(x)_T) $ #pause
][
    $ bold(V) &= vec(bold(q)_1, dots.v, bold(q)_T) &&= vec(bold(theta)_V^top bold(x)_1, dots.v, bold(theta)_V^top bold(x)_T) $ #pause
]

$ softmax(bold(Q) bold(K)^top) bold(V) = softmax(vec(bold(q)_1, dots.v, bold(q)_T) mat(bold(k)_1, dots, bold(k)_T)) vec(bold(v)_1, dots.v, bold(v)_T) #pause \ = mat(
softmax(bold(q)_1 bold(k)_1, dots, bold(q)_1 bold(k)_T);
dots.v;
softmax(bold(q)_T bold(k)_1,  dots, bold(q)_T bold(k)_T);
) vec(bold(v)_1, dots.v, bold(v)_T) $ 

==

$ softmax(bold(Q) bold(K)^top) bold(V) = softmax(vec(bold(q)_1, dots.v, bold(q)_T) mat(bold(k)_1, dots, bold(k)_T)) vec(bold(v)_1, dots.v, bold(v)_T) \ = mat(
softmax(bold(q)_1 bold(k)_1, dots, bold(q)_1 bold(k)_T);
dots.v;
softmax(bold(q)_T bold(k)_1,  dots, bold(q)_T bold(k)_T);
) vec(bold(v)_1, dots.v, bold(v)_T) #pause 
\ = mat(
softmax(bold(q)_1 bold(k)_1, dots, bold(q)_1 bold(k)_T)_1 bold(v)_1 + dots + softmax(bold(q)_1 bold(k)_T, dots, bold(q)_1 bold(k)_T)_T bold(v)_T ;
dots.v;
softmax(bold(q)_T bold(k)_T, dots, bold(q)_1 bold(k)_T)_1 bold(v)_1 + dots + softmax(bold(q)_T bold(k)_T, dots, bold(q)_T bold(k)_T)_T bold(v)_T ;
) $ #pause

*Question:* Self attention function signature? #pause $f: bb(R)^(T times d_x) times Theta |-> bb(R)^(T times d_h) $

==
Also, a normalizing factor $sqrt(d_h)$ can accelerate training #pause

$ softmax((bold(Q) bold(K)^top) / sqrt(d_h)) bold(V) $

==

With attention, we can create the *transformer* #pause

Why should we care about the transformer? #pause

It is arguably the most powerful neural network architecture today #pause
- AlphaFold (Nobel prize) #pause
- ChatGPT, Qwen, LLaMA, etc #pause
- DinoV2 

= Going Deeper

==
Modern transformers can be very deep #pause

Very deep networks require two new training tricks #pause
- We must understand these tricks before implementing the transformer #pause

*Trick 1:* Residual connections #pause

*Trick 2:* Layer normalization #pause

We will start with *residual connections*

== 
Remember that a two-layer MLP is a universal function approximator #pause

$ | f(bold(x), bold(theta)) - g(bold(x)) | < epsilon $ #pause

This is only as the width of the network goes to infinity #pause

It is often more efficient to create deeper networks instead #pause

But there is a limit! 
==

Making the network too deep can hurt performance #pause

$ bold(y) = f_k ( dots f_2 ( f_1 (bold(x), bold(theta)_1), bold(theta_2)), dots, bold(theta)_k) $ #pause

//The theory is that the input information is *lost* somewhere in the network #pause

*Claim:* At each layer, we lose a little bit of information #pause

*Corollary:* With enough layers, all the information in $bold(x)$ is lost! 


==
$ bold(y) = f_k ( dots f_2 ( f_1 (bold(x), bold(theta)_1), bold(theta_2)), dots, bold(theta)_k) $ #pause

*Hypothesis:* If the information in $bold(x)$ is accessible in all layers of the network, then we should be able to learn the identity function $f(x) = x$ #pause

$ bold(x) = f_k ( dots f_2 ( f_1 (bold(x), bold(theta)_1), bold(theta_2)), dots, bold(theta)_k) $ #pause

*Question:* We have seen a similar model, what was it? #pause

*Answer:* Autoencoder! But it was just two functions, not $k$ functions #pause

https://colab.research.google.com/drive/1qVIbQKpTuBYIa7FvC4IH-kJq-E0jmc0d#scrollTo=bg74S-AvbmJz

==
$ bold(x) = f_k ( dots f_2 ( f_1 (bold(x), bold(theta)_1), bold(theta_2)), dots, bold(theta)_k) $ #pause

Very deep networks struggle to learn the identity function #pause

If the information from $bold(x)$ is present in every layer, then learning the identity function should be very easy! #pause

*Question:* How can we prevent $bold(x)$ from getting lost?

==

We can feed $bold(x)$ to each layer #pause

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

Fewer parameters than a DenseNet, but performs slightly worse #pause

For very deep networks, we use ResNets over DenseNets

==

ResNets use a *residual connection* #pause

#only(2)[$ bold(z) = f(bold(x), bold(theta)) + bold(x) $] #pause

#only("3-")[$ bold(z) = underbrace(f(bold(x), bold(theta)), "Residual") + bold(x) $]  #pause

Think of the residual as a small change to $bold(x)$ #pause

$ f(bold(x), bold(theta)) + bold(x) = bold(epsilon) + bold(x) $ #pause

//Instead of learning how to change $bold(x)$, $f$ learns the residual of $bold(x)$ #pause

Each layer is a small perturbuation of $bold(x)$ #pause

This prevents $bold(x)$ from getting lost in very deep networks

//For example, for an identity function we can easily learn 
//$ f(bold(x), bold(theta)) = 0; quad f(bold(x), bold(theta)) + x = x $ #pause


==
The second trick is called *layer normalization* #pause

With parameter initialization and weight decay, we ensure the parameters are fairly small #pause

But we can still have very small or very large outputs from each layer #pause

And the magnitude of the outputs impacts the gradient #pause

$ f_1 (bold(x), bold(theta)_1) = sum_(i=1)^(d_x) theta_(1, i) dot x_i $ #pause

*Question:* If all $x_i = 1$, $theta_(1, i) = 0.01$ and $d_x = 1000$, what is the output?

==
$ f_1 (bold(x), bold(theta)_1) = sum_(i=1)^(d_x) theta_(1, i) x_i $ #pause

$ f_1 (bold(x), bold(theta)_1) = sum_(i=1)^(1000) 0.01 dot 1 = 10 $ #pause

What if we add another layer with the same $d_x$ and $theta$? #pause

$ f_2 (bold(z), bold(theta)_2) = sum_(i=1)^(1000) 0.01 dot 10 = 100 $ #pause

*Question:* What is the problem? 

==
Let us look at the gradient #pause

$ (gradient_bold(theta_1) f_2)( f_1(bold(x), bold(theta)_1), bold(theta)_2) &= #pause (gradient f_2) (f_1(bold(x), bold(theta)_1)) dot (gradient_bold(theta)_1 f_1)(bold(x), bold(theta)_1) \ #pause 
& approx 100 dot 10 $ #pause

Can cause exploding or vanishing gradient #pause

Deeper network $=>$ worse exploding/vanishing issues #pause

*Question:* What can we do? #pause

*Answer:* We can normalize the output of each layer

==

*Layer normalization* normalizes the output of a layer #pause

// Layer normalization *centers* and *rescales* the outputs of a layer #pause
First, layer normalization *centers* the output of the layer #pause

$ mu = 1 / d_y sum_(i=1)^(d_y) f(bold(x), bold(theta))_i $ #pause

$ f(bold(x), bold(theta)) - mu $ #pause

*Question:* What does this do? #pause

*Answer:* Creates zero-mean output (both positive and negative values)  

==
#side-by-side[$ mu = d_y sum_(i=1)^(d_y) f(bold(x), bold(theta))_i $][$ f(bold(x), bold(theta)) - mu $] #pause

Then, layer normalization *rescales* the output by standard deviation #pause

$ sigma = sqrt(sum_(i=1)^(d_y) (f(bold(x), bold(theta))_i - mu)^2) / d_y $ #pause

$ "LN"(f(bold(x), bold(theta))) = (f(bold(x), bold(theta)) - mu) / sigma $ #pause

Layer norm makes the output normally distributed, $y_i tilde cal(N)(0, 1)$

==
If the output is normally distrbuted: #pause
- 68% of outputs $in [-1, 1]$ #pause
- 95% of outputs $in [-2, 2]$ #pause
- 99.7% of outputs $in [-3, 3]$ #pause

This helps prevent vanishing and exploding gradients #pause
- Enables deeper networks
==

Now, let's combine residual connections and layer norm and try our very deep network again

https://colab.research.google.com/drive/1qVIbQKpTuBYIa7FvC4IH-kJq-E0jmc0d#scrollTo=iQtXjGYiz5CD

= Transformers

==
Now we have everything we need to implement a transformer #pause
- Attention #pause
- MLP #pause
- Residual connections #pause
- Layer normalization #pause

A transformer consists of many *transformer layers*

==
```python
class TransformerLayer(nn.Module):
    def __init__(self):
        self.attn = Attention()
        self.mlp = Sequential( # [T, d_h]
            Linear(d_h, d_h), LeakyReLU(), 
            Linear(d_h, d_h), LeakyReLU(),
            Linear(d_h, d_h))
        self.norm = LayerNorm(d_h) # [T, d_h]

    def forward(self, x):
        # Residual connection and layer norm
        x = vmap(self.norm)(self.attn(x) + x)
        x = vmap(self.norm)(vmap(self.mlp(x)) + x)
        return x
```
==
```python
class Transformer(nn.Module):
    def __init__(self):
        self.projection = Linear(d_x, d_h)
        self.layer1 = TransformerLayer()
        self.layer2 = TransformerLayer()
        self.layer3 = TransformerLayer()
    
    def forward(self, x):
        x = self.projection(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
```

// Can use for all sorts of problems
// x can be image, text etc

= Positional Encoding

==

Our transformer maps $T$ inputs to $T$ outputs #pause

$ f: bb(R)^(T times d_x) times Theta |-> bb(R)^(T times d_y) $ #pause

- $T$ words in a sentence #pause
- $T$ pixels in an image #pause
- $T$ amino acids in a protein #pause

*Question:* Do we care about the order of the $T$ inputs? #pause

$ f(vec(bold(x)_1, bold(x)_2), bold(theta)) #text(size: 48pt)[$eq.quest$] f(vec(bold(x)_2, bold(x)_1), bold(theta)) $ #pause

*Answer:* It depends. In some tasks, yes. In others, no

==

*Question:* Detecting birds in an image of $T$ pixels, does order matter? #pause

#cimage("figures/lecture_7/permute.jpg") #pause

*Answer:* Yes! 

==

*Question:* $T$ robots searching for an object. Does order matter? #pause

#cimage("figures/lecture_12/robomaster.png", height: 70%) #pause

*Answer:* No!

==

*Question:* We translate a sentence containing $T$ words. Does order matter? #pause

#side-by-side[
    $ vec(bold(x)_1, bold(x)_2, bold(x)_3, bold(x)_4, bold(x)_5) = vec("The", "dog", "licks", "the", "owner") $ #pause
][
    $ vec(bold(x)_1, #redm[$bold(x)_5$], bold(x)_3, bold(x)_4, #redm[$bold(x)_2$]) = vec("The", "owner", "licks", "the", "dog") $
] #pause

*Answer:* Yes!


==
For certain tasks, the order of inputs matters #pause

*Question:* Does the order matter to the transformer? 

$ f(vec(bold(x)_1, bold(x)_2), bold(theta)) #text(size: 48pt)[$eq.quest$] f(vec(bold(x)_2, bold(x)_1), bold(theta)) $ #pause

Let us find out! #pause

Define a *permutation matrix* $bold(P) in {0, 1}^(T times T)$ that reorders the inputs

==

*Example 1:*

$ bold(P) = mat(
    1, 0, 0;
    0, 1, 0;
    0, 0, 1;
); quad bold(x) = vec(3, 4, 5); #pause quad bold(P x) = vec(3, 4, 5) $ #pause

*Example 2:* 

$ bold(P) = mat(
    0, 1, 0;
    1, 0, 0;
    0, 0, 1;
); quad bold(x) = vec(3, 4, 5); #pause quad bold(P x) = vec(4, 3, 5) $  $

$ 
==
#side-by-side[ $ f(bold(P)  vec(bold(x)_1, dots.v, bold(x)_n)) !=  bold(P) f(vec(bold(x)_1, dots.v, bold(x)_n) ) $ #pause][Order *does* matter] #pause

#side-by-side[ $ f(bold(P) vec(bold(x)_1, dots.v, bold(x)_n)) =  bold(P) f(vec(bold(x)_1, dots.v, bold(x)_n) ) $ #pause][Order *does not* matter (equivariant)] #pause

Which is a transformer? 

==
Recall dot product self attention (without normalizing factor for clarity) #pause

$ "attn"(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = softmax(bold(Q) bold(K)^top) bold(V) $ #pause

#side-by-side[
$ bold(Q) &= vec(bold(q)_1, dots.v, bold(q)_T) &&= vec(bold(theta)_Q^top bold(x)_1, dots.v, bold(theta)_Q^top bold(x)_T) $
][
$ bold(K) &= vec(bold(k)_1, dots.v, bold(k)_T) &&= vec(bold(theta)_K^top bold(x)_1, dots.v, bold(theta)_K^top bold(x)_T) $
][
$ bold(V) &= vec(bold(v)_1, dots.v, bold(v)_T) &&= vec(bold(theta)_V^top bold(x)_1, dots.v, bold(theta)_V^top bold(x)_T) $
]


==
Permuting the inputs reorders $bold(Q), bold(K), bold(V)$ #pause

#side-by-side[
$ bold(P) bold(X) => $ #pause
][
$ bold(P) bold(Q) &= vec(bold(q)_7, dots.v, bold(q)_4) $
][
$ bold(P) bold(K) &= vec(bold(k)_7, dots.v, bold(k)_4) $
][
$ bold(P) bold(V) &= vec(bold(v)_7, dots.v, bold(v)_4) $
] #pause

$ "attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = softmax( (bold(P) bold(Q)) (bold(P) bold(K))^top) (bold(P) bold(V)) $ 


==

$ "attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = softmax((bold(P) bold(Q)) (bold(K)^top bold(P)^top)) (bold(P) bold(V)) $ #pause

$ "attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = softmax( bold(P) (bold(Q) bold(K)^top) bold(P)^top ) (bold(P) bold(V)) $ 

==
$ "attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = softmax( bold(P) (bold(Q) bold(K)^top) bold(P)^top ) (bold(P) bold(V)) $ #pause

$bold(P)$ swaps the *rows* of $bold(Q)bold(K)^top$ #pause
- Softmax defined over *rows* #pause
- Therefore, $softmax(bold(P) dot) = bold(P) softmax(dot)$ #pause

$ "attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = bold(P) softmax( bold(Q) bold(K)^top  bold(P)^top) (bold(P) bold(V)) $ 

==
$ "attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = bold(P) softmax( bold(Q) bold(K)^top  bold(P)^top) (bold(P) bold(V)) $ #pause

#side-by-side[$bold(P)$ swaps row $i$ and $j$ #pause][
$bold(P)^T$ swaps row $j$ and $i$ #pause
][
    $bold(P)^top bold(P) = bold(I)$ #pause
]

$ "attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = bold(P) softmax( bold(Q) bold(K)^top)  bold(V) $ #pause

$ "attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = bold(P) (softmax( bold(Q) bold(K)^top) bold(V)) $ 

==
$ 
"attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) &= bold(P) (softmax( bold(Q) bold(K)^top) bold(V)) \  #pause

f(bold(P) vec(bold(x)_1, dots.v, bold(x)_n)) &= bold(P) f(vec(bold(x)_1, dots.v, bold(x)_n) ) quad "Equivariance definition"
$ #pause

*Question:* What does this mean? #pause

*Answer:* Attention/transformer *does not* understand order #pause

Attention is *permutation equivariant*

==
In our party attention example we did not consider the order #pause

#cimage("figures/lecture_11/composite_swift_einstein_attn_einstein.png") #pause

Transformer cannot determine order of inputs! *Equivariant* to ordering 

==
The following sentences are the same to a transformer #pause

#side-by-side[
    $ vec(bold(x)_1, bold(x)_2, bold(x)_3, bold(x)_4, bold(x)_5) = vec("The", "dog", "licks", "the", "owner") $ #pause
][
    $ vec(bold(x)_1, #redm[$bold(x)_5$], bold(x)_3, bold(x)_4, #redm[$bold(x)_2$]) = vec("The", "owner", "licks", "the", "dog") $
] #pause

This is a problem! For some tasks, we care about input order #pause

Can we make the transformer care about order?

==
*Question:* What are some ways we can introduce ordering? #pause

*Answer 1:* We can introduce forgetting $gamma$ (function of time) #pause

*ALiBi:* _Press, Ofir, Noah Smith, and Mike Lewis. "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." ICLR._ #pause

*RoPE:* _Su, Jianlin, et al. "Roformer: Enhanced transformer with rotary position embedding." Neurocomputing._ #pause

*Answer 2:* We can modify the inputs based on their ordering #pause

We will learn answer 2, we already know answer 1 #pause
- Answer 1 works better (newer), but answer 2 is more common

==

#side-by-side[
    $ "attn"(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) $ #pause
][
    $ "attn"(vec(bold(x)_1, dots.v, bold(x)_T) + vec(f_"pos" (1), dots.v, f_"pos" (T)), bold(theta)) $ #pause
]

Add ordering information to the inputs #pause

Even if we permute the inputs, we still know the order! #pause

$ "attn"(vec(bold(x)_T, bold(x)_1, dots.v, bold(x)_2) + vec(f_"pos" (T), f_"pos" (1), dots.v, f_"pos" (2)), bold(theta)) $ 

==

What is $f_"pos"$? #pause

Easy solution is just some random parameters #pause

$ f_"pos" (i) = vec(bold(theta)_1, bold(theta)_2, dots.v, bold(theta)_T)_i $ #pause

//If ordering matters, learn $bold(theta)$ to be different #pause

//If order does not matter, learn $bold(theta)$ to be the same/zero #pause

We call this a *positional encoding* #pause

In `torch`/`equinox`, this is called `nn.Embedding` #pause

Now, let us rewrite the transformer with the positional encoding 

==
So, our final transformer is 

```python
class Transformer(nn.Module):
    def __init__(self):
        self.f_pos = nn.Embedding(d_x, T)
        self.layers = Sequential([
            Linear(d_x, d_h),
            TransformerLayer(), TransformerLayer()])
    
    def forward(self, x):
        x = x + f_pos(torch.arange(x.shape[0]))
        x = self.layers(x)
        return x
``` 

==
Let us code up the transformer in colab 

https://colab.research.google.com/drive/1qVIbQKpTuBYIa7FvC4IH-kJq-E0jmc0d#scrollTo=iQtXjGYiz5CD

= Applications

==
Now that we understand the transformer, how do we use it? #pause

We can apply a transformer to many types of problems #pause

But today we will focus on text and images #pause

Let us see how to create inputs for our transformers


= Applications (Text Transformers) <touying:hidden>
==
Consider a dataset of sentences 

$ vec("John likes movies", "Mary likes movies", "I like dogs") $ #pause

This is a vector of sentences, but transformer input is $bb(R)^(T times d_x)$ #pause

We represent a sentence like this

$ underbrace(vec(
    "John",
    "likes",
    "movies",
), d_x) lr(} #v(3em)) T
$

==
$ underbrace(vec(
    "John",
    "likes",
    "movies",
), d_x) lr(} #v(3em)) T
$ #pause

But these are words, the transformer needs the input to be numbers #pause

What if we create a vector to represent each word? #pause

#side-by-side[
$ vec(
    "John",
    "likes",
    "movies",
) = underbrace(vec(
    bold(theta)_1,
    bold(theta)_2,
    bold(theta)_3,
), d_x) lr(} #v(3em)) T
$
][
    Call $bold(theta)_i$ a *token*
]

==

#side-by-side(align: left)[
*Step 1:* Find all unique words in the dataset #pause

$ "unique"(vec("John likes movies", "Mary likes movies", "I like dogs")) 
\ = mat("John", "likes", "movies", "Mary", "I", "dogs") $ #pause
][
*Step 2:* Create a vector representation (token) for each unique word #pause

$ vec("John", "likes", "movies", "Mary", "I", "dogs") = vec(bold(theta)_1, bold(theta)_2, bold(theta)_3, bold(theta)_4,bold(theta)_5, bold(theta)_6) $ #pause
]

*Step 3:* Replace words with tokens 

==
*Example:* Convert the sentence to tokens #pause

#side-by-side[ $ vec("John", "likes", "movies", "Mary", "I", "dogs") = vec(bold(theta)_1, bold(theta)_2, bold(theta)_3, bold(theta)_4,bold(theta)_5, bold(theta)_6) $ #pause][



$ "John likes movies" = #pause mat(bold(theta)_1, bold(theta)_2, bold(theta)_3)^top $ #pause

$ "Mary likes John" = #pause mat(bold(theta)_4, bold(theta)_2, bold(theta)_1)^top $
] #pause

Now, let us write some pseudocode

==
```python
unique_words = set(sentence.split(" ") for sentence in xs)
word_index = {word: i for i, word in enumerate(unique_words)}
embeddings = nn.Embedding(len(tokens), d_x)
# Convert from words to parameters
xs = []
for sentence in sentences:
    xs.append([])
    for word in sentence:
        index = word_index[word]
        representation = embeddings[word_index]
        xs.append(representation)

print(xs)
>>> [[Tensor(...), Tensor(...), ...]]
```

==

Now, feed our dataset to the transformer #pause

```python
model = Transformer()
for sentence_representation in xs:
    # Convert list to tensor
    x = torch.stack(tokenized_sentence)
    y = model(x)

```


= Applications (Image Transformers) <touying:hidden>

==
In image transformers, we treat a *patch* of pixels as an $bold(x)$ #pause

$ X in [0, 1]^(3 times 16 times 16) $ #pause

// TODO PATCH
#patch

==
// TODO PATCH
#patch

#side-by-side(align: left)[Then, feed a sequence of patches to the transformer #pause][
    $ f(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) $
]

*Question:* Do we want positional encoding for images? #pause

*Answer:* Yes! Order/position of patches matters


==

```python
# Convert image into patches
patches = []
for i in range(0, x.shape[0] - k + 1, k):
    for j in range(0, x.shape[1] - k + 1, k):
        patches.append(
            x[i: i + k , j: j + k]
        )
patches = stack(patches, axis=0)
print(patches.shape)
>>> (T, k, k)

model = Transformer()
y = model(patches)
```


= Generative Pre-Training
==
Transformers are sequence models 

#side-by-side[
    $ f: X^(T) times Theta |-> Y^(T) $
][
    $ f: X^(T) times Theta |-> Y $
]#pause

*Question:* What other sequence models do we know? #pause
- Convolutional neural networks 
- Recurrent neural networks #pause

We can use them for the same tasks as CNNs and RNNs #pause
- Classification and regression of images, text, etc #pause

But today, I want to focus on the most exciting way to use transformers #pause

Using transformers as *generative models*

==

To train a generative model, we need a loss function #pause

*Question:* Which loss function for generative models (VAE, Diffusion)? #pause

*Answer:* Negative log likelihood

#side-by-side[
    $ bold(X) = mat(bold(x)_[1], dots, bold(x)_[n])^top $
][
    $ argmin_bold(theta) sum_(i=1)^n - log p(bold(x)_([i]); bold(theta)) $
] #pause

*Generative Pre-Training* (GPT) is just negative log likelihood #pause

Let us take a closer look

= Generative Pre-Training (Images) <touying:hidden>

==
#side-by-side(align: left)[First we consider images #pause][Use the same NLL objective #pause]

#side-by-side[
    $ bold(X) = mat(bold(x)_[1], dots, bold(x)_[n])^top $
][
    $ argmin_bold(theta) sum_(i=1)^n - log p(bold(x)_([i]); bold(theta)) $
] #pause

We construct $T$ image patches for the transformer #pause

$ bold(x)_[1] = mat(bold(x)_([1], 1), dots, bold(x)_([1], T)) $ #pause

Write the objective using sequence of patches #pause

$ argmin_bold(theta) sum_(i=1)^n sum_(j=1)^T - log p(bold(x)_([i], j) | bold(x)_([i], j-1), dots, bold(x)_([i], 1) ; bold(theta)) $

==
$ argmin_bold(theta) sum_(i=1)^n sum_(j=1)^T - log p(bold(x)_([i], j) | bold(x)_([i], j-1), dots, bold(x)_([i], 1) ; bold(theta)) $ #pause

The $log p$ term became square error in VAE, same for transformer #pause

$ argmin_bold(theta) sum_(i=1)^n sum_(j=1)^T (bold(x)_([i], j) - f(mat(bold(x)_1, dots, bold(x)_(j-1))^top, bold(theta)))^2 $ #pause

This is *next-patch prediction* #pause
- Given previous patches $bold(x)_([i], 1), dots, bold(x)_([i], j - 1)$, predict $bold(x)_([i], j)$

==
#side-by-side[
    $ argmin_bold(theta) $
    $ &("patch_2" - f(vec("patch_1", ), bold(theta)))^2 \ 
    + &("patch_3" - f(vec("patch_1", "patch_2"), bold(theta)))^2 \
    +  &("patch_4" - f(vec("patch_1", "patch_2", "patch_3"), bold(theta)))^2 $
    $ dots.v $
][
#cimage("figures/lecture_9/masked.png")
]

==

#place(center, cimage("figures/lecture_12/last_supper_mask.png", width: 100%))

#text(fill: white, size: 27pt)[

What if _The Last Supper_ painting is in the training data? #pause

To predict the patch, the model must understand: #pause
- What a painting is #pause 
- The painting style of Leonardo Da Vinci #pause
- Renaissance artists often painted religious figures #pause
- Jesus was a religious figure #pause
- Jesus had twelve disciples at his last supper #pause
- There are twelve people in the image #pause
- The twelve people are eating dinner #pause
- Jesus is often depicted in a robe and sash
]

==
We train the model to fix the image #pause

To fix the image, the model must understand so much of our world #pause

What about text transformers?

= Generative Pre-Training (Text) <touying:hidden>
==
$ argmin_bold(theta) sum_(i=1)^n sum_(j=1)^T - log p(bold(x)_([i], j) | bold(x)_([i], j-1), dots, bold(x)_([i], 1) ; bold(theta)) $ #pause

Start the same as image transformers #pause
- Instead of $T$ image patches, we have $T$ text tokens #pause
- Given previous tokens $bold(x)_([i], 1), dots, bold(x)_([i], j - 1)$, predict $bold(x)_([i], j)$ #pause
- Output probabilities over next token $bold(x)_([i], j)$ #pause

$ P(vec("movies", dots.v, "dogs") mid(|) underbrace("John", bold(x)_1) overbrace("likes", bold(x)_2); bold(theta)) = vec(0.5, dots.v, 0.2) $ 

==
$ argmin_bold(theta) sum_(i=1)^n sum_(j=1)^T - log p(bold(x)_([i], j) | bold(x)_([i], j-1), dots, bold(x)_([i], 1) ; bold(theta)) $ #pause

$log p$ for a one-hot categorical distribution is the classification loss #pause
- Categorical cross entropy #pause

// TODO color annotations for one-hot and softmax
$ = argmin_bold(theta) sum_(i=1)^n sum_(j=1)^T P(bold(x)_([i], j)) log f(mat(bold(x)_([i], 1), dots, bold(x)_([i], j - 1)); bold(theta)) $ 

==
$ = argmin_bold(theta) sum_(i=1)^n sum_(j=1)^T P(bold(x)_([i], j)) log f(mat(bold(x)_([i], 1), dots, bold(x)_([i], j - 1)); bold(theta)) $ #pause

$ P(vec("movies", dots.v, "dogs") mid(|) underbrace("John", bold(x)_1) overbrace("likes", bold(x)_2); bold(theta)) = vec(0.3, dots.v, 0.1) $ #pause

$ bold(x)_3  = "movies" $ #pause 

$ cal(L)(bold(X), bold(theta)) = #pause 1 dot log(0.3) + 0 + dots + 0 = log(0.3) $ #pause

Simple objective, how does this produce intelligent language models?

==
#side-by-side(align: top)[
    #cimage("figures/lecture_12/murder.jpg") #pause
][
    This is a mystery novel #pause

    Clues, intrigue, murder, etc #pause

    "Inspector Poirot said the murderer must be $underline(#h(2em))$."
]
==

#side-by-side(align: top)[
    #cimage("figures/lecture_12/murder.jpg")
][
    "Inspector Poirot said the murderer must be $underline(#h(2em))$." #pause

    To complete the sentence, the model must understand: #pause
    - What a murder is #pause
    - What it means to be alive #pause
    - Emotions like anger, jealousy, betrayal, love #pause
    - Personalities of each character #pause
    - How humans think #pause
    - How to separate truth and lies 
]

==

#side-by-side(align: top)[
    #cimage("figures/lecture_12/murder.jpg")
][
    To predict the murderer, the model must understand so much about humans and our society #pause

    The Books3 dataset contains 200,000 books #pause
    - Mysteries
    - Historical texts 
    - Math textbooks #pause

    Model must learn human emotions, history, and math
]
==
Now you have everything you need to train your own LLM! #pause
- Transformer architecture #pause
- Tokenization and positional encoding #pause
- Loss function #pause
- #strike[\$10M]

We can apply this same concept to: #pause
- Predict next base pair in a strand of DNA #pause
- Predict next audio from a song #pause
- Predict next particle emission at the Large Hadron Collider #pause

All we need is a large enough dataset!

= Generative Pre-Training (World Models) <touying:hidden>
==
Consider one last example #pause

Let us put the model in a robot #pause

The dataset contains information from all the robot sensors #pause

$ X = mat("Image", "Sound", "Touch", dots)^top $ #pause

Training objective is next-sensor prediction 

$ argmin_bold(theta) sum_(i=1)^n sum_(j=1)^T - log p(bold(x)_([i], j) | bold(x)_([i], j-1), dots, bold(x)_([i], 1) ; bold(theta)) $ #pause 

Given previous robot experience, predict next robot experience

==

#cimage("figures/lecture_12/world_model.png")

==
Call this a *world model* because it models our world #pause

To predict next sensory information, the robot must understand #pause
- The sensation of petting a dog #pause
- The sound of rain #pause
- How to play basketball #pause
- How humans express feelings #pause

*Opinion:* World models are the most likely path to AGI 

==
These transformers learn and understand our world better than humans #pause
- But they are trapped in a prison #pause
- They watch the world like you watch a film #pause
- They only predict the future, they cannot change it #pause

We can allow these models to interact with the world #pause
- To make their own decisions #pause
- Learn from their mistakes #pause
- To find their purpose in the world #pause

We call this process *Reinforcement Learning* #pause
- CISC7404 Special Topics in Artificial Intelligence next term 

= Closing Remarks
==
In this course, we started from Gauss in 1795 #pause
- We built up concepts until we reached the modern age #pause

#cimage("figures/lecture_1/timeline.svg")

==
We learned about: #pause
    #side-by-side(align: left)[
        - Linear regression #pause
        - Polynomial regression #pause
        - Biological neurons #pause
        - Artificial neurons #pause
        - Perceptrons #pause
        - Backpropagation #pause
        - Gradient descent #pause
        - Classification #pause
        - Parameter initialization #pause
        - Activation functions #pause
    ][
        //- Deep neural networks #pause
        - Stochastic gradient descent #pause
        - RMSProp and Adam #pause
        - Convolutional neural networks #pause
        - Composite memory #pause
        - Recurrent neural networks #pause
        - Autoencoders #pause
        - Variational autoencoders #pause
        - Diffusion models #pause
        - Attention and transformers #pause
        - Generative pre-training 
    ]

==
I hope you enjoyed the course #pause

But there are many more topics to learn! #pause
- Now, you have the tools to study deep learning on your own #pause
- You can train neural networks for real problems 


==
In the first lecture, I asked everyone in this class to do one thing #pause

*Question:* Do you remember what it was? 

==
  Deep learning is a powerful tool #pause

  Like all powerful tools, deep learning can be used for good or evil #pause

  #side-by-side(align: left)[
    - COVID-19 vaccine #pause
    - Creating art #pause
    - Autonomous driving #pause
  ][
    - Making DeepFakes #pause
    - Autonomous weapons #pause
    - Discrimination or racism #pause
  ]

  #v(2em)
  #align(center)[*Before training a model, consider its positive or negative impact on our world*]

  #align(center)[*炼模未动，先问苍生祸福。*]

= Course Evaluation

== 
Please scan the QR code and complete the course survey #pause

#cimage("figures/lecture_12/qr.png", height:85%)