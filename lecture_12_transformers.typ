#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm
#import "@preview/touying:0.5.3": *
#import themes.university: *
#import "common_touying.typ": *
#import "@preview/cetz:0.3.1"
#import "@preview/fletcher:0.5.1" as fletcher: node, edge

// TODO: Move permutation to attention slide
// cast it as a function over sets

#set math.vec(delim: "[")
#set math.mat(delim: "[")

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
  draw_filter(0, 0, image_values)
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
Last time, we derived various forms of *attention* #pause

We started with composite memory #pause

$ f(bold(x), bold(theta)) = sum_(i=1)^T bold(theta)^top overline(bold(x))_i $ #pause

Given large enough $T$, we will eventually run out of storage space #pause

The sum is a *lossy* operation that can store a limited amount of information

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
$ bold(K) 
    = mat(bold(k)_1, bold(k)_2, dots, bold(k)_T) 
    = mat(bold(theta)_K^top bold(x)_1, bold(theta)_K^top bold(x)_2, dots, bold(theta)_K^top bold(x)_T), 
    quad bold(K) in bb(R)^(d_h times T) 
$ #pause

$ bold(q) = bold(theta)^top_Q bold(x)_q,  quad bold(theta)_Q in bb(R)^(d_x times d_h), quad bold(q) in bb(R)^(d_h) $ #pause

$ lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) 
    &= softmax(bold(q)^top bold(K)) 
    = softmax(bold(q)^top mat(bold(k)_1, bold(k)_2, dots, bold(k)_T)) 
    \ &= softmax(mat( 
    bold(q)^top bold(k)_1,
    bold(q)^top bold(k)_2,
    dots,
    bold(q)^top bold(k)_T,
    //(bold(theta)_Q^top bold(x)_q)^top (bold(theta)_K^top bold(x)_1),
    //(bold(theta)_Q^top bold(x)_q)^top (bold(theta)_K^top bold(x)_2),
)) $ #pause

We call this *dot-product attention* #pause

Then we add attention back to the composite model

==

$ f(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = sum_(i=1)^T bold(theta)^top bold(x)_i dot lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda)_i
$ #pause

We relabel $bold(theta)$ to $bold(theta)_V$

$ f(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = sum_(i=1)^T bold(theta)_#redm[$V$]^top bold(x)_i dot lambda(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)_lambda)_i $ #pause

In dot-product attention, we call $bold(theta)_V^top bold(x)_i$ the *value*
==

In *dot product self attention* we create queries for all inputs #pause

$ bold(Q) = mat(bold(q)_1, bold(q)_2, dots, bold(q)_T) = mat(bold(theta)_Q^top bold(x)_1, bold(theta)_Q^top bold(x)_2, dots, bold(theta)_Q^top bold(x)_T), quad bold(Q) in bb(R)^(T times d_h) $ #pause

==
Attention looks messy, but we can rewrite it in matrix form #pause

$
bold(Q) &= mat(bold(q)_1, bold(q)_2, dots, bold(q)_T) &&= mat(bold(theta)_Q^top bold(x)_1, bold(theta)_Q^top bold(x)_2, dots, bold(theta)_Q^top bold(x)_T) \

bold(K) &= mat(bold(k)_1, bold(k)_2, dots, bold(k)_T) &&= mat(bold(theta)_K^top bold(x)_1, bold(theta)_K^top bold(x)_2, dots, bold(theta)_K^top bold(x)_T) \

bold(V) &= mat(bold(v)_1, bold(v)_2, dots, bold(v)_T) &&= mat(bold(theta)_V^top bold(x)_1, bold(theta)_V^top bold(x)_2, dots, bold(theta)_V^top bold(x)_T) quad

$ #pause

$ "attn"(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = softmax( (bold(Q) bold(K)^top) / sqrt(d_h)) bold(V) $ #pause

With attention, we can create the *transformer* #pause

==
Why should we care about the transformer? #pause

It is arguably the most powerful neural network architecture today #pause
- AlphaFold (Nobel prize) #pause
- ChatGPT, Qwen, LLaMA, etc #pause
- DinoV2 #pause

= Going Deeper


==
Modern transformers can be very deep #pause

Very deep networks require two new training tricks #pause

We must understand these tricks before implementing the transformer #pause

*Trick 1:* Residual connections #pause

*Trick 2:* Layer normalization #pause

We will start with the *residual connection*

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

At each layer, we lose a little bit of information #pause

With enough layers, all the information in $bold(x)$ is lost! 


==
$ bold(y) = f_k ( dots f_2 ( f_1 (bold(x), bold(theta)_1), bold(theta_2)), dots, bold(theta)_k) $ #pause

*Claim:* If the input information is present in all layers of the network, then we should be able to learn the identity function $f(x) = x$ #pause

$ bold(x) = f_k ( dots f_2 ( f_1 (bold(x), bold(theta)_1), bold(theta_2)), dots, bold(theta)_k) $ #pause

*Question:* We have seen a similar model, what was it? #pause

*Answer:* Autoencoder! But it was just two functions, not $k$ functions #pause

*Question:* Do you agree or disagree with the claim? #pause

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

$ gradient_bold(theta_1) f_2( f_1(bold(x), bold(theta)_1), bold(theta)_2) &= #pause gradient_bold(theta)_1 [f_2] (f_1(bold(x), bold(theta)_1)) dot gradient_bold(theta)_1 [f_1](bold(x), bold(theta)_1) \ #pause 
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
- 99.7% of outputs $in [-3, 3]$ #pause
- 99.99% of outputs $in [-4, 4]$ #pause
- 99.9999% of outputs $in [-5, 5]$ #pause

This helps prevent vanishing and exploding gradients
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
        self.mlp = Sequential(
            Linear(d_h, d_h), LeakyReLU(), Linear(d_h, d_h))
        self.norm = nn.LayerNorm(
            d_h, elementwise_affine=False)

    def forward(self, x):
        # Residual connection and layer norm
        x = self.norm(self.attn(x) + x)
        x = self.norm(self.mlp(x) + x)
        return x
```
==
```python
class Transformer(nn.Module):
    def __init__(self):
        self.layer1 = TransformerLayer()
        self.layer2 = TransformerLayer()
        self.layer3 = TransformerLayer()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
``` #pause

*Question:* What are the input/output shapes of the transformer? #pause

*Answer:* $f: bb(R)^(T times d_x) times Theta |-> bb(R)^(T times d_y)$



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

First, define a *permutation matrix* $bold(P) in {0, 1}^(T times T)$ that reorders the inputs #pause

==

*Example 1:*

$ bold(P) = mat(
    1, 0, 0;
    0, 1, 0;
    0, 0, 1;
); quad bold(a) = vec(3, 4, 5); #pause quad bold(P a) = vec(3, 4, 5) $ #pause

*Example 2:* 

$ bold(P) = mat(
    0, 1, 0;
    1, 0, 0;
    0, 0, 1;
); quad bold(a) = vec(3, 4, 5); #pause quad bold(P a) = vec(4, 3, 5) $  $

$ 
==
#side-by-side[ $ f(bold(P)  vec(bold(x)_1, dots.v, bold(x)_n)) !=  bold(P) f(vec(bold(x)_1, dots.v, bold(x)_n) ) $ #pause][Order *does* matter (not equivariant)] #pause

#side-by-side[ $ f(bold(P) vec(bold(x)_1, dots.v, bold(x)_n)) =  bold(P) f(vec(bold(x)_1, dots.v, bold(x)_n) ) $ #pause][Order *does not* matter (equivariant)] #pause

Which is a transformer? 

==
Recall dot product self attention #pause

$ "attn"(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = softmax( (bold(Q) bold(K)^top) / sqrt(d_h)) bold(V) $ #pause

$
bold(Q) &= mat(bold(q)_1, bold(q)_2, dots, bold(q)_T) &&= mat(bold(theta)_Q^top bold(x)_1, bold(theta)_Q^top bold(x)_2, dots, bold(theta)_Q^top bold(x)_T) \

bold(K) &= mat(bold(k)_1, bold(k)_2, dots, bold(k)_T) &&= mat(bold(theta)_K^top bold(x)_1, bold(theta)_K^top bold(x)_2, dots, bold(theta)_K^top bold(x)_T) \

bold(V) &= mat(bold(v)_1, bold(v)_2, dots, bold(v)_T) &&= mat(bold(theta)_V^top bold(x)_1, bold(theta)_V^top bold(x)_2, dots, bold(theta)_V^top bold(x)_T) quad
$ 


==
Permuting the inputs reorders $bold(Q), bold(K), bold(V)$ #pause

$ "attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = softmax( ((bold(P) bold(Q)) (bold(P) bold(K))^top) / sqrt(d_h)) (bold(P) bold(V)) $ #pause

$
bold(P) bold(Q) &= mat(bold(q)_T, bold(q)_1, dots, bold(q)_2) &&= mat(bold(theta)_Q^top bold(x)_T, bold(theta)_Q^top bold(x)_1, dots, bold(theta)_Q^top bold(x)_2) \

bold(P) bold(K) &= mat(bold(k)_T, bold(k)_1, dots, bold(k)_2) &&= mat(bold(theta)_K^top bold(x)_T, bold(theta)_K^top bold(x)_1, dots, bold(theta)_K^top bold(x)_2) \

bold(P) bold(V) &= mat(bold(v)_T, bold(v)_1, dots, bold(v)_2) &&= mat(bold(theta)_V^top bold(x)_T, bold(theta)_V^top bold(x)_1, dots, bold(theta)_V^top bold(x)_2) quad
$ 


==
$ "attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = softmax( ((bold(P) bold(Q)) (bold(P) bold(K))^top) / sqrt(d_h)) (bold(P) bold(V)) $ #pause

$ "attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = softmax( bold(P) (bold(Q) (bold(P) bold(K))^top) / sqrt(d_h)) (bold(P) bold(V)) $ #pause

$ "attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = softmax( bold(P) (bold(Q) (bold(K)^top bold(P)^top)) / sqrt(d_h)) (bold(P) bold(V)) $ 

==
$ "attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = softmax( bold(P) (bold(Q) (bold(K)^top bold(P)^top)) / sqrt(d_h)) (bold(P) bold(V)) $ #pause

$ "attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = softmax( bold(P) (bold(Q) bold(K)^top)  / sqrt(d_h) bold(P)^top ) (bold(P) bold(V)) $ #pause

$ "attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = bold(P)  softmax( (bold(Q) bold(K)^top)  / sqrt(d_h) bold(P)^top ) (bold(P) bold(V)) $ 

==
$ "attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = bold(P)  softmax( (bold(Q) bold(K)^top)  / sqrt(d_h) bold(P)^top ) (bold(P) bold(V)) $ #pause

$ "attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = bold(P)  softmax( (bold(Q) bold(K)^top)  / sqrt(d_h) ) bold(P)^top (bold(P) bold(V)) $ #pause

#side-by-side[$bold(P)$ swaps row $i$ and $j$ #pause][
$bold(P)^T$ swaps row $j$ and $i$ #pause
][
    $bold(P)^top bold(P) = bold(I)$ #pause
]

$ "attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) = bold(P)  softmax( (bold(Q) bold(K)^top)  / sqrt(d_h) )  bold(V) $

==
$ 
"attn"(bold(P) vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) & = bold(P)  softmax( (bold(Q) bold(K)^top)  / sqrt(d_h) )  bold(V) \  #pause

f(bold(P) vec(bold(x)_1, dots.v, bold(x)_n)) &= bold(P) f(vec(bold(x)_1, dots.v, bold(x)_n) ) 
$ #pause

*Question:* What does this mean? #pause

*Answer:* Attention/transformer *does not* understand order. Equivariant, order *does not* matter to the transformer.

==
This makes sense, in our party attention example we never consider the order #pause

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

*Answer 1:* We can introduce forgetting (function of time) #pause

*ALiBi:* _Press, Ofir, Noah Smith, and Mike Lewis. "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." International Conference on Learning Representations._ #pause

*RoPE:* _Su, Jianlin, et al. "Roformer: Enhanced transformer with rotary position embedding." Neurocomputing._ #pause

*Answer 2:* We can modify the inputs based on their ordering #pause

We will focus on answer 2 because it is more common

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

In `torch`, this is called `nn.Embedding` #pause

Now, let us rewrite the transformer with the positional encoding 

==
So, our final transformer is 

```python
class Transformer(nn.Module):
    def __init__(self):
        self.f_pos = nn.Embedding(d_x, T)
        self.layer1 = TransformerLayer()
        self.layer2 = TransformerLayer()
    
    def forward(self, x):
        x = x + f_pos(torch.arange(x.shape[0]))
        x = self.layer1(x)
        x = self.layer2(x)
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


= Text Transformers


==
Consider a dataset of sentences 

$ vec("John likes movies", "Mary likes movies", "I like dogs") $ #pause

This is a vector of sentences, but a transformer input is $bb(R)^(T times d_x)$ #pause

What if we represent a sentence like this

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

==

#side-by-side(align: left)[
*Step 1:* Find all unique words in the dataset #pause

$ "unique"(vec("John likes movies", "Mary likes movies", "I like dogs")) 
\ = mat("John", "likes", "movies", "Mary", "I", "dogs") $ #pause
][
*Step 2:* Create a vector representation for each unique word #pause

$ vec("John", "likes", "movies", "Mary", "I", "dogs") = vec(bold(theta)_1, bold(theta)_2, bold(theta)_3, bold(theta)_4,bold(theta)_5, bold(theta)_6) $ #pause
]

*Step 3:* Replace words with vector representations

==
*Example:* Convert the sentence to vector representations #pause

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


= Image Transformers

==
In image transformers, we treat a *patch* of pixels as an $bold(x)$ #pause

$ X in [0, 1]^(3 times 16 times 16) $ #pause

#patch

==
#patch

#side-by-side(align: left)[Then, feed a sequence of patches to the transformer #pause][
    $ f(vec(bold(x)_1, dots.v, bold(x)_T), bold(theta)) $
]


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
*Question:* How do we train transformers? #pause

*Answer:* Can train just like other neural networks #pause
- Classification loss #pause
- Regression loss #pause

In practice, transformers require lots of training data #pause

Transformer scale very well -- add more data, model becomes stronger #pause

Today, dataset size limits transformers #pause

There are not enough graduate students to label training data!

==

*Question:* How can we train transformers with finite students/datasets? #pause 

*Answer:* We can use *unsupervised learning* #pause

The internet contains billions of unlabeled sentences and images #pause

Mask/hide part of the input, train the model to predict the missing part #pause

Another name for this is *generative pre-training* (GPT) #pause

We *generate* the missing data #pause

This method is *extremely* powerful

==

#side-by-side[#cimage("figures/lecture_9/masked.png")][
    _He, Kaiming, et al. "Masked autoencoders are scalable vision learners." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022._ #pause
]

==

In GPT, we optimize the following objective #pause

$ argmin_bold(theta) cal(L)(vec(bold(x)_1, dots.v, bold(x)_(T)), bold(theta)) = argmin_bold(theta) [ -log P(bold(x)_(T) | bold(x)_1, dots, bold(x)_(T - 1); bold(theta)) ] $ #pause

$ P(vec("movies", "Mary", "dogs") mid(|) "John", "likes"; bold(theta)) = vec(0.5, 0.3, 0.2) $ #pause

#side-by-side[ $bold(x)_T$ = "movies" #pause][$ P("movies" | "John", "likes"; bold(theta)) = 0.5 $] #pause

Update $bold(theta)$ so that $P("movies" | "John", "likes"; bold(theta)) = 0.6$ 

==
#side-by-side(align: left)[What about for images? #pause][ Use the same objective #pause]

$ argmin_bold(theta) cal(L)(vec(bold(x)_1, dots.v, bold(x)_(T)), bold(theta)) = argmin_bold(theta) [-log P(bold(x)_T | bold(x)_1, dots, bold(x)_(T - 1); bold(theta))] $ #pause

$ = argmin_bold(theta) [-log P("pixel_3" | "pixel_1", "pixel_2"; bold(theta))] $ #pause

//We can imagine the model outputs the mean of a normal distribution, then minimizing the square error is equivalent to minimizing the log probability

Minimizing the square error is equivalent to minimizing the (Gaussian) negative log probability

//Minimizing the square error is equivalent to matching two normal distributions #pause
//The square error represents a normal distribution over pixel values #pause

$ = argmin_bold(theta) (f(vec("pixel_1", "pixel_2"), bold(theta)) - "pixel_3" )^2 $

==

Why does this work so well? #pause

Let us see what the models learn with a GPT loss #pause

Let us start with the image transformer

==
Anyone familiar with Da Vinci's painting _The Last Supper_? #pause

#cimage("figures/lecture_12/last_supper.jpg", width: 80%)

==
#cimage("figures/lecture_12/last_supper_mask.png", width: 80%) #pause

*Question*: What concepts does the vision transformer need to understand to predict the missing pixels?

==

#place(center, cimage("figures/lecture_12/last_supper_mask.png", width: 100%))

#text(fill: white)[

Concepts the model must understand: #pause
- This is a picture of a painting #pause 
- The painting style is from the renaissance era #pause
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

This is the power of generative pre-training #pause

What about text transformers?

==
#side-by-side(align: top)[
    #cimage("figures/lecture_12/murder.jpg")
][
    This is a mystery novel #pause

    Clues, intrigue, murder, etc #pause

    "Ah, said inspector Poirot, the murderer must be $underline(#h(4em))$." #pause

    To complete the sentence, the model must understand:
]
==

#side-by-side(align: top)[
    #cimage("figures/lecture_12/murder.jpg")
][
    To complete the sentence, the model must understand: #pause
    - What a murder is #pause
    - What it means to be alive #pause
    - Emotions like anger, jealousy, betrayal, love #pause
    - Personalities of each character #pause
    - Why a human would murder another human #pause
    - How humans react to emotions #pause
    - How to tell if someone lies 
]

==

#side-by-side(align: top)[
    #cimage("figures/lecture_12/murder.jpg")
][
    To predict the murderer, the model must understand so much about humans and our society #pause

    The Books3 dataset contains 200,000 books #pause

    We train the model to predict the ending of all these books
]

==
We can apply this same concept to: #pause
- Predict missing base pairs in a strand of DNA #pause
- Predict missing audio from a song #pause
- Predict the outcome of particle collisions at the Large Hadron Supercollider #pause

All we need is a large enough dataset!

==
What if we put the model in a robot? #pause

Give the model what the robot sees and what the robot does #pause

Predict what the robot will see next #pause

Call this a *world model* because it models the structure of our world #pause

==

#cimage("figures/lecture_12/world_model.png")

==
The world model must understand: #pause
- How to control the robot #pause
- How to pet a dog #pause
- Dogs have feelings #pause
- Petting makes dogs happy #pause
- Dogs smile when happy


==
Soon, I will apply for a grant to train a world model #pause

If I win, I will need help creating a robot dataset #pause

I will need some humans to control our robots in the world #pause

If you are interested, give me your email after class

==
These transformers learn and understand the structure of our world #pause

But their understanding is trapped #pause

They can only finish sentences or complete pictures #pause

How can we use this strong understanding to help humans? #pause
- Identify pictures of cancer #pause
- Make scientific discoveries #pause
- Minimize human suffering #pause

Today, we use *reinforcement learning*

We will formally introduce reinforcement learning next lecture

= Closing Remarks

==
This is the last in-person lecture #pause

I will record a video on reinforcement learning next week #pause

I will be here from 7:00PM on December 2 for questions/discussion on reinforcement learning

==
In this course, we started from Gauss in 1795 #pause

We built up concepts until we reached the modern age #pause

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
        - Many activation functions #pause
    ][
        //- Deep neural networks #pause
        - Stochastic gradient descent #pause
        - RMSProp and Adam #pause
        - Convolutional neural networks #pause
        - Composite memory #pause
        - Recurrent neural networks #pause
        - Autoencoders #pause
        - Variational autoencoders #pause
        - Graph neural networks #pause
        - Attention and transformers #pause
        - Generative pre-training 
    ]

==
I hope you enjoyed the course! #pause

But there are many more topics to learn! #pause

Now, you have the tools to study deep learning on your own #pause

You have the tools to train neural networks for real problems #pause


==
In the first lecture, I asked everyone in this class for something #pause

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
    - Weapon guidance systems #pause
    - Discrimination #pause
  ]

  #v(2em)
  #align(center)[*Before training a model, think about whether it is good or bad for the world*]

= Course Evaluation

== 
Department instructed me to ask you for course feedback #pause

We take this feedback seriously #pause

Your feedback will impact future courses (and my job) #pause

If you like the course, please say it! #pause

Please be specific on what you like and do not like #pause

Your likes/dislikes will change your future courses 

== 

I must leave the room to let you fill out this form #pause

Please scan the QR code and complete the survey #pause

I will return in 10 minutes to see if everyone has finished #pause

https://isw.um.edu.mo/siaweb 

==
If you participated in class, come see me after class
    - Answered a question 
    - Asked a question

If you would like to collect data for a world model, come see me after class 