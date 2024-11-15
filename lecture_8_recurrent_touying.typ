#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm
#import "@preview/touying:0.5.3": *
#import themes.university: *
#import "common_touying.typ": *
#import "@preview/cetz:0.2.2"
#import "@preview/fletcher:0.5.1" as fletcher: node, edge
#import "@preview/ctheorems:1.1.2": *
#import "@preview/numbly:0.1.0": numbly

#let cimage(..args) = { 
  align(center + horizon, image(..args))
}
#let redm(x) = {
  text(fill: color.red, $#x$)
}

#let bluem(x) = {
  text(fill: color.blue, $#x$)
}

#let greenm(x) = {
  text(fill: color.green, $#x$)
}

#let cetz-canvas = touying-reducer.with(reduce: cetz.canvas, cover: cetz.draw.hide.with(bounds: true))
#let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)

#set math.vec(delim: "[")
#set math.mat(delim: "[")


// Trace theory input only
// Composite memory (no neurons)
// Neural networks
// TODO training over sequence

#show: university-theme.with(
  aspect-ratio: "16-9",
  // config-common(handout: true),
  config-info(
    title: [Recurrent Neural Networks],
    subtitle: [CISC 7026 - Introduction to Deep Learning],
    author: [Steven Morad],
    //date: datetime.today(),
    institution: [University of Macau],
    logo: emoji.school,
  ),
  header-right: none,
  header: self => utils.display-current-heading(level: 1)
)

#set heading(numbering: numbly("{1}.", default: "1.1"))

#title-slide()

== Outline <touying:hidden>

#components.adaptive-columns(
    outline(title: none, indent: 1em, depth: 1)
)

= Review

==
We treat images as a vector, with no relationship between nearby pixels #pause

These images are equivalent to a neural network

#cimage("figures/lecture_7/permute.jpg") #pause

It is a miracle that our neural networks could classify clothing!

//#components.progressive-outline()
//= Recurrent Neural Networks

= Sequence Modeling

== 

We previously used convolution to model signals #pause

Some signals, such as audio, occur over time #pause

Convolution approaches temporal data from an electrical engineering approach #pause

Today, we will process temporal data using a psychological approach

Convolution makes use of locality and translation equivariance properties #pause

This makes learning more efficient, but not all problems benefit from locality and equivariance

== 

*Example 1:* You like dinosaurs as a child, you grow up and study dinosaurs for work #pause
    - Not local! Two events occur separated by 20 years #pause

*Example 2:* Your parent changes your diaper #pause
    - Not equivariant! Ok if you are a baby, different meaning if you are an adult! #pause

*Example 3:* You love person A, then 5 years later marry person B #pause
    - Not equivariant or local! If you marry person B then love person A, it is different #pause

*Question:* Any other examples? 

== 
If your problem has local and equivariant structure, use convolution #pause

For other problems, we need something else! #pause

Humans experience time and process temporal data #pause

Can we use this to come up with a new neural network architecture? 

= Trace Theory

== 
How do humans experience time? #pause

Humans create memories #pause

We experience time by reasoning over our memories

==
#slide[
    #image("figures/lecture_8/locke.jpeg", height: 100%) 
][
    John Locke (1690) believed that conciousness and identity arise from memories #pause

    If all your memories were erased, you would be a different person #pause

    Without the ability to reason over memories, we would react to stimuli like bacteria
]

== 
So how do we model memories in humans? #pause

$ f: underbrace(H, "Memories") times overbrace(X, "Sensory information") times underbrace(Theta, "Neurons") |-> overbrace(H, "Updated memories") $ #pause

All your memories represented as a vector $bold(h) in H$ #pause

Everything you currently sense (sight, touch, sound, emotions) is a vector $bold(x) in X$ #pause

We update our memories following 

$ bold(h)_t = f(bold(x)_t, bold(h)_(t-1)) $ 

==
$ bold(h)_t = f(bold(x)_t, bold(h)_(t-1)) $

#image("figures/lecture_8/insideout.jpg", height: 85%)

==
TODO: Only recurrent model first

 After we have constructed our memories $bold(h)_t$, we do not recall all information at once #pause

*Example:* I ask you your favorite ice cream flavor #pause

You recall previous times you ate ice cream, but not your phone number #pause

We should model this too

$ g: H times X times Theta |-> Y $ #pause

$ bold(y)_t = g(bold(h)_t, bold(x)_t, bold(theta)) $

==
$ f: H times X times Theta |-> H; quad g: H times X times Theta |-> Y $

The function $f$ is *recurrent* because it outputs a future input #pause

$ #redm[$bold(h)_t$] = f(bold(x)_t, bold(h)_(t-1)); quad bold(y)_t = g(bold(x)_t, bold(h)_t) $ #pause

$ #greenm[$bold(h)_(t+1)$] = f(bold(x)_(t+1), #redm[$bold(h)_(t)$]); quad bold(y)_(t+1) = g(bold(x)_(t+1), bold(h)_(t + 1)) $ #pause

$ bold(h)_(t+2) = f(bold(x)_(t+2), #greenm[$bold(h)_(t + 1)$]); quad bold(y)_(t+2) = g(bold(x)_(t+2), bold(h)_(t + 2)) $  #pause

$ dots.v $ #pause

We call $f,g$ *recurrent neural networks* (RNN)

= Elman Networks

= Backpropagation Through Time

= Squashing States

= LSTM

= GRU

= Linear Recurrent Models

= Coding