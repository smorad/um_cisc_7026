#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/cetz:0.2.2": canvas, draw, plot
#import "common.typ": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm

#set math.vec(delim: "[")
#set math.mat(delim: "[")

#let ag = (
  [Review],
  [Unsupervised Learning],
  [Compression],
  [Autoencoders],
  [Variational Models],
  [World Models],
  [Coding]
)

#show: university-theme.with(
  aspect-ratio: "16-9",
  short-title: "CISC 7026: Introduction to Deep Learning",
  short-author: "Steven Morad",
  short-date: "Lecture 8: Autoencoders"
)

#title-slide(
  title: [Autoencoders and Generative Models],
  subtitle: "CISC 7026: Introduction to Deep Learning",
  institution-name: "University of Macau",
)


#aslide(ag, none)
#aslide(ag, 0)

// TODO Review

#aslide(ag, 0)
#aslide(ag, 1)
#sslide[
    #cimage("figures/lecture_8/supervised_unsupervised.png")
]

#sslide[
    Unsupervised learning is not an accurate term, because there is some supervision #pause

    "I now call it *self-supervised learning*, because *unsupervised* is both a loaded and confusing term. … Self-supervised learning uses way more supervisory signals than supervised learning, and enormously more than reinforcement learning. That’s why calling it “unsupervised” is totally misleading." - Yann LeCun, Godfather of Deep Learning 

    We will use the term *self-supervised* learning, although many textbooks still call it unsupervised learning #pause
]

#sslide[
    In supervised learning, humans provide the model with a dataset containing inputs $bold(X)$ and corresponding outputs $bold(Y)$

    $ bold(X) = vec(x_[1], x_[2], dots.v, x_[n]) quad bold(Y) = vec(y_[1], y_[2], dots.v, y_[n]) $
]

#sslide[
    In self-supervised learning, the outputs (and sometimes inputs) are *not* provided by humans #pause

    // TODO

    The model learns without human supervision
]

#sslide[
    Semi-supervised learning is responsible for today's most powerful models #pause



    The models can learn from the entire internet, without needing humans to feed them data #pause

    They can learn from datasets that would take decades for humans to create #pause
]

#sslide[
    How do these models work? #pause

    They learn the structure of the data #pause

    If the structure of the data is every picture in the world, they learn about the structure of the world

    // TODO GPT/DinoV2 images
]