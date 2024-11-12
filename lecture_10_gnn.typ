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
  [Quiz],
  [Introduction to Prof. Li],
  [Graph Neural Networks],
)

#show: university-theme.with(
  aspect-ratio: "16-9",
  short-title: "CISC 7026: Introduction to Deep Learning",
  short-author: "Steven Morad",
  short-date: "Lecture 9: Autoencoders"
)

#title-slide(
  title: [Graph Neural Networks],
  subtitle: "CISC 7026: Introduction to Deep Learning",
  institution-name: "University of Macau",
)

#aslide(ag, none)

#aslide(ag, 0)

#sslide[
  #cimage("figures/lecture_9/shrek.jpg") 
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
  Then, we discussed variational autoencoders #pause

  However, to save time we will review these and write code next time
]

#aslide(ag, 0)
#aslide(ag, 1)

#sslide[
  All phones and laptops away or face down #pause

  No cheating, you will get 0 and I tell Dean #pause

  I will hand out the quizzes face down #pause

  Turn them over when I say start #pause

  You will have 20 minutes #pause

  Good luck!
]

#aslide(ag, 1)
#aslide(ag, 2)

#sslide[
    Prof. Qingbiao Li did his PhD at Cambridge and a postdoc at Oxford #pause

    He has two Master's degrees, one from University of Edinburgh and one from Imperial College London #pause

    His interests are in deep learning for medicine and multirobot system #pause

    Today, he will teach you about *Graph Neural Networks*
]