#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "@preview/cetz:0.4.0"
#import "@preview/fletcher:0.5.8" as fletcher: node, edge
#import "common.typ": *

// For students: you may want to change this to true
// otherwise you will get one slide for each new line
#let handout = true

// cetz and fletcher bindings for touying
#let cetz-canvas = touying-reducer.with(reduce: cetz.canvas, cover: cetz.draw.hide.with(bounds: true))
#let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-common(handout: handout),
  config-info(
    title: [Introduction],
    subtitle: [CISC 7026 - Introduction to Deep Learning],
    author: [Steven Morad],
    institution: [University of Macau],
    logo: image("figures/common/bolt-logo.png", width: 4cm)
  ),
  header-right: none,
  header: self => utils.display-current-heading(level: 1)
)



#title-slide()

= Outline <touying:hidden>

#components.adaptive-columns(
    outline(title: none, indent: 1em, depth: 1)
)


= Course Timeline
==
This course follows the history of deep learning #pause

#cimage("figures/lecture_1/timeline.svg")


= Deep Learning Successes
==
Deep learning is becoming very popular worldwide

#cimage("figures/lecture_1/ml_chart.jpg", height: 70%)

#text(size: 12pt)[Credit: Stanford University 2024 AI Index Report]

==
#side-by-side[
  #cimage("figures/lecture_1/hinton_nobel.jpeg", height: 100%) #pause
][
  John Hopfield and Geoffrey Hinton won the 2024 Nobel prize in physics #pause

  They won the prize for their work on deep learning
]

==
#side-by-side[
  #cimage("figures/lecture_1/demis.jpeg", height: 100%) #pause
][
  Hassabis (DeepMind), Jumper, and Baker won the 2024 Nobel prize in chemistry #pause

  Also for their work on deep learning #pause
]


==
#side-by-side[
  Science fiction is now possible through deep learning #pause

  It creates art
][
  #cimage("figures/lecture_1/ai_generated.png")
]



==
It beat the world champions at difficult video games like DotA 2

#cimage("figures/lecture_1/openai5.jpeg", height: 70%)

#link("https://youtu.be/eHipy_j29Xw?si=iM8QVB6_P-ROUU1Y")

==
It is learning to use tools and break rules

#link("https://youtu.be/kopoLzvh5jY?si=keH4i8noY4zUVNrP")

// 55 mins blabbing, no break

==
It is operating fully autonomous taxis in many cities

#cimage("figures/lecture_1/waymo.png", height: 60%)

#link("https://www.youtube.com/watch?v=Zeyv1bN9v4A")

==
Maybe it is doing your homework, then explaining itself

#cimage("figures/lecture_1/homework.png", height:80%)

==
It is making you lose money in the stock market

#cimage("figures/lecture_1/stocks.jpeg", height: 80%)

==
It is telling your doctor if you have cancer

#cimage("figures/lecture_1/cancer.jpeg", height: 80%)

==
We are solving more and more problems using deep learning #pause

Deep learning is creeping into our daily lives #pause

In many cases, deep models outperform humans #pause

Our deep models keep improving as we get more data #pause

*Opinion:* In the next 10 years, our lives will look very different


==
Throughout this course, you will be training your own deep models #pause

After the course, you will be experts at deep learning #pause

Deep learning is a powerful tool #pause

Like all powerful tools, deep learning can be used for evil #pause

#v(2em)
#align(center)[*Request:* Before you train a deep model, ask yourself whether it is good or bad for the world]


= What is Deep Learning?
==
How does deep learning work? #pause

It consists of four parts:
+ Dataset
+ Deep neural network
+ Loss function
+ Optimization procedure

==
The *dataset* provides a set inputs and associated outputs
#align(center)[
  #grid(
    columns: 2,
    column-gutter: 4cm,
    row-gutter: 1cm,
    cimage("figures/lecture_1/dog.png", height: 60%),
    cimage("figures/lecture_1/muffin.png", height: 60%),
    align(center)[#text(size: 32pt)[Dog]],
    align(center)[#text(size: 32pt)[Muffin]],
  )
]

==
The *deep neural network* maps inputs to outputs

#cimage("figures/lecture_1/dog_to_caption_nn.png", height: 80%)

==
The *loss function* describes how "wrong" the neural network is. We call this "wrongness" the *loss*. #pause

#cimage("figures/lecture_1/loss_function_nn.png", height: 70%)

==
The *optimization procedure* changes the neural network to reduce the loss #pause

#cimage("figures/lecture_1/optimizer_nn.png", height: 70%)

==

Deep learning consists of four parts:
+ Dataset
+ Deep neural network
+ Loss function
+ Optimization procedure #pause

In this course, we will cover these parts in detail


= Terminology: AI, ML, DL

==
#side-by-side[
  #cimage("figures/lecture_1/ai_ml_dl.svg") #pause
][
Deep Learning is a type of Machine Learning #pause

Machine learning defines *what* we are trying to do #pause

Deep learning defines *how* we do it #pause

Before we can understand deep learning, we must become familiar with machine learning
]

= Machine Learning
==
*Task:* Write a program to determine if a picture contains a dog. #pause

*Question:* How would you program this? #pause

#cimage("figures/lecture_1/dog_muffin.jpeg", width: 50%) #pause

Would your method still work? #pause

We often know *what* we want, but we do not know *how*

==
We give machine learning the *what* #pause

And it tells us the *how* #pause

In other words, we tell an ML model *what* we want it to do #pause

And it learns *how* to do it

==
We often know *what* we want, but we do not know *how* #pause

We have many pictures of either dogs or muffins $x in X$ #pause

We want to know if the picture is [dog | muffin] $y in Y$ #pause

We learn a function or mapping from $X$ to $Y$ #pause

$ f: X |-> Y $ #pause

Machine learning tells us how to find $f$

== 
#cimage("figures/lecture_1/dog_muffin.svg", width: 100%)
==

Consider some more interesting functions #pause

$ f: "image" |-> "caption" $ #pause
$ f: "caption" |-> "image" $ #pause
$ f: "English" |-> "Chinese" $ #pause
$ f: "law" |-> "change in climate" $ #pause
$ f: "voice command" |-> "robot action" $ #pause

*Question:* Can anyone suggest other interesting functions?

==
Why do we call it machine *learning*? #pause

We learn the function $f$ from the *data* $x in X, y in Y$ #pause

More specifically, we learn function *parameters* $Theta$ #pause

$ f: X, Theta |-> Y $

==
More specifically, we learn function *parameters* $Theta$

$ f: X, Theta |-> Y $

#only(2)[$ f("你好吗", vec(theta_1, theta_2, dots.v)) = "You good?" $]

#only((3))[
  $ f( #image("figures/lecture_1/dog.png", height: 20%), vec(theta_1, theta_2, dots.v) ) = "Dog" $
]
#only((4))[
  $ f( #image("figures/lecture_1/muffin.png", height: 20%), vec(theta_1, theta_2, dots.v) ) = "Muffin" $
]
#only((5,6))[
  $ f("Dog", vec(theta_1, theta_2, dots.v) ) = #image("figures/lecture_1/dog.png", height: 20%) $ 
]
#only(6)[
  Machine learning learns the parameters that solve difficult problems
]

==
*Summary:* #pause
+ Certain problems are difficult to solve with programming #pause
  - Dog or muffin? #pause
+ Machine learning provides a framework to solve difficult problems #pause
  - We learn the parameters $theta$ for some function $f(x, theta) = y$

==
You will use Python with machine learning libraries in this course #pause

We will specifically focus on: #pause
  - JAX #pause
  - PyTorch #pause

You should become comfortable using these libraries #pause
  - Read tutorials online #pause
  - Play with the libraries

==
Both JAX and PyTorch are libraries for the Python language #pause

They are both based on `numpy`, which itself is based on `MATLAB` #pause

Both libraries are *tensor processing* libraries #pause
  - Designed for linear algebra and taking derivatives #pause

To install, use `pip` #pause
  - `pip install torch` #pause
  - `pip install jax jaxlib`

==
Create vectors, matrices, or tensors in `jax`

```python
  import jax.numpy as jnp
  a = jnp.array(1) # Scalar
  b = jnp.array([1, 2]) # Vector
  C = jnp.array([[1,2], [3,4]]) # 2x2 Matrix
  D = jnp.ones((3,3,3)) # 3x3x3 Tensor
```

You can determine the dimensions of a variable using `shape`

```python
  b.shape # Prints (2,)
  C.shape # Prints (2,2)
  D.shape # prints (3,3,3)
```

==
Create vectors, matrices, or tensors in `pytorch`

```python
  import torch
  a = torch.tensor(1) # Scalar
  b = torch.tensor([1, 2]) # Vector
  C = torch.tensor([[1,2], [3,4]]) # 2x2 Matrix
  D = torch.ones((3,3,3)) # 3x3x3 Tensor
```

You can determine the dimensions of a variable using `shape`

```python
  b.shape # Prints (2,)
  C.shape # Prints (2,2)
  D.shape # prints (3,3,3)
```
==
Most operations in `jax` and `pytorch` are *vectorized* #pause
  - Executed in parallel, very fast #pause

```python
  import jax.numpy as jnp
  
  s = 5 * jnp.array([1, 2])
  print(s) # jnp.array(5, 10)
  x = jnp.array([1, 2]) + jnp.array([3, 4])
  print(x) # jnp.array([4, 6])
  y = jnp.array([1, 2]) * jnp.array([3, 4]) # Careful!
  print(y) # jnp.array([3, 8])
  z = jnp.array([[1], [2]]) @ jnp.array([[3, 4]])
  print(z) # A^t B (dot product), jnp.array([[11]])

```
==
`pytorch` is very similar to `jax`

```python
  import torch

  s = 5 * torch.tensor([1, 2])
  print(s) # torch.tensor(5, 10)
  x = torch.tensor([1, 2]) + torch.tensor([3, 4])
  print(x) # torch.tensor([4, 6])
  y = torch.tensor([1, 2]) * torch.tensor([3, 4]) # Careful!
  print(y) # torch.tensor([3, 8])
  z = torch.tensor([[1], [2]]) @ torch.tensor([[3, 4]])
  print(z) # A^t B (dot product), torch.tensor([[11]])
```
==
  You can also call various methods on arrays/tensors
  ```python
  import jax.numpy as jnp

  x = jnp.array([[1, 2], [3, 4]]).sum(axis=0) 
  print(x) # Sum across leading axis, array([4, 6])
  y = jnp.array([[1, 2], [3, 4]]).mean()
  print(y) # Mean across all axes, array(2.5)
  z = jnp.array([[1, 2], [3, 4]]).reshape((4,))
  print(z) # jnp.array([1, 2, 3, 4])
  ```
==
Same thing for `pytorch`
```python
import torch

x = torch.tensor([[1, 2], [3, 4]]).sum(axis=0) 
print(x) # Sum across leading axis, array([4, 6])
y = torch.tensor([[1, 2], [3, 4]]).mean()
print(y) # Mean across all axes, array(2.5)
z = torch.tensor([[1, 2], [3, 4]]).reshape((4,))
print(z) # torch.tensor([1, 2, 3, 4])
```

==
These libraries can produce tricky error messages! #pause
#text(size: 20pt)[
```python
>>> jnp.array([[1,2]]) @ jnp.array([[3,4]]) # (1, 2) x (1, 2)
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
File "/local/scratch/sm2558/miniconda3/envs/jax/lib/python3.11/site-packages/jax/_src/numpy/array_methods.py", line 256, in deferring_binary_op
  return binary_op(*args)
          ^^^^^^^^^^^^^^^^
File "/local/scratch/sm2558/miniconda3/envs/jax/lib/python3.11/site-packages/jax/_src/numpy/lax_numpy.py", line 3192, in matmul
  out = lax.dot_general(
        ^^^^^^^^^^^^^^^^
TypeError: dot_general requires contracting dimensions to have the same shape, got (2,) and (1,).
```
]

= Coding
==
  Let us do a Google Colab tutorial!
==
  Let us set up a local `conda` environment
==

*Homework:* #pause
- Review linear algebra if you are not familiar #pause
- Install `pytorch` and `jax` onto your computer, or use Google Colab #pause
- Play around with them #pause
  - What does `@` do if you have a `(3, 3, 3)` tensor? #pause
  - What does `*` do if one tensor has fewer dimensions? #pause
  - How do you invert a matrix? #pause
  - What does `jnp.arange, jnp.zeros, jnp.full` do? #pause
    - `torch.arange, torch.zeros, torch.full` #pause
- Read the documentation #pause