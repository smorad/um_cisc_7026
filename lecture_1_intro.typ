#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/cetz:0.2.2": canvas, draw
#import "common.typ": *




#show: university-theme.with(
  aspect-ratio: "16-9",
  short-title: "CISC 7026: Introduction to Deep Learning",
  short-author: "Steven Morad",
  short-date: "Lecture 1: Introduction"
)
#show: slide_template

#title-slide(
  // Section time: 34 mins at leisurely pace
  title: [Introduction],
  subtitle: "CISC 7026: Introduction to Deep Learning",
  institution-name: "University of Macau",
  //logo: image("logo.jpg", width: 25%)
)

/*
#slide[
  #utils.register-section[section1]
  #utils.register-section[section2]
  #utils.polylux-outline()
  #utils.current-section
]
*/

#slide(title: [Overview])[
  + Brief chat
  + Course Information
  + Course Structure
  + Lecture
  #pdfpc.speaker-note("bork")
]

#slide(title: [Overview])[
  + *Brief chat*
  + Course Information
  + Course Structure
  + Lecture
]


#slide(title: [Brief Chat])[
  This is my first course at UM #pause
  
  I taught a course on Deep Reinforcement Learning at Cambridge #pause

  I am not perfect, and I am still learning how to teach effectively #pause
  
  Please provide feedback privately to me #pause
  - Email smorad at um.edu.mo
  - Chat after class #pause

  I would like to make the class *interactive* #pause

  The best way to learn is to *ask questions* and have *discussions*
]

#slide(title: [Brief Chat])[
  I will tell you about myself, and why I am interested in deep learning #pause

  Then, *you* will tell me why you are interested in deep learning #pause

  It will help me alter the course towards your goals
]

#slide(title: [Brief Chat])[
  I was always interested in space and robotics #pause

  #side-by-side[#cimage("figures/lecture_1/az.jpg", height: 60%)][#cimage("figures/lecture_1/speer.png", height: 60%)]
]

#slide[
  After school, I realized much of the classical robotics that we learn in school *does not work* in reality #pause

  #cimage("figures/lecture_1/curiosity.jpg", height: 50%) #pause

  Today's robots are stupid -- important robots are human controlled
]

#slide[
  Since then, I have focused on creating less stupid robots #pause

  #side-by-side[#cimage("figures/lecture_1/cambridge.jpg", height: 70%)][#cimage("figures/lecture_1/robomaster.jpg", height: 70%)] #pause

  Robots that *learn* from their mistakes
]

// 16 mins when I blab
#slide[
  I am interested in *deep learning* because I want to make smarter robots #pause

  There are many tasks that humans do not like to do, that robots can do #pause

  #v(2em)
  #align(center)[What do you want to learn? Why?]
]

// #slide(title: [Brief Chat])[
//   I am interested in Deep Learning because I think it is the most promising approach for automating tasks that we do not want to do #pause

//   I am interested in *deep reinforcement learning* and *memory* #pause
//   - Reinforcement learning trains agents that know how to interact with the world #pause
//   - Memory gives them the ability to remember your favorite food, where the agents live, etc #pause  

//   I am interested in this course because it enables me to continue my research #pause

//   #align(center)[What are you here to learn? What interests you?]
// ]

#slide(title: [Brief Chat])[
  I am starting a lab, and looking for research students focusing on deep reinforcement learning and robotics #pause

  If you finish the course and find it too easy, send me an email!
]

#slide(title: [Overview])[
  + *Brief chat*
  + Course Information
  + Course Structure
  + Lecture
]

// ~8 mins at very slow pace
#slide(title: [Overview])[
  + Brief chat
  + *Course Information*
  + Course Structure
  + Lecture
]

#slide(title: [Course Info])[
  Most communication will happen over Moodle #pause
  - I will try and post lecture slides after each lecture #pause
  - Assignments #pause
  - Grading
]

#slide(title: [Course Info])[
  *Prerequisites*: #pause
  - Programming in python #pause
    - Should be able to implement a stack, etc #pause
  - Linear algebra #pause
    - Multiply matrices, invert matrices, solve systems of equations, etc #pause
  - Multivariable calculus #pause
    - Computing gradients $mat((partial f) / (partial x_1), (partial f) / (partial x_2), dots)^top $ #pause
  *Good to Know:* #pause
  - Probability #pause
    - Bayes rule, conditional probabilities $P(a | b) = (P(b | a) P(a)) / P(b)$
]

#slide(title: [Course Info])[
    *Grading (subject to change):* #pause
  - 70% assignments #pause
  - 20% quiz #pause
  - 10% attendance and participation
    - Name plates
]

#slide(title: [Course Info])[
  *Office Hours:* Monday and Tuesday 11:00 - 12:00, E11 4026 #pause

  Review assignments early, so you can attend Tuesday office hours #pause

  Monday office hours will be crowded before deadlines #pause
  - You will not have much time if you have not started!
]
// 26 mins blabbing

#slide(title: [Overview])[
  + Brief chat
  + *Course Information*
  + Course Structure
  + Lecture
]

#slide(title: [Overview])[
  + Brief chat
  + Course Information
  + *Course Structure*
  + Lecture
]

#slide(title: [Course Structure])[
  This course is structured after the _Dive into Deep Learning_ textbook and Prof. Dingqi Yang's slides #pause

  The textbook is available for free online at https://d2l.ai #pause

  If you get confused by a lecture, try reading the corresponding chapter #pause

  Also available in Chinese at https://zh.d2l.ai #pause

  We will start from the basics and learn the theory behind Deep Learning #pause

  Your assignments will teach you two popular Deep Learning libraries: #pause
  - PyTorch #pause
  - JAX
]

//32 mins very slow
#slide(title: [Course Structure])[
  3 hours is a very long time to pay attention #pause

  I will place slides titled "Relax", every $tilde 45$ minutes #pause

  At the Relax slides, we may take a $tilde 15$ minute break #pause

  You can: #pause
  - Leave the classroom #pause
  - Use the toilet #pause
  - Walk around #pause
  - Ask me questions #pause
  
  This format is subject to change
]

#slide(title: [Course Structure])[
  We will be following the history of machine learning #pause

  #cimage("figures/lecture_1/timeline.svg", height: 70%)
]

// 38 blabbing

/*
#slide(title: [Course Structure])[
  We will be touching on the following topics (subject to change) #pause

  - Introduction to machine learning #pause
  - Linear regression and classification #pause
  - Neural Networks and the Perceptron #pause
  - Convolutional Neural Networks #pause
  - Recurrent Neural Networks #pause
  - Attention and Transformers #pause
  - Reinforcement Learning
]
*/

//#focus-slide[Questions?]

#slide(title: [Overview])[
  + Brief chat
  + Course Information
  + *Course Structure*
  + Lecture
]

#slide(title: [Overview])[
  + Brief chat
  + Course Information
  + Course Structure
  + *Lecture*
]
// 25 mins very slow
// 39 mins blabbing

#focus-slide[Relax]

// ~17 mins very slow

#title-slide(
  // Section time: 34 mins at leisurely pace
  title: [Introduction],
  subtitle: "CISC 7026: Introduction to Deep Learning",
  institution-name: "University of Macau",
  //logo: image("logo.jpg", width: 25%)
)

#slide(title: [Overview])[
  + Deep Learning Successes #pause
  + What is Deep Learning? #pause
  + Differences between AI, ML, DL #pause
  + Define Machine Learning #pause
  + Machine Learning Libraries
]
// TODO remove supervised/unsupervised and classification

#slide(title: [Overview])[
  + *Deep Learning Successes*
  + What is Deep Learning?
  + Differences between AI, ML, DL
  + Define Machine Learning
  + Machine Learning Libraries
]


#slide(title: [Successes])[
  Deep learning is becoming very popular worldwide

  #cimage("figures/lecture_1/ml_chart.jpg", height: 70%)

  #text(size: 12pt)[Credit: Stanford University 2024 AI Index Report]
]

#slide(title: [Successes])[
  Many things that were once considered science fiction are now possible through deep learning. It can draw pictures for us

  #cimage("figures/lecture_1/ai_generated.png", height: 70%)
]

#slide(title: [Successes])[
  It can beat the world champions at difficult video games like DotA 2

  #cimage("figures/lecture_1/openai5.jpeg", height: 70%)

  #link("https://youtu.be/eHipy_j29Xw?si=iM8QVB6_P-ROUU1Y")
]

#slide(title: [Successes])[
  It is learning to use tools and break rules

  #link("https://youtu.be/kopoLzvh5jY?si=keH4i8noY4zUVNrP")
]

// 55 mins blabbing, no break

#slide(title: [Successes])[
  It is operating fully autonomous taxis in four cities

  #cimage("figures/lecture_1/waymo.png", height: 60%)

  #link("https://www.youtube.com/watch?v=Zeyv1bN9v4A")
]

#slide(title: [Successes])[
  Maybe it is doing your homework, then explaining itself

  #cimage("figures/lecture_1/homework.png", height:80%)
]

#slide(title: [Successes])[
  It is making you lose money in the stock market

  #cimage("figures/lecture_1/stocks.jpeg", height: 80%)
]

#slide(title: [Successes])[
  It is telling your doctor if you have cancer

  #cimage("figures/lecture_1/cancer.jpeg", height: 80%)
]

#slide(title: [Successes])[
  We are solving more and more problems using deep learning #pause
  
  Deep learning is creeping into our daily lives #pause
  
  In many cases, deep models outperform humans #pause

  Our deep models keep improving as we get more data #pause

  *Opinion:* In the next 10-20 years, our lives will look very different
]

// ~26 mins very slow

#slide(title: [Successes])[

  Throughout this course, you will be training your own deep models #pause

  After the course, you will be experts at deep learning #pause

  Deep learning is a powerful tool #pause

  Like all powerful tools, deep learning can be used for evil #pause

  #v(2em)
  #align(center)[*Request:* Before you train a deep model, ask yourself whether it is good or bad for the world]
]

// 1h10m blabbing

#slide(title: [Overview])[
  + *Deep Learning Successes*
  + What is Deep Learning?
  + Differences between AI, ML, DL
  + Define Machine Learning
  + Machine Learning Libraries
]
#slide(title: [Overview])[
  + Deep Learning Successes
  + *What is Deep Learning?*
  + Differences between AI, ML, DL
  + Define Machine Learning
  + Machine Learning Libraries
]

#slide(title: [DL at a Glance])[
  At a high level, how does deep learning work? #pause

  It consists of four parts:
  + Dataset
  + Deep neural network
  + Loss function
  + Optimization procedure
]

#slide(title: [DL at a Glance])[
  The dataset provides a set inputs and associated outputs
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
]

#slide(title: [DL at a Glance])[
  The *neural network* learns to map the inputs to outputs

  #cimage("figures/lecture_1/dog_to_caption_nn.png", height: 80%)
]

#slide(title: [DL at a Glance])[
  The *loss function* describes how "wrong" the neural network is. We call this "wrongness" the *loss*. #pause

  #cimage("figures/lecture_1/loss_function_nn.png", height: 70%)
]

#slide(title: [DL at a Glance])[
  The *optimization procedure* changes the neural network to reduce the loss #pause

  #cimage("figures/lecture_1/optimizer_nn.png", height: 70%)
]

#slide(title: [DL at a Glance])[
  At a high level, how does deep learning work?

  It consists of four parts:
  + Dataset
  + Deep neural network
  + Loss function
  + Optimization procedure
]

// 1h19 blabbing no break

#slide(title: [Overview])[
  + Deep Learning Successes
  + *What is Deep Learning?*
  + Differences between AI, ML, DL
  + Define Machine Learning
  + Machine Learning Libraries
]

// 45 mins slow?
#focus-slide[Relax]

#slide(title: [Overview])[
  + Deep Learning Successes
  + What is Deep Learning?
  + *Differences between AI, ML, DL*
  + Define Machine Learning
  + Machine Learning Libraries
]

#slide(title: [AI, ML, DL])[
  #side-by-side[
    #cimage("figures/lecture_1/ai_ml_dl.svg") #pause
  ][
  Deep Learning is a type of Machine Learning #pause

  Machine learning defines *what* we are trying to do #pause

  Deep learning defines *how* we do it #pause

  Before we can understand deep learning, we must become familiar with machine learning
  ]
]

#slide(title: [Overview])[
  + Deep Learning Successes
  + What is Deep Learning?
  + *Differences between AI, ML, DL*
  + Define Machine Learning
  + Machine Learning Libraries
]


#slide(title: [Overview])[
  + Deep Learning Successes
  + What is Deep Learning?
  + Differences between AI, ML, DL
  + *Define Machine Learning*
  + Machine Learning Libraries
]

#slide(title: [ML])[
  *Task:* Write a program to determine if a picture contains a dog. #pause

  *Question:* How would you program this? #pause
  
  #cimage("figures/lecture_1/dog_muffin.jpeg", width: 50%) #pause

  Would your method still work? #pause

  We often know *what* we want, but we do not know *how*
]

#slide(title: [ML])[
  We give machine learning the *what* #pause

  And it tells us the *how* #pause

  In other words, we tell an ML model *what* we want it to do #pause

  And it learns *how* to do it
]


#slide(title: [ML])[
  We often know *what* we want, but we do not know *how* #pause

  We have many pictures of either dogs or muffins $x in X$ #pause

  We want to know if the picture is [dog | muffin] $y in Y$ #pause

  We learn a function or mapping from $X$ to $Y$ #pause

  $ f: X |-> Y $ #pause

  Machine learning tells us how to find $f$
]

#slide(title: [ML])[
  #cimage("figures/lecture_1/dog_muffin.svg", width: 100%)
]

/*
#slide(title: [ML])[
  In machine learning, our goal is to learn a function $f$ #pause

  Often, our functions look different than you might expect #pause

  #side-by-side[
      #text(size: 60pt)[$ f(x) = x^2 $] #pause
  ][
      #cimage("figures/lecture_1/dog_muffin.svg", width: 100%) #pause
  ]

  #v(1em)
  #align(center)[But they are indeed both mathematical functions!]
]
*/

#slide(title: [ML])[
  Consider some more interesting functions #pause

  $ f: "image" |-> "caption" $ #pause
  $ f: "caption" |-> "image" $ #pause
  $ f: "English" |-> "Chinese" $ #pause
  $ f: "law" |-> "change in climate" $ #pause
  $ f: "voice command" |-> "robot action" $ #pause

  *Question:* Can anyone suggest other interesting functions?
]

// ~45 mins slow

#slide(title: [ML])[
  Why do we call it machine *learning*? #pause

  We learn the function $f$ from the *data* $x in X, y in Y$ #pause

  More specifically, we learn function *parameters* $Theta$ #pause

  $ f: X, Theta |-> Y $
]

#slide[
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
]

#slide(title: [ML])[
  *Summary:* #pause
  + Certain problems are difficult to solve with programming #pause
    - Dog or muffin? #pause
  + Machine learning provides a framework to solve difficult problems #pause
    - We learn the parameters $theta$ for some function $f(x, theta) = y$
]

#focus-slide[Relax]

#slide(title: [ML])[
  You will use Python with machine learning libraries in this course #pause

  We will specifically focus on: #pause
    - JAX #pause
    - PyTorch #pause

  You should become comfortable using these libraries #pause
    - Read tutorials online #pause
    - Play with the libraries
]

// #slide(title: [ML])[
//   + JAX vs PyTorch
//     + Based on numpy and matlab
//   + Matrix operations
// ]

#slide[
  Both JAX and PyTorch are libraries for the Python language #pause

  They are both based on `numpy`, which itself is based on `MATLAB` #pause

  Both libraries are *tensor processing* libraries #pause
    - Designed for linear algebra and taking derivatives #pause

  To install, use `pip` #pause
    - `pip install torch` #pause
    - `pip install jax jaxlib`
]

#slide[
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
]

#slide[
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
]

#slide[
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
]

#slide[ 
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
]

#slide[
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
]

#slide[
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
]

#slide[
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
]

#slide[
  Let us do a Google Colab tutorial!
]

#slide[
  Let us set up a local `conda` environment
]

#slide[
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
]

#slide[
  There might be a quiz on `jax` and `pytorch` operations next lecture
]
// 1h25m + 30 min break = 1h55m total