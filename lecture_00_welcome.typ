#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "@preview/cetz:0.4.0"
#import "@preview/fletcher:0.5.8" as fletcher: node, edge
#import "common.typ": *

// For students: you may want to change this to true
// otherwise you will get one slide for each new line
#let handout = false

// cetz and fletcher bindings for touying
#let cetz-canvas = touying-reducer.with(reduce: cetz.canvas, cover: cetz.draw.hide.with(bounds: true))
#let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-common(handout: handout),
  config-info(
    title: [Welcome],
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


= Course Background

==

This is my second time teaching this course at UM #pause

Please provide feedback privately to me #pause
- Email smorad at um.edu.mo
- Chat after class #pause

I would like to make the class *interactive* #pause

The best way to learn is to *ask questions* and have *discussions*

==
I will tell you about myself, and why I am interested in deep learning #pause

Then, *you* will tell me why you are interested in deep learning #pause

It will help me alter the course towards your goals

==
  I was always interested in space and robotics #pause

  #side-by-side[#cimage("figures/lecture_1/az.jpg", height: 60%)][#cimage("figures/lecture_1/speer.png", height: 60%)]

==
  After school, I realized much of the classical robotics that we learn in school *does not work* in reality #pause

  #cimage("figures/lecture_1/curiosity.jpg", height: 50%) #pause

  Today's robots are stupid -- important robots are human controlled
==
Since then, I have focused on creating less stupid robots #pause

#side-by-side[#cimage("figures/lecture_1/cambridge.png", height: 70%)][#cimage("figures/lecture_1/robomaster.jpg", height: 70%)] #pause

Robots that *learn* from their mistakes

// 16 mins when I blab
==
I am interested in *deep learning* because I want to make smarter robots #pause

There are many tasks that humans do not like to do, that robots can do #pause

#v(2em)
#align(center)[What do you want to learn? Why?]

==
I lead the Behavior Optimization and Learning Theory (BOLT) Lab #pause 

I am looking for research students focusing on deep reinforcement learning and robotics #pause

If the course is too easy, send me an email or come to office hours #pause
- I give all applicants a 3-4 week project to measure their capabilities

= Course Information
==
Most communication will happen over Moodle #pause
- I will try and post lecture slides after each lecture #pause
- Assignments #pause
- Grading

==
*Prerequisites*: #pause
- Programming in python #pause
  - Should be able to implement a stack, etc #pause
  - GPT/DeepSeek is *not enough*, you will fail #pause
    - I pick rare libraries that DeepSeek does not understand #pause
- Linear algebra #pause
  - Multiply matrices #pause 
  - Invert matrices #pause
  - Solve systems of equations $A x = b$ #pause
- Multivariable calculus #pause
  - Computing gradients $mat((partial f) / (partial x_1), (partial f) / (partial x_2), dots)^top $ 

==
*Good to Know:* #pause
- Probability and statistics #pause
  - Bayes rule, conditional probabilities $P(a | b) = (P(b | a) P(a)) / P(b)$

- Numerical/array programming #pause
  - Numpy, matlab, octave, etc.

==
  *Grading:* #pause
- 30% assignments #pause
- 30% exams #pause
- 30% final project #pause
- 10% participation #pause
  - 5% group participation #pause
  - 5% individual participation 

==
5% individual participation #pause
  - For asking questions or answering questions in class #pause
  - If you never talk in class, *you will get 0 points* #pause
  - Better learning when you answer questions (especially if wrong) #pause

Some of you are shy or have poor English skills #pause
  - To succeed today in deep learning, you need two things: #pause
    - Confidence - not afraid to speak your ideas #pause
    - English - all best papers/conferences currently in English #pause
      - Maybe in 10 years, best papers will be in Chinese, *but not today* #pause
  - Andrew Ng, Yann LeCun, Yoshua Bengio, Geoffrey Hinton, etc #pause
    - Many great researchers forgotten without confidence 

==
*Office Hours:* Thursday 10:00-12:00 #pause

Review assignments early, so you can attend office hours #pause

Office hours may be crowded before deadlines #pause
- You will not have much time if you have not started!

// 26 mins blabbing

= Course Structure
==
I designed all the course material myself #pause
- Assignments and content inspired by Prof. Dingqi Yang #pause
- You can view presentation source code online 
  - https://github.com/smorad/um_cisc_7026 #pause
- I will upload slides to moodle after lecture 

==
*Planned topics:*

#side-by-side(align: left)[
    - (08.22): Course Introduction
    - (08.29): Linear Regression
    - (09.05): Neural Networks 
    - (09.12): Backpropagation and Optimization
    - (09.19): Exam 1
    - (09.26): Classification
    - (10.03): Training Tricks 
    - (10.10): Convolutional Neural Networks

][
    - (10.17): Exam 2
    - (10.24): Recurrent Neural Networks
    - (10.31): Autoencoders and Generative Models
    - (11.07): Diffusion Models 
    - (11.14): Attention and Transformers
    - (11.21): Exam 3 
    - (11.28): Foundation Models 
]

==
*Homework assignments:* #pause

- (08.22 - 08.29): (Optional) Array Programming
- (08.29 - 09.12): Linear Regression
- (09.12 - 09.26): Neural Networks and Backpropagation
- (09.26 - 10.10): MLP Regression
- (10.10 - 10.24): Convolutional MNIST Classification
- (10.24 - 11.07): RNN Stock Market Prediction 
- (10.31 - 11.07): Final Project Plan
- (11.07 - 12.05): Final Project



==
If you do not like my teaching style, it is ok #pause

You can instead follow the  _Dive into Deep Learning_ textbook #pause
- Available for free online at https://d2l.ai #pause
- Syllabus contains corresponding textbook chapter #pause
- Also available in Chinese at https://zh.d2l.ai 

==
//32 mins very slow
First time I taught this course, all lectures were 3 hours #pause
- Students hate this #pause
- Stop paying attention around 2 hours #pause
- After exam students cannot learn #pause

I will try to keep lectures less than 2.5 hours #pause
- I will always stay later to answer questions #pause
- After exam there will be no lecture, you can go sleep #pause

I might also provide short breaks #pause
- Leave the classroom #pause
- Use the toilet #pause
- Ask me questions

= Questions or Comments?
==
https://ummoodle.um.edu.mo/pluginfile.php/1298433/mod_resource/content/7/syllabus.pdf
// Approx 30 mins total (2025/2026)