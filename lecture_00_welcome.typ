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
  Most of the classical robotics learned in school *does not work* in reality #pause

  #cimage("figures/lecture_1/curiosity.jpg", height: 70%) #pause

  Today's robots are stupid -- important robots are human controlled
==
Since then, I have focused on creating less stupid robots #pause

#side-by-side[#cimage("figures/lecture_1/cambridge.png", height: 70%)][#cimage("figures/lecture_1/robomaster.jpg", height: 70%)] #pause

Robots that *learn* from their mistakes

// 16 mins when I blab
==

There are many tasks that humans do not like, but must be done #pause
- We can solve these tasks with intelligent robots #pause
- Humans can focus on passions like sports, art, studies, etc #pause

I am interested in *deep learning* because I believe it is the only way to create intelligent robots that learn from their mistakes #pause 

#v(2em)
#align(center)[*Question:* Why are you interested in deep learning?]

==
I lead the Behavior Optimization and Learning Theory (BOLT) Lab #pause 

I am looking for a research student focusing on deep reinforcement learning and robotics *with a strong mathematical background* #pause

If the course is too easy, send me an email or come to office hours #pause
- I give all applicants a 3-4 week project to measure their capabilities

= Prerequisites

==
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


= Grading
==
- 30% assignments #pause
- 30% exams #pause
- 30% final project #pause
- 10% participation #pause
  - 5% group participation #pause
  - 5% individual participation 


= Grading - Assignments <touying:hidden>
==
Turn in assignments on time! #pause

Late penalties:
  - -25% 0-1 days late
  - -50% 1-2 days late
  - -75% 2-3 days late
  - -100% 3+ days late 

= Grading - Exams <touying:hidden>
==
There are 3 exams in this course, I will only score your best two exams #pause

*Example 1:* You are sick and miss exam 2, but you take exam 1 and 3 #pause

#side-by-side[Exam 1: 70/100][Exam 2: 0/100][Exam 3: 90/100][Score: 80/100] #pause

*Example 2:* You take all three exams #pause

#side-by-side[Exam 1: 70/100][Exam 2: 80/100][Exam 3: 90/100][Score: 85/100] #pause

*Example 3:* If you are very smart you can skip exam 3 #pause

#side-by-side[Exam 1: 100/100][Exam 2: 100/100][Exam 3: 0/100][Score: 100/100]


= Grading - Final Project <touying:hidden>
==
Final project instructions already on Moodle #pause
- Listen to lectures to understand deep learning #pause
- Form of a group of 4 or 5 members #pause
- Think of an interesting deep learning project #pause
- Create and submit your project plan #pause
- Submit your final project
// 26 mins blabbing

= Grading - Participation <touying:hidden>
==
5% group participation, 5% individual participation #pause
  - All students share group score, work together! #pause
  - Individual participation for asking/answering questions in class #pause
  - If you never speak, *you will get 0 individiual participation points* #pause

It is my job to prepare you for success in deep learning #pause
  - Some of you are shy or have poor English skills #pause
  - To succeed, you need *confidence* and *English skills* #pause
      - Maybe in 10 years, best papers will be in Chinese, *but not today* #pause
    - Andrew Ng, Yann LeCun, and Yoshua Bengio #pause
      - Learned English as a second language and confident speakers #pause 
      - Other great scientists are forgotten 

= Course Structure
= Course Structure - Office Hours <touying:hidden>
==
*Office Hours:* Thursday 14:00-16:00 #pause

Review assignments early, so you can attend office hours #pause

Office hours may be crowded before deadlines #pause
- You will not have much time if you have not started!


= Course Structure - Planned Topics <touying:hidden>
==
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

= Course Structure - Homework Assigments <touying:hidden>
==
- (08.22 - 08.29): (Optional) Array Programming
- (08.29 - 09.12): Linear Regression
- (09.12 - 09.26): Neural Networks and Backpropagation
- (09.26 - 10.10): MLP Regression
- (10.10 - 10.24): Convolutional MNIST Classification
- (10.24 - 11.07): RNN Stock Market Prediction 
- (10.31 - 11.07): Final Project Plan
- (11.07 - 12.05): Final Project


= Course Structure - Resources <touying:hidden>
==
Most communication will happen over Moodle #pause
- I will try and post lecture slides after each lecture #pause
- Assignments #pause
- Grading
==
I designed the course material myself #pause
- Assignments inspired by Prof. Dingqi Yang #pause
- You can view presentation source code online 
  - https://github.com/smorad/um_cisc_7026 #pause
- I will upload slides to moodle after lecture 

==
If you do not like my teaching style, it is ok #pause

You can instead follow the  _Dive into Deep Learning_ textbook #pause
- Available for free online at https://d2l.ai #pause
- Syllabus contains corresponding textbook chapter #pause
- Also available in Chinese at https://zh.d2l.ai 

= Course Structure - Breaks <touying:hidden>
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

= Cheating
==
I take cheating *very seriously* #pause

If caught, you get a 0 in the course and *fail the course* #pause
  - Will drop your GPA, and may cause removal from Master program #pause

I failed cheating students last year, it is not worth it #pause
  - I already drop your lowest exam score

==

The value of education is in learning, not the degree #pause
- Just study and do your best #pause

Cheating is not possible for Baidu/DeepSeek/etc interview #pause
- Invite you on-campus for 5-hour whiteboard interview #pause
- Cannot cheat, cannot get lucky #pause
- Only way to succeed is to understand the material

==
#cimage("figures/lecture_1/interview2.png")
==
#cimage("figures/lecture_1/interview.png")

= Questions or Comments?
==
https://ummoodle.um.edu.mo/pluginfile.php/1298433/mod_resource/content/7/syllabus.pdf
// Approx 30 mins total (2025/2026)

= Introduction to Deep Learning