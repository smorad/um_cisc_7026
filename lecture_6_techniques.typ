#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/cetz:0.2.2": canvas, draw, plot
#import "common.typ": *
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm

#set math.vec(delim: "[")
#set math.mat(delim: "[")
#let agenda(index: none) = {
  let ag = (
    [Review],
    [Deeper neural networks],
    [Parameter initialization],
    [Regularization],
    [Residual networks],
    [Adaptive optimization],
    [Activation functions],
    [Coding]
  )
  for i in range(ag.len()){
    if index == i {
      enum.item(i + 1)[#text(weight: "bold", ag.at(i))]
    } else {
      enum.item(i + 1)[#ag.at(i)]
    }
  }
}

#show: university-theme.with(
  aspect-ratio: "16-9",
  short-title: "CISC 7026: Introduction to Deep Learning",
  short-author: "Steven Morad",
  short-date: "Lecture 6: Modern Techniques"
)

#title-slide(
  // Section time: 34 mins at leisurely pace
  title: [Modern Techniques],
  subtitle: "CISC 7026: Introduction to Deep Learning",
  institution-name: "University of Macau",
  //logo: image("logo.jpg", width: 25%)
)