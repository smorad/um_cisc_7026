#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/cetz:0.2.2": canvas, draw

#set text(size: 25pt)
#set math.vec(delim: "[")
#set math.mat(delim: "[")
#let argmin = $op("arg min", limits: #true)$

#let cimage(..args) = { 
  align(center + horizon, image(..args))
}

#let side-by-side(columns: none, gutter: 1em, align: center + horizon, ..bodies) = {
  let bodies = bodies.pos()
  let columns = if columns ==  none { (1fr,) * bodies.len() } else { columns }
  if columns.len() != bodies.len() {
    panic("number of columns must match number of content arguments")
  }

  grid(columns: columns, gutter: gutter, align: align, ..bodies)
}

#let slide_template(doc) = {
  set text(size: 25pt)
  set math.vec(delim: "[")
  set math.mat(delim: "[")
  doc
}

#let sslide(content) = {
  //slide(title: utils.current-section)[
  slide[
    #content
  ]
}

#let redm(x) = {
  text(fill: color.red, $#x$)
}

/*
#let sections-state = state("polylux-sections", ())
#let bold-outline(enum-args: (:), padding: 0pt) = locate( loc => {
  let sections = sections-state.final(loc)
  pad(padding, enum(
    ..enum-args,
    ..sections.map(section => {
      link(section.loc, section.body))
      }
  ))
})
*/