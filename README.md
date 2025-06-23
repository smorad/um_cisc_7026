# Steven's Introduction to Deep Learning Course
This repository contains the source code for my UM CISC 7026 course. I will update the code each time I teach the course. If you want to refer to a specific course year, please look for the specified tag (e.g., `fall-2024`).

## Compiling the Material
I utilize `typst` for slide generation. You can compile the slides on your own.

If on MacOS, you can install `typst` using `brew`.
```bash
brew install typst@0.13.1
```

Alternatively, you can install `typst` from [this GitHub link](https://github.com/typst/typst/releases/tag/v0.13.1). 

To compile the material is quite simple. Navigate to this repository using the command line, and run
```bash
typst compile path_to_lecture.typ
```

This will produce a PDF labeled `path_to_lecture.pdf`

**Note:** The generated slides have transitions (often one line per slide). If you would like to generate slides without transitions for study purposes, consider setting `#let handout = False` at the beginning of the source file.