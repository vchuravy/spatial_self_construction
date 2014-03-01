A Julia implementation of the autopoesis framework.

Install Julia from http://julialang.org/ this programm has been tested with v0.3.0-prerelease ether build it from source https://github.com/JuliaLang/julia or download a package for your platform.

### Mac OS X

The other option is to install Julia via homebrew [url]
After installing homebrew follow the instructions from [url]

I would recommend installing the current devel version with support for apples accelerate

brew install julia --HEAD --accelerate

You should also install matplotlib -> brew install matplotlib

### Installing julia packages

After doing so open a terminal and execute julia

Then install the dependencies:

 - MAT
 - NumericExtensions
 - ProgressMeter
 - Datetime
 - PyPlot

Pkg.update()
Pkg.add("DEPENDENCYNAME")


afterwards you can execute julia main.jl