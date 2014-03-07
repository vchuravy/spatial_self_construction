A Julia implementation of the autopoesis framework.

Install Julia from http://julialang.org/ this programm has been tested with v0.3.0-prerelease ether build it from source https://github.com/JuliaLang/julia or download a package for your platform.

### Mac OS X
#### Mac > 10.7
 - Download the 0.3-prerelease from [http://julialang.org]
 - Install it
 - rename the installed Application to Julia-0.3.0
 - modify your path (in .bash_profile) to include /Applications/Julia-0.3.0.app/Contents/Resources/julia/bin
 - test if the command julia works on a terminal.


The other option is to install Julia via homebrew [http://brew.sh/]
After installing homebrew follow the instructions from [https://github.com/staticfloat/homebrew-julia]

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
 - OpenCL

```julia
Pkg.update()
Pkg.add("DEPENDENCYNAME")
```

### Running the simulator

#### From the terminal

```bash
cd path/to/autopoesis
julia -L main.jl -e 'main()'
```

#### From the REPL or IJulia

```julia
require("/full/path/to/autopoesis/main.jl")
main()
```

#### Load directly, but open a REPL
```bash
julia -L main.jl
>> main()
```

### Visualization

The simulator uses matplotlib via PyPlot.jl for plotting. Since plotting slows the program down significantly it is disable by default.

To enable it call

```julia
main(enableVis=true)
```

If you also want to enable the direction field plotting

use

```julia
main(enableVis=true, enableDirFieldVis=true)
```

### Loading Files

If you want to load results from previous runs, copy them to the data folder and then call

```julia
main(fileName="nameOfFile", loadTime=iStep) #Without the .mat ending, loadTime in the saved interval
```