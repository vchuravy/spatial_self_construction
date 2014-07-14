# Publications

 - `ALIFE14`: **Quantifying robustness in a spatial model of metabolism-boundary co-construction** by 
  *Eran Agmon, Alexander J Gates, Valentin Churavy and Randall D. Beer*
   - The publication can be found at [MIT Press](http://mitpress.mit.edu/sites/default/files/titles/content/alife14/ch082.html)
   - The version used is to produce the data is tagged as [alife14](https://github.com/vchuravy/spatial_self_construction/tree/alife14)

# Documentation

If you find anything missing in the documentation or find yourself unable to reproduce results please feel free to contact < v [dot] churavy [at] gmail [dot] com > or file an issue here on Github.

# Implementation

The implementation was done in Julia and OpenCL. If your computer does not support OpenCL the computation will be done purely in Julia. All our computations have been done for performance reasons with the OpenCL version.

## License
The source for this project is published under the MIT license.

# Reproducing experiments.

To ensure the reproducibility of the experimental results one has to make sure that the same software versions are used. Even then we can not guarantee perfect reproducibility due to possible variance between machines. So far we have not encountered any of those issues and have reproduced the results on different machines and operating systems. If you encounter any issues feel free to contact us. Below we have noted the software versions used for our experiments.

## Julia version

The program should be able to run on any Julia version from 0.3 upwards. If not please contact us so that we can update it.
Specifically all experiments currently have been done against Julia version 0.3.0-prerelease+1785 (4e48d5b) to Julia version 0.3.0-prerelease+2121 (8ea753e).

## Packages used

We extend our thanks to the Julia community for producing many excellent packages.
We have used the following packages in the specified versions. Due to possible API changes in lower or higher version we recommend only using those. If you encounter any incompatibilities in higher versions please open an issue and we will try to resolve it.

 - Cartesian                     0.1.5
 - Datetime                      0.1.2
 - Distributions                 0.4.2
 - JSON                          0.3.5
 - MAT                           0.2.3
 - NumericExtensions             0.6.0
 - OpenCL                        0.2.4
 - PyPlot                        1.2.2

### Installing specific versions

If you want to install specific versions of packages you can use the Julia package manager to install them.

```julia
julia> Pkg.add("OpenCL", VersionNumber(0, 2, 4))
```

# Installation
Install Julia from http://julialang.org/ this programm has been tested with v0.3.0-prerelease ether build it from source https://github.com/JuliaLang/julia or download a package for your platform.

## Mac OS X
### Mac > 10.7
#### Directly
 - Download the 0.3-prerelease from [http://julialang.org]
 - Install it
 - rename the installed Application to Julia-0.3.0
 - modify your path (in .bash_profile) to include /Applications/Julia-0.3.0.app/Contents/Resources/julia/bin
 - test if the command julia works on a terminal.

#### Homebrew
The other option is to install Julia via homebrew [http://brew.sh/]
After installing homebrew follow the instructions from [https://github.com/staticfloat/homebrew-julia]

I would recommend installing the current devel version with support for apples accelerate

brew install julia --HEAD --accelerate

You should also install matplotlib -> brew install matplotlib

## Installing Julia packages

After doing so open a terminal and execute julia

Then install the dependencies:

 - MAT
 - NumericExtensions
 - Datetime
 - PyPlot
 - OpenCL
 - Distributions

 If this is the first time you installed Julia you can just copy the file REQUIRE into the folder ~/.julia/v0.3/

```bash
julia -e "Pkg.init()"
cp REQUIRE ~/.julia/v0.3/
```

otherwise add the dependencies via the Julia PKG manager.

```julia
Pkg.update()
Pkg.add("DEPENDENCYNAME")
```

# Running the simulator

## From the terminal

```bash
cd path/to/autopoesis
julia -L main.jl -e 'main()'
```

## From the REPL or IJulia

```julia
require("/full/path/to/autopoesis/main.jl")
main()
```

## Load directly, but open a REPL
```bash
julia -L main.jl
>> main()
```

## Visualization

The simulator uses Matplotlib via PyPlot.jl for plotting. Since plotting slows the program down significantly it is disable by default.

To enable it call

```julia
main(enableVis=true)
```

## Loading Files

If you want to load results from previous runs, copy them to the data folder and then call

```julia
main(fileName="nameOfFile", loadTime=iStep) #Without the .mat ending, loadTime in the saved interval
```

## Overwriting settings

You can overwrite settings the following way.

- Create a Julia dictionary with the layout "parameter name" => value
- Include the dictionary in the main call

As a example to increase the maximum runtime from a default value of 15000 time steps to 30000 time steps while also increasing the interval of visualization to every hundreds step.

```julia
config = {"timeTotal" => 30000, "visInterval" = 100}
main(config, enableVis=true)
```

You can change every value specified in config.jl -> baseConfig

## Causing disturbances
To study the behaviour of a model instance we added the possibility to perturb the system at any point in time with a limited selection of disturbances.

If you only want to study the effect one specific disturbance on the system you can do the following. If you want to apply a range of disturbances to a system please take a look under cluster.

```julia
config = {"visInterval" = 1}
disturbance = {time => [name, arg]}
main(config, disturbance, enableVis=true)
```

Where name and possible arguments are given below and time is the value when the disturbance is happening as a float. Please note that it is indeed possible to add sequential disturbances, but not to apply two disturbances at the same point in time.

### Gaussian blur
 - Name : ```:gaussian_blur```
 - Parameters:
   - sigma: radius

### Punch
 - Name : ```:punch_local```
 - Parameters:
   - x: x coordinate of impact
   - y: y coordinate of impact
   - alpha: Amplitude
   - beta: wideness

### Random noise
 - Name : ```:global```
 - Parameters:
   - mu: Amplitude of the Gaussian
   - sig: Standard deviation

### Tear

#### Tear membrane
 - Name : ```:tear_membrane```
 - Parameters:
   - tearsize:

#### Tear directionfield
 - Name : ```:tear_dfield```
 - Parameters:
   - tearsize:

#### Tear directionfield and membrane
 - Name : ```:tear```
 - Parameters:
   - tearsize:

## Running tests as a cluster

To run multiple tests simultaneously you have to provide a configuration file (examples can be found under cluster_configs)

### Layout

```json
{
  "fileName.csv" : {
    "name" : "disturbance name",
    "parameter1" : value,
    "parameter2" : {
            "min" : 0.0,
            "max" : 2.0,
            "step" : 0.5
        }
  }
}
```
In this case parameter1 is kept constant and parameter2 is varied over the range 0.0 to 2.0 with a step of 0.5

A special parameter called "times" is used for stochastic disturbances, it specifies how often each disturbance should be tested.

### Executing

```
julia -L driver.jl -e 'runCluster("fileName", "outputFolder",  configName="cluster_configs/example_global.json", loadTime=12)
```

- outputFolder defaults to ./results
- configName defaults to cluster_configs/all.json
- loadTime defaults to -1 and thus to the last time step