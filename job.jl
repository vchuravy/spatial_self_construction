include("main.jl")

function runProcess(config, dist, out)
    r = main(config, dist, true, loadTime = -1, resultFolder=out)
    if length(dist) == 1
        time, value = first(dist)
        name = first(value)
        args = value[2:end]
        return ((name, args), r)
    else
        return ((nothing, dist), r)
    end
end