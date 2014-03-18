include("main.jl")

function runProcess(config, dist, out, loadTime)
    r = main(config, dist, true, loadTime = loadTime, resultFolder=out)
    if length(dist) == 1
        time, value = first(dist)
        name = first(value)
        args = value[2:end]
        return ((name, args), r)
    else
        return ((nothing, dist), r)
    end
end