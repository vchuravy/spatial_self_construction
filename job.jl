include("main.jl")

function runProcess(config, val, out, loadTime)
    f, dist = val

    r = @time main(config, dist, true, loadTime = loadTime, resultFolder=out)
    params = first(values(dist))
    return (f, params, r)
end