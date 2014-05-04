using Cartesian

macro at_proc(p, ex)
   quote
           remotecall( $p, ()->eval(Main,$(Expr(:quote,ex))))
   end
end

# TODO: find a better way to handle arbitrary dimensions.
function createDisturbance(name :: Symbol, ranges)
    dim = length(ranges)
    if dim == 1
        createDisturbance1(name, ranges)
    elseif dim == 2
        createDisturbance2(name, ranges)
    elseif dim == 3
        createDisturbance3(name, ranges)
    elseif dim == 4
        createDisturbance4(name, ranges)
    else
        error("Can't handle dim $dim")
    end
end

function createDisturbance4(name :: Symbol, ranges)
    vals = Dict[]

    @nloops 4 i d -> ranges[d] begin
        args = @ntuple 4 i
        push!(vals, {100.0 => [name, args...]})
    end

    return vals
end

function createDisturbance3(name :: Symbol, ranges)
    vals = Dict[]

    @nloops 3 i d -> ranges[d] begin
        args = @ntuple 3 i
        push!(vals, {100.0 => [name, args...]})
    end

    return vals
end

function createDisturbance2(name :: Symbol, ranges)
    vals = Dict[]

    @nloops 2 i d -> ranges[d] begin
        args = @ntuple 2 i
        push!(vals, {100.0 => [name, args...]})
    end

    return vals
end

function createDisturbance1(name :: Symbol, range)
    vals = Dict[]

    for i in range
        push!(vals, {100.0 => [name, i]})
    end

    return vals
end

function torange(c :: Dict)
    min = c["min"]
    max = c["max"]
    step = c["step"]
    return min:step:max
end

function torange(c :: Real)
    return c
end

function writedlm_row(io::IO, row, dlm = ',')
    pb = PipeBuffer()
    state = start(row)
    while !done(row, state)
        (x, state) = next(row, state)
        Base.writedlm_cell(pb, x, dlm, false)
        done(row, state) ? write(pb,'\n') : print(pb,dlm)
    end
    (nb_available(pb) > (16*1024)) && write(io, takebuf_array(pb))
    write(io, takebuf_array(pb))
    nothing
end

function apply_punch_down!(A, x0, y0, a, b)
                d1, d2 = size(A)
                @assert x0 in 1:d1
                @assert y0 in 1:d2
                #dist(x, y) = sqrt((x - x0)^2 + (y-y0)^2)
                #dist(x, y) = abs(x-x0) + abs(y-y0)
                dist(x, y) = floor(sqrt((x - x0)^2 + (y-y0)^2))
                #punch(d) = 1- sech(1/b * d) ^ a
                punch(x)=a*exp(-x^2/(2*b^2))+1
                for j in 1:d2
                    for i in 1:d1
                        d = dist(i, j)
                        A[i,j] *= punch(d)
                    end
                end
            end
