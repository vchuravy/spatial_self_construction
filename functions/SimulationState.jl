type SimulationState{T <: FloatingPoint}
    Mfield :: Matrix{T}
    Afield :: Matrix{T}
    Wfield :: Matrix{T}
    Ffield :: Matrix{T}
    θfield :: Matrix{T}

    Mpot    :: Matrix{T}
    Mpot1   :: Matrix{T}
    Mpot2   :: Matrix{T}
    Wpot    :: Matrix{T}
    Apot    :: Matrix{T}
    Apot1   :: Matrix{T}


    Wlap :: Matrix{T}
    Alap :: Matrix{T}
    Mlap :: Matrix{T}
    Flap :: Matrix{T}

    dW :: Matrix{T}
    dA :: Matrix{T}
    dM :: Matrix{T}
    dF :: Matrix{T}

    Mflow :: Matrix{Matrix{T}}
    Wflow :: Matrix{Matrix{T}}
    Aflow :: Matrix{Matrix{T}}
    Area :: Array{T, 3}

    function SimulationState(Mfield :: Matrix, Afield :: Matrix, Wfield :: Matrix, Ffield :: Matrix, θfield :: Matrix)
        s = size(Mfield)
        @assert s == size(Afield)
        @assert s == size(Wfield)
        @assert s == size(Ffield)
        @assert s == size(θfield)

        create() = Array(T, s...)
        create_flow() = begin
            Flow = Array(Matrix{T}, s...)

            for i in 1:length(Flow)
                Flow[i] = Array(T, (3,3))
            end
            return Flow
        end

        new(convert(Array{T}, Mfield),
            convert(Array{T}, Afield),
            convert(Array{T}, Wfield),
            convert(Array{T}, Ffield),
            convert(Array{T}, θfield),
            create(),
            create(),
            create(),
            create(),
            create(),
            create(),
            create(),
            create(),
            create(),
            create(),
            create(),
            create(),
            create(),
            create(),
            create_flow(),
            create_flow(),
            create_flow(),
            Array(T, 4, s...))
    end
end

function SimulationState{T <: FloatingPoint}(:: Type{T}, d1 :: Int, d2 :: Int)
     create() = Array(T, d1, d2)
     SimulationState{T}(create(), create(), create(), create(), create())
end