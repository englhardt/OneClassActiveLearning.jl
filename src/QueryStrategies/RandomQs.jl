struct RandomQs <: QueryStrategy end

function qs_score(qs::RandomQs, x::Array{T, 2}, labels::Dict{Symbol, Array{Int, 1}})::Array{Float64, 1} where T <: Real
    return rand(size(x, 2))
end
