struct DecisionBoundaryPQs <: ModelBasedPQs
    occ::OCClassifier
end

function qs_score(qs::DecisionBoundaryPQs, x::Array{T, 2}, labels::Dict{Symbol, Array{Int, 1}})::Array{Float64, 1} where T <: Real
    return -abs.(SVDD.predict(qs.occ, x))
end
