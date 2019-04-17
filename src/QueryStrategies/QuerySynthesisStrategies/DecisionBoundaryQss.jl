struct DecisionBoundaryQss <: ModelBasedQss
    occ::SVDD.OCClassifier
    optimizer::QuerySynthesisOptimizer
    DecisionBoundaryQss(occ; optimizer=nothing) = new(occ, optimizer)
end

function qs_score_function(qs::DecisionBoundaryQss, data::Array{T, 2}, labels::Dict{Symbol, Array{Int, 1}})::Function where T <: Real
    return x -> -abs.(SVDD.predict(qs.occ, x))
end
