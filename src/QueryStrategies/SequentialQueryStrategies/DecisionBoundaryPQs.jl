struct DecisionBoundaryPQs <: ModelBasedPQs
    occ::SVDD.OCClassifier
end

decion_boundary_pqs(predictions) = -abs.(predictions)

function qs_score(qs::DecisionBoundaryPQs,
                  x::Array{T, 2},
                  labels::Dict{Symbol, Array{Int, 1}})::Array{Float64, 1} where {T <: Real}
    return decion_boundary_pqs(SVDD.predict(qs.occ, x))
end

function qs_score(qs::DecisionBoundaryPQs,
                  x::Array{T, 2},
                  labels::Dict{Symbol, Array{Int, 1}},
                  subspaces::Vector{Vector{Int}}) where {T <: Real}
    size(x,1) == size(qs.occ.data,1) || throw(DimensionMismatch("Number of dimensions in x must match the number of dimensions of the classifier."))
    return map(idx -> decion_boundary_pqs(SVDD.predict(qs.occ, x[subspaces[idx], :], idx)), eachindex(subspaces))
end
