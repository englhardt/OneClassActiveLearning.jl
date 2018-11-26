"""
Original publication:
V. Barnabé-Lortie, C. Bellinger, and N. Japkowicz.
Active learning for one-class classification. In 2015
IEEE 14th International Conference on Machine
Learning and Applications (ICMLA), pages 390–395,
Dec 2015.
"""
struct HighConfidencePQs <: ModelBasedPQs
    occ::OCClassifier
end

function qs_score(qs::HighConfidencePQs, x::Array{T, 2}, labels::Dict{Symbol, Array{Int, 1}})::Array{Float64, 1} where T <: Real
    return SVDD.predict(qs.occ, x)
end

function qs_score(qs::HighConfidencePQs,
                  x::Array{T, 2},
                  labels::Dict{Symbol, Array{Int, 1}},
                  subspaces::Vector{Vector{Int}}) where T <: Real
    size(x,1) == size(qs.occ.data,1) || throw(DimensionMismatch("Number of dimensions in x must match the number of dimensions of the classifier."))
    return map(idx -> SVDD.predict(qs.occ, x[subspaces[idx], :], idx), eachindex(subspaces))
end
