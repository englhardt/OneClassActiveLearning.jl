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
    return predict(qs.occ, x)
end
