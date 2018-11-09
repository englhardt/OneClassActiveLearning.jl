"""
Original publication:
N. Görnitz, M. Kloft, K. Rieck, and U. Brefeld.
Toward supervised anomaly detection. Journal of
Artificial Intelligence Research (JAIR), pages
235–262, Jan. 2013.
"""
struct NeighborhoodBasedPQs <: HybridPQs
    occ::OCClassifier
    knn::Array{Int,2}
    η::Float64
    k::Int
    NeighborhoodBasedPQs(occ::OCClassifier, x::Array{T, 2}; η=0.5, k=5) where T <: Real = new(occ, knn_indices(x; k=k), η, k)
end

function qs_score(qs::NeighborhoodBasedPQs, x::Array{T, 2}, labels::Dict{Symbol, Array{Int, 1}})::Array{Float64,1} where T <: Real
    @assert size(qs.knn, 2) == size(x, 2)
    Lin_indices = SVDD.merge_pools(labels, :Lin)
    τ_nb = -(0.5 .+ 1 / (2 * qs.k) * [length(qs.knn[:, i] ∩ Lin_indices) for i in 1:size(x, 2)])
    return qs.η * -abs.(SVDD.predict(qs.occ, x)) + (1 - qs.η) * τ_nb
end
