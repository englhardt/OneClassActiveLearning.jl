"""
Original publication:
L. Yin, H. Wang, and W. Fan. Active learning based
support vector data description method for robust
novelty detection. Knowledge-Based Systems, pages
40–52, Aug. 2018.
"""
struct BoundaryNeighborCombinationPQs <: HybridPQs
    occ::OCClassifier
    nn_dist::Array{Float64,1}
    η::Float64
    p::Float64
    BoundaryNeighborCombinationPQs(occ::OCClassifier, x::Array{T, 2}; η=0.7, p=0.15) where T <: Real = new(occ, knn_mean_dist(x; k=1), η, p)
end

function qs_score(qs::BoundaryNeighborCombinationPQs, x::Array{T, 2}, labels::Dict{Symbol, Array{Int, 1}})::Array{Float64, 1} where T <: Real
    @assert length(qs.nn_dist) == size(x, 2)
    if rand() < qs.p
        return rand(size(x,2))
    end
    prediction = abs.(SVDD.predict(qs.occ, x))
    d_hyp = (prediction .- minimum(prediction[labels[:U]])) ./ maximum(prediction[labels[:U]])
    d_nn = (qs.nn_dist .- minimum(qs.nn_dist[labels[:U]])) ./ maximum(qs.nn_dist[labels[:U]])
    return qs.η * (-d_hyp) + (1 - qs.η) * (-d_nn)
end
