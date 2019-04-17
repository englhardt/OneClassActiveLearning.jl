"""
Original publication:
L. Yin, H. Wang, and W. Fan. Active learning based
support vector data description method for robust
novelty detection. Knowledge-Based Systems, pages
40–52, Aug. 2018.
"""
struct BoundaryNeighborCombinationPQs{M <: SVDD.OCClassifier} <: HybridPQs
    occ::M
    nn_dist
    η::Float64
    p::Float64
end

BoundaryNeighborCombinationPQs{M}(args...;kwargs...) where M = BoundaryNeighborCombinationPQs(args...; kwargs...)

function BoundaryNeighborCombinationPQs(occ::M, x::Array{T, 2}; η=0.7, p=0.15) where {T <: Real, M <:SVDD.OCClassifier}
    BoundaryNeighborCombinationPQs(occ, knn_mean_dist(x; k=1), η, p)
end

function BoundaryNeighborCombinationPQs(occ::M, x::Array{T, 2}; η=0.7, p=0.15) where {T <: Real, M <: SVDD.SubOCClassifier}
    nn_dist = [QueryStrategies.knn_mean_dist(x[s,:]; k=1) for s in occ.subspaces]
    BoundaryNeighborCombinationPQs(occ, nn_dist, η, p)
end

function bnc(nn_dist, η, p, labels, predictions)
    pred = abs.(predictions)
    d_hyp = (pred .- minimum(pred[labels[:U]])) ./ maximum(pred[labels[:U]])
    d_nn = (nn_dist .- minimum(nn_dist[labels[:U]])) ./ maximum(nn_dist[labels[:U]])
    return η * (-d_hyp) + (1 - η) * (-d_nn)
end

function qs_score(qs::BoundaryNeighborCombinationPQs{M},
                  x::Array{T, 2},
                  labels::Dict{Symbol, Array{Int, 1}})::Array{Float64, 1} where {T <: Real, M <: SVDD.OCClassifier}
    @assert length(qs.nn_dist) == size(x, 2)
    if rand() < qs.p
        return qs_score(RandomPQs(), x, labels)
    end
    return bnc(qs.nn_dist, qs.η, qs.p, labels, SVDD.predict(qs.occ, x))
end

function qs_score(qs::BoundaryNeighborCombinationPQs{M},
                  x::Array{T, 2},
                  labels::Dict{Symbol, Array{Int, 1}},
                  subspaces::Vector{Vector{Int}}) where {T <: Real, M <: SVDD.SubOCClassifier}
    length(qs.nn_dist) == length(subspaces) || throw(DimensionMismatch("Number of nn_dist arrays must match the number of subspaces."))
    all(length.(qs.nn_dist) .== size(x, 2)) || throw(DimensionMismatch("Number of neighbors in nn_dist must match number of observations in x."))
    size(qs.occ.data, 1) == size(x, 1) || throw(DimensionMismatch("Number of dimensions in x must match the number of dimensions of the classifier."))
    # return random for all observations (and not per subspace)
    if rand() < qs.p
        return qs_score(RandomPQs(), x, labels, subspaces)
    end
    map(i -> bnc(qs.nn_dist[i], qs.η, qs.p, labels, SVDD.predict(qs.occ, x[subspaces[i], :], i)), eachindex(subspaces))
end
