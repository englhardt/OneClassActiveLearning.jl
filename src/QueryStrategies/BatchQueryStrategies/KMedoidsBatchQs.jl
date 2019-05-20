struct KMedoidsBatchQs <: BatchPQs
    k::Int

    function KMedoidsBatchQs(; k::Int)::KMedoidsBatchQs
        (k < 1) && throw(ArgumentError("Invalid batch size k=$(k)."))
        return new(k)
    end
end

"""
For maximum diversity in batch selection.
Compute k-medoids clustering with k=batchsize, return cluster medoids
"""
function select_batch(qs::KMedoidsBatchQs, x::Array{T, 2}, labels::Dict{Symbol, Vector{Int}}, candidate_indices::Vector{Int})::Vector{Int} where T <: Real
    num_observations = length(candidate_indices)
    if num_observations <= qs.k
        return candidate_indices
    end

    candidates = x[:, candidate_indices]
    distances = Distances.pairwise(Distances.Euclidean(), candidates, dims=2)
    clustering = kmedoids(distances, qs.k)

    medoid_indices = clustering.medoids

    return candidate_indices[medoid_indices]
end
