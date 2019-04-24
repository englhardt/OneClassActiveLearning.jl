struct AllRandomBatchQs <: BatchPQs
    k::Int

    function AllRandomBatchQs(;k::Int=0)::AllRandomBatchQs
        (k < 1) && throw(ArgumentError("Invalid batch size k=$(k)."))
        return new(k)
    end
end

function select_batch(qs::AllRandomBatchQs, x::Array{T, 2}, labels::Dict{Symbol, Vector{Int}}, candidate_indices::Vector{Int})::Vector{Int} where T <: Real
    num_observations = length(candidate_indices)
    if num_observations <= qs.k
        return candidate_indices
    end

    return sample(candidate_indices, qs.k, replace=false, ordered=true)
end
