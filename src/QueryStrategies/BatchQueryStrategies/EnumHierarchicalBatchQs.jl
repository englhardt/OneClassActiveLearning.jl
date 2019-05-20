struct EnumHierarchicalBatchQs <: MultiObjectiveBatchQs
    model::SVDD.OCClassifier
    inf_measure::SequentialPQs
    rep_measure::Function
    div_measure::Function
    k::Int

    function EnumHierarchicalBatchQs(model::SVDD.OCClassifier, informativeness::SequentialPQs; representativeness::Symbol, diversity::Symbol,
        k::Int)::EnumHierarchicalBatchQs
        (k < 1) && throw(ArgumentError("Invalid batch size k=$(k)."))

        representativeness_measure = get_rep_measure(representativeness)
        diversity_measure = get_div_measure(diversity)
        return new(model, informativeness, representativeness_measure, diversity_measure, k)
    end
end

"""
Idea: Find 4*k elements with largest representativeness,
    select 2*k elements with largest informativeness,
    select batch of size k with maximum diversity
"""
function select_batch(qs::EnumHierarchicalBatchQs, x::Array{T, 2}, labels::Dict{Symbol, Vector{Int}}, candidate_indices::Vector{Int})::Vector{Int} where T <: Real
    num_observations = length(candidate_indices)
    if num_observations <= qs.k
        return candidate_indices
    end

    # representativeness
    rep_scores = qs.rep_measure(qs.model, x, labels, candidate_indices)
    most_representative_indices = candidate_indices[sortperm(rep_scores, rev=true)[1:min(4*qs.k, num_observations)]]

    #informativeness
    inf_scores = qs_score(qs.inf_measure, x, labels)[most_representative_indices]
    batch_candidate_indices = most_representative_indices[sortperm(inf_scores, rev=true)[1:min(2*qs.k, num_observations)]]

    best_batch = nothing
    best_score = -Inf
    for batch in subsets(batch_candidate_indices, qs.k)
        batch_diversity = qs.div_measure(qs.model, x, batch)
        if (batch_diversity > best_score)
            best_batch = batch
            best_score = batch_diversity
        end
    end
    return best_batch
end
