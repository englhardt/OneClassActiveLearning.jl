struct GreedyHierarchicalBatchQs <: MultiObjectiveBatchQs
    model::SVDD.OCClassifier
    inf_measure::SequentialPQs
    rep_measure::Function
    div_measure::Function
    k::Int

    function GreedyHierarchicalBatchQs(model::SVDD.OCClassifier, informativeness::SequentialPQs; representativeness::Symbol, diversity::Symbol,
        k::Int)::GreedyHierarchicalBatchQs
        (k < 1) && throw(ArgumentError("Invalid batch size k=$(k)."))

        representativeness_measure = get_rep_measure(representativeness)
        diversity_measure = get_iterative_div_measure(diversity)
        return new(model, informativeness, representativeness_measure, diversity_measure, k)
    end
end

"""
Idea: Find 4*k elements with largest representativeness,
    select 2*k elements with largest informativeness,
    iteratively generate batch by greedyly adding sample with maximum diversity
        to already selected batch samples
"""
function select_batch(qs::GreedyHierarchicalBatchQs, x::Array{T, 2}, labels::Dict{Symbol, Vector{Int}}, candidate_indices::Vector{Int})::Vector{Int} where T <: Real
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

    div_scores = Float64[]
    batch_samples = [batch_candidate_indices[1]]
    for iteration in 2:qs.k

        div_scores = qs.div_measure(qs.model, x, candidate_indices, batch_samples[end], div_scores)
        # find candidate with best score
        best_sample_index = candidate_indices[argmax(div_scores)]

        push!(batch_samples, best_sample_index)
    end
    return batch_samples
end
