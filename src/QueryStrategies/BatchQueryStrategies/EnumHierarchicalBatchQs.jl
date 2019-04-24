mutable struct EnumHierarchicalBatchQs <: MultiObjectiveBatchQs
    model::SVDD.OCClassifier
    inf_measure::SequentialPQs
    rep_measure::F1 where F1 <: Function
    div_measure::F2 where F2 <: Function
    k::Int

    function EnumHierarchicalBatchQs(model::SVDD.OCClassifier, informativeness::SequentialPQs; representativeness::Symbol=nothing, diversity::Symbol=nothing,
        k::Int=0)::EnumHierarchicalBatchQs
        # check basic params
        (model == nothing) && throw(ArgumentError("No model specified."))
        (k < 1) && throw(ArgumentError("Invalid batch size k=$(k)."))

        not_initialized = x->throw(ErrorException("Calling not initialized function."))
        strategy = new(model, informativeness, not_initialized, not_initialized, k)

        # set up measures
        set_rep_measure!(strategy, representativeness)
        set_enumerative_div_measure!(strategy, diversity)

        return strategy
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

    # representativeness first
    rep_scores = qs.rep_measure(x, labels, candidate_indices)
    most_representative_indices = candidate_indices[sortperm(rep_scores, rev=true)[1:min(4*qs.k, num_observations)]]

    #informativeness second
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
