struct IterativeBatchQs <: MultiObjectiveBatchQs
    model::SVDD.OCClassifier
    inf_measure::SequentialPQs
    rep_measure::Function
    div_measure::Function
    normalization::Function
    k::Int
    λ_inf::Float64
    λ_rep::Float64
    λ_div::Float64

    function IterativeBatchQs(model::SVDD.OCClassifier, informativeness::SequentialPQs; representativeness::Symbol, diversity::Symbol,
        k::Int, λ_inf::T1 where T1<:Real=0.33, λ_rep::T2 where T2 <: Real=0.33, λ_div::T3 where T3 <: Real=0.33)::IterativeBatchQs
        (k < 1) && throw(ArgumentError("Invalid batch size k=$(k)."))

        λ_inf, λ_rep, λ_div = normalize_weights(λ_inf, λ_rep, λ_div)
        representativeness_measure = get_rep_measure(representativeness)
        diversity_measure = get_iterative_div_measure(diversity)
        return new(model, informativeness, representativeness_measure, diversity_measure, min_max_normalization, k, λ_inf, λ_rep, λ_div)
    end
end

"""
Select best batch with weighted sum of requirements.
Iterative selection: diversity is only computed to previously selected observations.
"""
function select_batch(qs::IterativeBatchQs, x::Array{T, 2}, labels::Dict{Symbol, Vector{Int}}, candidate_indices::Vector{Int})::Vector{Int} where T <: Real
    num_observations = length(candidate_indices)
    if num_observations <= qs.k
        return candidate_indices
    end

    #informativeness
    inf_scores_normalized = qs.normalization(qs_score(qs.inf_measure, x, labels)[candidate_indices])

    # representativeness
    rep_scores_normalized = qs.normalization(qs.rep_measure(qs.model, x, labels, candidate_indices))

    batch_samples = Int[]
    div_scores_normalized = Float64[]
    for iteration in 1:qs.k
        combined_scores = Vector{Float64}(undef, num_observations)
        if length(batch_samples) == 0
            # first sample for new batch cannot compute diversity to existing samples
            combined_scores = qs.λ_inf * inf_scores_normalized + qs.λ_rep * rep_scores_normalized
        else
            div_scores_normalized = qs.normalization(qs.div_measure(qs.model, x, candidate_indices, batch_samples[end], div_scores_normalized))
            combined_scores = qs.λ_inf * inf_scores_normalized + qs.λ_rep * rep_scores_normalized + qs.λ_div * div_scores_normalized
            batch_sample_indices = [ind for (ind, val) in enumerate(candidate_indices) if val in batch_samples]
            combined_scores[batch_sample_indices] .= -Inf
        end
        best_sample_index = candidate_indices[argmax(combined_scores)]

        push!(batch_samples, best_sample_index)
    end
    return batch_samples
end
