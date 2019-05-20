struct EnumerativeBatchQs <: MultiObjectiveBatchQs
    model::SVDD.OCClassifier
    inf_measure::SequentialPQs
    rep_measure::Function
    div_measure::Function
    normalization::Function
    k::Int
    λ_inf::Float64
    λ_rep::Float64
    λ_div::Float64

    function EnumerativeBatchQs(model::SVDD.OCClassifier, informativeness::SequentialPQs; representativeness::Symbol, diversity::Symbol,
        k::Int, λ_inf::T1 where T1<:Real=0.33, λ_rep::T2 where T2 <: Real=0.33, λ_div::T3 where T3 <: Real=0.33)::EnumerativeBatchQs
        # check basic params
        (model == nothing) && throw(ArgumentError("No model specified."))
        (k < 1) && throw(ArgumentError("Invalid batch size k=$(k)."))

        λ_inf, λ_rep, λ_div = normalize_weights(λ_inf, λ_rep, λ_div)
        representativeness_measure = get_rep_measure(representativeness)
        diversity_measure = get_enumerative_div_measure(diversity)
        return new(model, informativeness, representativeness_measure, diversity_measure, min_max_normalization, k, λ_inf, λ_rep, λ_div)
    end
end

"""
Select best batch with weighted sum of requirements.

Enumerative selection: Compute weighted sum for all possible candidates.
Warning: This may take a while.
"""
function select_batch(qs::EnumerativeBatchQs, x::Array{T, 2}, labels::Dict{Symbol, Vector{Int}}, candidate_indices::Vector{Int})::Vector{Int} where T <: Real
    num_observations = length(candidate_indices)
    if num_observations <= qs.k
        return candidate_indices
    end
    #informativeness
    inf_scores_normalized = qs.normalization(qs_score(qs.inf_measure, x, labels)[candidate_indices])

    # representativeness
    rep_scores_normalized = qs.normalization(qs.rep_measure(qs.model, x, labels, candidate_indices))

    best_batch = Vector{Int}()
    best_score = -Inf

    for batch in subsets(1:num_observations, qs.k)
        inf_score = sum(inf_scores_normalized[batch])/qs.k
        rep_score = sum(rep_scores_normalized[batch])/qs.k
        div_score = qs.div_measure(qs.model, x, candidate_indices[batch])
        combined_score = qs.λ_inf * inf_score + qs.λ_rep * rep_score + qs.λ_div * div_score

        if combined_score > best_score
            best_score = combined_score
            best_batch = candidate_indices[batch]
        end
    end
    return best_batch
end
