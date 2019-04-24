mutable struct EnumerativeBatchQs <: MultiObjectiveBatchQs
    model::SVDD.OCClassifier
    inf_measure::SequentialPQs
    rep_measure::Function
    div_measure::Function
    normalization::Function
    k::Int
    λ_inf::Float64
    λ_rep::Float64
    λ_div::Float64

    function EnumerativeBatchQs(model::SVDD.OCClassifier, informativeness::SequentialPQs; representativeness::Symbol=nothing, diversity::Symbol=nothing,
        k::Int=0, λ_inf::T1 where T1<:Real =0.33, λ_rep::T2 where T2 <: Real=0.33, λ_div::T3 where T3 <: Real=0.33)::EnumerativeBatchQs
        # check basic params
        (model == nothing) && throw(ArgumentError("No model specified."))
        (k < 1) && throw(ArgumentError("Invalid batch size k=$(k)."))

        # min max normalization
        function normalization(x::Vector{Float64})::Vector{Float64}
            return (x .- min(x...)) ./ (max(x...) - min(x...))
        end

        # normalize λs
        λ_s = λ_inf+λ_rep+λ_div
        λ_inf = λ_inf / λ_s
        λ_rep = λ_rep / λ_s
        λ_div = λ_div / λ_s

        not_initialized = x->throw(ErrorException("Calling not initialized function."))
        strategy = new(model, informativeness, not_initialized, not_initialized, normalization, k, λ_inf, λ_rep, λ_div)

        # set up measures
        set_rep_measure!(strategy, representativeness)
        set_enumerative_div_measure!(strategy, diversity)

        return strategy
    end
end

function select_batch(qs::EnumerativeBatchQs, x::Array{T, 2}, labels::Dict{Symbol, Vector{Int}}, candidate_indices::Vector{Int})::Vector{Int} where T <: Real
    num_observations = length(candidate_indices)
    if num_observations <= qs.k
        return candidate_indices
    end
    #informativeness needs to be computed once every iteration
    inf_scores_normalized = qs.normalization(qs_score(qs.inf_measure, x, labels)[candidate_indices])

    # representativeness needs to be computed once
    rep_scores_normalized = qs.normalization(qs.rep_measure(x, labels, candidate_indices))

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
