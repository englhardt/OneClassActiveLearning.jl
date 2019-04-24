mutable struct IterativeBatchQs <: MultiObjectiveBatchQs
    model::SVDD.OCClassifier
    inf_measure::SequentialPQs
    rep_measure::F1 where F1 <: Function
    div_measure::F2 where F2 <: Function
    norm::F3 where F3 <: Function
    k::Int
    λ_inf::Float64
    λ_rep::Float64
    λ_div::Float64

    function IterativeBatchQs(model::SVDD.OCClassifier, informativeness::SequentialPQs; representativeness::Symbol=nothing, diversity::Symbol=nothing,
        k::Int=0, λ_inf::T1 where T1<:Real =0.33, λ_rep::T2 where T2 <: Real=0.33, λ_div::T3 where T3 <: Real=0.33)::IterativeBatchQs
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
        set_iterative_div_measure!(strategy, diversity)

        return strategy
    end
end

function select_batch(qs::IterativeBatchQs, x::Array{T, 2}, labels::Dict{Symbol, Vector{Int}}, candidate_indices::Vector{Int})::Vector{Int} where T <: Real
    num_observations = length(candidate_indices)
    if num_observations <= qs.k
        return candidate_indices
    end

    #informativeness needs to be computed once every iteration
    inf_scores = qs_score(qs.inf_measure, x, labels)[candidate_indices]

    # representativeness needs to be computed once
    rep_scores = qs.rep_measure(x, labels, candidate_indices)

    batch_samples = []
    # empty array hack
    # if array is set to nothing, method dispatching does not work
    div_scores = Float64[]

    for iteration in 1:qs.k
        combined_scores = Vector{Float64}(undef, num_observations)
        if length(batch_samples) == 0
            # first sample for new batch cannot compute diversity to existing samples
            combined_scores = qs.λ_inf * qs.norm(inf_scores) + qs.λ_rep * qs.norm(rep_scores)
        else
            div_scores = qs.div_measure(qs.model, candidate_indices, batch_samples[end], div_scores)
            # use normalization function to make value ranges comparable
            combined_scores = qs.λ_inf * qs.norm(inf_scores) + qs.λ_rep * qs.norm(rep_scores) + qs.λ_div * qs.norm(div_scores)
            # ignore samples already in current batch
            batch_sample_indices = [ind for (ind, val) in enumerate(candidate_indices) if val in batch_samples]
            combined_scores[batch_sample_indices] .= -Inf
        end
        # find candidate with best score
        best_sample_index = candidate_indices[argmax(combined_scores)]

        push!(batch_samples, best_sample_index)
    end
    return batch_samples
end
