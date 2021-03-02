struct EnsembleBatchQs <: BatchPQs
    model::SVDD.OCClassifier
    k::Int
    sequential_strategy::SequentialPQs
    solver::JuMP.MOI.OptimizerWithAttributes
    model_indices::Vector{Vector{Int}}
    batch_models::Vector{SVDD.OCClassifier}

    function EnsembleBatchQs(model::SVDD.OCClassifier, sequential_strategy::SequentialPQs; k::Int, solver::JuMP.MOI.OptimizerWithAttributes)::EnsembleBatchQs
        (k < 1) && throw(ArgumentError("Invalid batch size k=$(k)."))

        labels = MLLabelUtils.labelmap2vec(model.pools)
        num_observations = size(model.data, 2)
        samples_per_batch = ceil(Int, num_observations/k)

        model_indices = Vector{Vector{Int}}()
        batch_models = Vector{SVDD.OCClassifier}()
        for i in 1:k
            new_model = deepcopy(model)
            push!(batch_models, new_model)
            # sample 1 / k indices
            indices = sample(1:num_observations, samples_per_batch, replace=false)
            push!(model_indices, indices)
        end
        return new(model, k, sequential_strategy, solver, model_indices, batch_models)
    end
end

"""
Select best batch by selecting best observation from each ensemble member.
"""
function select_batch(qs::EnsembleBatchQs, x::Array{T, 2}, labels::Dict{Symbol, Vector{Int}}, candidate_indices::Vector{Int})::Vector{Int} where T <: Real
    num_observations = length(candidate_indices)
    if num_observations <= qs.k
        return candidate_indices
    end

    label_list = MLLabelUtils.labelmap2vec(labels)

    batch_samples = []
    for i in 1:qs.k
        # set ensemble model parameters (data + labels)
        indices = qs.model_indices[i]
        model = qs.batch_models[i]
        SVDD.set_data!(model, qs.model.data[:, indices])
        SVDD.set_pools!(model, MLLabelUtils.labelmap(label_list[indices]))

        STDOUT_orig, STDERR_orig = stdout, stderr
        redirect_stdout(); redirect_stderr()
        status = SVDD.fit!(model, qs.solver)
        redirect_stdout(STDOUT_orig); redirect_stderr(STDERR_orig)

        if status !== JuMP.MathOptInterface.OPTIMAL
            throw(ErrorException("Ensemble model could not find optimal solution."))
        end

        candidate_scores = qs_score(qs.sequential_strategy, x, labels)[candidate_indices]
        descending_indices = sortperm(candidate_scores; rev=true)
        push!(batch_samples, first(x for x in candidate_indices[descending_indices] if x ∉ batch_samples))
    end
    return batch_samples
end
