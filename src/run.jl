
function init_from_experiment(experiment, data, labels, res)
    debug(LOGGER, "[INIT] Start initialization of experiment.")
    pools = copy(experiment[:param][:initial_pools])
    split_strategy = experiment[:split_strategy]

    # model
    train_data, train_pools, _ = get_train(split_strategy, data, pools)
    debug(LOGGER, "[INIT] Initializing model '$(experiment[:model])' with $(format_observations(train_data)) observations.")
    model = SVDD.instantiate(eval(experiment[:model][:type]), train_data, train_pools, experiment[:model][:param])
    SVDD.initialize!(model, eval(experiment[:model][:init_strategy]))
    set_model_fitted!(res, model)
    SVDD.set_adjust_K!(model, experiment[:param][:adjust_K])
    solver = experiment[:param][:solver]
    debug(LOGGER, "[INIT] Model solver for this experiment is '$(typeof(solver))'.")

    # query strategy
    debug(LOGGER, "[INIT] Initializing QS '$(experiment[:query_strategy][:type])'.")
    query_data, _, _ = get_query(split_strategy, data, pools)
    qs = QueryStrategies.initialize_qs(eval(experiment[:query_strategy][:type]), model, query_data, experiment[:query_strategy][:param])

    debug(LOGGER, "[INIT] Initializing oracle.")
    if isa(experiment[:oracle], Oracle)
        oracle = experiment[:oracle]
    else
        oracle = Oracles.initialize_oracle(eval(experiment[:oracle][:type]), data, labels, experiment[:oracle][:param])
    end

    info(LOGGER, "[INIT] Initialization done.")
    return (model, pools, solver, qs, split_strategy, oracle)
end

function check_active_learn_args(data, labels)
    MLLabelUtils.islabelenc(labels, OneClassActiveLearning.LABEL_ENCODING) || throw(ArgumentError("Argument labels is in the wrong encoding."))
    return size(data, 2) == length(labels) || throw(ArgumentError("Number of observations ($(size(data,2))) does not equal number of labels $(length(labels))."))
end

function active_learn(experiment::Dict{Symbol, Any})
    data, labels = OneClassActiveLearning.load_data(experiment[:data_file])
    return active_learn(experiment, data, labels)
end

function active_learn(experiment::Dict{Symbol, Any}, data::Array{T, 2}, labels::Vector{Symbol}) where T <: Real
    check_active_learn_args(data, labels)
    log_experiment_info(experiment)

    res = Result(experiment)
    set_worker_info!(res)
    set_data_stats!(res, data, experiment[:split_strategy])

    model, pools, solver, qs, split_strategy, oracle = try
        init_from_experiment(experiment, data, labels, res)
    catch e
        warn(LOGGER, e)
        if isa(e, KDEException)
            res.status[:exit_code] = :kde_error
        elseif isa(e, MissingLabelTypeException)
            res.status[:exit_code] = :missing_label_type
        else
            throw(e)
        end
        return res
    end

    train_data, train_pools, _ = get_train(split_strategy, data, pools)
    SVDD.set_data!(model, train_data)
    SVDD.set_pools!(model, MLLabelUtils.labelmap(train_pools))

    classify_precision = get(experiment[:param], :classify_precision, SVDD.OPT_PRECISION)
    debug(LOGGER, "Classify precision: $classify_precision")

    debug(LOGGER, "Start active learning cycle with $(experiment[:param][:num_al_iterations]) queries.")
    for i in 0:experiment[:param][:num_al_iterations]
        info(LOGGER, "Iteration $(i)")
        debug(LOGGER, "Memory consumption $(round(Int, Sys.free_memory() / 2^20)) MB / $(round(Int, Sys.total_memory() / 2^20)) MB")

        debug(LOGGER, "[FIT] Start fitting model on $(format_observations(train_data)) observations.")
        # Workaround: redirect solver output
        stdout_orig, stderr_orig = stdout, stderr
        redirect_stdout(); redirect_stderr()
        status, time_fit, mem_fit = @timed SVDD.fit!(model, solver)
        redirect_stdout(stdout_orig); redirect_stderr(stderr_orig)
        debug(LOGGER, "[FIT] Fitting done ($(time_fit) s, $(format_bytes(mem_fit))).")

        ValueHistories.@trace res.al_history i time_fit mem_fit
        if status !== JuMP.MathOptInterface.OPTIMAL
            warn(LOGGER, "Not solved to optimality. Solver status: $status.")
            res.status[:exit_code] = :solver_error
            return res
        end

        test_data, _, test_indices = get_test(split_strategy, data, pools)
        debug(LOGGER, "[TEST] Testing by predicting $(format_observations(test_data)) observations.")
        predictions = SVDD.predict(model, test_data)
        push_evaluation!(res.al_history, i, predictions, labels[test_indices], classify_precision)
        debug(LOGGER, "[TEST] Testing done.")

        if i < experiment[:param][:num_al_iterations]
            query_data, query_pools, query_indices = try
                get_query(split_strategy, data, pools)
            catch e
                warn(LOGGER, e)
                if isa(e, KDEException)
                     res.status[:exit_code] = :kde_error
                elseif isa(e, MissingLabelTypeException)
                    res.status[:exit_code] = :missing_label_type
                else
                    res.status[:exit_code] = :unknown_qs_error
                end
                return res
            end
            if :U ∉ query_pools && !isa(qs, QuerySynthesisStrategy)
                info(LOGGER, "Aborting '$(experiment[:hash])' after $(i) iterations because no more unlabeled observations are left.")
                al_summarize!(res)
                res.status[:exit_code] = :early_abort
                return res
            end
            debug(LOGGER, "[QS] Starting query strategy on $(format_observations(query_data)) observations.")
            if i == 0
                query, time_qs, mem_qs = @timed get_query_object_helper(qs, query_data, query_pools, query_indices)
            else
                query, time_qs, mem_qs = @timed get_query_object_helper(qs, query_data, query_pools, query_indices, values(res.al_history, :query_history))
            end
            debug(LOGGER, "[QS] Query strategy finished ($(time_qs) s, $(format_bytes(mem_qs))).")
            query_label = Oracles.ask_oracle(oracle, query)

            # tmp workaround
            data, pools, labels = process_query!(isa(query, Array) ? query : [query],
                                         isa(query_label, Array) ? query_label : [query_label],
                                         model,
                                         split_strategy,
                                         data,
                                         pools,
                                         labels)

            push_query!(res.al_history, i, query, query_label, time_qs, mem_qs)
            isa(query, Int) ? debug(LOGGER, "[QS] Query(id = $(query), label = $(query_label))") :
                             debug(LOGGER, "[QS] Query(label = $(query_label))")
            debug(LOGGER, "[QS] Query strategy done.")
        end
        debug(LOGGER, "Finished iteration $(i).")
    end
    debug(LOGGER, "Finished active learning cycle.")

    al_summarize!(res)
    debug(LOGGER, "Summary done.")
    info(LOGGER, "Finished experiment '$(experiment[:hash])'.")
    res.status[:exit_code] = :success
    return res
end

function process_query!(query_data::Array{T, 2},
                        query_labels::Vector{Symbol},
                        model, split_strategy, data, pools, labels) where T <: Real
    size(query_data, 2) == length(query_labels) || throw(DimensionMismatch("Number of queries does not match number of labels."))
    size(data, 1) == size(query_data, 1) ||throw(DimensionMismatch("Data dimensionality does not match query dimensionality."))

    n_old = size(data,2)
    append!(split_strategy.train, trues(length(query_labels)))
    append!(split_strategy.test, falses(length(query_labels)))
    data = hcat(data, query_data)
    pools = vcat(pools, fill(:U, length(query_labels)))
    labels = vcat(labels, query_labels)
    n_new = size(data, 2)
    global_query_ids = collect((n_old + 1):n_new)

    process_query!(global_query_ids, query_labels, model, split_strategy, data, pools, labels)
end

function process_query!(global_query_ids::Vector{Int},
                        query_labels::Vector{Symbol},
                        model, split_strategy, data, pools, labels)
    length(global_query_ids) == length(query_labels) || throw(DimensionMismatch("Number of queries does not match number of labels."))

    pools_before = copy(pools)
    train_mask_before = OneClassActiveLearning.calc_mask(split_strategy.train_strat, split_strategy.train, pools)

    query_pool_labels = convert_labels_to_learning(query_labels)
    pools[global_query_ids] .= query_pool_labels
    train_data, train_pools, _ = get_train(split_strategy, data, pools)

    train_mask_after = OneClassActiveLearning.calc_mask(split_strategy.train_strat, split_strategy.train, pools)

    # indices that are in train before and after, relative to the updated train
    global_remaining_indices = findall(train_mask_before .& train_mask_after)
    old_idx_remaining_train = map(id -> get_local_idx(id, split_strategy, pools_before, Val(:train)), global_remaining_indices)
    new_idx_remaining_train = map(id -> get_local_idx(id, split_strategy, pools, Val(:train)), global_remaining_indices)

    filtered_query_ids = filter_query_id(global_query_ids, split_strategy, query_pool_labels, Val(:train))
    # check if query is relevant for train subset and model
    if isempty(filtered_query_ids)
        return data, pools, labels
    end
    train_query_ids = map(id -> get_local_idx(id, split_strategy, pools, Val(:train)), filtered_query_ids)

    SVDD.update_with_feedback!(model, train_data, train_pools, train_query_ids, old_idx_remaining_train, new_idx_remaining_train)

    return data, pools, labels
end

function get_query_object_helper(qs::Q,
                                 query_data::Array{T, 2},
                                 query_pools::Vector{Symbol},
                                 query_indices::Vector{Int},
                                 history::Vector{Int}=Int[])::Int where T <: Real where Q <: Union{PoolQs, SubspaceQs}
    return QueryStrategies.get_query_object(qs, query_data, query_pools, query_indices, history)
end

function get_query_object_helper(qs::QuerySynthesisStrategy,
                                 query_data::Array{T, 2},
                                 query_pools::Vector{Symbol},
                                 query_indices::Vector{Int},
                                 history::Vector{Array{T, 2}}=Vector{Array{T, 2}}())::Array{T, 2} where T <: Real
    return QueryStrategies.get_query_object(qs, query_data, query_pools, history)
end

function push_query!(al_history::ValueHistories.MVHistory, i, query, query_label, time_qs, mem_qs)
    push!(al_history, :query_history, i, query)
    ValueHistories.@trace al_history i query_label time_qs mem_qs
    return nothing
end

function push_evaluation_cm!(al_history, i, cm)
    push!(al_history, :cm, i, cm)
    for e in [:cohens_kappa, :matthews_corr, :f1_score, :tpr, :fpr]
        push!(al_history, e, i, eval(e)(cm))
    end
end

function push_evaluation!(al_history::ValueHistories.MVHistory, i, predictions::Vector{Vector{Float64}}, labels, classify_precision)
    cm = ConfusionMatrix(SVDD.classify(predictions, Val(:Global), opt_precision=classify_precision), labels)
    push_evaluation_cm!(al_history, i, cm)
    return nothing
end

function push_evaluation!(al_history::ValueHistories.MVHistory, i, predictions, labels, classify_precision)
    cm = ConfusionMatrix(SVDD.classify.(predictions, opt_precision=classify_precision), labels)
    push_evaluation_cm!(al_history, i, cm)
    push!(al_history, :auc, i, roc_auc(predictions, labels))
    for k in [0.01, 0.02, 0.05, 0.1, 0.2]
        auc_fpr = OneClassActiveLearning.roc_auc(predictions, labels, fpr = k)
        auc_fpr_normalized = OneClassActiveLearning.roc_auc(predictions, labels, fpr = k, normalize = true)
        push!(al_history, Symbol("auc_fpr_$(k)"), i, auc_fpr)
        push!(al_history, Symbol("auc_fpr_normalized_$(k)"), i, auc_fpr_normalized)
    end
    return nothing
end
