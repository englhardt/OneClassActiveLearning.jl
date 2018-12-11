struct ConfusionMatrix
    pos_class::Symbol
    neg_class::Symbol

    tp::Int
    fp::Int
    tn::Int
    fn::Int
end

function Base.show(io::IO, cm::ConfusionMatrix)
    pad = 9
    tp = rpad("TP: $(cm.tp)", pad)
    fp = rpad("FP: $(cm.fp)", pad)
    fn = rpad("FN: $(cm.fn)", pad)
    tn = rpad("TN: $(cm.tn)", pad)

    pos_class = rpad(cm.pos_class, 8)
    neg_class = rpad(cm.neg_class, 8)
    blank = rpad("", pad)

    println(io, """                 Actual
                     $blank $(pos_class) $(neg_class)
                   Predicted  ------------------
                   $(pos_class)  | $(tp) $(fp)
                   $(neg_class)  | $(fn) $(tn) """)
end

function Base.show(io::IO, cms::Array{T, 1}) where T <: ConfusionMatrix
    for cm in cms
        println(io,"TP: $(cm.tp) FP: $(cm.fp) TN: $(cm.tn) FN $(cm.fn) [P = :$(cm.pos_class), N = :$(cm.neg_class)]")
    end
end

get_positive(x::ConfusionMatrix) = x.tp + x.fn

get_negative(x::ConfusionMatrix) = x.fp + x.tn

tpr(x::ConfusionMatrix) = x.tp / (x.tp + x.fn)
sensitivity = tpr
recall = tpr

fpr(x::ConfusionMatrix) = x.fp / (x.fp + x.tn)
specificity(x) = 1 - fpr(x)
tnr = specificity

get_n(x::ConfusionMatrix) = get_positive(x) + get_negative(x)

f1_score(x::ConfusionMatrix) = 2 * x.tp / (2 * x.tp + x.fp + x.fn)

ConfusionMatrix(tp, fp, tn, fn; pos_class = :outlier, neg_class = :inlier) = ConfusionMatrix(pos_class, neg_class, tp, fp, tn, fn)

Base.convert(x::Type{ConfusionMatrix}, d::Dict) = ConfusionMatrix(d)

function ConfusionMatrix(d::Dict)
    return ConfusionMatrix(convert(Int64, d["tp"]),
                           convert(Int64, d["fp"]),
                           convert(Int64, d["tn"]),
                           convert(Int64, d["fn"]))
end

function ConfusionMatrix(classification, ground_truth; pos_class = :outlier, neg_class = :inlier)
    @assert islabelenc(classification, LABEL_ENCODING)
    @assert islabelenc(ground_truth, LABEL_ENCODING)
    @assert length(classification) == length(ground_truth)

    tp = sum((classification .== pos_class) .& (ground_truth .== pos_class))
    fp = sum((classification .== pos_class) .& (ground_truth .== neg_class))
    tn = sum((classification .== neg_class) .& (ground_truth .== neg_class))
    fn = sum((classification .== neg_class) .& (ground_truth .== pos_class))

    return ConfusionMatrix(pos_class, neg_class, tp, fp, tn, fn)
end

function cohens_kappa(x::ConfusionMatrix)
    p_0 = (x.tp + x.tn) / (get_negative(x) + get_positive(x))
    p_e = ((x.tp + x.fp) * get_positive(x) + (x.tn + x.fn) * get_negative(x)) / (get_negative(x) + get_positive(x))^2
    return (p_0 - p_e) / (1 - p_e)
end

function matthews_corr(x::ConfusionMatrix)::Float64
    d = sqrt((x.tp + x.fp) * (x.tp + x.fn) * (x.tn + x.fp) * (x.tn + x.fn))
    return d == 0 ? 0.0 : (x.tp * x.tn - x.fp * x.fn) / d
end

function roc_auc(predictions::Vector{T}, ground_truth::Vector{Symbol}; fpr = 1.0, normalize = false) where T <: Real
    @assert length(predictions) == length(ground_truth)
    @assert islabelenc(ground_truth, LABEL_ENCODING)
    lm = labelmap(ground_truth)
    r = ROCAnalysis.roc(predictions[lm[:outlier]], predictions[lm[:inlier]])
    return AUC(r, pfa=fpr, normalize = normalize)
end
