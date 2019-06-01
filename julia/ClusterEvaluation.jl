

#本模块用于评估聚类的结果
#module ClusterEvaluation
using LinearAlgebra,Distributed
#using PyCall
using Clustering
using ScikitLearn
@sk_import cluster:OPTICS
@sk_import cluster: (KMeans,DBSCAN,AgglomerativeClustering,Birch,AffinityPropagation)
@sk_import metrics: (silhouette_score,calinski_harabasz_score,davies_bouldin_score)

#对数据执行KMeans聚类并返回聚类结果
function doKMeans(X::Matrix,k)
    # R = kmeans(X', k ; display=:iter)
    # a = assignments(R)
    # return a
    model = KMeans(init="k-means++", n_clusters=k)
    fit!(model, X)
    cls=predict(model,X)
    return cls
end

#对数据执行KMeans聚类并返回聚类结果
function doDBscan(X::Matrix)
    model = DBSCAN(eps=0.3, min_samples=10)
    fit!(model, X)
    #println(model.labels_)
    return model.labels_
end

#对数据执行KMeans聚类并返回聚类结果
function doHierarchical(X::Matrix)
    model =AgglomerativeClustering(linkage="ward")
    fit!(model, X)
    #println(model.labels_)
    return model.labels_
end

#对数据执行KMeans聚类并返回聚类结果
function doAffinity(X::Matrix)
    model= AffinityPropagation(preference=-50)
    fit!(model, X)
    #println(model.labels_)
    return model.labels_
end

#对数据执行KMeans聚类并返回聚类结果
function doBirch(X::Matrix,k)
    model =Birch(branching_factor=50, n_clusters=k, threshold=0.5, compute_labels=true)
    fit!(model, X)
    cls=predict(model,X)
    return cls
end

#对数据执行KMeans聚类并返回聚类结果
function doOptics(X::Matrix)
    model =OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)
    fit!(model, X)
    #println(model.labels_)
    return model.labels_
end



#计算加权q值
function evaluateWQ(X::Matrix,cls::Vector)
    #最聚类结果的种类
    tx=unique(cls)
    #数据的行、列数
    (rows,cols)=size(X)
    q= @distributed (+) for k in tx
        #取本类别的数据
        dfx=X[cls.==k,:]
        F=svd(dfx)
        #q值乘以本类目的权重
        wq=(F.S[1]/sum(F.S))*(size(dfx,1)/rows)
        wq
    end
    return q
end

#计算dunn指数
function evaluateDunn(X::Matrix,cls::Vector)
    #最聚类结果的种类
    tx=unique(cls)
    #数据的行、列数
    (rows,cols)=size(X)
    grData=Dict{Int32,Array{Float64,2}}()
    for k in tx
        #取本类别的数据
        dfx=X[cls.==k,:]
        grData[k]=dfx
    end
    dfs=collect(values(grData))
    num=length(dfs)
    dunnMin=-1#类间最小距离
    dunnMax=-1#类内最大距离
    for i in 1:num-1 #计算类间最大距离
        for j in i+1:num
            df1=dfs[i]
            df2=dfs[j]
            for r1 in 1:size(df1,1)
                row1=df1[r1,:]
                for r2 in 1:size(df2,1)
                    row2=df2[r2,:]
                    dis=norm(row1-row2)
                    if dis<dunnMin || dunnMin==-1
                        dunnMin=dis
                    end
                end
            end
        end
    end
    for df in dfs #计算类内最大距离
        rnum=size(df,1)
        for r1 in 1:rnum-1
            row1=df[r1,:]
            for r2 in r1+1:rnum
                row2=df[r2,:]
                dis=norm(row1-row2)
                if dis>dunnMax
                    dunnMax=dis
                end
            end
        end
    end
    #println("dunnMin:$dunnMin,dunnMax:$dunnMax")
    return dunnMin/dunnMax
end



#计算silhouette_score
function evaluateSilhouette(X::Matrix,cls::Vector)
    #silhouette_score
    score=silhouette_score(X, cls, metric="euclidean")
    return score
end

#计算calinski_harabasz_score
function evaluateCalinski(X::Matrix,cls::Vector)
    #silhouette_score
    score=calinski_harabasz_score(X, cls)
    return score
end

#计算davies_bouldin_score
function evaluateDBI(X::Matrix,cls::Vector)
    #silhouette_score
    score=davies_bouldin_score(X, cls)
    return score
end

#end
