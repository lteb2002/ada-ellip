



using RDatasets: dataset

iris = dataset("datasets", "iris")

# ScikitLearn.jl expects arrays, but DataFrames can also be used - see
# the corresponding section of the manual
X = convert(Matrix, iris[[:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]])
y = convert(Vector, iris[:Species])
using ScikitLearn
@sk_import cluster: (KMeans,OPTICS)
model = KMeans(init="k-means++", n_clusters=10)
fit!(model, X)
cls=predict(model,X)

# using ClusterEvaluation
# cls = ClusterEvaluation.doKMeans(X,10)
using Clustering
# cluster X into 20 clusters using K-means
R = kmeans(X', 10; maxiter=200, display=:iter)
@assert nclusters(R) == 10 # verify the number of clusters

a = assignments(R) # get the assignments of points to clusters
c = counts(R) # get the cluster sizes
M = R.centers # get the cluster centers




include("../ClusterEvaluation.jl")
#cls=doKMeans(X,10)
cls1=doDBscan(X)
cls2=doHierarchical(X)
cls3=doAffinity(X)
#cls=doBirch(X,10)
#cls=doOptics(X)

q= evaluateWQ(X,cls)
dunn = evaluateDunn(X,cls)
si = evaluateSilhouette(X,cls)
dbi=evaluateDBI(X,cls)
cal=evaluateCalinski(X,cls)
println("q:$q")
println("dunn:$dunn")
