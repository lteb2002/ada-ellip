

module RereUtils
rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps())
alldata, allabels = MLDatasets.MNIST.traindata(Float64)
typeof(alldata)
data = reshape(permutedims(alldata[:, :, 1:2500], (3, 1, 2)),
               2500, size(alldata, 1)*size(alldata, 2))
X = rescale(data, dims=1)


end
