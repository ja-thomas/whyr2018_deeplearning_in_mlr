library(reshape2)
library(ggplot2)
library(mlr)
library(mlrMBO)
library(mxnet)
source("../data/read_data.R")
source("https://raw.githubusercontent.com/mlr-org/mlr-extralearner/master/R/RLearner_classif_mxff.R")


train = load_image_file('../data/train-images-idx3-ubyte')
test = load_image_file('../data/t10k-images-idx3-ubyte')
train$y = load_label_file('../data/train-labels-idx1-ubyte')
test$y = load_label_file('../data/t10k-labels-idx1-ubyte')
data = as.data.frame(rbind(cbind(train$x, train$y), cbind(test$x, test$y)))
colnames(data) = c(colnames(data)[-785], "y")

task = makeClassifTask(id = "fmnist", data = data, target = "y")
# Simple CNN in mlr

lenet = makeLearner(cl = "classif.mxff",
  # architecture
  layers = 3,
  conv.layer1 = TRUE,
  num.layer1 = 20,
  conv.kernel1 = c(5, 5),
  act1 = "tanh",
  pool.kernel1 = c(2, 2),
  pool.stride1 = c(2, 2),
  conv.layer2 = TRUE,
  num.layer2 = 50,
  conv.kernel2 = c(5, 5),
  act2 = "tanh",
  pool.kernel2 = c(2, 2),
  pool.stride2 = c(2, 2),
  conv.layer3 = FALSE,
  num.layer3 = 500,
  act3 = "tanh",
  conv.data.shape = c(28, 28),
  # training specs
  optimizer = "sgd",
  learning.rate = 0.01,
  momentum = 0.9,
  num.round = 200,
  eval.metric = mx.metric.accuracy,
  validation.ratio = 0.2,
  epoch.end.callback = mx.callback.early.stop(bad.steps = 5, maximize = TRUE),
  ctx = mx.gpu(), # mx.gpu()
  kvstore = "local")

par.set = makeParamSet(
  makeNumericParam(id = "learning.rate", lower = 0.05, upper = 0.3),
  makeNumericParam(id = "momentum", lower = 0.7, upper = 0.99),
  makeIntegerParam(id = "num.layer1", lower = 2, upper = 7, trafo = function(x) 2^x),
  makeIntegerParam(id = "num.layer2", lower = 2, upper = 7, trafo = function(x) 2^x)
)

ctrl = makeMBOControl()
# set time budget in seconds
ctrl = setMBOControlTermination(ctrl, time.budget = 100)
tune.ctrl = makeTuneControlMBO(mbo.control = ctrl)
result = tuneParams(learner = lrn, task = task, resampling = hout,
  par.set = par.set, control = tune.ctrl, show.info = TRUE, measure = acc)

saveRDS(result, file = "../data/simpleCNN.rds")



data = mx.symbol.Variable('data')
# first conv layer
conv1 = mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=c(2,2), stride=c(2,2))
# second conv layer
conv2 = mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=c(2,2), stride=c(2,2))
# first fullc layer
flatten = mx.symbol.flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
# softmax loss
lenet.sym = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')


lrn = makeLearner("classif.mxff",
  symbol = lenet.sym,
  conv.layer1 = TRUE,
  conv.data.shape = c(28, 28),
  # training specs
  optimizer = "sgd",
  learning.rate = 0.01,
  momentum = 0.9,
  num.round = 200,
  eval.metric = mx.metric.accuracy,
  validation.ratio = 0.2,
  epoch.end.callback = mx.callback.early.stop(bad.steps = 5, maximize = TRUE),
  ctx = mx.gpu(),
  kvstore = "local")

resample(lrn.resnet, task, hout, acc)


# Resnet
source("https://raw.githubusercontent.com/apache/incubator-mxnet/master/example/image-classification/symbol_resnet-28-small.R")

resnet.sym = get_symbol(num_classes = 10)

lrn.resnet = makeLearner(cl = "classif.mxff",
  symbol = resnet.sym,
  conv.data.shape = c(28, 28),
  conv.layer1 = TRUE,
  num.round = 1000,
  eval.metric = mx.metric.accuracy,
  validation.ratio = 0.2,
  epoch.end.callback = mx.callback.early.stop(bad.steps = 5, maximize = TRUE),
  ctx = mx.gpu()
)

resnet.mod = train(lrn.resnet, task)

resample(lrn.resnet, task, hout, acc)
graph.viz(symbol = resnet.sym, direction = "LR", graph.height.px = 150, type = "graph")

par.set = makeParamSet(
  makeNumericParam(id = "learning.rate", lower = 0.05, upper = 0.3),
  makeDiscreteParam(id = "optimizer", c("adam", "sgd", "rmsprop")),
  makeNumericParam(id = "beta1", lower = 0.8, upper = 0.95,
    # params can be dependent, e.g.: beta1 does only apply to "adam", see ?mx.opt.adam
    requires = quote(optimizer == "adam")),
  makeNumericParam(id = "gamma1", lower = 0.8, upper = 0.95,
    # vice versa to adam
    requires = quote(optimizer == "rmsprop"))
)

ctrl = makeMBOControl()
# set max amount of optim iterations
ctrl = setMBOControlTermination(ctrl, iters = 5)
tune.ctrl = makeTuneControlMBO(mbo.control = ctrl)
result.resnet = tuneParams(learner = lrn.resnet, task = task, resampling = cv3,
  par.set = par.set, control = tune.ctrl, show.info = TRUE)

saveRDS(result.resnet, file = "../data/resnet.rds")



# Inception

source("https://raw.githubusercontent.com/apache/incubator-mxnet/master/example/image-classification/symbol_inception-bn-28-small.R")
inc.sym = get_symbol(10)

lrn.inc = makeLearner(cl = "classif.mxff",
  symbol = inc.sym,
  conv.data.shape = c(28, 28),
  conv.layer1 = TRUE,
  num.round = 1000,
  eval.metric = mx.metric.accuracy,
  validation.ratio = 0.2,
  epoch.end.callback = mx.callback.early.stop(bad.steps = 10, maximize = TRUE),
  ctx = mx.gpu(),
  optimizer = "adam"
)

resample(lrn.resnet, task, hout, acc)
