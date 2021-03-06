% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/xgb.plot.deepness.R
\name{xgb.plot.deepness}
\alias{xgb.plot.deepness}
\title{Plot model trees deepness}
\usage{
xgb.plot.deepness(filename_dump = NULL, model = NULL)
}
\arguments{
\item{filename_dump}{the path to the text file storing the model. Model dump must include the gain per feature and per tree (parameter \code{with.stats = T} in function \code{xgb.dump}).}

\item{model}{dump generated by the \code{xgb.train} function. Avoid the creation of a dump file.}
}
\value{
Two graphs showing the distribution of the model deepness.
}
\description{
Generate a graph to plot the distribution of deepness among trees.
}
\details{
Display both the number of \code{leaf} and the distribution of \code{weighted observations}
by tree deepness level.
The purpose of this function is to help the user to find the best trad-off to set
the \code{max.depth} and \code{min_child_weight} parameters according to the bias / variance trade-off.

See \link{xgb.train} for more information about these parameters.

The graph is made of two parts:

\itemize{
 \item Count: number of leaf per level of deepness;
 \item Weighted cover: noramlized weighted cover per Leaf (weighted number of instances).
}

This function is very inspired from this blog post \url{http://aysent.github.io/2015/11/08/random-forest-leaf-visualization.html}
}
\examples{
data(agaricus.train, package='xgboost')

bst <- xgboost(data = agaricus.train$data, label = agaricus.train$label, max.depth = 15,
                 eta = 1, nthread = 2, nround = 30, objective = "binary:logistic",
                 min_child_weight = 50)

xgb.plot.deepness(model = bst)

}

